from __future__ import annotations

import numpy as np

from .inference import GaussianGraphGP, gaussian_prior_mean, learn_kernel_weights
from .kernels import TrajectoryKernels, build_trajectory_kernel, faiss_neighbors


def otsu_threshold(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or (not np.all(np.isfinite(values))):
        raise ValueError("Otsu requires a non-empty finite score vector")
    values = np.sort(values)
    if values[0] == values[-1]:
        return float(values[0])
    scale = max(abs(values[0]), abs(values[-1]))
    scaled = values / scale
    scaled -= scaled[0]
    cut = np.flatnonzero(values[:-1] < values[1:])
    cumulative = np.cumsum(scaled)
    left_count = cut + 1
    objective = cumulative[cut] ** 2 / left_count + (
        cumulative[-1] - cumulative[cut]
    ) ** 2 / (len(values) - left_count)
    index = cut[int(np.argmax(objective))]
    threshold = float(values[index] / 2.0 + values[index + 1] / 2.0)
    return float(values[index]) if threshold >= values[index + 1] else threshold


class GRAM:
    def __init__(self, artifacts, noisy_labels: np.ndarray, config: dict, seed: int):
        self.labels = np.asarray(noisy_labels, dtype=np.int64).copy()
        self.config = dict(config)
        self.seed = seed
        self.num_classes = artifacts.probability.shape[1]
        count = len(self.labels)
        neighbors, _ = faiss_neighbors(
            artifacts.embedding,
            min(config["diagnostic_neighbors"], count - 1),
            normalize=True,
        )
        diagnostics = np.column_stack(
            (
                1.0 - artifacts.probability[np.arange(count), self.labels],
                artifacts.early_loss,
                1.0 - (self.labels[neighbors] == self.labels[:, None]).mean(axis=1),
            )
        )
        self.prior_mean = gaussian_prior_mean(diagnostics)
        self.kernels = TrajectoryKernels(
            build_trajectory_kernel(
                artifacts.margin_trajectory,
                rank=config["margin_graph_rank"],
                k=config["graph_neighbors"],
            ).features,
            build_trajectory_kernel(
                artifacts.gradient_trajectory,
                rank=config["gradient_graph_rank"],
                k=config["graph_neighbors"],
            ).features,
        )
        self.weights = np.asarray(config["initial_kernel_weights"], dtype=np.float64)
        self.verified = np.zeros(count, dtype=bool)
        self.status = np.zeros(count, dtype=np.int64)
        self.corrected_labels = self.labels.copy()
        self.batch = 0
        self._posterior = None
        self._gp = None

    def posterior(self):
        if self._posterior is None:
            observed = np.flatnonzero(self.verified)
            if len(observed):
                fit = learn_kernel_weights(
                    self.kernels,
                    self.prior_mean,
                    observed,
                    self.status[observed],
                    self.weights,
                    self.config,
                )
                self.weights = fit.weights.copy()
            self._gp = GaussianGraphGP(
                self.kernels,
                self.prior_mean,
                self.weights,
                self.config["observation_noise_variance"],
            )
            self._posterior = self._gp.posterior(observed, self.status[observed])
        return self._posterior

    def select(self, count: int):
        candidates = np.flatnonzero(~self.verified)
        if count <= 0 or count > len(candidates):
            raise ValueError(
                "query count must be positive and no larger than the unverified pool"
            )
        probability, _, _ = self.posterior()
        informative = self._gp.informative_value(candidates)
        acquisition = probability + self.config["iv_beta"] * informative
        tie = np.random.default_rng(self.seed + self.batch).random(len(candidates))
        selected = candidates[np.lexsort((tie, -acquisition[candidates]))[:count]]
        return selected, probability, informative, acquisition

    def observe(self, ids: np.ndarray, expert_labels: np.ndarray):
        ids = np.asarray(ids, dtype=np.int64)
        expert_labels = np.asarray(expert_labels, dtype=np.int64)
        if ids.ndim != 1 or not ids.size or expert_labels.shape != ids.shape:
            raise ValueError(
                "oracle indices and labels must be non-empty aligned vectors"
            )
        if (
            np.any(ids < 0)
            or np.any(ids >= len(self.labels))
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("oracle indices must be unique and in range")
        if np.any(self.verified[ids]):
            raise ValueError("samples cannot be verified twice")
        if np.any(expert_labels < 0) or np.any(expert_labels >= self.num_classes):
            raise ValueError("oracle labels are outside the class range")
        self.verified[ids] = True
        self.status[ids] = self.labels[ids] != expert_labels
        self.corrected_labels[ids] = expert_labels
        self.batch += 1
        self._posterior = None
        self._gp = None

    def predict(self):
        probability, _, _ = self.posterior()
        threshold = otsu_threshold(probability)
        predicted = probability > threshold
        predicted[self.verified] = self.status[self.verified].astype(bool)
        return probability.copy(), predicted, threshold
