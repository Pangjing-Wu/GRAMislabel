from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import rankdata

from .kernels import TrajectoryKernels


@dataclass
class KernelWeightFit:
    weights: np.ndarray
    objective: float | None
    iterations: int
    converged: bool
    learned: bool
    reason: str


def _weighted_features(kernels: TrajectoryKernels, weights: np.ndarray) -> np.ndarray:
    blocks = []
    for weight, features in zip(
        weights[:2], (kernels.margin_features, kernels.gradient_features)
    ):
        if weight > 0.0 and features.shape[1]:
            blocks.append(math.sqrt(float(weight)) * features)
    if not blocks:
        return np.empty((len(kernels.margin_features), 0), dtype=np.float64)
    return np.column_stack(blocks)


class GaussianGraphGP:

    def __init__(
        self,
        kernels: TrajectoryKernels,
        prior_mean: np.ndarray,
        weights: np.ndarray,
        observation_noise_variance: float,
    ):
        self.kernels = kernels
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.observation_noise_variance = float(observation_noise_variance)
        self._covariance_features = None
        self._observed_ids = None
        if self.weights.shape != (3,) or np.any(self.weights < 0.0):
            raise ValueError("kernel weights must be three non-negative values")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("kernel weights must sum to one")
        if self.observation_noise_variance <= 0.0:
            raise ValueError("observation noise variance must be positive")

    def posterior(self, observed_ids: np.ndarray, values: np.ndarray):
        observed_ids = np.asarray(observed_ids, dtype=np.int64)
        targets = 2.0 * np.asarray(values, dtype=np.float64) - 1.0
        features = _weighted_features(self.kernels, self.weights)
        identity_weight = float(self.weights[2])
        total_noise = identity_weight + self.observation_noise_variance
        rank = features.shape[1]
        if len(observed_ids) and rank:
            observed = features[observed_ids]
            precision = np.eye(rank) + observed.T @ observed / total_noise
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
            residual = targets - self.prior_mean[observed_ids]
            projected = observed.T @ residual
            coefficient = linalg.cho_solve(
                (factor, True), projected / total_noise, check_finite=False
            )
        else:
            observed = np.empty((len(observed_ids), rank), dtype=np.float64)
            factor = np.eye(rank)
            residual = targets - self.prior_mean[observed_ids]
            coefficient = np.zeros(rank, dtype=np.float64)
        latent_mean = self.prior_mean + features @ coefficient
        if len(observed_ids) and identity_weight:
            independent_mean = (
                identity_weight * (residual - observed @ coefficient) / total_noise
            )
            latent_mean[observed_ids] += independent_mean
        if rank:
            latent = linalg.solve_triangular(
                factor, features.T, lower=True, check_finite=False
            ).T
            graph_variance = np.einsum("ij,ij->i", latent, latent)
        else:
            latent = features
            graph_variance = np.zeros(len(self.prior_mean), dtype=np.float64)
        self._covariance_features = latent
        self._observed_ids = observed_ids.copy()
        variance = graph_variance + identity_weight
        if len(observed_ids) and identity_weight:
            shrinkage = self.observation_noise_variance / total_noise
            variance[observed_ids] = (
                np.square(shrinkage) * graph_variance[observed_ids]
                + identity_weight * shrinkage
            )
        probability = ndtr(
            latent_mean
            / np.sqrt(np.maximum(variance + self.observation_noise_variance, 1e-12))
        )
        return (probability, variance, latent_mean)

    def informative_value(self, unverified_ids: np.ndarray) -> np.ndarray:
        if self._covariance_features is None:
            raise RuntimeError("compute the GP posterior before informative value")
        unverified_ids = np.asarray(unverified_ids, dtype=np.int64)
        if unverified_ids.ndim != 1 or len(np.unique(unverified_ids)) != len(
            unverified_ids
        ):
            raise ValueError("unverified indices must be a one-dimensional unique set")
        if np.any(unverified_ids < 0) or np.any(unverified_ids >= len(self.prior_mean)):
            raise ValueError("unverified index is outside the GP population")
        if np.intersect1d(unverified_ids, self._observed_ids).size:
            raise ValueError("informative value pool must exclude verified samples")
        result = np.zeros(len(self.prior_mean), dtype=np.float64)
        if not len(unverified_ids):
            return result
        latent = self._covariance_features[unverified_ids]
        gram = latent.T @ latent
        graph_variance = np.einsum("ij,ij->i", latent, latent)
        identity_weight = float(self.weights[2])
        column_norm_squared = (
            np.einsum("ij,ij->i", latent @ gram, latent)
            + 2.0 * identity_weight * graph_variance
            + identity_weight**2
        )
        variance = graph_variance + identity_weight
        result[unverified_ids] = np.maximum(column_norm_squared, 0.0) / (
            len(unverified_ids) * (variance + self.observation_noise_variance)
        )
        return result


def _simplex_weights(theta: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    logits = np.r_[np.asarray(theta, dtype=np.float64), 0.0]
    logits -= logits.max()
    active_weights = np.exp(logits)
    active_weights /= active_weights.sum()
    weights = np.zeros(3, dtype=np.float64)
    weights[np.asarray(active, dtype=np.int64)] = active_weights
    return weights


def _theta_from_weights(weights: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    active_weights = np.maximum(
        np.asarray(weights, dtype=np.float64)[np.asarray(active)], 1e-12
    )
    active_weights /= active_weights.sum()
    return np.log(active_weights[:-1]) - math.log(active_weights[-1])


def learn_kernel_weights(
    kernels: TrajectoryKernels,
    prior_mean: np.ndarray,
    observed_ids: np.ndarray,
    values: np.ndarray,
    previous_weights: np.ndarray,
    config: dict,
) -> KernelWeightFit:
    active_names = tuple(config["active_kernel_components"])
    component_index = {"margin": 0, "gradient": 1, "identity": 2}
    active = tuple((component_index[name] for name in active_names))
    initial = np.asarray(config["initial_kernel_weights"], dtype=np.float64)
    observed_ids = np.asarray(observed_ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if len(observed_ids) < config["minimum_queried_for_weight_learning"]:
        return KernelWeightFit(
            previous_weights, None, 0, True, False, "insufficient_queries"
        )
    if (
        config["require_both_status_classes_for_weight_learning"]
        and len(np.unique(values)) < 2
    ):
        return KernelWeightFit(
            previous_weights, None, 0, True, False, "single_status_class"
        )
    targets = 2.0 * values - 1.0
    residual = targets - np.asarray(prior_mean, dtype=np.float64)[observed_ids]
    margin = kernels.margin_features[observed_ids]
    gradient = kernels.gradient_features[observed_ids]
    margin_gram = margin.T @ margin
    gradient_gram = gradient.T @ gradient
    cross_gram = margin.T @ gradient
    margin_projection = margin.T @ residual
    gradient_projection = gradient.T @ residual
    residual_square = float(residual @ residual)
    reference = initial
    theta = _theta_from_weights(previous_weights, active)

    def objective(candidate: np.ndarray) -> float:
        weights = _simplex_weights(candidate, active)
        margin_weight, gradient_weight, identity_weight = weights
        diagonal = identity_weight + config["observation_noise_variance"]
        blocks = []
        projections = []
        if margin_weight > 0.0 and margin.shape[1]:
            blocks.append("margin")
            projections.append(math.sqrt(margin_weight) * margin_projection)
        if gradient_weight > 0.0 and gradient.shape[1]:
            blocks.append("gradient")
            projections.append(math.sqrt(gradient_weight) * gradient_projection)
        if blocks == ["margin", "gradient"]:
            cross = math.sqrt(margin_weight * gradient_weight) * cross_gram
            gram = np.block(
                [
                    [margin_weight * margin_gram, cross],
                    [cross.T, gradient_weight * gradient_gram],
                ]
            )
        elif blocks == ["margin"]:
            gram = margin_weight * margin_gram
        elif blocks == ["gradient"]:
            gram = gradient_weight * gradient_gram
        else:
            gram = np.empty((0, 0), dtype=np.float64)
        if len(gram):
            precision = np.eye(len(gram)) + gram / diagonal
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
            projected = np.concatenate(projections)
            correction = float(
                projected
                @ linalg.cho_solve((factor, True), projected, check_finite=False)
            )
            quadratic = residual_square / diagonal - correction / (diagonal * diagonal)
            log_determinant = len(residual) * math.log(diagonal) + 2.0 * float(
                np.log(np.diag(factor)).sum()
            )
        else:
            quadratic = residual_square / diagonal
            log_determinant = len(residual) * math.log(diagonal)
        negative_log_likelihood = 0.5 * (quadratic + log_determinant) / len(residual)
        penalty = config["weight_regularization"] * float(
            np.square(weights - reference).sum()
        )
        return negative_log_likelihood + penalty

    initial_objective = objective(theta)
    result = minimize(
        objective, theta, method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-10}
    )
    candidate_weights = _simplex_weights(result.x, active)
    candidate_objective = float(result.fun)
    if not np.isfinite(candidate_objective) or candidate_objective > initial_objective:
        candidate_weights = np.asarray(previous_weights, dtype=np.float64)
        candidate_objective = initial_objective
    return KernelWeightFit(
        candidate_weights,
        candidate_objective,
        int(result.nit),
        bool(result.success),
        True,
        str(result.message),
    )


def prior_probability(diagnostics: np.ndarray) -> np.ndarray:
    combined = diagnostics.mean(axis=1)
    probability = rankdata(combined, method="average") / (len(combined) + 1.0)
    return np.clip(probability, 0.0001, 1.0 - 0.0001)


def gaussian_prior_mean(diagnostics: np.ndarray) -> np.ndarray:
    return 2.0 * prior_probability(diagnostics) - 1.0
