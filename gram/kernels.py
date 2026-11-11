from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def faiss_neighbors(values: np.ndarray, k: int, *, normalize: bool):
    import faiss

    matrix = np.ascontiguousarray(values, dtype=np.float32).copy()
    if normalize:
        faiss.normalize_L2(matrix)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = max(64, 4 * (k + 1))
    index.add(matrix)
    distance, indices = index.search(matrix, min(k + 2, len(matrix)))
    selected_ids = np.empty((len(matrix), k), dtype=np.int64)
    selected_distance = np.empty((len(matrix), k), dtype=np.float64)
    for row in range(len(matrix)):
        keep = indices[row] != row
        ids = indices[row][keep][:k]
        distances = distance[row][keep][:k]
        if len(ids) != k:
            raise RuntimeError("nearest-neighbor search returned too few neighbors")
        selected_ids[row] = ids
        selected_distance[row] = distances
    return (selected_ids, selected_distance)


@dataclass
class PriorGraph:
    neighbors: np.ndarray
    features: np.ndarray


@dataclass
class TrajectoryKernels:
    margin_features: np.ndarray
    gradient_features: np.ndarray


def _trajectory_knn_graph(values: np.ndarray, k: int = 50):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or (not np.all(np.isfinite(values))):
        raise ValueError("trajectory values must be a finite N x T matrix")
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)
    k = min(k, len(values) - 1)
    neighbors, squared_distance = faiss_neighbors(standardized, k, normalize=False)
    bandwidth_squared = np.maximum(squared_distance[:, -1], np.finfo(np.float64).eps)
    row = np.repeat(np.arange(len(values)), k)
    column = neighbors.reshape(-1)
    denominator = np.sqrt(bandwidth_squared[row] * bandwidth_squared[column])
    weight = np.exp(
        -0.5
        * squared_distance.reshape(-1)
        / np.maximum(denominator, np.finfo(np.float64).eps)
    )
    adjacency = sparse.csr_matrix((weight, (row, column)), shape=(len(values),) * 2)
    return (adjacency.maximum(adjacency.T).tocsr(), neighbors)


def _heat_kernel_features(adjacency: sparse.csr_matrix, rank: int) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized_adjacency = inverse @ adjacency @ inverse
    sample_count = adjacency.shape[0]
    active_rank = min(rank, sample_count - 1)
    if active_rank <= 0:
        raise ValueError("a graph kernel requires at least two samples")
    if sample_count <= rank + 1:
        values, vectors = np.linalg.eigh(normalized_adjacency.toarray())
        values, vectors = (values[-active_rank:], vectors[:, -active_rank:])
    else:
        values, vectors = eigsh(normalized_adjacency, k=active_rank, which="LA")
    order = np.argsort(values)[::-1]
    laplacian_values = np.maximum(1.0 - values[order], 0.0)
    features = vectors[:, order] * np.exp(-0.5 * laplacian_values)[None, :]
    trace = float(np.einsum("ij,ij->", features, features))
    features *= math.sqrt(sample_count / max(trace, 1e-12))
    return features


def build_trajectory_kernel(
    trajectory: np.ndarray, rank: int = 64, k: int = 50
) -> PriorGraph:
    adjacency, neighbors = _trajectory_knn_graph(trajectory, k)
    features = _heat_kernel_features(adjacency, rank)
    return PriorGraph(neighbors=neighbors, features=features)
