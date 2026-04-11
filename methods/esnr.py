from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    if labels.size == 0:
        raise ValueError("labels must be non-empty")

    return labels


def build_label_indicator(labels: np.ndarray) -> np.ndarray:
    """
    Build label indicator matrix L of shape (m, n), where:
      - m = number of unique labels
      - n = number of nodes

    Row i corresponds to the i-th sorted unique class label.
    Column j corresponds to node j.
    """
    labels = _validate_labels(labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    m = len(classes)
    n = len(labels)

    L = np.zeros((m, n), dtype=float)
    L[inverse, np.arange(n)] = 1.0
    return L


def adjacency_matrix_from_graph(G: nx.Graph) -> np.ndarray:
    """
    Convert NetworkX graph to a dense adjacency matrix using node order 0..n-1.
    """
    n = G.number_of_nodes()
    expected_nodes = list(range(n))
    actual_nodes = sorted(G.nodes())

    if actual_nodes != expected_nodes:
        raise ValueError(
            "Graph nodes must be contiguous integers 0..n-1 for ESNR."
        )

    return nx.to_numpy_array(G, nodelist=expected_nodes, dtype=float)


def build_aggregated_matrix(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the label-aggregated matrix C = L A.
    """
    labels = _validate_labels(labels)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")

    if A.shape[0] != len(labels):
        raise ValueError(
            f"Adjacency size {A.shape[0]} does not match label length {len(labels)}"
        )

    L = build_label_indicator(labels)
    return L @ A


def biwhiten(
    C: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Biwhiten matrix C using Sinkhorn-style left/right scaling.

    Returns
    -------
    C_prime : np.ndarray
        Biwhitened matrix.
    meta : dict
        Diagnostics including iterations and convergence status.
    """
    if C.ndim != 2:
        raise ValueError(f"C must be 2D, got shape {C.shape}")

    m, n = C.shape
    if m == 0 or n == 0:
        raise ValueError(f"C must be non-empty, got shape {C.shape}")

    x = np.ones(m, dtype=float)
    y = np.ones(n, dtype=float)

    converged = False
    iterations = 0

    for it in range(max_iter):
        # y <- m / (C^T x)
        denom_y = C.T @ x
        denom_y = np.maximum(denom_y, epsilon)
        y = m / denom_y

        # x <- n / (C y)
        denom_x = C @ y
        denom_x = np.maximum(denom_x, epsilon)
        x = n / denom_x

        # paper-style residuals
        row_residual = np.max(np.abs(x * (C @ y) - n))
        col_residual = np.max(np.abs(y * (C.T @ x) - m))

        iterations = it + 1
        if row_residual <= tol and col_residual <= tol:
            converged = True
            break

    C_prime = (np.sqrt(x)[:, None]) * C * (np.sqrt(y)[None, :])

    return C_prime, {
        "iterations": iterations,
        "converged": converged,
        "row_residual": float(row_residual),
        "col_residual": float(col_residual),
    }


def compute_esnr_from_C(
    C: np.ndarray,
    *,
    stability_epsilon: float = 1e-9,
    biwhiten_tol: float = 1e-6,
    biwhiten_max_iter: int = 1000,
) -> dict[str, Any]:
    """
    Compute ESNR from aggregated matrix C.

    Pipeline:
      1. add epsilon for strict positivity
      2. biwhiten
      3. mean-center
      4. compute singular values
      5. threshold at sqrt(m) + sqrt(n)
      6. average normalized excess above threshold
    """
    if C.ndim != 2:
        raise ValueError(f"C must be 2D, got shape {C.shape}")

    m, n = C.shape
    if m == 0 or n == 0:
        raise ValueError(f"C must be non-empty, got shape {C.shape}")

    C_stable = C.astype(float, copy=True) + stability_epsilon

    C_prime, meta = biwhiten(
        C_stable,
        tol=biwhiten_tol,
        max_iter=biwhiten_max_iter,
    )

    C_centered = C_prime - np.mean(C_prime)

    singular_values = np.linalg.svd(C_centered, compute_uv=False)
    alpha = float(np.sqrt(m) + np.sqrt(n))

    numerator = np.maximum(singular_values - alpha, 0.0)
    contributions = np.divide(
        numerator,
        singular_values,
        out=np.zeros_like(singular_values),
        where=singular_values > 0,
    )

    esnr = float(np.mean(contributions))
    n_outliers = int(np.sum(singular_values > alpha))
    outlier_mass = float(np.sum(numerator))

    return {
        "esnr": esnr,
        "singular_values": singular_values,
        "threshold": alpha,
        "n_classes": m,
        "n_nodes": n,
        "n_outlier_singular_values": n_outliers,
        "outlier_mass": outlier_mass,
        **meta,
    }


def compute_esnr_from_graph(
    G: nx.Graph,
    labels: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Full ESNR pipeline from graph + labels.
    """
    labels = _validate_labels(labels)

    if G.number_of_nodes() != len(labels):
        raise ValueError(
            f"Graph node count {G.number_of_nodes()} does not match label length {len(labels)}"
        )

    A = adjacency_matrix_from_graph(G)
    C = build_aggregated_matrix(A, labels)
    return compute_esnr_from_C(C, **kwargs)