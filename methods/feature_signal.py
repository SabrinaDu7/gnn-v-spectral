from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    if labels.size == 0:
        raise ValueError("labels must be non-empty")

    return labels


def _validate_features(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if X.shape[0] != len(labels):
        raise ValueError(
            f"Number of feature rows {X.shape[0]} does not match label length {len(labels)}"
        )

    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature column")

    return X


def _validate_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    if train_idx.ndim != 1:
        raise ValueError(f"train_idx must be 1D, got shape {train_idx.shape}")

    if test_idx.ndim != 1:
        raise ValueError(f"test_idx must be 1D, got shape {test_idx.shape}")

    if train_idx.size == 0:
        raise ValueError("train_idx must be non-empty")

    if test_idx.size == 0:
        raise ValueError("test_idx must be non-empty")

    if np.any(train_idx < 0) or np.any(train_idx >= n_nodes):
        raise ValueError("train_idx contains out-of-range values")

    if np.any(test_idx < 0) or np.any(test_idx >= n_nodes):
        raise ValueError("test_idx contains out-of-range values")

    return train_idx, test_idx


def _make_logistic_regression(random_state: int = 0) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )


def compute_feature_only_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Train a feature-only logistic regression classifier and evaluate on held-out nodes.
    """
    labels = _validate_labels(labels)
    X = _validate_features(X, labels)
    train_idx, test_idx = _validate_indices(train_idx, test_idx, len(labels))

    clf = _make_logistic_regression(random_state=random_state)
    clf.fit(X[train_idx], labels[train_idx])

    y_pred = clf.predict(X[test_idx])

    macro_f1 = float(f1_score(labels[test_idx], y_pred, average="macro"))
    accuracy = float(accuracy_score(labels[test_idx], y_pred))

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def compute_shuffled_label_null(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_trials: int = 20,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Estimate a null baseline by shuffling the training labels and recomputing
    feature-only macro-F1 multiple times.
    """
    labels = _validate_labels(labels)
    X = _validate_features(X, labels)
    train_idx, test_idx = _validate_indices(train_idx, test_idx, len(labels))

    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    rng = np.random.default_rng(random_state)
    null_scores = []

    for trial in range(n_trials):
        shuffled_train_labels = labels[train_idx].copy()
        rng.shuffle(shuffled_train_labels)

        clf = _make_logistic_regression(random_state=random_state + trial)
        clf.fit(X[train_idx], shuffled_train_labels)

        y_pred = clf.predict(X[test_idx])
        score = float(f1_score(labels[test_idx], y_pred, average="macro"))
        null_scores.append(score)

    null_scores = np.asarray(null_scores, dtype=float)

    return {
        "null_macro_f1_scores": null_scores,
        "null_macro_f1_mean": float(np.mean(null_scores)),
        "null_macro_f1_std": float(np.std(null_scores)),
        "n_null_trials": int(n_trials),
    }


def compute_feature_signal(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_null_trials: int = 20,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Compute feature signal from node features alone.

    Pipeline:
      1. train feature-only logistic regression
      2. evaluate held-out macro-F1 and accuracy
      3. estimate shuffled-label null macro-F1
      4. compute raw improvement over null
      5. compute normalized improvement over null
    """
    metrics = compute_feature_only_metrics(
        X,
        labels,
        train_idx,
        test_idx,
        random_state=random_state,
    )

    null = compute_shuffled_label_null(
        X,
        labels,
        train_idx,
        test_idx,
        n_trials=n_null_trials,
        random_state=random_state,
    )

    macro_f1 = metrics["macro_f1"]
    null_mean = null["null_macro_f1_mean"]

    raw_signal = float(macro_f1 - null_mean)

    denom = 1.0 - null_mean
    normalized_signal = float(raw_signal / denom) if denom > 0 else 0.0

    return {
        "feature_macro_f1": macro_f1,
        "feature_accuracy": metrics["accuracy"],
        "feature_null_macro_f1_mean": null_mean,
        "feature_null_macro_f1_std": null["null_macro_f1_std"],
        "feature_signal_raw": raw_signal,
        "feature_signal_norm": normalized_signal,
        "n_null_trials": null["n_null_trials"],
    }