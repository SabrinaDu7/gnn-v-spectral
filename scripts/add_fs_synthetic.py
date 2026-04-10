from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from methods.feature_signal import compute_feature_signal


ROOT = Path("data/cache/synthetic80")
META_CSV = ROOT / "metadata" / "feature_informativeness_experiment_table.csv"


def ensure_fs_columns(df: pd.DataFrame) -> pd.DataFrame:
    fs_columns = [
        "feature_macro_f1",
        "feature_macro_f1_std",
        "feature_accuracy",
        "feature_accuracy_std",
        "feature_null_macro_f1_mean",
        "feature_null_macro_f1_std",
        "feature_signal_raw",
        "feature_signal_raw_std",
        "feature_signal_norm",
        "feature_signal_norm_std",
        "feature_n_splits",
        "feature_n_null_trials",
    ]
    for col in fs_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def compute_feature_signal_aggregated(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 2,
    test_size: float = 0.3,
    n_null_trials: int = 3,
    random_state: int = 0,
) -> dict[str, float | int]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature rows ({X.shape[0]}) do not match label length ({y.shape[0]})"
        )

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Need at least 2 classes for FS computation")

    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError(
            f"Need at least 2 examples in every class, got min class count {min_class_count}"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    split_results: list[dict[str, float]] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        result = compute_feature_signal(
            X,
            y,
            train_idx,
            test_idx,
            n_null_trials=n_null_trials,
            random_state=random_state + split_id,
        )
        split_results.append(result)

    def mean_of(key: str) -> float:
        return float(np.mean([r[key] for r in split_results]))

    def std_of(key: str) -> float:
        return float(np.std([r[key] for r in split_results]))

    return {
        "feature_macro_f1": mean_of("feature_macro_f1"),
        "feature_macro_f1_std": std_of("feature_macro_f1"),
        "feature_accuracy": mean_of("feature_accuracy"),
        "feature_accuracy_std": std_of("feature_accuracy"),
        "feature_null_macro_f1_mean": mean_of("feature_null_macro_f1_mean"),
        "feature_null_macro_f1_std": mean_of("feature_null_macro_f1_mean"),
        "feature_signal_raw": mean_of("feature_signal_raw"),
        "feature_signal_raw_std": std_of("feature_signal_raw"),
        "feature_signal_norm": mean_of("feature_signal_norm"),
        "feature_signal_norm_std": std_of("feature_signal_norm"),
        "feature_n_splits": int(n_splits),
        "feature_n_null_trials": int(n_null_trials),
    }


def main() -> None:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_CSV}")

    df = pd.read_csv(META_CSV)
    df = ensure_fs_columns(df)

    n_rows = len(df)
    print(f"Loaded {n_rows} rows from {META_CSV}")

    for idx, row in df.iterrows():
        graph_id = row["graph_id"]
        feature_code = row["feature_informativeness_code"]
        feature_frac = row["feature_informativeness_frac"]

        feature_path = ROOT / row["feature_path"]
        label_path = ROOT / row["label_path"]

        print(
            f"[{idx + 1}/{n_rows}] "
            f"{graph_id} | feature_info={feature_code} ({feature_frac:.1f})"
        )

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        X = np.load(feature_path)
        y = np.load(label_path)

        fs = compute_feature_signal_aggregated(
            X,
            y,
            n_splits=2,
            test_size=0.3,
            n_null_trials=3,
            random_state=0,
        )

        for key, value in fs.items():
            df.at[idx, key] = value

        print(
            f"  FS norm: {fs['feature_signal_norm']:.4f}, "
            f"macro-F1: {fs['feature_macro_f1']:.4f}"
        )

    df.to_csv(META_CSV, index=False)
    print(f"\nUpdated metadata written to {META_CSV}")


if __name__ == "__main__":
    main()