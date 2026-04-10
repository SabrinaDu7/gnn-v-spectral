from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.real_world.loaders import RealWorldGraph
from data.real_world.characterize import feature_signal_properties


ROOT = Path("data/cache/realworld")
META_CSV = ROOT / "metadata" / "graph_index_realworld.csv"


def load_cached_realworld_graph(row: pd.Series) -> RealWorldGraph:
    edge_path = ROOT / row["edge_path"]
    label_path = ROOT / row["label_path"]

    edges = pd.read_csv(edge_path)
    labels = np.load(label_path)

    features = None
    feature_path = row.get("feature_path", None)
    if pd.notna(feature_path) and str(feature_path).strip() != "":
        features = np.load(ROOT / feature_path)

    metadata = {
        "graph_id": row["graph_id"],
        "dataset": row["dataset"],
        "n_nodes": int(row["n_nodes"]),
        "n_edges": int(row["num_edges"]),
        "num_classes": int(row["num_classes"]),
        "has_features": bool(row["has_features"]),
        "feature_dim": None if features is None else int(features.shape[1]),
    }

    return RealWorldGraph(
        graph_id=row["graph_id"],
        dataset=row["dataset"],
        edges=edges,
        labels=labels,
        features=features,
        metadata=metadata,
    )


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


def main() -> None:
    df = pd.read_csv(META_CSV)
    df = ensure_fs_columns(df)

    for idx, row in df[df["graph_id"] == "ppi"].iterrows():
        graph_id = row["graph_id"]

        print(f"Processing {graph_id}...")

        graph = load_cached_realworld_graph(row)
        fs = feature_signal_properties(
            graph,
            n_splits=5,
            test_size=0.3,
            n_null_trials=20,
            random_state=0,
        )

        for key, value in fs.items():
            df.at[idx, key] = value

        print(
            f"  FS norm: {fs['feature_signal_norm']}, "
            f"macro-F1: {fs['feature_macro_f1']}"
        )

    df.to_csv(META_CSV, index=False)
    print(f"Updated metadata written to {META_CSV}")


if __name__ == "__main__":
    main()