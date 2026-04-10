from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.real_world.loaders import load_real_world_graph
from data.real_world.characterize import (
    extract_largest_connected_component,
    basic_graph_properties,
)

ROOT = Path("data/cache/realworld")
META_CSV = ROOT / "metadata" / "graph_index_realworld.csv"


def main() -> None:
    graph = load_real_world_graph(
        "ppi_adapted",
        raw_dir="data/raw/ppi",
        graph_index=0,
        label_index=0,
    )
    graph = extract_largest_connected_component(graph)

    # Rename to final stable project name
    graph.graph_id = "ppi"
    graph.dataset = "ppi"

    props = basic_graph_properties(graph)

    # Save files in existing realworld layout
    out_dir = ROOT / "ppi"
    (out_dir / "edges").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)

    graph.edges.to_csv(out_dir / "edges" / "ppi_edges.csv", index=False)
    np.save(out_dir / "labels" / "ppi_labels.npy", graph.labels)
    np.save(out_dir / "features" / "ppi_features.npy", graph.features)

    # Load existing metadata table
    df = pd.read_csv(META_CSV)

    row = {
        "graph_id": "ppi",
        "dataset": "ppi",
        "variant": "full_gcc",
        "task": "node_classification",
        "label_name": "ppi_label_0",
        "edge_path": "ppi/edges/ppi_edges.csv",
        "label_path": "ppi/labels/ppi_labels.npy",
        "feature_path": "ppi/features/ppi_features.npy",
        "has_features": True,
        "feature_type": "biological_node_features",
        "feature_dim": 50,
        "n_nodes": int(props["n_nodes"]),
        "num_edges": int(props["n_edges"]),
        "avg_degree": float(props["avg_degree"]),
        "density": float(props["density"]),
        "num_connected_components": int(props["num_connected_components"]),
        "largest_cc_size": int(props["largest_component_size"]),
        "largest_cc_fraction": float(props["largest_component_fraction"]),
        "num_classes": int(props["num_classes"]),
        "notes": (
            "Adapted single-graph transductive PPI dataset. "
            "Selected graph_index=0 and label_index=0 from PyG PPI, then took GCC."
        ),
        "esnr": float(props["esnr"]),
        "esnr_n_outliers": int(props["esnr_n_outliers"]),
        "esnr_outlier_mass": float(props["esnr_outlier_mass"]),
        "esnr_converged": bool(props["esnr_converged"]),
        "esnr_iterations": int(props["esnr_iterations"]),
    }

    # Replace existing row if present, otherwise append
    if (df["graph_id"] == "ppi").any():
        df.loc[df["graph_id"] == "ppi", :] = pd.DataFrame([row]).values
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(META_CSV, index=False)

    print("Saved ppi files and updated graph_index_realworld.csv")
    print(row)


if __name__ == "__main__":
    main()