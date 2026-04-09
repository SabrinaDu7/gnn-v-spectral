from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from methods.esnr import compute_esnr_from_graph


ROOT = Path("data/cache/synthetic80")


def load_graph(edge_path: Path, label_path: Path) -> tuple[nx.Graph, np.ndarray]:
    edges = pd.read_csv(edge_path)
    labels = np.load(label_path)

    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    return G, labels


def build_graph_id(base_idx: int, noise_code: str, noise_type: str, family: str) -> str:
    if noise_type == "clean":
        return f"graph{base_idx:03d}_000_clean_{family}"
    return f"graph{base_idx:03d}_{noise_code}_{noise_type}_{family}"


def resolve_paths(base_idx: int, noise_code: str, noise_type: str, family: str) -> tuple[Path, Path]:
    graph_id = build_graph_id(base_idx, noise_code, noise_type, family)
    edge_path = ROOT / family / noise_type / "edges" / f"{graph_id}.csv"
    label_path = ROOT / family / noise_type / "labels" / f"{graph_id}_labels.npy"
    return edge_path, label_path


def inspect_one(base_idx: int, family: str) -> pd.DataFrame:
    rows = []

    settings = [("clean", "000")]
    settings += [("random", code) for code in ["005", "010", "015", "020", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070", "075", "080"]]
    settings += [("targeted_betweenness", code) for code in ["005", "010", "015", "020", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070", "075", "080"]]

    for noise_type, noise_code in settings:
        edge_path, label_path = resolve_paths(base_idx, noise_code, noise_type, family)

        if not edge_path.exists() or not label_path.exists():
            print(f"Skipping missing files: {edge_path.name}")
            continue

        G, labels = load_graph(edge_path, label_path)
        result = compute_esnr_from_graph(G, labels)

        svals = result["singular_values"]
        row = {
            "family": family,
            "base_idx": base_idx,
            "noise_type": noise_type,
            "noise_code": noise_code,
            "noise_frac": int(noise_code) / 100.0,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "esnr": result["esnr"],
            "n_outliers": result["n_outlier_singular_values"],
            "outlier_mass": result["outlier_mass"],
            "converged": result["converged"],
            "iterations": result["iterations"],
            "threshold": result["threshold"],
            "sv1": svals[0] if len(svals) > 0 else np.nan,
            "sv2": svals[1] if len(svals) > 1 else np.nan,
            "sv3": svals[2] if len(svals) > 2 else np.nan,
            "sv4": svals[3] if len(svals) > 3 else np.nan,
            "sv5": svals[4] if len(svals) > 4 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    dfs = []

    for family in ["sbm", "lfr"]:
        for base_idx in [1, 2, 3]:
            df = inspect_one(base_idx, family)
            dfs.append(df)

            print(f"\n=== {family.upper()} graph{base_idx:03d} ===")
            print(
                df[[
                    "noise_type", "noise_code", "n_edges", "esnr",
                    "n_outliers", "outlier_mass", "sv1", "sv2", "sv3", "sv4", "sv5"
                ]].to_string(index=False)
            )

    out = pd.concat(dfs, ignore_index=True)
    out_path = Path("results/esnr_synthetic_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nSaved results to {out_path}")