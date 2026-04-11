from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from methods.esnr import compute_esnr_from_graph


def load_graph(edge_path: Path, label_path: Path) -> tuple[nx.Graph, np.ndarray]:
    edges = pd.read_csv(edge_path)
    labels = np.load(label_path)

    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    G.add_edges_from(zip(edges["src"], edges["dst"]))

    return G, labels


def inspect_graph(edge_path: Path, label_path: Path) -> None:
    G, labels = load_graph(edge_path, label_path)
    result = compute_esnr_from_graph(G, labels)

    print(f"\nGraph: {edge_path.name}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"ESNR: {result['esnr']:.6f}")
    print(f"Threshold: {result['threshold']:.6f}")
    print(f"Outlier singular values: {result['n_outlier_singular_values']}")
    print(f"Outlier mass: {result['outlier_mass']:.6f}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Top singular values: {result['singular_values'][:10]}")


if __name__ == "__main__":
    root = Path("data/cache/synthetic80")

    examples = [
        (
            root / "sbm/clean/edges/graph001_000_clean_sbm.csv",
            root / "sbm/clean/labels/graph001_000_clean_sbm_labels.npy",
        ),
        (
            root / "sbm/random/edges/graph001_015_random_sbm.csv",
            root / "sbm/random/labels/graph001_015_random_sbm_labels.npy",
        ),
        (
            root / "sbm/random/edges/graph001_045_random_sbm.csv",
            root / "sbm/random/labels/graph001_045_random_sbm_labels.npy",
        ),
        (
            root / "sbm/targeted_betweenness/edges/graph001_045_targeted_betweenness_sbm.csv",
            root / "sbm/targeted_betweenness/labels/graph001_045_targeted_betweenness_sbm_labels.npy",
        ),
    ]

    for edge_path, label_path in examples:
        inspect_graph(edge_path, label_path)