# Graph characterization: sparsity, heterophily, ESNR
from __future__ import annotations

import numpy as np
import pandas as pd

from .loaders import RealWorldGraph


def degree_sequence(graph: RealWorldGraph) -> np.ndarray:
    edges = graph.edges
    n = graph.metadata["n_nodes"]

    deg = np.zeros(n, dtype=int)
    src_counts = edges["src"].value_counts()
    dst_counts = edges["dst"].value_counts()

    for node, count in src_counts.items():
        deg[int(node)] += int(count)
    for node, count in dst_counts.items():
        deg[int(node)] += int(count)

    return deg


def class_counts(graph: RealWorldGraph) -> dict[int, int]:
    labels = graph.labels
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def basic_graph_properties(graph: RealWorldGraph) -> dict:
    n = graph.metadata["n_nodes"]
    m = graph.metadata["n_edges"]
    deg = degree_sequence(graph)

    density = 0.0 if n <= 1 else (2 * m) / (n * (n - 1))

    props = {
        "graph_id": graph.graph_id,
        "dataset": graph.dataset,
        "n_nodes": n,
        "n_edges": m,
        "num_classes": graph.metadata["num_classes"],
        "class_counts": class_counts(graph),
        "has_features": graph.metadata["has_features"],
        "feature_dim": graph.metadata["feature_dim"],
        "avg_degree": float(np.mean(deg)),
        "min_degree": int(np.min(deg)),
        "max_degree": int(np.max(deg)),
        "density": float(density),
    }
    return props


def print_basic_graph_properties(graph: RealWorldGraph) -> None:
    props = basic_graph_properties(graph)

    print(f"Graph ID:      {props['graph_id']}")
    print(f"Dataset:       {props['dataset']}")
    print(f"Nodes:         {props['n_nodes']}")
    print(f"Edges:         {props['n_edges']}")
    print(f"Classes:       {props['num_classes']}")
    print(f"Class counts:  {props['class_counts']}")
    print(f"Has features:  {props['has_features']}")
    print(f"Feature dim:   {props['feature_dim']}")
    print(f"Avg degree:    {props['avg_degree']:.3f}")
    print(f"Min degree:    {props['min_degree']}")
    print(f"Max degree:    {props['max_degree']}")
    print(f"Density:       {props['density']:.6f}")