from __future__ import annotations

from collections import deque

import numpy as np

from .loaders import RealWorldGraph

import networkx as nx
from methods.esnr import compute_esnr_from_graph


def degree_sequence(graph: RealWorldGraph) -> np.ndarray:
    n = graph.metadata["n_nodes"]
    deg = np.zeros(n, dtype=int)

    if len(graph.edges) == 0:
        return deg

    src = graph.edges["src"].to_numpy(dtype=int)
    dst = graph.edges["dst"].to_numpy(dtype=int)

    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)

    return deg


def class_counts(graph: RealWorldGraph) -> dict[int, int]:
    values, counts = np.unique(graph.labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def adjacency_list(graph: RealWorldGraph) -> list[list[int]]:
    n = graph.metadata["n_nodes"]
    adj = [[] for _ in range(n)]

    if len(graph.edges) == 0:
        return adj

    src = graph.edges["src"].to_numpy(dtype=int)
    dst = graph.edges["dst"].to_numpy(dtype=int)

    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    return adj


def connected_components(graph: RealWorldGraph) -> list[list[int]]:
    n = graph.metadata["n_nodes"]
    adj = adjacency_list(graph)
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue

        q = deque([start])
        visited[start] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        components.append(comp)

    components.sort(key=len, reverse=True)
    return components


def connected_component_sizes(graph: RealWorldGraph) -> list[int]:
    return [len(comp) for comp in connected_components(graph)]


def extract_node_induced_subgraph(graph: RealWorldGraph, nodes: list[int] | np.ndarray, graph_id_suffix: str) -> RealWorldGraph:
    """
    Create a node-induced subgraph and reindex nodes to 0..k-1.
    """
    nodes = np.array(sorted(nodes), dtype=int)
    node_set = set(nodes.tolist())
    old_to_new = {old: new for new, old in enumerate(nodes)}

    edges = graph.edges
    mask = edges["src"].isin(node_set) & edges["dst"].isin(node_set)
    sub_edges = edges.loc[mask].copy()

    sub_edges["src"] = sub_edges["src"].map(old_to_new)
    sub_edges["dst"] = sub_edges["dst"].map(old_to_new)
    sub_edges = sub_edges.reset_index(drop=True)

    sub_labels = graph.labels[nodes]

    sub_features = None
    if graph.features is not None:
        sub_features = graph.features[nodes]

    sub_metadata = dict(graph.metadata)
    sub_metadata["graph_id"] = f"{graph.graph_id}_{graph_id_suffix}"
    sub_metadata["n_nodes"] = int(len(nodes))
    sub_metadata["n_edges"] = int(len(sub_edges))
    sub_metadata["num_classes"] = int(len(np.unique(sub_labels)))
    sub_metadata["notes"] = (
        (graph.metadata.get("notes", "") + " | ").strip(" |")
        + f"Node-induced subgraph: {graph_id_suffix}"
    )

    return RealWorldGraph(
        graph_id=f"{graph.graph_id}_{graph_id_suffix}",
        dataset=graph.dataset,
        edges=sub_edges,
        labels=sub_labels,
        features=sub_features,
        metadata=sub_metadata,
    )


def extract_largest_connected_component(graph: RealWorldGraph) -> RealWorldGraph:
    comps = connected_components(graph)
    if not comps:
        raise ValueError("Graph has no connected components.")

    gcc_nodes = comps[0]
    return extract_node_induced_subgraph(graph, gcc_nodes, "gcc")

def to_networkx_graph(graph: RealWorldGraph) -> nx.Graph:
    """
    Convert a RealWorldGraph edge list to an undirected NetworkX graph
    with nodes 0..n-1.
    """
    n = graph.metadata["n_nodes"]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if len(graph.edges) > 0:
        src = graph.edges["src"].to_numpy(dtype=int)
        dst = graph.edges["dst"].to_numpy(dtype=int)
        G.add_edges_from(zip(src, dst))

    return G


def basic_graph_properties(graph: RealWorldGraph) -> dict:
    n = graph.metadata["n_nodes"]
    m = graph.metadata["n_edges"]
    deg = degree_sequence(graph)

    density = 0.0 if n <= 1 else (2 * m) / (n * (n - 1))
    comp_sizes = connected_component_sizes(graph)

    G = nx.Graph()
    edges = graph.edges

    G.add_edges_from(zip(edges["src"], edges["dst"]))

    G_nx = to_networkx_graph(graph)
    esnr_stats = compute_esnr_from_graph(G_nx, graph.labels)

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
        "median_degree": float(np.median(deg)),
        "min_degree": int(np.min(deg)),
        "max_degree": int(np.max(deg)),
        "num_isolated_nodes": int(np.sum(deg == 0)),
        "density": float(density),
        "num_connected_components": int(len(comp_sizes)),
        "largest_component_size": int(comp_sizes[0]) if comp_sizes else 0,
        "largest_component_fraction": float(comp_sizes[0] / n) if comp_sizes and n > 0 else 0.0,
        "component_sizes_top10": comp_sizes[:10],
        "esnr": float(esnr_stats["esnr"]),
        "esnr_n_outliers": int(esnr_stats["n_outlier_singular_values"]),
        "esnr_outlier_mass": float(esnr_stats["outlier_mass"]),
        "esnr_converged": bool(esnr_stats["converged"]),
        "esnr_iterations": int(esnr_stats["iterations"]),
    }
    return props


def print_basic_graph_properties(graph: RealWorldGraph) -> None:
    props = basic_graph_properties(graph)

    print(f"Graph ID:                  {props['graph_id']}")
    print(f"Dataset:                   {props['dataset']}")
    print(f"Nodes:                     {props['n_nodes']}")
    print(f"Edges:                     {props['n_edges']}")
    print(f"Classes:                   {props['num_classes']}")
    print(f"Class counts:              {props['class_counts']}")
    print(f"Has features:              {props['has_features']}")
    print(f"Feature dim:               {props['feature_dim']}")
    print(f"Average degree:            {props['avg_degree']:.3f}")
    print(f"Median degree:             {props['median_degree']:.3f}")
    print(f"Min degree:                {props['min_degree']}")
    print(f"Max degree:                {props['max_degree']}")
    print(f"Isolated nodes:            {props['num_isolated_nodes']}")
    print(f"Density:                   {props['density']:.6f}")
    print(f"Connected components:      {props['num_connected_components']}")
    print(f"Largest component size:    {props['largest_component_size']}")
    print(f"Largest component fraction:{props['largest_component_fraction']:.4f}")
    print(f"Top 10 component sizes:    {props['component_sizes_top10']}")

def filter_classes_by_min_size(
    graph: RealWorldGraph,
    min_size: int,
    graph_id_suffix: str | None = None,
) -> RealWorldGraph:
    """
    Keep only nodes whose class appears at least `min_size` times.
    Then relabel the remaining classes to contiguous ids 0..C-1.
    """
    counts = class_counts(graph)
    keep_classes = sorted([cls for cls, cnt in counts.items() if cnt >= min_size])

    if not keep_classes:
        raise ValueError(f"No classes remain after filtering with min_size={min_size}")

    keep_mask = np.isin(graph.labels, keep_classes)
    kept_nodes = np.where(keep_mask)[0]

    suffix = graph_id_suffix or f"min{min_size}"
    subgraph = extract_node_induced_subgraph(graph, kept_nodes, suffix)

    old_labels = subgraph.labels.copy()
    new_classes = sorted(np.unique(old_labels))
    relabel_map = {old: new for new, old in enumerate(new_classes)}
    new_labels = np.array([relabel_map[int(x)] for x in old_labels], dtype=int)

    subgraph.labels = new_labels
    subgraph.metadata = dict(subgraph.metadata)
    subgraph.metadata["original_class_ids_retained"] = [int(x) for x in new_classes]
    subgraph.metadata["min_class_size_filter"] = int(min_size)
    subgraph.metadata["num_classes"] = int(len(new_classes))
    subgraph.metadata["notes"] = (
        (subgraph.metadata.get("notes", "") + " | ").strip(" |")
        + f" Retained only classes with size >= {min_size} and relabeled to contiguous ids."
    )

    return subgraph