from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def _validate_graph_and_labels(G: nx.Graph, labels: np.ndarray) -> np.ndarray:
    """
    Validate that labels align with node ids 0..n-1 and return labels as int64.
    """
    labels = np.asarray(labels, dtype=np.int64)

    n_nodes = G.number_of_nodes()
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}.")

    if len(labels) != n_nodes:
        raise ValueError(
            f"Label length {len(labels)} does not match node count {n_nodes}."
        )

    expected_nodes = list(range(n_nodes))
    actual_nodes = sorted(G.nodes())
    if actual_nodes != expected_nodes:
        raise ValueError(
            "Graph nodes must be contiguous integers 0..n-1 for label alignment."
        )

    return labels


def compute_basic_graph_stats(G: nx.Graph) -> dict[str, Any]:
    """
    Compute graph-level structural statistics for an undirected graph.
    """
    n_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    avg_degree = (2 * num_edges / n_nodes) if n_nodes > 0 else 0.0
    density = nx.density(G) if n_nodes > 1 else 0.0
    sparsity = 1.0 - density

    if n_nodes == 0:
        num_connected_components = 0
        largest_cc_size = 0
        largest_cc_fraction = 0.0
    else:
        connected_components = list(nx.connected_components(G))
        num_connected_components = len(connected_components)
        largest_cc_size = max(len(component) for component in connected_components)
        largest_cc_fraction = largest_cc_size / n_nodes

    return {
        "n_nodes": n_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "sparsity": sparsity,
        "num_connected_components": num_connected_components,
        "largest_cc_size": largest_cc_size,
        "largest_cc_fraction": largest_cc_fraction,
    }


def compute_label_aware_stats(G: nx.Graph, labels: np.ndarray) -> dict[str, Any]:
    """
    Compute label-aware statistics for an undirected graph with one community label per node.

    Definitions
    -----------
    intercommunity_edge_fraction:
        Fraction of edges whose endpoints belong to different planted communities.

    heterophily:
        Average, across nodes with nonzero degree, of the fraction of neighbors
        that belong to a different planted community.

    In this benchmark these are related but not identical:
    - intercommunity_edge_fraction is edge-level
    - heterophily is node-level
    """
    labels = _validate_graph_and_labels(G, labels)

    num_edges = G.number_of_edges()
    num_communities = int(len(np.unique(labels)))

    if num_edges == 0:
        intercommunity_edge_fraction = 0.0
    else:
        cross_edges = sum(1 for u, v in G.edges() if labels[u] != labels[v])
        intercommunity_edge_fraction = cross_edges / num_edges

    node_cross_fractions: list[float] = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)

        if degree == 0:
            continue

        cross_neighbors = sum(1 for neighbor in neighbors if labels[neighbor] != labels[node])
        node_cross_fractions.append(cross_neighbors / degree)

    heterophily = float(np.mean(node_cross_fractions)) if node_cross_fractions else 0.0

    return {
        "num_communities": num_communities,
        "intercommunity_edge_fraction": intercommunity_edge_fraction,
        "heterophily": heterophily,
    }


def compute_all_graph_stats(G: nx.Graph, labels: np.ndarray) -> dict[str, Any]:
    """
    Compute all graph characterization fields used in the synthetic benchmark metadata.
    """
    basic_stats = compute_basic_graph_stats(G)
    label_stats = compute_label_aware_stats(G, labels)
    return {**basic_stats, **label_stats}


__all__ = [
    "compute_basic_graph_stats",
    "compute_label_aware_stats",
    "compute_all_graph_stats",
]