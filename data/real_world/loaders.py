# Loaders for real-world datasets (STRING, ABIDE, etc.)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RealWorldGraph:
    graph_id: str
    dataset: str
    edges: pd.DataFrame              # columns: src, dst
    labels: np.ndarray               # shape (n,)
    features: Optional[np.ndarray]   # shape (n, d) or None
    metadata: dict


def _require_edge_columns(edges: pd.DataFrame) -> None:
    required = {"src", "dst"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required edge columns: {missing}")


def _remove_self_loops(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.loc[edges["src"] != edges["dst"]].copy()


def _symmetrize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    flipped = edges.rename(columns={"src": "dst", "dst": "src"})
    both = pd.concat([edges, flipped], ignore_index=True)
    return both


def _deduplicate_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    canon = pd.DataFrame({
        "src": np.minimum(edges["src"].to_numpy(), edges["dst"].to_numpy()),
        "dst": np.maximum(edges["src"].to_numpy(), edges["dst"].to_numpy()),
    })
    canon = canon.drop_duplicates().reset_index(drop=True)
    return canon


def _collect_node_ids(
    edges: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    node_ids = set(edges["src"]).union(set(edges["dst"]))
    if labels_df is not None and "node" in labels_df.columns:
        node_ids.update(labels_df["node"].tolist())
    if features_df is not None and "node" in features_df.columns:
        node_ids.update(features_df["node"].tolist())
    return np.array(sorted(node_ids))


def _reindex_edges(edges: pd.DataFrame, node_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    mapping = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    reindexed = edges.copy()
    reindexed["src"] = reindexed["src"].map(mapping)
    reindexed["dst"] = reindexed["dst"].map(mapping)
    return reindexed, mapping


def _validate_graph(graph: RealWorldGraph) -> None:
    _require_edge_columns(graph.edges)

    n_nodes = graph.metadata["n_nodes"]

    if graph.labels.shape[0] != n_nodes:
        raise ValueError(
            f"labels has length {graph.labels.shape[0]} but n_nodes={n_nodes}"
        )

    if graph.features is not None and graph.features.shape[0] != n_nodes:
        raise ValueError(
            f"features has {graph.features.shape[0]} rows but n_nodes={n_nodes}"
        )

    if graph.edges[["src", "dst"]].isnull().any().any():
        raise ValueError("Edge list contains null values")

    if (graph.edges["src"] == graph.edges["dst"]).any():
        raise ValueError("Graph still contains self-loops after preprocessing")


def _finalize_graph(
    *,
    graph_id: str,
    dataset: str,
    edges: pd.DataFrame,
    labels: np.ndarray,
    features: Optional[np.ndarray],
    label_name: str,
    is_directed_original: bool,
    notes: Optional[str] = None,
) -> RealWorldGraph:
    edges = edges[["src", "dst"]].copy()
    edges = _remove_self_loops(edges)
    edges = _symmetrize_edges(edges)
    edges = _deduplicate_undirected_edges(edges)

    node_ids = _collect_node_ids(edges)
    edges, mapping = _reindex_edges(edges, node_ids)

    n_nodes = len(node_ids)

    metadata = {
        "graph_id": graph_id,
        "dataset": dataset,
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "num_classes": int(len(np.unique(labels))),
        "label_name": label_name,
        "has_features": features is not None,
        "feature_dim": None if features is None else int(features.shape[1]),
        "is_directed_original": is_directed_original,
        "was_symmetrized": True,
        "removed_self_loops": True,
        "notes": notes,
    }

    graph = RealWorldGraph(
        graph_id=graph_id,
        dataset=dataset,
        edges=edges.reset_index(drop=True),
        labels=labels,
        features=features,
        metadata=metadata,
    )

    _validate_graph(graph)
    return graph


import json


def load_polblogs(raw_dir: str | Path) -> RealWorldGraph:
    """
    Load and standardize PolBlogs from raw files stored in:

        <raw_dir>/adjacency.tsv
        <raw_dir>/labels.tsv

    Returns an undirected, unweighted, simple graph representation that is
    consistent with the synthetic graph pipeline.
    """
    raw_dir = Path(raw_dir)

    edge_path = raw_dir / "adjacency.tsv"
    label_path = raw_dir / "labels.tsv"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing raw edge file: {edge_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing raw label file: {label_path}")

    # PyG reads the first two tab-separated columns from adjacency.tsv.
    edges = pd.read_csv(
        edge_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["src", "dst"],
    )

    # labels.tsv is a single-column label vector, one label per node.
    labels = pd.read_csv(
        label_path,
        sep="\t",
        header=None,
    ).iloc[:, 0].to_numpy(dtype=int)

    n_nodes = int(labels.shape[0])

    # Because PolBlogs can contain isolated nodes, define the node set from labels.
    # This assumes nodes are indexed 0..n-1 in the raw files.
    if len(edges) > 0:
        raw_min = int(edges[["src", "dst"]].min().min())
        raw_max = int(edges[["src", "dst"]].max().max())
        if raw_min < 0 or raw_max >= n_nodes:
            raise ValueError(
                f"Raw PolBlogs edges out of range for labels: "
                f"min={raw_min}, max={raw_max}, n_nodes={n_nodes}"
            )

    # Standardize to simple undirected graph.
    edges = _remove_self_loops(edges)
    edges = _symmetrize_edges(edges)
    edges = _deduplicate_undirected_edges(edges)

    metadata = {
        "graph_id": "polblogs",
        "dataset": "polblogs",
        "n_nodes": n_nodes,
        "n_edges": int(len(edges)),
        "num_classes": int(len(np.unique(labels))),
        "label_name": "political_ideology",
        "has_features": False,
        "feature_dim": None,
        "is_directed_original": True,
        "was_symmetrized": True,
        "removed_self_loops": True,
        "notes": "Raw graph has no native node features. Processed as simple undirected graph.",
    }

    graph = RealWorldGraph(
        graph_id="polblogs",
        dataset="polblogs",
        edges=edges.reset_index(drop=True),
        labels=labels,
        features=None,
        metadata=metadata,
    )

    _validate_graph(graph)
    return graph


def _build_lastfm_feature_matrix(feature_json_path: str | Path, n_nodes: int) -> np.ndarray:
    """
    Build a dense binary feature matrix from the LastFM Asia JSON file.

    Raw format:
        {
          "0": [artist_id_1, artist_id_2, ...],
          "1": [...],
          ...
        }

    We compress the raw artist IDs into contiguous column indices
    based on the sorted set of all artist IDs that appear.
    """
    feature_json_path = Path(feature_json_path)

    with open(feature_json_path, "r") as f:
        raw = json.load(f)

    # Normalize node keys to int
    raw = {int(k): v for k, v in raw.items()}

    # Basic sanity check
    missing_nodes = sorted(set(range(n_nodes)) - set(raw.keys()))
    extra_nodes = sorted(set(raw.keys()) - set(range(n_nodes)))

    if missing_nodes:
        raise ValueError(f"Missing feature entries for nodes: {missing_nodes[:10]} ...")
    if extra_nodes:
        raise ValueError(f"Unexpected feature entries for out-of-range nodes: {extra_nodes[:10]} ...")

    # Collect all unique artist IDs
    all_artist_ids = sorted({artist for artists in raw.values() for artist in artists})
    artist_to_col = {artist_id: j for j, artist_id in enumerate(all_artist_ids)}

    features = np.zeros((n_nodes, len(all_artist_ids)), dtype=np.uint8)

    for node_id in range(n_nodes):
        artist_ids = raw[node_id]
        for artist_id in artist_ids:
            features[node_id, artist_to_col[artist_id]] = 1

    return features

def load_lastfm_asia(raw_dir: str | Path) -> RealWorldGraph:
    """
    Load and standardize LastFM Asia from raw SNAP files stored in:

        <raw_dir>/lastfm_asia_edges.csv
        <raw_dir>/lastfm_asia_target.csv
        <raw_dir>/lastfm_asia_features.json

    Returns a simple undirected graph representation consistent with the
    synthetic graph pipeline.
    """
    raw_dir = Path(raw_dir)

    edge_path = raw_dir / "lastfm_asia_edges.csv"
    label_path = raw_dir / "lastfm_asia_target.csv"
    feature_path = raw_dir / "lastfm_asia_features.json"

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing raw edge file: {edge_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing raw label file: {label_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing raw feature file: {feature_path}")

    # Load edges
    edges = pd.read_csv(edge_path)
    expected_edge_cols = {"node_1", "node_2"}
    if not expected_edge_cols.issubset(edges.columns):
        raise ValueError(f"Expected edge columns {expected_edge_cols}, got {list(edges.columns)}")

    edges = edges.rename(columns={"node_1": "src", "node_2": "dst"})[["src", "dst"]].astype(int)

    # Load labels
    labels_df = pd.read_csv(label_path)
    expected_label_cols = {"id", "target"}
    if not expected_label_cols.issubset(labels_df.columns):
        raise ValueError(f"Expected label columns {expected_label_cols}, got {list(labels_df.columns)}")

    labels_df = labels_df.sort_values("id").reset_index(drop=True)

    node_ids = labels_df["id"].to_numpy(dtype=int)
    if not np.array_equal(node_ids, np.arange(len(node_ids))):
        raise ValueError("Expected LastFM node ids to be exactly 0..n-1 after sorting by id")

    labels = labels_df["target"].to_numpy(dtype=int)
    n_nodes = int(labels.shape[0])

    # Build binary feature matrix from sparse JSON representation
    features = _build_lastfm_feature_matrix(feature_path, n_nodes)

    if features.shape[0] != n_nodes:
        raise ValueError(
            f"Feature rows ({features.shape[0]}) do not match label length ({n_nodes})"
        )

    if len(edges) > 0:
        raw_min = int(edges[["src", "dst"]].min().min())
        raw_max = int(edges[["src", "dst"]].max().max())
        if raw_min < 0 or raw_max >= n_nodes:
            raise ValueError(
                f"Raw LastFM edges out of range for labels/features: "
                f"min={raw_min}, max={raw_max}, n_nodes={n_nodes}"
            )

    # Standardize to simple undirected graph
    edges = _remove_self_loops(edges)
    edges = _symmetrize_edges(edges)
    edges = _deduplicate_undirected_edges(edges)

    metadata = {
        "graph_id": "lastfm_asia",
        "dataset": "lastfm_asia",
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edges)),
        "num_classes": int(len(np.unique(labels))),
        "label_name": "country",
        "has_features": True,
        "feature_dim": int(features.shape[1]),
        "is_directed_original": False,
        "was_symmetrized": True,
        "removed_self_loops": True,
        "notes": (
            "Raw graph has sparse artist-like features stored in JSON and country labels. "
            "Features were converted to a dense binary matrix over observed artist IDs. "
            "Processed as simple undirected graph."
        ),
    }

    graph = RealWorldGraph(
        graph_id="lastfm_asia",
        dataset="lastfm_asia",
        edges=edges.reset_index(drop=True),
        labels=labels,
        features=features,
        metadata=metadata,
    )

    _validate_graph(graph)
    return graph

def load_facebook_residence(raw_dir: str | Path, campus_name: str) -> RealWorldGraph:
    """
    Load one Facebook100 campus using residence as the target label.

    Important:
    - remove residence/dorm/house from features to avoid leakage
    """
    raise NotImplementedError("Implement after LastFM.")


def save_real_world_graph(graph: RealWorldGraph, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)

    edges_dir = out_dir / "edges"
    labels_dir = out_dir / "labels_processed"
    metadata_dir = out_dir / "metadata"
    features_dir = out_dir / "features"

    edges_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    edge_file = edges_dir / f"{graph.graph_id}_edges.csv"
    label_file = labels_dir / f"{graph.graph_id}_labels.npy"
    metadata_file = metadata_dir / f"{graph.graph_id}_metadata.json"

    graph.edges.to_csv(edge_file, index=False)
    np.save(label_file, graph.labels)

    if graph.features is not None:
        feature_file = features_dir / f"{graph.graph_id}_features.npy"
        np.save(feature_file, graph.features)

    with open(metadata_file, "w") as f:
        json.dump(graph.metadata, f, indent=2)

def load_real_world_graph(name: str, raw_dir: str | Path, **kwargs) -> RealWorldGraph:
    name = name.lower()

    if name == "polblogs":
        return load_polblogs(raw_dir)
    if name == "lastfm_asia":
        return load_lastfm_asia(raw_dir)
    if name == "facebook_residence":
        return load_facebook_residence(raw_dir, **kwargs)

    raise ValueError(f"Unknown dataset: {name}")