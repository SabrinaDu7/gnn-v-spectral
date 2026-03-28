""" Data Loading"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int, Float
from dataclasses import dataclass

from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
import torch_geometric.transforms as T

from data.generators.io import load_edge_index
from methods.spectral.embeddings import kcut_eigenspectrum

DEFAULT_DATASET_ROOT = "data/cache/synthetic"


@dataclass
class GraphData():
    graph: BaseData
    graph_id: str

    noise_fraction: float
    num_classes: int
    labels: Int[torch.Tensor, "n_nodes"]

    whole_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"]
    kcut_eigenspectrum: Float[torch.Tensor, "n_nodes n_eigenvectors"]
    regularized_eigenspectrum: Float[torch.Tensor, "n_nodes n_nodes"]

    whole_eigenvals: Float[torch.Tensor, "n_nodes"]
    kcut_eigenvals: Float[torch.Tensor, "n_eigenvectors"]
    regularized_eigenvals: Float[torch.Tensor, "n_nodes"]

    features: Float[torch.Tensor, "n_nodes feature_dim"] | None = None

    train_idx: Int[torch.Tensor, "num_train_nodes"] | None = None
    val_idx: Int[torch.Tensor, "num_valid_nodes"] | None = None
    test_idx: Int[torch.Tensor, "num_test_nodes"]| None = None


#### Dataloading ####
def load_graph_data(
    metadata_csv: str | Path,
    graph_id: str,
    spectra_pt: str | Path,
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> GraphData:
    """
    Load a single graph by graph_id from a metadata CSV into a GraphData object.

    Parameters
    ----------
    metadata_csv : str | Path
        Path to graph_index_{family}.csv.
    graph_id : str
        Row identifier in the metadata CSV.
    spectra_pt : str | Path
        Path to the precomputed .pt file containing whole and regularized
        eigenspectra (keys: whole_V, whole_evals, reg_V, reg_evals).
    dataset_root : str | Path
        Root used to resolve relative edge_path and label_path from the CSV.

    Returns
    -------
    GraphData
    """
    dataset_root = Path(dataset_root)
    row = pd.read_csv(metadata_csv).set_index("graph_id").loc[graph_id]

    edge_path  = dataset_root / row["edge_path"]
    label_path = dataset_root / row["label_path"]

    labels     = torch.from_numpy(np.load(label_path))
    num_nodes  = len(labels)
    edge_index = load_edge_index(edge_path)
    graph      = Data(edge_index=edge_index, num_nodes=num_nodes)

    spectra     = torch.load(spectra_pt, weights_only=True)
    whole_V     = spectra["whole_V"]
    whole_evals = spectra["whole_evals"]
    reg_V       = spectra["reg_V"]
    reg_evals   = spectra["reg_evals"]

    kcut_V, kcut_evals = kcut_eigenspectrum(
        edge_index, num_nodes, all_V=whole_V, all_eigenvalues=whole_evals
    )

    return GraphData(
        graph=graph,
        graph_id=graph_id,
        noise_fraction=float(row["noise_frac"]),
        num_classes=int(row["num_communities"]),
        labels=labels,
        whole_eigenspectrum=whole_V,
        kcut_eigenspectrum=kcut_V,
        regularized_eigenspectrum=reg_V,
        whole_eigenvals=whole_evals,
        kcut_eigenvals=kcut_evals,
        regularized_eigenvals=reg_evals,
    )
