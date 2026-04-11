"""Simple Graph Convolution (SGC) for transductive community detection.

Code adapted from https://github.com/Tiiiger/SGC

"""

from __future__ import annotations

from typing import Self

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from sklearn.metrics import adjusted_rand_score
from torch_geometric.nn import SGConv

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig


class _SGCModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_hops: int,
    ) -> None:
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=k_hops, cached=True)

    def forward(
        self,
        x: Float[torch.Tensor, "n_nodes in_channels"],
        edge_index: Int[torch.Tensor, "2 n_edges"],
    ) -> Float[torch.Tensor, "n_nodes out_channels"]:
        return self.conv(x, edge_index)


class SGC(BaseMethod):
    """
    SGC (Wu et al., 2019) for transductive node classification / community detection.

    Collapses k graph-convolution steps into a single pre-computed feature
    propagation, then fits a linear classifier. Acts as a middle-ground between
    spectral and GNN-based methods.

    Relevant config fields: lr, epochs, k_hops.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self._model: _SGCModule | None = None

    def fit(
        self,
        data: GraphData,
        *,
        study_name: str | None = None,
        optuna_storage_path: str | None = None,
        **kwargs,
    ) -> Self:
        """
        Pre-propagate features for config.k_hops steps; train linear classifier
        on data.train_idx nodes for config.epochs steps.

        Parameters
        ----------
        data : GraphData
        study_name : str | None
            Optuna study name; passed through to hyperparameter search if used.
        optuna_storage_path : str | None
            Path to Optuna storage backend; passed through if used.

        Returns
        -------
        Self
        """
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = data.features.to(device)
        edge_index = data.graph.edge_index.to(device)
        labels = data.labels.to(device)
        train_idx = data.train_idx.to(device)

        in_channels = x.size(1)
        self._model = _SGCModule(
            in_channels=in_channels,
            out_channels=cfg.num_classes,
            k_hops=cfg.k_hops,
        ).to(device)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self._model.train()
        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            logits = self._model(x, edge_index)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

        self._model.eval()
        return self

    def score(
        self,
        data: GraphData,
        *,
        use_test_idx: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate SGC predictions on data.val_idx (or data.test_idx) nodes.

        Parameters
        ----------
        data : GraphData
        use_test_idx : bool
            If True, evaluate on data.test_idx instead of data.val_idx.

        Returns
        -------
        dict[str, float]
            Keys: "ARI".
        """
        device = next(self._model.parameters()).device
        x = data.features.to(device)
        edge_index = data.graph.edge_index.to(device)
        idx = data.test_idx if use_test_idx else data.val_idx

        with torch.no_grad():
            self._model.eval()
            logits = self._model(x, edge_index)

        preds = logits.argmax(dim=-1).cpu()
        labels = data.labels.cpu()
        ari = adjusted_rand_score(labels[idx].numpy(), preds[idx].numpy())
        return {"ARI": float(ari)}
