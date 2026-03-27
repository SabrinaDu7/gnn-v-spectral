"""Simple Graph Convolution (SGC) for transductive community detection."""

from __future__ import annotations

from typing import Self

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig


class SGC(BaseMethod):
    """
    SGC (Wu et al., 2019) for transductive node classification / community detection.

    Collapses k graph-convolution steps into a single pre-computed feature
    propagation, then fits a linear classifier. Acts as a middle-ground between
    spectral and GNN-based methods.

    Relevant config fields: hidden_dim, num_layers, lr, epochs, dropout, k_hops.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig) -> None:
        raise NotImplementedError

    def fit(self, data: GraphData) -> Self:
        """
        Pre-propagate features for config.k_hops steps; train linear classifier
        on data.train_idx nodes for config.epochs steps.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        Self
        """
        raise NotImplementedError

    def score(self, data: GraphData) -> dict[str, float]:
        """
        Evaluate SGC predictions on data.valid_idx nodes.

        ARI and NMI computed via sklearn.metrics.
        relative_ARI is float("nan"); filled in at pipeline level.

        Parameters
        ----------
        data : GraphData

        Returns
        -------
        dict[str, float]
            Keys: "ARI", "NMI", "relative_ARI".
        """
        raise NotImplementedError
