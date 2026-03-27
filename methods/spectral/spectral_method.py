"""Single class for all 6 spectral embedding × classifier combinations."""

from __future__ import annotations

from typing import Literal, Self

from data import GraphData
from methods.base import BaseMethod, ExperimentConfig


class SpectralMethod(BaseMethod):
    """
    Spectral community detection with configurable embedding and classifier.

    Covers all 6 spectral benchmark methods by composing an embedding type
    with a downstream classifier. The embedding is computed on the full graph;
    the classifier is fit on train_idx nodes only.

    Parameters
    ----------
    config : ExperimentConfig
        config.n_eigenvectors must be set (not None) when embedding_type="kcut".
    embedding_type : {"whole", "kcut", "regularized"}
        Which Laplacian spectrum variant to use as node features:
          "whole"       — full eigenspectrum of the normalised Laplacian
          "kcut"        — bottom-k eigenvectors (spectral k-way cut)
          "regularized" — regularized Laplacian spectrum (tau-shift)
    classifier_type : {"lr", "lp"}
        "lr" — logistic regression fit on train_idx embeddings
        "lp" — label propagation seeded from train_idx ground-truth labels

    Notes
    -----
    embedding_type and classifier_type are keyword-only to prevent silent
    argument-order bugs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        embedding_type: Literal["whole", "kcut", "regularized"],
        classifier_type: Literal["lr", "lp"],
    ) -> None:
        raise NotImplementedError

    def fit(self, data: GraphData) -> Self:
        """
        Compute spectral embedding on the full graph; fit classifier on train_idx.

        For classifier_type="lr":
            Fits a logistic regression on the embeddings of data.train_idx nodes.
        For classifier_type="lp":
            Initialises label propagation with ground-truth labels at data.train_idx.

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
        Predict community labels for data.valid_idx and compute metrics.

        ARI and NMI computed via sklearn.metrics against data.labels[valid_idx].
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
