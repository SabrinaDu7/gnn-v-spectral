"""
Per-condition Optuna hyperparameter tuning.

Tunes hyperparameters once per experimental condition (not per graph),
then the best params are reused across all 5 base graphs at that condition.

Tuning runs CPU-only to allow safe multiprocessing with many workers.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import optuna
import pandas as pd
import torch
from jaxtyping import Float

from data import GraphData, load_graph_data, DEFAULT_DATASET_ROOT
from methods import METHOD_REGISTRY, ExperimentConfig
from methods.spectral.spectral_method import SpectralMethod

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Search spaces ────────────────────────────────────────────────────────────

_RF_SPACE: dict = {
    "n_estimators": ("categorical", [50, 100, 200, 500]),
    "rf_max_depth": ("categorical", [5, 10, 20, 30, None]),
    "rf_min_samples_leaf": ("int", 1, 4),
}

SEARCH_SPACES: dict[str, dict] = {
    "gcn": {
        "lr": ("log_float", 1e-4, 1e-1),
        "hidden_dim": ("categorical", [16, 32, 64, 128, 256]),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.8),
        "weight_decay": ("log_float", 1e-5, 1e-2),
    },
    "gat": {
        "lr": ("log_float", 1e-4, 1e-1),
        "hidden_dim": ("categorical", [16, 32, 64, 128, 256]),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.8),
        "weight_decay": ("log_float", 1e-5, 1e-2),
        "num_heads": ("categorical", [1, 2, 4, 8]),
    },
    "sgc": {
        "k_hops": ("int", 1, 5),
        "lr": ("log_float", 1e-4, 1e-1),
        "weight_decay": ("log_float", 1e-5, 1e-2),
    },
    "whole_lr": {"lr_C": ("log_float", 1e-3, 1e3)},
    "kcut_lr": {"lr_C": ("log_float", 1e-3, 1e3)},
    "regularized_lr": {"lr_C": ("log_float", 1e-3, 1e3)},
    "whole_rf": _RF_SPACE,
    "kcut_rf": _RF_SPACE,
    "regularized_rf": _RF_SPACE,
}

# ── Condition key columns ────────────────────────────────────────────────────

STRUCTURAL_TUNE_COLS = ["family", "structural_noise_type", "structural_noise_code"]
FEATURE_TUNE_COLS = [
    "family", "structural_noise_type", "structural_noise_code",
    "feature_informativeness_code",
]

# ── Default config values ────────────────────────────────────────────────────

_DEFAULT_CONFIG = dict(
    seed=42,
    hidden_dim=64,
    num_layers=2,
    lr=0.01,
    epochs=200,
    dropout=0.5,
    num_heads=8,
    k_hops=2,
    n_estimators=100,
    weight_decay=0.0,
    lr_C=1.0,
    rf_max_depth=None,
    rf_min_samples_leaf=1,
)


# ── Core functions ───────────────────────────────────────────────────────────


def suggest_hyperparams(
    trial: optuna.Trial,
    *,
    model_key: str,
) -> dict:
    """Map search space definitions to ``trial.suggest_*`` calls."""
    space = SEARCH_SPACES.get(model_key, {})
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown search spec kind: {kind!r}")
    return params


def build_config(*, num_classes: int, **overrides) -> ExperimentConfig:
    """Create an ExperimentConfig from defaults with Optuna overrides applied."""
    merged = {**_DEFAULT_CONFIG, **overrides, "num_classes": num_classes}
    return ExperimentConfig(**merged)


def make_objective(
    model_key: str,
    data: GraphData,
    *,
    precomputed_embeddings: Float[torch.Tensor, "n_nodes n_eigenvectors"] | None = None,
):
    """Return a callable Optuna objective for *model_key* on *data*.

    Spectral methods reuse *precomputed_embeddings* across all trials to
    avoid redundant O(n^3) eigendecompositions.
    """

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_hyperparams(trial, model_key=model_key)
        config = build_config(num_classes=data.num_classes, **overrides)
        model = METHOD_REGISTRY[model_key](config)

        try:
            if isinstance(model, SpectralMethod) and precomputed_embeddings is not None:
                model.fit(data, embeddings=precomputed_embeddings)
            else:
                model.fit(data)

            metrics = model.score(data)
            val_ari = metrics.get("ARI")
            if val_ari is None or math.isnan(val_ari):
                return -1.0
            return float(val_ari)
        except Exception:
            return -1.0

    return objective


def tune_condition(
    model_key: str,
    data: GraphData,
    *,
    n_trials: int = 40,
    precomputed_embeddings: Float[torch.Tensor, "n_nodes n_eigenvectors"] | None = None,
    seed: int = 42,
) -> dict:
    """Run one Optuna study and return best params + best val ARI."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = make_objective(
        model_key, data, precomputed_embeddings=precomputed_embeddings,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }


# ── Multiprocessing worker ──────────────────────────────────────────────────


def _tune_worker(args: tuple) -> tuple:
    """Picklable worker: tune one (condition, model) pair on CPU."""
    (
        condition_key,
        model_key,
        family,
        graph_id,
        feature_path_str,
        n_trials,
        seed,
    ) = args

    # Force CPU for safe multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)

    metadata_csv = str(
        Path(DEFAULT_DATASET_ROOT) / "metadata" / f"graph_index_{family}.csv"
    )
    feature_path = feature_path_str if feature_path_str else None
    if feature_path is not None:
        feature_path = str(Path(DEFAULT_DATASET_ROOT) / feature_path)

    data = load_graph_data(
        metadata_csv=metadata_csv, graph_id=graph_id,
        features_pt=feature_path, seed=1,
    )

    # Pre-extract embeddings for spectral methods
    embedding_map = {
        "whole": data.whole_eigenspectrum,
        "kcut": data.kcut_eigenspectrum,
        "regularized": data.regularized_eigenspectrum,
    }
    embedding_type = model_key.rsplit("_", 1)[0] if "_" in model_key and model_key not in ("sgc", "gcn", "gat") else None
    precomputed = embedding_map.get(embedding_type) if embedding_type else None

    result = tune_condition(
        model_key, data,
        n_trials=n_trials,
        precomputed_embeddings=precomputed,
        seed=seed,
    )
    return (condition_key, model_key, result)


def tune_all_conditions(
    experiment_table: pd.DataFrame,
    *,
    model_keys: list[str],
    n_trials: int = 40,
    n_jobs: int = 10,
    experiment_type: Literal["structural_noise", "feature_informativeness"],
    seed: int = 42,
) -> dict[tuple, dict[str, dict]]:
    """Tune all (condition, model) pairs in parallel.

    Returns a nested dict: ``{condition_key: {model_key: best_params}}``.
    """
    if experiment_type == "structural_noise":
        tune_cols = STRUCTURAL_TUNE_COLS
    else:
        tune_cols = FEATURE_TUNE_COLS

    # Pick first base graph per condition as the tuning representative
    representatives = (
        experiment_table
        .sort_values("graph_id")
        .groupby(tune_cols, sort=False)
        .first()
        .reset_index()
    )

    # Build work items
    work_items = []
    for _, rep_row in representatives.iterrows():
        condition_key = tuple(str(rep_row[c]) for c in tune_cols)
        graph_id = rep_row["graph_id"]
        family = rep_row["family"]
        feature_path = str(rep_row.get("feature_path", "")) if "feature_path" in rep_row.index else ""
        for model_key in model_keys:
            work_items.append((
                condition_key,
                model_key,
                family,
                graph_id,
                feature_path,
                n_trials,
                seed,
            ))

    total = len(work_items)
    logger.info(
        "Tuning %d (condition × model) pairs with %d trials each, %d workers ...",
        total, n_trials, n_jobs,
    )

    # Run in parallel
    best_params_map: dict[tuple, dict[str, dict]] = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, (cond_key, mk, result) in enumerate(pool.map(_tune_worker, work_items), 1):
            if cond_key not in best_params_map:
                best_params_map[cond_key] = {}
            best_params_map[cond_key][mk] = result["best_params"]
            if i % 50 == 0 or i == total:
                logger.info("  Tuning progress: %d / %d", i, total)

    logger.info("Tuning complete. %d conditions tuned.", len(best_params_map))
    return best_params_map
