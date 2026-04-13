"""Precompute whole and regularized eigenspectra for cached graphs.

Synthetic: saves one .pt file per graph under
    data/cache/synthetic/{family}/{noise_type}/spectra/{graph_id}.pt

Real-world: saves one .pt file per graph under
    data/cache/realworld/spectra/{graph_id}.pt

Both .pt files share the same schema:
    {
        "whole_V":     Float[Tensor, "num_nodes num_nodes"],
        "whole_evals": Float[Tensor, "num_nodes"],
        "reg_V":       Float[Tensor, "num_nodes n_eigenvectors_plus_1"],
        "reg_evals":   Float[Tensor, "n_eigenvectors_plus_1"],
    }

Usage:
    uv run data/precompute_spectra.py synthetic-config --family lfr --noise-type clean
    uv run data/precompute_spectra.py real-world-config
"""

# TODO: Need a way to verify correctness. Probably use old assignment function to plot eigenspectra and then check with plot from saved eigenspectra

from __future__ import annotations

import tyro
from pathlib import Path
from typing import Literal, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.generators.io import load_edge_index
from methods.spectral.embeddings import regularized_eigenspectrum, whole_eigenspectrum

FAMILIES = ("lfr", "sbm")
NOISE_TYPES = ("clean", "random", "targeted_betweenness")

# Maps CLI --name values to the `dataset` column in graph_index_realworld.csv.
_RW_NAME_TO_DATASET: dict[str, str] = {
    "polblogs": "polblogs",
    "lastfm_asia": "lastfm_asia",
    "facebook_penn94": "facebook100",
    "ppi": "ppi",
    "amazon_computers": "amazon_computers",
    "amazon_photo": "amazon_photo",
    "cora": "cora",
    "dblp": "dblp",
    "github": "github",
}


def _save_spectra(
    *,
    graph_id: str,
    edge_path: Path,
    label_path: Path, # Just to compute nodes
    out_path: Path,
    device: torch.device,
) -> None:
    """Compute and save whole + regularized eigenspectra for one graph."""
    labels = np.load(label_path)
    num_nodes = int(len(labels))
    edge_index = load_edge_index(edge_path)

    print(f"  {graph_id} ({num_nodes} nodes) ...", end=" ", flush=True)

    whole_V, whole_evals = whole_eigenspectrum(edge_index, num_nodes, device=device)
    reg_V, reg_evals = regularized_eigenspectrum(edge_index, num_nodes, device=device)

    torch.save(
        {
            "whole_V": whole_V,
            "whole_evals": whole_evals,
            "reg_V": reg_V,
            "reg_evals": reg_evals,
        },
        out_path,
    )
    print("done")


def precompute(
    *,
    root: Path,
    families: tuple[str, ...],
    noise_types: tuple[str, ...],
    device: torch.device,
) -> None:
    for idx, family in enumerate(families):
        out_dir = root / family / noise_types[idx] / "spectra"
        out_dir.mkdir(exist_ok=True)

        meta_path = root / "metadata" / f"graph_index_{family}.csv"
        if not meta_path.exists():
            print(f"[skip] no metadata found at {meta_path}")
            continue

        df = pd.read_csv(meta_path)
        rows = df[df["noise_type"].isin(noise_types)]
        print(f"[{family}] {len(rows)} graphs to process")

        for _, row in rows.iterrows():
            graph_id = str(row["graph_id"])
            out_path = out_dir / f"{graph_id}.pt"
            if out_path.exists():
                print(f"  [skip] {out_path} already exists")
                continue

            _save_spectra(
                graph_id=graph_id,
                edge_path=root / Path(str(row["edge_path"])),
                label_path=root / Path(str(row["label_path"])),
                out_path=out_path,
                device=device
            )


def precompute_real_world(*, root: Path, dataset_filter: str | None = None, device: torch.device) -> None:
    """
    Precompute spectra for real-world graphs cached under root.

    Reads graph_index_realworld.csv from {root}/metadata/, optionally
    filtering rows by the `dataset` column value.
    Edge and label paths in the CSV are relative to root.
    Saves spectra to {root}/spectra/{graph_id}.pt.
    """
    general_dataset_name = root.name.split("/")[-1] # e.g. "realworld" from "data/cache/realworld/"
    meta_path = root / "metadata" / f"graph_index_{general_dataset_name}.csv"

    if not meta_path.exists():
        print(f"[skip] no metadata found at {meta_path}")
        return


    df = pd.read_csv(meta_path)
    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter]
    df = df.sort_values("n_nodes", ascending=True).reset_index(drop=True)

    label = dataset_filter if dataset_filter is not None else "all"
    print(f"[realworld/{label}] {len(df)} graphs to process")


    for _, row in df.iterrows():
        graph_id = str(row["graph_id"])
        spectra_dir = root / row["dataset"] / "spectra"
        spectra_dir.mkdir(parents=True, exist_ok=True)

        out_path = spectra_dir / f"{graph_id}.pt"
        print(out_path)

        if out_path.exists():
            print(f"  [skip] {out_path} already exists")
            continue

        _save_spectra(
            graph_id=graph_id,
            edge_path=root / Path(str(row["edge_path"])),
            label_path=root / Path(str(row["label_path"])),
            out_path=out_path,
            device=device,
        )


@dataclass
class SyntheticConfig:
    name: Literal["all", "lfr", "sbm"] = "all"
    noise_type: Literal["clean", "random", "targeted_betweenness"] = "clean"
    root: Path = Path("data/cache/synthetic")
    device: str = "cpu" # "cpu" or "cuda"


@dataclass
class RealWorldConfig:
    name: Literal["all", "facebook_penn94", "lastfm_asia", "polblogs", "ppi"] = "all"
    root: Path = Path("data/cache/realworld")
    device: str = "cpu" # "cpu" or "cuda"


def main() -> None:
    config = tyro.cli(Union[SyntheticConfig, RealWorldConfig])
    device = torch.device(config.device)

    if isinstance(config, SyntheticConfig):
        families = FAMILIES if config.name == "all" else (config.name,)
        noise_types = NOISE_TYPES if config.noise_type == "all" else (config.noise_type,)
        precompute(root=config.root, families=families, noise_types=noise_types, device=device)
    else:
        dataset_filter = None if config.name == "all" else _RW_NAME_TO_DATASET[config.name]
        precompute_real_world(root=config.root, dataset_filter=dataset_filter, device=device)


if __name__ == "__main__":
    main()
