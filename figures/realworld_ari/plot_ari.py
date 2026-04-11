"""Plot grouped bar charts of train/val/test ARI per method for a given dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "realworld_ari"
FIGURES_DIR = Path(__file__).parent

BAR_WIDTH = 0.25
GROUP_GAP = 0.15  # extra spacing between method groups
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
DATASETS = [
    "facebook_penn94_residence_min50_gcc",
    "lastfm_asia",
    "polblogs_gcc",
    "ppi",
]
DATASET_LABELS = {
    "facebook_penn94_residence_min50_gcc": "facebook",
    "lastfm_asia": "lastfm",
    "polblogs_gcc": "polblogs",
    "ppi": "ppi",
}
DATASET_COLORS = {
    "facebook_penn94_residence_min50_gcc": "#4C72B0",
    "lastfm_asia": "#DD8452",
    "polblogs_gcc": "#55A868",
    "ppi": "#C44E52",
}


def plot_ari_by_dataset(dataset: str, *, save: bool = True) -> matplotlib.figure.Figure:
    """Generate a grouped bar chart of ARI (train/val/test) for each method.

    Parameters
    ----------
    dataset:
        Dataset name matching the CSV filename stem, e.g. ``"polblogs_gcc"``.
    save:
        If True, save the figure to ``figures/<dataset>_ari.png``.

    Returns
    -------
    plt.Figure
    """
    csv_path = RESULTS_DIR / f"{dataset}.csv"
    df = pd.read_csv(csv_path, index_col="method")

    methods = df.index.tolist()
    splits = ["train", "val", "test"]
    n = len(methods)

    # x positions: each group is centered, with extra gap between groups
    group_width = 3 * BAR_WIDTH + GROUP_GAP
    centers = np.arange(n) * group_width
    offsets = np.array([-BAR_WIDTH, 0, BAR_WIDTH])

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 6))

    for split, offset, color in zip(splits, offsets, SPLIT_COLORS.values()):
        values = df[split].to_numpy(dtype=float)
        bars = ax.bar(
            centers + offset,
            np.where(np.isnan(values), 0, values),
            width=BAR_WIDTH,
            label=split,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            label_y = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("ARI")
    ax.set_title(f"ARI by method — {dataset}")
    ax.legend(title="split")
    ax.set_ylim(min(df.min(numeric_only=True).min() - 0.15, -0.05), 1.18)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / f"{dataset}_ari.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


def plot_ari_by_split(split: str, *, save: bool = True) -> matplotlib.figure.Figure:
    """Generate a grouped bar chart of ARI across datasets for each method.

    One bar per dataset within each method group, for a single split.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    save:
        If True, save the figure to ``figures/ari_by_split_<split>.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    # Load all datasets; align on the union of methods preserving order from first file
    frames: dict[str, pd.Series] = {}
    for ds in DATASETS:
        df = pd.read_csv(RESULTS_DIR / f"{ds}.csv", index_col="method")
        frames[ds] = df[split]

    combined = pd.DataFrame(frames)  # rows=methods, columns=datasets
    methods = combined.index.tolist()
    n = len(methods)
    n_datasets = len(DATASETS)

    total_bar_span = n_datasets * BAR_WIDTH
    group_width = total_bar_span + GROUP_GAP
    centers = np.arange(n) * group_width
    offsets = np.linspace(-total_bar_span / 2 + BAR_WIDTH / 2,
                          total_bar_span / 2 - BAR_WIDTH / 2,
                          n_datasets)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    for ds, offset in zip(DATASETS, offsets):
        values = combined[ds].to_numpy(dtype=float)
        color = DATASET_COLORS[ds]
        bars = ax.bar(
            centers + offset,
            np.where(np.isnan(values), 0, values),
            width=BAR_WIDTH,
            label=DATASET_LABELS[ds],
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            label_y = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("ARI")
    ax.set_title(f"ARI by method — {split} split")
    ax.legend(title="dataset")
    all_vals = combined.to_numpy(dtype=float)
    ax.set_ylim(min(np.nanmin(all_vals) - 0.15, -0.05), 1.18)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / f"{split}_ari.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    for ds in DATASETS:
        plot_ari_by_dataset(ds)
    for sp in ("train", "val", "test"):
        plot_ari_by_split(sp)
