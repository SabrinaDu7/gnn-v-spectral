from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


META_CSV = Path("data/cache/synthetic80/metadata/feature_informativeness_experiment_table.csv")
OUTPUT_PATH = Path("figures/synthetic_fs_sanity.png")


def main() -> None:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_CSV}")

    df = pd.read_csv(META_CSV)

    required_cols = {
        "family",
        "base_graph_id",
        "structural_noise_type",
        "structural_noise_frac",
        "feature_informativeness_frac",
        "feature_signal_norm",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pick one base graph for a clean first sanity check
    df = df[df["base_graph_id"] == "graph001"].copy()

    # Focus on a few representative conditions
    conditions = [
        ("sbm", "random", 0.15),
        ("sbm", "random", 0.45),
        ("lfr", "random", 0.15),
        ("lfr", "random", 0.45),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_any = False

    for family, noise_type, noise_frac in conditions:
        subset = df[
            (df["family"] == family)
            & (df["structural_noise_type"] == noise_type)
            & (df["structural_noise_frac"] == noise_frac)
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("feature_informativeness_frac")

        label = f"{family.upper()} | {noise_type} | noise={noise_frac:.2f}"

        ax.plot(
            subset["feature_informativeness_frac"],
            subset["feature_signal_norm"],
            marker="o",
            label=label,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No matching rows found for the selected conditions.")

    ax.set_xlabel("Feature informativeness")
    ax.set_ylabel("Feature signal (FS)")
    ax.set_title("Synthetic sanity check: FS vs feature informativeness")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()