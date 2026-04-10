from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METADATA_CSV = Path("data/cache/realworld/metadata/graph_index_realworld.csv")
OUTPUT_PATH = Path("figures/realworld_esnr_vs_fs.png")

GRAPH_IDS = [
    "lastfm_asia",
    "facebook_penn94_residence_min50_gcc",
    "ppi",
]


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)

    needed_cols = {"graph_id", "esnr", "feature_signal_norm"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    plot_df = df[df["graph_id"].isin(GRAPH_IDS)].copy()

    if plot_df.empty:
        raise ValueError("No matching graph_ids found in metadata.")

    plot_df = plot_df.dropna(subset=["esnr", "feature_signal_norm"])

    if plot_df.empty:
        raise ValueError("All selected rows have missing ESNR or feature signal values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        plot_df["esnr"],
        plot_df["feature_signal_norm"],
        s=100,
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["graph_id"],
            (row["esnr"], row["feature_signal_norm"]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Structural signal (ESNR)")
    ax.set_ylabel("Feature signal (FS)")
    ax.set_title("Real-world graphs in ESNR vs FS space")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")
    print(plot_df[["graph_id", "esnr", "feature_signal_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()