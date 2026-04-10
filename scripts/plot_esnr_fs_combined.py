from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SYN_META_CSV = Path("data/cache/synthetic80/metadata/feature_informativeness_experiment_table.csv")
REAL_META_CSV = Path("data/cache/realworld/metadata/graph_index_realworld.csv")
OUTPUT_PATH = Path("figures/esnr_vs_fs_combined.png")

REAL_GRAPH_IDS = [
    "lastfm_asia",
    "facebook_penn94_residence_min50_gcc",
    "ppi",
]

REAL_LABELS = {
    "lastfm_asia": "LastFM",
    "facebook_penn94_residence_min50_gcc": "Penn94",
    "ppi": "PPI",
}


def main() -> None:
    if not SYN_META_CSV.exists():
        raise FileNotFoundError(f"Missing synthetic metadata file: {SYN_META_CSV}")
    if not REAL_META_CSV.exists():
        raise FileNotFoundError(f"Missing real-world metadata file: {REAL_META_CSV}")

    syn_df = pd.read_csv(SYN_META_CSV)
    real_df = pd.read_csv(REAL_META_CSV)

    required_syn = {"family", "esnr", "feature_signal_norm"}
    required_real = {"graph_id", "esnr", "feature_signal_norm"}
    missing_syn = required_syn - set(syn_df.columns)
    missing_real = required_real - set(real_df.columns)

    if missing_syn:
        raise ValueError(f"Missing synthetic columns: {missing_syn}")
    if missing_real:
        raise ValueError(f"Missing real-world columns: {missing_real}")

    syn_df = syn_df.dropna(subset=["esnr", "feature_signal_norm"]).copy()

    real_df = real_df[real_df["graph_id"].isin(REAL_GRAPH_IDS)].copy()
    real_df = real_df.dropna(subset=["esnr", "feature_signal_norm"])

    fig, ax = plt.subplots(figsize=(9, 7))

    # Synthetic background
    family_styles = {
        "sbm": {"label": "Synthetic SBM", "marker": "o"},
        "lfr": {"label": "Synthetic LFR", "marker": "s"},
    }

    for family, style in family_styles.items():
        subset = syn_df[syn_df["family"] == family]
        if subset.empty:
            continue

        ax.scatter(
            subset["esnr"],
            subset["feature_signal_norm"],
            s=20,
            alpha=0.25,
            marker=style["marker"],
            label=style["label"],
        )

    # Real-world foreground
    ax.scatter(
        real_df["esnr"],
        real_df["feature_signal_norm"],
        s=140,
        marker="X",
        label="Real-world",
        zorder=5,
    )

    for _, row in real_df.iterrows():
        label = REAL_LABELS.get(row["graph_id"], row["graph_id"])
        ax.annotate(
            label,
            (row["esnr"], row["feature_signal_norm"]),
            xytext=(8, 8),
            textcoords="offset points",
            zorder=6,
        )

    ax.set_xlabel("Structural signal (ESNR)")
    ax.set_ylabel("Feature signal (FS)")
    ax.set_title("Synthetic and real-world graphs in ESNR vs FS space")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Saved plot to: {OUTPUT_PATH}")
    print("\nReal-world points:")
    print(real_df[["graph_id", "esnr", "feature_signal_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()