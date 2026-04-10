from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.real_world.loaders import load_real_world_graph
from data.real_world.characterize import (
    extract_largest_connected_component,
    basic_graph_properties,
    feature_signal_properties,
)


def main() -> None:
    rows = []

    # first scan a reasonable range of label indices
    for label_index in range(121):
        try:
            graph = load_real_world_graph(
                "ppi_adapted",
                raw_dir="data/raw/ppi",
                graph_index=0,
                label_index=label_index,
            )
            graph = extract_largest_connected_component(graph)

            labels = graph.labels
            values, counts = np.unique(labels, return_counts=True)

            # skip degenerate labels
            if len(values) != 2:
                continue

            neg = int(counts[0]) if values[0] == 0 else int(counts[1])
            pos = int(counts[1]) if values[1] == 1 else int(counts[0])
            pos_frac = pos / len(labels)

            # cheap screening
            if pos < 100:
                continue
            if pos_frac < 0.2 or pos_frac > 0.8:
                continue

            props = basic_graph_properties(graph)

            fs = feature_signal_properties(
                graph,
                n_splits=2,
                test_size=0.3,
                n_null_trials=3,
                random_state=0,
            )

            rows.append({
                "label_index": label_index,
                "n_nodes": props["n_nodes"],
                "n_edges": props["n_edges"],
                "pos_count": pos,
                "neg_count": neg,
                "pos_frac": pos_frac,
                "esnr": props["esnr"],
                "feature_signal_norm": fs["feature_signal_norm"],
                "feature_macro_f1": fs["feature_macro_f1"],
            })

            print(
                f"label={label_index:03d} "
                f"pos_frac={pos_frac:.3f} "
                f"esnr={props['esnr']:.3f} "
                f"fs={fs['feature_signal_norm']:.3f}"
            )

        except Exception as e:
            print(f"label={label_index:03d} failed: {e}")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["feature_signal_norm", "esnr"],
        ascending=[False, False],
    )
    print("\nTop candidates:")
    print(df.head(10).to_string(index=False))

    out_path = Path("data/cache/realworld/metadata/ppi_label_scan.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved scan to {out_path}")


if __name__ == "__main__":
    main()