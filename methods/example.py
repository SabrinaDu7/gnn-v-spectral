"""Usage example: load all four real-world graphs and compute test ARI for every method."""

from __future__ import annotations

from data import load_graph_data
from methods.registry import METHOD_REGISTRY, ExperimentConfig

METADATA_CSV  = "data/cache/realworld/metadata/graph_index_realworld.csv"
REAL_WORLD  = True

GRAPH_IDS = [
    # "polblogs_gcc",
    # "lastfm_asia",
    # "facebook_penn94_residence_min50_gcc",
    "ppi",
    # "graph001_045_targeted_betweenness_sbm",
]

for graph_id in GRAPH_IDS:
    print(f"\n{'='*60}")
    print(f"Graph: {graph_id}")
    print(f"{'='*60}")

    data = load_graph_data(
        metadata_csv=METADATA_CSV,
        graph_id=graph_id,
        dataset_root="data/cache/" + ("realworld" if REAL_WORLD else "synthetic"),
    )

    config = ExperimentConfig(
        num_classes=data.num_classes,
        seed=0,
        hidden_dim=32,
        num_layers=2,
        lr=1e-2,
        epochs=50,
        dropout=0.0,
        num_heads=2,
        k_hops=2,
        n_estimators=100,
    )

    methods = {name: ctor(config) for name, ctor in METHOD_REGISTRY.items()}

    for method_name, method in methods.items():
        embedding = None
        for spectral_type in ("whole", "kcut", "regularized"):
            if method_name.startswith(spectral_type):
                embedding = getattr(data, f"{spectral_type}_eigenspectrum")
                break

        method.fit(data, embeddings=embedding)
        result_train = method.score(data, split="train")
        result_val = method.score(data, split="val")
        result_test = method.score(data, split="test")
        print(f"  {method_name:<20s}  ARI={result_train['ARI']:.4f} (train)")
        print(f"  {method_name:<20s}  ARI={result_val['ARI']:.4f} (val)")
        print(f"  {method_name:<20s}  ARI={result_test['ARI']:.4f} (test)")