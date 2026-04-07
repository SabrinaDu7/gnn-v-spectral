"""Usage example: instantiate all methods and run one end-to-end."""

from __future__ import annotations

from data import load_graph_data
from methods.registry import METHOD_REGISTRY, ExperimentConfig

# ---------------------------------------------------------------------------
# Load a real graph from the synthetic cache
# ---------------------------------------------------------------------------
data = load_graph_data(
    metadata_csv="data/cache/synthetic/metadata/graph_index_sbm.csv",
    graph_id="graph001_045_targeted_betweenness_sbm",
)

# ---------------------------------------------------------------------------
# Config — all fields populated so every method can be instantiated
# ---------------------------------------------------------------------------
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
    n_estimators=100, # For Random Forest classifier
)

# ---------------------------------------------------------------------------
# Instantiate all 9 methods from the registry
# ---------------------------------------------------------------------------
methods = {name: ctor(config) for name, ctor in METHOD_REGISTRY.items()}
print("Instantiated methods:", list(methods.keys()))

# ---------------------------------------------------------------------------
# Run fit + score on "whole_lr"
# ---------------------------------------------------------------------------
spectral_type = "whole"  # Choose from "whole", "kcut", or "regularized"
embedding = getattr(data, f"{spectral_type}_eigenspectrum")

method_type_spectral = f"{spectral_type}_lr"
method_type_gnn = "sgc"

method = methods[method_type_gnn]
method.fit(data, embeddings=embedding) # gnn methods ignore the embeddings argument and use one-hot encodings as features

val_score  = method.score(data, split="val")
test_score = method.score(data, split="test")
train_score  = method.score(data, split="train")

print("val  score:", val_score)
print("test score:", test_score)
print("train score:", train_score)
