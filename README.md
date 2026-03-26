# When do Graph Neural Networks Outperform Spectral Methods under Structural Noise?

## Repo Structure
gnn-v-spectral/
│
├── data/                          # Jamie
│   ├── generators/
│   │   ├── sbm.py                 # SBM graph generation
│   │   ├── lfr.py                 # LFR graph generation
│   │   └── perturbations.py      # Random + targeted edge deletion
│   ├── real-world/
│   │   ├── loaders.py             # STRING, ABIDE, etc.
│   │   └── characterize.py        # Sparsity, heterophily, ESNR
│   └── cache/                     # Generated graphs saved here (gitignored)
│
├── methods/                       # Sabrina
│   ├── spectral/
│   │   ├── embeddings.py          # Full, filtered, regularized spectrum
│   │   └── classifiers.py        # Logistic regression, label propagation
│   ├── gnns/
│   │   ├── gcn.py
│   │   ├── gat.py
│   │   └── sgc.py
│   ├── hybrid/
│   │   ├── spectral-pe.py         # Spectral positional encodings → GNN
│   │   ├── lpagg.py               # Label propagation post-processing
│   │   └── appnp.py
│   ├── registry.py                # METHOD-REGISTRY lives here
│   └── metrics.py                 # ARI, NMI, ESNR, evaluate-all()
│
├── pipeline/                      # Finn
│   xx
│
├── notebooks/                     # Shared, for EDA and plotting
│
├── results/                       # Gitignored, populated at runtime
│
├── tests/
│
├── configs/                       # Saved ExperimentConfig JSON files
│   ├── sbm-baseline.json
│   └── lfr-high-noise.json
│
├── pyproject.toml
└── README.md


