import numpy as np

from data.generators.sbm import SBMConfig, generate_sbm


def test_generate_sbm_basic():
    config = SBMConfig()
    G, labels, metadata = generate_sbm(config, seed=0)

    assert G.number_of_nodes() == 1000
    assert len(labels) == 1000
    assert metadata["family"] == "sbm"
    assert metadata["num_communities"] == 5
    assert metadata["p_in"] > metadata["p_out"]

    assert min(G.nodes()) == 0
    assert max(G.nodes()) == 999

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64

    unique_labels = np.unique(labels)
    assert len(unique_labels) == 5
    assert set(unique_labels) == {0, 1, 2, 3, 4}

from data.generators.lfr import LFRConfig, generate_lfr


def test_generate_lfr_basic():
    config = LFRConfig()
    G, labels, metadata = generate_lfr(config, seed=0)

    assert G.number_of_nodes() == 1000
    assert len(labels) == 1000
    assert metadata["family"] == "lfr"
    assert metadata["num_communities"] >= 2

    assert min(G.nodes()) == 0
    assert max(G.nodes()) == 999

    assert sum(metadata["community_sizes"]) == 1000



from data.generators.characterize import compute_all_graph_stats

def test_compute_all_graph_stats_sbm():
    G, labels, _ = generate_sbm(SBMConfig(), seed=0)
    stats = compute_all_graph_stats(G, labels)

    assert stats["n_nodes"] == 1000
    assert stats["num_edges"] == G.number_of_edges()
    assert stats["avg_degree"] > 0
    assert 0.0 <= stats["density"] <= 1.0
    assert 0.0 <= stats["sparsity"] <= 1.0
    assert stats["num_connected_components"] >= 1
    assert 0.0 <= stats["largest_cc_fraction"] <= 1.0
    assert stats["num_communities"] == 5
    assert 0.0 <= stats["intercommunity_edge_fraction"] <= 1.0
    assert 0.0 <= stats["heterophily"] <= 1.0


from pathlib import Path
import pandas as pd

from data.generators.characterize import compute_all_graph_stats
from data.generators.io import (
    format_base_graph_id,
    format_noise_code,
    make_graph_id,
    make_output_paths,
    save_graph_edgelist,
    save_labels,
    write_metadata_csv,
)
from data.generators.sbm import SBMConfig, generate_sbm


def test_io_end_to_end(tmp_path: Path):
    G, labels, metadata = generate_sbm(SBMConfig(), seed=0)
    stats = compute_all_graph_stats(G, labels)

    base_graph_id = format_base_graph_id(1)
    noise_code = format_noise_code(0.0)
    graph_id = make_graph_id(base_graph_id, noise_code, "clean", "sbm")
    paths = make_output_paths(tmp_path, "sbm", "clean", graph_id)

    save_graph_edgelist(G, labels, paths["edge_path"])
    save_labels(labels, paths["label_path"])

    assert paths["edge_path"].exists()
    assert paths["label_path"].exists()

    df = pd.read_csv(paths["edge_path"])
    assert list(df.columns) == ["src", "dst", "same_comm", "comm_pair"]

    loaded_labels = np.load(paths["label_path"])
    assert len(loaded_labels) == G.number_of_nodes()

    row = {
        "graph_id": graph_id,
        "family": "sbm",
        "base_graph_id": base_graph_id,
        "seed": 0,
        "noise_type": "clean",
        "noise_code": noise_code,
        "noise_frac": 0.0,
        "edge_path": str(paths["edge_path"]),
        "label_path": str(paths["label_path"]),
        **stats,
        **metadata,
    }

    write_metadata_csv([row], paths["metadata_path"])
    assert paths["metadata_path"].exists()
    