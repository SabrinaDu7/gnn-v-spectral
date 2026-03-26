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