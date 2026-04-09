import numpy as np
import networkx as nx

from methods.esnr import compute_esnr_from_graph


def make_two_block_graph(
    block_size: int = 10,
    p_in: float = 0.8,
    p_out: float = 0.05,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = 2 * block_size
    labels = np.array([0] * block_size + [1] * block_size)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            same = labels[i] == labels[j]
            p = p_in if same else p_out
            if rng.random() < p:
                G.add_edge(i, j)

    return G, labels


def make_er_graph(n: int, p: float, seed: int = 0):
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    if sorted(G.nodes()) != list(range(n)):
        G.add_nodes_from(range(n))
    return G


def test_esnr_structured_exceeds_random():
    G_structured, labels = make_two_block_graph(
        block_size=10,
        p_in=0.8,
        p_out=0.05,
        seed=1,
    )

    # approximate density match
    n = len(labels)
    m_structured = G_structured.number_of_edges()
    p_er = 2 * m_structured / (n * (n - 1))

    G_random = make_er_graph(n=n, p=p_er, seed=2)

    structured_result = compute_esnr_from_graph(G_structured, labels)
    random_result = compute_esnr_from_graph(G_random, labels)

    assert structured_result["esnr"] > random_result["esnr"]


def test_esnr_true_labels_exceed_shuffled_labels():
    G, true_labels = make_two_block_graph(
        block_size=12,
        p_in=0.8,
        p_out=0.05,
        seed=2,
    )

    shuffled_labels = true_labels.copy()
    rng = np.random.default_rng(123)
    rng.shuffle(shuffled_labels)

    true_result = compute_esnr_from_graph(G, true_labels)
    shuffled_result = compute_esnr_from_graph(G, shuffled_labels)

    assert true_result["esnr"] > shuffled_result["esnr"]