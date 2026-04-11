import numpy as np

from methods.feature_signal import compute_feature_signal


def make_feature_dataset(
    n_per_class: int = 50,
    n_features: int = 8,
    signal_strength: float = 2.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    n = len(labels)

    X = rng.normal(size=(n, n_features))
    X[:n_per_class, 0] += signal_strength
    X[n_per_class:, 0] -= signal_strength

    return X, labels


def test_feature_signal_true_features_exceed_shuffled_labels():
    X, labels = make_feature_dataset(
        n_per_class=60,
        n_features=10,
        signal_strength=2.0,
        seed=1,
    )

    n = len(labels)
    train_idx = np.arange(0, int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)

    true_result = compute_feature_signal(
        X,
        labels,
        train_idx,
        test_idx,
        n_null_trials=20,
        random_state=0,
    )

    shuffled_labels = labels.copy()
    rng = np.random.default_rng(123)
    rng.shuffle(shuffled_labels)

    shuffled_result = compute_feature_signal(
        X,
        shuffled_labels,
        train_idx,
        test_idx,
        n_null_trials=20,
        random_state=0,
    )

    assert true_result["feature_signal_raw"] > shuffled_result["feature_signal_raw"]


def test_feature_signal_informative_features_exceed_random_features():
    X_signal, labels = make_feature_dataset(
        n_per_class=60,
        n_features=10,
        signal_strength=2.0,
        seed=2,
    )

    rng = np.random.default_rng(999)
    X_random = rng.normal(size=X_signal.shape)

    n = len(labels)
    train_idx = np.arange(0, int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)

    signal_result = compute_feature_signal(
        X_signal,
        labels,
        train_idx,
        test_idx,
        n_null_trials=20,
        random_state=0,
    )

    random_result = compute_feature_signal(
        X_random,
        labels,
        train_idx,
        test_idx,
        n_null_trials=20,
        random_state=0,
    )

    assert signal_result["feature_signal_raw"] > random_result["feature_signal_raw"]