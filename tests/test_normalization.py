import numpy as np

from humanoid_vla.normalization import NormStats, compute_norm_stats


def test_round_trip():
    rng = np.random.default_rng(0)
    stats = NormStats(
        state_mean=rng.normal(size=6).astype(np.float32),
        state_std=rng.uniform(0.5, 2.0, size=6).astype(np.float32),
        action_mean=rng.normal(size=3).astype(np.float32),
        action_std=rng.uniform(0.5, 2.0, size=3).astype(np.float32),
    )
    a = rng.normal(size=(10, 3)).astype(np.float32)
    np.testing.assert_allclose(stats.denormalize_action(stats.normalize_action(a)), a, rtol=1e-5)


def test_compute_stats_correctness():
    rng = np.random.default_rng(1)
    states = [rng.normal(loc=2.0, scale=3.0, size=(50, 4)) for _ in range(5)]
    actions = [rng.normal(loc=-1.0, scale=0.5, size=(50, 2)) for _ in range(5)]
    stats = compute_norm_stats(iter(states), iter(actions))

    all_states = np.concatenate(states)
    all_actions = np.concatenate(actions)
    np.testing.assert_allclose(stats.state_mean, all_states.mean(0), rtol=1e-4)
    np.testing.assert_allclose(stats.state_std, all_states.std(0), rtol=1e-3)
    np.testing.assert_allclose(stats.action_mean, all_actions.mean(0), rtol=1e-4)


def test_constant_dim_gets_std_floor():
    # Frozen joints produce zero-variance dims; std must be floored, not zero.
    states = [np.ones((10, 3))]
    actions = [np.zeros((10, 2))]
    stats = compute_norm_stats(iter(states), iter(actions))
    assert (stats.state_std > 0).all()
    assert (stats.action_std > 0).all()
    normalized = stats.normalize_state(np.ones(3))
    assert np.isfinite(normalized).all()


def test_serialization_round_trip():
    stats = NormStats.identity(4, 2)
    restored = NormStats.from_dict(stats.to_dict())
    np.testing.assert_array_equal(stats.state_mean, restored.state_mean)
    np.testing.assert_array_equal(stats.action_std, restored.action_std)
