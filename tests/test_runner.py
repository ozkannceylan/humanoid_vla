import numpy as np

from humanoid_vla.runner import TemporalEnsembler


def _make(chunk_size=4, action_dim=2, max_steps=20, k=0.01, chunk_exec=2):
    return TemporalEnsembler(chunk_size, action_dim, max_steps, k=k, chunk_exec=chunk_exec)


def test_single_chunk_passthrough():
    ens = _make()
    chunk = np.arange(8, dtype=np.float64).reshape(4, 2)
    ens.add_chunk(0, chunk)
    # With one chunk, weighting cancels: outputs equal the chunk rows.
    for i in range(4):
        np.testing.assert_allclose(ens.get_action(i), chunk[i])


def test_uncovered_step_returns_zeros():
    ens = _make()
    np.testing.assert_array_equal(ens.get_action(5), np.zeros(2))


def test_overlap_weighting_matches_reference():
    # Reference implementation: the inline loop the legacy scripts used.
    chunk_size, action_dim, k = 4, 2, 0.01
    ens = _make(chunk_size, action_dim, k=k)
    rng = np.random.default_rng(0)
    chunks = {0: rng.normal(size=(4, 2)), 2: rng.normal(size=(4, 2))}

    total = 24
    action_sum = np.zeros((total, action_dim))
    weight_sum = np.zeros(total)
    for start, chunk in chunks.items():
        ens.add_chunk(start, chunk)
        for i in range(chunk_size):
            w = np.exp(-k * i)
            action_sum[start + i] += w * chunk[i]
            weight_sum[start + i] += w

    for step in range(6):
        expected = action_sum[step] / weight_sum[step] if weight_sum[step] > 0 else np.zeros(2)
        np.testing.assert_allclose(ens.get_action(step), expected)


def test_newer_chunk_dominates_overlap():
    # At an overlapping step, the newer chunk has a smaller within-chunk offset
    # and therefore higher weight.
    ens = _make(chunk_size=4, k=0.5)
    ens.add_chunk(0, np.zeros((4, 2)))
    ens.add_chunk(2, np.ones((4, 2)))
    blended = ens.get_action(2)  # old chunk offset 2 (w=e^-1) vs new offset 0 (w=1)
    assert (blended > 0.5).all()


def test_reset_from_clears_future_only():
    ens = _make()
    ens.add_chunk(0, np.ones((4, 2)))
    ens.reset_from(1)
    np.testing.assert_allclose(ens.get_action(1), np.ones(2))
    np.testing.assert_array_equal(ens.get_action(2), np.zeros(2))


def test_needs_replan_cadence():
    ens = _make(chunk_exec=5)
    assert [s for s in range(12) if ens.needs_replan(s)] == [0, 5, 10]


def test_shape_validation():
    ens = _make()
    try:
        ens.add_chunk(0, np.zeros((3, 2)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong chunk shape")
