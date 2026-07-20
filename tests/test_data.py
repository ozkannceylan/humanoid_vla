import numpy as np
import torch

from humanoid_vla.data import ChunkDataset, EpisodeStore


def test_store_infers_dims_and_tasks(demo_dir):
    store = EpisodeStore(demo_dir)
    assert store.state_dim == 28
    assert store.action_dim == 14
    assert store.task_labels == ["grasp the red cube", "pick up the red cube"]
    assert len(store.episodes) == 8


def test_filter_success(demo_dir):
    store = EpisodeStore(demo_dir, filter_success=True)
    assert len(store.episodes) == 7


def test_split_stratified_and_disjoint(demo_dir):
    store = EpisodeStore(demo_dir)
    train_idx, val_idx = store.split(val_frac=0.25, seed=0)
    assert not set(train_idx) & set(val_idx)
    assert len(train_idx) + len(val_idx) == 8
    # Stratified: every task appears in validation.
    val_tasks = {store.episodes[i]["task"] for i in val_idx}
    assert val_tasks == set(store.task_labels)
    # Deterministic given the seed.
    assert store.split(val_frac=0.25, seed=0) == (train_idx, val_idx)
    assert store.split(val_frac=0.25, seed=1) != (train_idx, val_idx)


def test_norm_stats_from_train_split_only(demo_dir):
    store = EpisodeStore(demo_dir)
    train_idx, _ = store.split(val_frac=0.25, seed=0)
    stats = store.norm_stats(train_idx)
    assert stats.state_mean.shape == (28,)
    assert stats.action_std.shape == (14,)
    assert (stats.action_std > 0).all()


def test_dataset_shapes_and_chunk_padding(demo_dir):
    store = EpisodeStore(demo_dir)
    ds = ChunkDataset(store, list(range(8)), chunk_size=20)
    assert len(ds) == 8 * 12
    img, state, task_id, chunk = ds[len(ds) - 1]  # last frame → fully padded chunk
    assert img.shape == (3, 224, 224)
    assert state.shape == (28,)
    assert isinstance(task_id, int)
    assert chunk.shape == (20, 14)
    # Padding repeats the final action.
    assert torch.equal(chunk[-1], chunk[11])


def test_augmentation_changes_pixels_not_labels(demo_dir):
    store = EpisodeStore(demo_dir)
    plain = ChunkDataset(store, [0], chunk_size=5, augment=False)
    aug = ChunkDataset(store, [0], chunk_size=5, augment=True)
    torch.manual_seed(0)
    img_a, state_a, tid_a, chunk_a = aug[0]
    img_p, state_p, tid_p, chunk_p = plain[0]
    assert img_a.shape == img_p.shape
    assert not torch.equal(img_a, img_p)  # jitter fired
    assert torch.equal(state_a, state_p)
    assert torch.equal(chunk_a, chunk_p)
    assert tid_a == tid_p


def test_state_matches_source_arrays(demo_dir):
    store = EpisodeStore(demo_dir)
    ds = ChunkDataset(store, [2], chunk_size=5)
    _, state, _, _ = ds[0]
    ep = store.episodes[2]
    expected = np.concatenate([ep["positions"][0], ep["velocities"][0]])
    np.testing.assert_allclose(state.numpy(), expected)
