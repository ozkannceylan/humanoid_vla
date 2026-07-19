"""Shared fixtures: synthetic HDF5 demo episodes matching the real schema."""

import numpy as np
import pytest

CAM_H, CAM_W = 48, 64  # small stand-in for 480x640 (resized to 224 anyway)


def _write_episode(path, T, dim, task, success=True, seed=0):
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("obs/joint_positions", data=rng.normal(size=(T, dim)).astype(np.float32))
        f.create_dataset("obs/joint_velocities", data=rng.normal(size=(T, dim)).astype(np.float32))
        f.create_dataset(
            "obs/camera_frames",
            data=rng.integers(0, 255, size=(T, CAM_H, CAM_W, 3), dtype=np.uint8),
        )
        f.create_dataset("action", data=rng.normal(size=(T, dim)).astype(np.float32))
        f.attrs["task_description"] = task
        f.attrs["success"] = success


@pytest.fixture
def demo_dir(tmp_path):
    """8 tiny bimanual-shaped episodes (dim=14) across 2 tasks; one failure."""
    tasks = ["pick up the red cube", "grasp the red cube"]
    for i in range(8):
        _write_episode(
            tmp_path / f"episode_{i:04d}.hdf5",
            T=12,
            dim=14,
            task=tasks[i % 2],
            success=(i != 7),
            seed=i,
        )
    return str(tmp_path)
