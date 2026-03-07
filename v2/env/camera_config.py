from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraDefinition:
    name: str
    pos: tuple[float, float, float]
    euler_deg: tuple[float, float, float]
    fovy: float = 70.0


STATIC_CAMERA_PRESETS: dict[str, CameraDefinition] = {
    "topdown": CameraDefinition(
        name="static_topdown_camera",
        pos=(0.30, -0.10, 1.20),
        euler_deg=(0.0, 0.0, 0.0),
        fovy=55.0,
    ),
    "corner": CameraDefinition(
        name="static_corner_camera",
        pos=(-0.05, -0.85, 1.50),
        euler_deg=(-35.0, 0.0, 20.0),
        fovy=65.0,
    ),
}

WRIST_CAMERA = CameraDefinition(
    name="wrist_camera",
    pos=(0.03, 0.0, 0.02),
    euler_deg=(-90.0, 0.0, 0.0),
    fovy=80.0,
)


def get_static_camera_definition(camera_type: str) -> CameraDefinition:
    try:
        return STATIC_CAMERA_PRESETS[camera_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported static camera type: {camera_type}") from exc


def jitter_vector(rng: np.random.Generator, magnitude: float, dims: int = 3) -> np.ndarray:
    return rng.uniform(-magnitude, magnitude, size=dims)
