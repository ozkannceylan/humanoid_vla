from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCENE_XML = PROJECT_ROOT / "sim" / "g1_with_camera.xml"
SOURCE_MODEL_XML = PROJECT_ROOT / "sim" / "models" / "g1_29dof.xml"
GENERATED_ASSETS_DIR = PROJECT_ROOT / "v2" / ".generated"

# ── Robot layout ─────────────────────────────────────────
NUM_ACTUATORS = 29

# Arm actuator indices (into data.ctrl[])
LEFT_ARM_CTRL = np.array([15, 16, 17, 18, 19, 20, 21], dtype=np.int32)
RIGHT_ARM_CTRL = np.array([22, 23, 24, 25, 26, 27, 28], dtype=np.int32)
BOTH_ARMS_CTRL = np.concatenate([LEFT_ARM_CTRL, RIGHT_ARM_CTRL])  # 14

# Arm DOF indices (into data.qvel[])
LEFT_ARM_DOF = np.array([21, 22, 23, 24, 25, 26, 27], dtype=np.int32)
RIGHT_ARM_DOF = np.array([28, 29, 30, 31, 32, 33, 34], dtype=np.int32)

ACTUATED_DOF_START = 6   # qvel offset — floating base uses 6 dof
ACTUATED_DOF_END = 35

# PD controller gains (29 actuators)
# fmt: off
KP = np.array([
    44, 44, 44, 70, 25, 25,         # left leg
    44, 44, 44, 70, 25, 25,         # right leg
    44, 25, 25,                      # waist
    40, 40, 40, 40, 10, 10, 10,     # left arm
    40, 40, 40, 40, 10, 10, 10,     # right arm
], dtype=np.float64)
KD = np.array([
     4,  4,  4,  7, 2.5, 2.5,
     4,  4,  4,  7, 2.5, 2.5,
     4, 2.5, 2.5,
     4,  4,  4,  4,  1,  1,  1,    # left arm
     4,  4,  4,  4,  1,  1,  1,    # right arm
], dtype=np.float64)
# fmt: on

# ── Physics ──────────────────────────────────────────────
PHYSICS_HZ = 500
DEFAULT_CONTROL_HZ = 30
SUBSTEPS = PHYSICS_HZ // DEFAULT_CONTROL_HZ  # ~16-17

# ── Scene objects ────────────────────────────────────────
TABLE_BODY_NAME = "table"
TABLE_CENTER = np.array([0.3, -0.1, 0.8], dtype=np.float64)
TABLE_HALF_EXTENTS = np.array([0.2, 0.2, 0.4], dtype=np.float64)

BOX_BODY_NAME = "green_box"
BOX_GEOM_NAME = "box_geom"
BOX_VISUAL_GEOM_NAME = "box_visual"
BOX_JOINT_NAME = "box_joint"
BOX_NOMINAL_POS = np.array([0.3, 0.0, 0.875], dtype=np.float64)

LEFT_HAND_SITE = "left_hand_site"
RIGHT_HAND_SITE = "right_hand_site"
LEFT_PALM_GEOM = "left_palm_pad"
RIGHT_PALM_GEOM = "right_palm_pad"

# Camera
EGO_CAMERA_NAME = "ego_camera"

# ── Trajectory ──────────────────────────────────────────
# Offsets relative to box centre for bimanual squeeze approach
PRE_APPROACH_OFFSET_Y = 0.15   # 15 cm to each side
PRE_APPROACH_OFFSET_Z = 0.12   # 12 cm above
APPROACH_OFFSET_Y     = 0.12   # 12 cm to each side (5 cm from surface)
SQUEEZE_OFFSET_Y      = 0.04   # 4 cm from centre → 3.5 cm inside surface
LIFT_OFFSET_Z         = 0.15   # 15 cm lift height

# Success criteria
MIN_LIFT_CM = 3.0
MIN_FORCE_N = 2.0


def ensure_generated_dir() -> Path:
    GENERATED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_ASSETS_DIR


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def euler_deg_to_quat(euler_deg: np.ndarray) -> np.ndarray:
    euler = np.deg2rad(euler_deg).astype(np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, "xyz")
    return quat


def mat_to_quat(mat9: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=np.float64))
    return quat


def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return radius * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=np.float64)
