"""Central registry for task labels and evaluation thresholds.

Every magic number that previously lived inline in ``scripts/`` should migrate
here so thresholds are defined exactly once.
"""

# ── Canonical task labels (must match `task_description` attrs in the HDF5 demos) ──

SINGLE_ARM_TASK_LABELS = [
    "reach the red cube",
    "grasp the red cube",
    "pick up the red cube",
    "place the red cube on the blue plate",
]

BIMANUAL_TASK_LABELS = [
    "pick up the green box with both hands",
]

ALL_TASK_LABELS = SINGLE_ARM_TASK_LABELS + BIMANUAL_TASK_LABELS

# ── Success / trigger thresholds (metres unless noted) ──

# Distance below which the scripted auto-grasp fires in kinematic single-arm mode.
AUTO_GRASP_DIST = 0.04
# Hand-to-cube distance that counts as a successful reach. Kept intentionally at
# the auto-grasp threshold + 2 cm margin; see docs/CODEBASE_REVIEW.md §2.5 for
# why these two must be calibrated together.
REACH_SUCCESS_DIST = 0.06
# Cube height that counts as "picked up".
PICK_HEIGHT_Z = 0.90
# Cube height that counts as "back on the table" for place.
PLACE_MAX_Z = 0.87
# Table surface height used when settling objects.
TABLE_SURFACE_Z = 0.825
# Bimanual: minimum lift (cm) and per-palm normal force (N) for success.
BIMANUAL_MIN_LIFT_CM = 3.0
BIMANUAL_MIN_FORCE_N = 2.0

# ── Control rates ──

CONTROL_HZ = 30
PHYSICS_HZ = 500

# ── Inference defaults (temporal ensembling) ──

CHUNK_SIZE = 20
CHUNK_EXEC = 5  # steps executed before re-planning
ENSEMBLE_K = 0.01  # exponential decay factor for chunk weighting

# ── Image pipeline ──

IMAGE_SIZE = 224
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
