"""humanoid_vla — training and evaluation stack for the G1 humanoid VLA project.

Installable package (``pip install -e .``) that replaces the loose ``scripts/``
modules for everything training-related:

- ``humanoid_vla.train``        — unified, reproducible trainer (single-arm + bimanual)
- ``humanoid_vla.data``         — unified HDF5 dataset with episode-level train/val split
- ``humanoid_vla.models.act``   — ACT policy v2 (spatial vision tokens, text conditioning)
- ``humanoid_vla.runner``       — the single temporal-ensembling implementation
- ``humanoid_vla.stats``        — Wilson confidence intervals for evaluation
- ``humanoid_vla.instructions`` — paraphrase corpus for language conditioning
- ``humanoid_vla.nl_parser``    — natural-language command routing (used by the ROS node)
"""

__version__ = "0.2.0"

from humanoid_vla.constants import BIMANUAL_TASK_LABELS, SINGLE_ARM_TASK_LABELS

__all__ = [
    "__version__",
    "SINGLE_ARM_TASK_LABELS",
    "BIMANUAL_TASK_LABELS",
]
