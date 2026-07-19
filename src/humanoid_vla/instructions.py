"""Instruction paraphrase corpus for language conditioning (Upgrade Plan Phase 2).

Each canonical task label carries a set of *training* paraphrases (sampled per
episode during training so the policy sees varied phrasings) and a disjoint set
of *held-out* paraphrases used only at evaluation time. Reporting success on
the held-out split is the paraphrase-robustness metric that an integer task-id
embedding fails by construction (cf. LIBERO-Para, which measures 22–52 pp drops
for policies that shortcut language).
"""

from __future__ import annotations

import random

from humanoid_vla.constants import BIMANUAL_TASK_LABELS, SINGLE_ARM_TASK_LABELS

_REACH, _GRASP, _PICK, _PLACE = SINGLE_ARM_TASK_LABELS
_BIMANUAL = BIMANUAL_TASK_LABELS[0]

# fmt: off
INSTRUCTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    _REACH: {
        "train": [
            "reach the red cube",
            "reach toward the red cube",
            "move your hand to the red cube",
            "extend your arm to the red cube",
            "bring your hand close to the red cube",
            "reach out to the red block",
            "move the gripper to the red cube",
            "approach the red cube with your hand",
            "get your hand near the red cube",
            "stretch your arm toward the red block",
            "reach for the small red cube on the table",
            "move your right hand toward the red object",
        ],
        "heldout": [
            "put your hand next to the red cube",
            "go touch the red block",
            "extend toward the crimson cube",
            "close the distance to the red cube",
        ],
    },
    _GRASP: {
        "train": [
            "grasp the red cube",
            "grab the red cube",
            "take hold of the red cube",
            "grip the red cube",
            "close your hand around the red cube",
            "grab the red block",
            "get a grip on the red cube",
            "clutch the red cube",
            "take the red cube in your hand",
            "secure the red cube in your gripper",
            "grasp the small red block on the table",
            "wrap your fingers around the red cube",
        ],
        "heldout": [
            "seize the red cube",
            "snatch up the red block",
            "get hold of the crimson cube",
            "clasp the red cube firmly",
        ],
    },
    _PICK: {
        "train": [
            "pick up the red cube",
            "lift the red cube",
            "pick the red cube up off the table",
            "raise the red cube",
            "lift the red block off the table",
            "pick up the red block",
            "lift up the red cube",
            "hoist the red cube",
            "take the red cube off the table",
            "elevate the red cube",
            "pick up the small red cube",
            "lift the red object into the air",
        ],
        "heldout": [
            "bring the red cube up",
            "get the red block off the table",
            "raise the crimson cube into the air",
            "scoop up the red cube",
        ],
    },
    _PLACE: {
        "train": [
            "place the red cube on the blue plate",
            "put the red cube on the blue plate",
            "set the red cube down on the blue plate",
            "move the red cube onto the blue plate",
            "place the red block on the blue plate",
            "put the red cube onto the blue dish",
            "drop the red cube on the blue plate",
            "set the red block on the blue plate",
            "transfer the red cube to the blue plate",
            "deposit the red cube on the blue plate",
            "carry the red cube over to the blue plate",
            "put the red block down on the blue dish",
        ],
        "heldout": [
            "rest the red cube on the blue plate",
            "lay the red block on the blue dish",
            "move the crimson cube onto the blue plate",
            "position the red cube over the blue plate and release it",
        ],
    },
    _BIMANUAL: {
        "train": [
            "pick up the green box with both hands",
            "lift the green box with both hands",
            "use both hands to pick up the green box",
            "grab the green box with two hands",
            "lift the box with both arms",
            "pick up the box using both hands",
            "hold the green box with both hands and lift it",
            "squeeze the green box between your hands and raise it",
            "lift the green container with two hands",
            "raise the green box using both arms",
            "pick the big green box up with both hands",
            "carry the green box with two hands",
        ],
        "heldout": [
            "hoist the green box with both arms",
            "clamp the box between your palms and lift",
            "use two hands to raise the green box",
            "lift up the large green box with both hands",
        ],
    },
}
# fmt: on


def get_instructions(task_label: str, split: str = "train") -> list[str]:
    """Return the paraphrase list for a canonical task label.

    Args:
        task_label: canonical label (a key of :data:`INSTRUCTION_TEMPLATES`).
        split: ``"train"`` or ``"heldout"``.
    """
    if task_label not in INSTRUCTION_TEMPLATES:
        raise KeyError(f"Unknown task label: {task_label!r}")
    if split not in ("train", "heldout"):
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")
    return list(INSTRUCTION_TEMPLATES[task_label][split])


def sample_instruction(task_label: str, rng: random.Random | None = None) -> str:
    """Sample one training paraphrase for a task (uniform)."""
    options = INSTRUCTION_TEMPLATES[task_label]["train"]
    return (rng or random).choice(options)
