"""Natural-language command routing for the ROS 2 task manager.

Replaces the buggy keyword matcher that lived in ``task_manager_node.py``:

- ``"green"`` / bare ``"box"`` no longer hijack single-arm commands
  ("reach the green cube" used to misroute to bimanual);
- unknown commands return ``(None, None)`` instead of silently executing
  "pick up the red cube".

This module is dependency-free so it can be unit-tested without ROS.
"""

from __future__ import annotations

from humanoid_vla.constants import BIMANUAL_TASK_LABELS, SINGLE_ARM_TASK_LABELS

# Phrases that unambiguously request the two-handed task.
_BIMANUAL_PHRASES = (
    "both hands",
    "both arms",
    "two hands",
    "two hand",
    "bimanual",
    "dual arm",
)

# Verbs that, combined with "box", imply the bimanual box task
# ("lift the box", "pick up the green box", ...).
_BOX_VERBS = ("lift", "pick", "grab", "hold", "raise")

_SINGLE_ARM_ALIASES = {
    "reach": SINGLE_ARM_TASK_LABELS[0],
    "grasp": SINGLE_ARM_TASK_LABELS[1],
    "pick": SINGLE_ARM_TASK_LABELS[2],
    "place": SINGLE_ARM_TASK_LABELS[3],
    "put": SINGLE_ARM_TASK_LABELS[3],
}


def parse_task_command(text: str) -> tuple[str | None, str | None]:
    """Parse a natural-language command into ``(mode, task_label)``.

    Returns:
        ("single_arm" | "bimanual", canonical task label), or ``(None, None)``
        if the command cannot be routed. Callers must surface the failure to
        the user instead of guessing.
    """
    text_lower = " ".join(text.strip().lower().split())
    if not text_lower:
        return None, None

    # Explicit bimanual intent.
    if any(phrase in text_lower for phrase in _BIMANUAL_PHRASES):
        return "bimanual", BIMANUAL_TASK_LABELS[0]

    # "lift/pick/... the box" style commands: box manipulation is bimanual-only.
    if "box" in text_lower and any(verb in text_lower for verb in _BOX_VERBS):
        return "bimanual", BIMANUAL_TASK_LABELS[0]

    # Exact canonical labels.
    if text_lower in SINGLE_ARM_TASK_LABELS:
        return "single_arm", text_lower
    if text_lower in BIMANUAL_TASK_LABELS:
        return "bimanual", text_lower

    # Single-arm verb aliases. "place"/"put" checked before "pick" so that
    # "put the cube on the plate" is not swallowed by the "pick" alias.
    for alias in ("place", "put", "reach", "grasp", "pick"):
        if alias in text_lower:
            return "single_arm", _SINGLE_ARM_ALIASES[alias]

    return None, None
