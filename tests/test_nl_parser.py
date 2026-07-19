import pytest

from humanoid_vla.nl_parser import parse_task_command


@pytest.mark.parametrize(
    ("text", "mode", "label"),
    [
        ("pick up the red cube", "single_arm", "pick up the red cube"),
        ("reach", "single_arm", "reach the red cube"),
        ("grasp the red cube", "single_arm", "grasp the red cube"),
        ("place it on the plate", "single_arm", "place the red cube on the blue plate"),
        ("put the cube on the plate", "single_arm", "place the red cube on the blue plate"),
        (
            "pick up the green box with both hands",
            "bimanual",
            "pick up the green box with both hands",
        ),
        ("lift the box", "bimanual", "pick up the green box with both hands"),
        ("bimanual grasp", "bimanual", "pick up the green box with both hands"),
        ("use two hands to lift it", "bimanual", "pick up the green box with both hands"),
    ],
)
def test_routing(text, mode, label):
    assert parse_task_command(text) == (mode, label)


def test_green_cube_regression():
    # Regression: "green" alone used to hijack single-arm commands to bimanual.
    mode, label = parse_task_command("reach the green cube")
    assert mode == "single_arm"
    assert label == "reach the red cube"


def test_unknown_commands_are_rejected():
    # Regression: unknown text used to silently execute "pick up the red cube".
    assert parse_task_command("dance the macarena") == (None, None)
    assert parse_task_command("") == (None, None)
    assert parse_task_command("   ") == (None, None)


def test_box_without_verb_is_not_bimanual():
    assert parse_task_command("where is the box") == (None, None)
