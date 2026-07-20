import random

import pytest

from humanoid_vla.constants import ALL_TASK_LABELS
from humanoid_vla.instructions import (
    INSTRUCTION_TEMPLATES,
    get_instructions,
    sample_instruction,
)


def test_all_tasks_covered():
    assert set(INSTRUCTION_TEMPLATES) == set(ALL_TASK_LABELS)


@pytest.mark.parametrize("label", ALL_TASK_LABELS)
def test_corpus_size_and_disjointness(label):
    train = get_instructions(label, "train")
    heldout = get_instructions(label, "heldout")
    assert len(train) >= 10, f"{label}: need >=10 training paraphrases"
    assert len(heldout) >= 3, f"{label}: need >=3 held-out paraphrases"
    assert not set(train) & set(heldout), f"{label}: train/heldout overlap"
    assert len(set(train)) == len(train), f"{label}: duplicate train paraphrases"


@pytest.mark.parametrize("label", ALL_TASK_LABELS)
def test_canonical_label_is_a_training_instruction(label):
    assert label in get_instructions(label, "train")


def test_sample_instruction_deterministic_with_rng():
    rng = random.Random(0)
    a = sample_instruction(ALL_TASK_LABELS[0], rng)
    rng = random.Random(0)
    b = sample_instruction(ALL_TASK_LABELS[0], rng)
    assert a == b


def test_invalid_inputs():
    with pytest.raises(KeyError):
        get_instructions("fly to the moon")
    with pytest.raises(ValueError):
        get_instructions(ALL_TASK_LABELS[0], "test")
