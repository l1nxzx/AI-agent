import random

import pytest

from tinylm import persona
from tinylm.persona import (HOLDOUT_EVERY, INTENTS, PersonaBuilder, pick_unknown_user_turns, split_phrasings,
                            validate_bank)

STORIES = {"train": [f"Once upon a time there was a happy dog named Max number {i}. The dog ran in the park and played "
                     f"with a ball all day. It was a very good day for the little dog." for i in range(30)] +
                    [f"A small cat sat on a mat number {i}. The cat was very happy. A bird came to say hello to the cat." for i in range(30)],
           "val": [f"A girl found a ball in the garden number {i}. She played with the ball all day and she was happy about it, "
                   f"and then she went home to eat." for i in range(20)]}
UNKNOWN = {"train": [f"Can you recommend a good hotel in the city number {i} please" for i in range(30)],
           "val": [f"Which camera should I buy for travel photos number {i} please" for i in range(10)]}


@pytest.fixture(scope="module")
def builder():
    return PersonaBuilder(STORIES, UNKNOWN)


def test_bank_valid_and_ascii():
    validate_bank()
    for name, d in INTENTS.items():
        for text in d["user"] + d["bot"]:
            assert text.isascii(), (name, text)
    for d in (persona.FAVORITES, persona.LIKES):
        for lst in d.values():
            assert all(t.isascii() for t in lst)
    assert all(t.isascii() for t in persona.WHAT_IS.values()) and all(t.isascii() for t in persona.UNKNOWN_BOT)


def test_deterministic(builder):
    assert builder.build(50, "train", 1) == builder.build(50, "train", 1)
    assert builder.build(50, "train", 1) != builder.build(50, "train", 2)


def test_structure(builder):
    for split in ("train", "val"):
        for conv in builder.build(300, split, 3):
            roles = [r for r, _ in conv]
            assert roles == ["user", "assistant"] * (len(conv) // 2) and 1 <= len(conv) // 2 <= 4
            assert all(t.strip() and t.isascii() for _, t in conv)


def test_validation_phrasings_are_held_out():
    for d in INTENTS.values():
        tr, va = set(split_phrasings(d["user"], "train")), set(split_phrasings(d["user"], "val"))
        assert va and tr and not (tr & va)
        assert len(tr) + len(va) == len(d["user"])


def test_names_are_remembered(builder):
    found = 0
    for conv in builder.build(600, "train", 5):
        text = " ".join(t for _, t in conv)
        if "Your name is" in text or "You told me your name is" in text or "I remember! You are" in text:
            found += 1
            name = [n for n in persona.NAMES if f"is {n}." in text or f"are {n}." in text]
            assert name and name[0].lower() in conv[0][1].lower()
    assert found > 10


def test_stories_come_from_pool(builder):
    seen = 0
    for conv in builder.build(500, "train", 7):
        for (_, user), (_, bot) in zip(conv[::2], conv[1::2]):
            if "story" in user.lower() and len(bot) > 100:
                seen += 1
                assert "Max" in bot or "cat" in bot or "bird" in bot
    assert seen > 5


def test_unknown_user_turn_filter():
    convs = [[("user", "Hi there, how are you doing on this fine day my friend"), ("assistant", "x")],
             [("user", "Can you recommend a good hotel near the beach for my family please"), ("assistant", "y")],
             [("user", "ok thanks"), ("assistant", "z")]]
    assert pick_unknown_user_turns(convs) == ["Can you recommend a good hotel near the beach for my family please"]
