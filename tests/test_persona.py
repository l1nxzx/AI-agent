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
        texts = d["user"] + d["bot"] if "pairs" not in d else [t for pair in d["pairs"] for t in pair]
        for text in texts:
            assert text.isascii(), (name, text)
    assert all(t.isascii() for t in persona.UNKNOWN_BOT + persona.STORY_INTRO + persona.NAME_INTRO_BOT)


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
        users = d["user"] if "pairs" not in d else [q for q, _ in d["pairs"]]
        tr, va = set(split_phrasings(users, "train")), set(split_phrasings(users, "val"))
        assert va and tr and not (tr & va)
        assert len(tr) + len(va) == len(users)


def test_names_are_remembered(builder):
    found = 0
    for conv in builder.build(600, "train", 5):
        text = " ".join(t for _, t in conv)
        if "Your name is" in text or "You told me your name is" in text or "I remember! You are" in text:
            found += 1
            said = conv[-1][1].rstrip(".").split()[-1]          # the name in the final reply
            assert said.lower() in conv[0][1].lower()           # was given by the user in the first turn
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


def test_names_include_made_up_ones(builder):
    firsts = {c[0][1].lower() for c in builder.build(800, "train", 9) if "name" in " ".join(t for _, t in c).lower()}
    assert any(not any(n.lower() in f for n in persona.NAMES) for f in firsts)


def test_identity_subset_only_uses_identity_intents():
    sub = PersonaBuilder(STORIES, UNKNOWN, intents=persona.IDENTITY_INTENTS)
    assert set(sub.names) == set(persona.IDENTITY_INTENTS)
    convs = sub.build(400, "train", 4)
    text = " ".join(t for c in convs for _, t in c)
    assert "bulldozer" not in text and "gummy bear" not in text      # jokes are not in the identity set
    assert "Petryk" in text


def test_unknown_intent_name_rejected():
    with pytest.raises(ValueError):
        PersonaBuilder(STORIES, UNKNOWN, intents=["greeting", "nope"])


def test_identity_is_consistent_and_honest(builder):
    """Every bot reply to 'who are you' names Petryk or says it is a language model, and nothing claims to be
    human or to have internet access."""
    d = INTENTS["who_are_you"]
    assert all(("Petryk" in r) for r in d["bot"])
    allbot = " ".join(r for v in INTENTS.values() for r in (v["bot"] if "pairs" not in v else [b for _, b in v["pairs"]])).lower()
    assert "i am a person" not in allbot and "i am human" not in allbot
    for q, r in INTENTS["abilities"]["pairs"]:
        assert not r.lower().startswith("yes, i am connected")


def test_replies_match_their_questions_for_paired_intents(builder):
    """In paired intents a reply must come with its own question, never a random one."""
    pairs = dict(INTENTS["abilities"]["pairs"])
    for conv in builder.build(1500, "train", 11):
        for (_, user), (_, bot) in zip(conv[::2], conv[1::2]):
            for q, r in pairs.items():
                if bot == r:
                    assert user.lower().rstrip("?.!").replace("'", "") == q.lower().rstrip("?.!").replace("'", "")
