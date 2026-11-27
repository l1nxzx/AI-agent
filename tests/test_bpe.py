import random

import pytest

from tinylm.bpe import BOT, END, ENDOFTEXT, SPECIAL_TOKENS, USER, BPETokenizer

CORPUS = [
    "Once upon a time there was a little dog. The dog loved to play in the park.",
    "The dog said, \"I can't wait!\" Then the dog ran home to see 3 friends.",
    "user hello there. how are you today? I am fine, thank you!",
] * 50


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300)


def test_vocab_size_and_layout(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - len(SPECIAL_TOKENS)
    ids = [tok.token_id(s) for s in SPECIAL_TOKENS]
    assert ids == list(range(300 - len(SPECIAL_TOKENS), 300))


@pytest.mark.parametrize("text", [
    "Once upon a time",
    "  leading and   multiple   spaces\n\nnewlines\t tabs",
    "unicode: café, 日本語, and symbols ~!@#$%^&*()_+",
    "",
    "numbers 12345 and don't, we're, it's",
])
def test_roundtrip_is_lossless(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_random_bytes_text(tok):
    rng = random.Random(0)
    chars = "abc de.,!?\n'\"0123é中"
    for _ in range(50):
        s = "".join(rng.choice(chars) for _ in range(rng.randint(0, 60)))
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress(tok):
    text = "The dog loved to play in the park."
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges
    assert a.fingerprint() == b.fingerprint()


def test_special_tokens_not_injected_by_default(tok):
    ids = tok.encode("hello <|end|> world")
    assert tok.token_id(END) not in ids
    assert tok.decode(ids) == "hello <|end|> world"


def test_special_tokens_when_allowed(tok):
    ids = tok.encode(f"{USER}hi{BOT}hello{END}{ENDOFTEXT}", allow_special=True)
    assert ids[0] == tok.token_id(USER)
    assert ids[-1] == tok.token_id(ENDOFTEXT)
    assert ids[-2] == tok.token_id(END)
    assert tok.decode(ids) == f"{USER}hi{BOT}hello{END}{ENDOFTEXT}"


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.merges == tok.merges
    assert tok2.fingerprint() == tok.fingerprint()
    text = "Once upon a time, the dog ran."
    assert tok2.encode(text) == tok.encode(text)


def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer.train(["ab ab ab ab cd"], vocab_size=256 + len(SPECIAL_TOKENS) + 1)
    # " a" and "ab" ... the most frequent adjacent byte pair is (a, b) with 4 occurrences
    assert t.merges[0] == (ord("a"), ord("b"))


def test_vocab_too_small_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train(["x"], vocab_size=100)
