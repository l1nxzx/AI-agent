import numpy as np
import pytest

from tinylm.bpe import BOT, END, ENDOFTEXT, USER, BPETokenizer
from tinylm.data import DataSource, encode_conversation, encode_story, write_dataset

STORIES = [f"Once upon a time there was a dog number {i}. The dog was happy." for i in range(40)]
CONVS = [[("user", "Hi there"), ("assistant", "Hello! How can I help you today?"),
          ("user", f"Tell me about {i}"), ("assistant", "Sure, here is some information.")] for i in range(20)]


@pytest.fixture(scope="module")
def tok():
    texts = STORIES + [c for conv in CONVS for _, c in conv]
    return BPETokenizer.train(texts, vocab_size=300)


@pytest.fixture(scope="module")
def data_dir(tok, tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_dataset(str(d), tok, STORIES[:30], STORIES[30:], CONVS[:15], CONVS[15:])
    return str(d)


def test_conversation_format(tok):
    ids = encode_conversation(tok, CONVS[0])
    text = tok.decode(ids)
    assert text.startswith(f"{USER}Hi there{END}{BOT}Hello!")
    assert text.endswith(f"{END}{ENDOFTEXT}")
    assert ids[0] == tok.token_id(USER)


def test_story_ends_with_endoftext(tok):
    assert encode_story(tok, STORIES[0])[-1] == tok.token_id(ENDOFTEXT)


def test_batches_are_shifted_and_deterministic(data_dir):
    ds = DataSource(data_dir, context_len=16, chat_fraction=0.5, seed=1)
    x, y, n_chat = ds.train_batch(step=3, micro=0, batch_size=8)
    assert x.shape == y.shape == (8, 16)
    # targets are the inputs shifted left by one token
    assert (x[:, 1:] == y[:, :-1]).all()
    x2, y2, n2 = ds.train_batch(step=3, micro=0, batch_size=8)
    assert (x == x2).all() and n_chat == n2
    x3, _, _ = ds.train_batch(step=4, micro=0, batch_size=8)
    assert not (x == x3).all()


def test_val_batches_fixed(data_dir):
    ds = DataSource(data_dir, context_len=16, chat_fraction=0.5, seed=1)
    a, _ = ds.val_batch("chat", 0, 4)
    b, _ = ds.val_batch("chat", 0, 4)
    assert (a == b).all()


def test_ids_in_range(data_dir, tok):
    ds = DataSource(data_dir, context_len=16, chat_fraction=0.5, seed=1)
    for name, arr in ds.arrays.items():
        assert int(np.max(arr)) < tok.vocab_size
