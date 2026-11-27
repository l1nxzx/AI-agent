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
    # targets are the inputs shifted left by one token (positions the loss ignores are -100)
    assert ((x[:, 1:] == y[:, :-1]) | (y[:, :-1] == -100)).all()
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


def test_bot_only_mask_covers_exactly_bot_tokens(tok):
    from tinylm.data import encode_conversation_with_mask
    ids, mask = encode_conversation_with_mask(tok, CONVS[0])
    assert ids == encode_conversation(tok, CONVS[0])
    learned = [i for i, m in zip(ids, mask) if m]
    text = tok.decode(learned)
    assert text == "Hello! How can I help you today?" + END + "Sure, here is some information." + END
    assert not mask[0] and not mask[-1]   # <|user|> marker and <|endoftext|> are never learned


def test_loss_masking_in_batches(data_dir):
    only_chat = DataSource(data_dir, context_len=16, chat_fraction=1.0, seed=1, chat_loss="bot_only")
    _, y, _ = only_chat.train_batch(0, 0, 8)
    assert (y == -100).any() and (y != -100).any()
    full = DataSource(data_dir, context_len=16, chat_fraction=1.0, seed=1, chat_loss="all")
    _, y2, _ = full.train_batch(0, 0, 8)
    assert not (y2 == -100).any()
    stories = DataSource(data_dir, context_len=16, chat_fraction=0.0, seed=1)
    _, y3, _ = stories.train_batch(0, 0, 8)
    assert not (y3 == -100).any()   # stories are always fully learned


def test_clean_dailydialog():
    from tinylm.data import clean_dailydialog
    assert clean_dailydialog("Say , Jim , how about a walk ?") == "Say, Jim, how about a walk?"
    assert clean_dailydialog("Let ' s go . I don’t mind .") == "Let's go. I don't mind."
    assert clean_dailydialog("I guess you are right.But what now ?") == "I guess you are right. But what now?"


def test_read_dailydialog(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tinylm.data import read_dailydialog
    p = tmp_path / "dd.parquet"
    pq.write_table(pa.table({"utterances": [["Hi !", "Hello .", "How are you ?"], ["A ?", "B ."]]}), p)
    convs = read_dailydialog(str(p))
    assert convs[0] == [("user", "Hi!"), ("assistant", "Hello.")]          # trailing user turn dropped
    assert convs[1] == [("user", "Hello."), ("assistant", "How are you?")]  # first utterance dropped variant
    assert convs[2] == [("user", "A?"), ("assistant", "B.")]
