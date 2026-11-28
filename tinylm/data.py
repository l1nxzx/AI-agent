"""Data download, format checks, tokenization and the training batch sampler.

Pipeline (run by `python -m tinylm.prepare_data`):
  1. Check that each Hugging Face dataset exists and has the expected files and columns.
  2. Download the parquet files we need.
  3. Train the BPE tokenizer on a sample of stories plus all chat text.
  4. Encode everything and write flat uint16 token files:
        stories_train.bin  stories_val.bin  chat_train.bin  chat_val.bin
     Documents are separated by <|endoftext|>. Chat turns look like:
        <|user|> text <|end|> <|bot|> text <|end|> ... <|endoftext|>
  5. Write meta.json with counts and the tokenizer fingerprint.

No persona or system prompt is added anywhere: the chat data is used exactly as published, with the
role names mapped to the <|user|> and <|bot|> markers.
"""
import json
import os
import re
import time

import numpy as np

from .bpe import BOT, END, ENDOFTEXT, USER, BPETokenizer
from .logutil import get_logger

TINYSTORIES_REPO = "roneneldan/TinyStories"
SMOLTALK_REPO = "HuggingFaceTB/smoltalk"

TS_TRAIN_PREFIX = "data/train-"
TS_VALID_PREFIX = "data/validation-"
CHAT_TRAIN_FILE = "data/everyday-conversations/train-00000-of-00001.parquet"
CHAT_TEST_FILE = "data/everyday-conversations/test-00000-of-00001.parquet"

# DailyDialog (about 13k human-written everyday dialogues). The Hub copy is a script dataset, so we read the
# automatically converted parquet files from the refs/convert/parquet revision.
DAILYDIALOG_REPO = "roskoN/dailydialog"
DAILYDIALOG_REVISION = "refs/convert/parquet"
DAILYDIALOG_TRAIN = "full/train/0000.parquet"
DAILYDIALOG_VAL = "full/validation/0000.parquet"

# NQ-Open: real, short factual questions that people typed into Google (about 88k). We use only the questions
# (never the answers) as examples of what a very small model should decline to answer. License: CC BY-SA 3.0.
NQ_REPO = "google-research-datasets/nq_open"
NQ_TRAIN = "nq_open/train-00000-of-00001.parquet"
NQ_VAL = "nq_open/validation-00000-of-00001.parquet"

ROLE_TO_MARKER = {"user": USER, "assistant": BOT}
BIN_FILES = ["stories_train.bin", "stories_val.bin", "chat_train.bin", "chat_val.bin"]


# ---------------------------------------------------------------------------------------------
# Dataset checks and download
# ---------------------------------------------------------------------------------------------
def check_and_locate_files(smoke=False, num_story_shards=1, dailydialog=False, nq=False):
    """Ask the Hub which files exist. Returns the file names we will download. Raises with a clear
    message if a dataset or an expected file is missing."""
    from huggingface_hub import HfApi

    log = get_logger()
    api = HfApi()
    out = {}
    for repo in (TINYSTORIES_REPO, SMOLTALK_REPO):
        if not api.repo_exists(repo, repo_type="dataset"):
            raise RuntimeError(f"dataset {repo} does not exist on the Hugging Face Hub")
    ts_files = sorted(api.list_repo_files(TINYSTORIES_REPO, repo_type="dataset"))
    st_files = set(api.list_repo_files(SMOLTALK_REPO, repo_type="dataset"))
    train_shards = [f for f in ts_files if f.startswith(TS_TRAIN_PREFIX) and f.endswith(".parquet")]
    valid_shards = [f for f in ts_files if f.startswith(TS_VALID_PREFIX) and f.endswith(".parquet")]
    if not train_shards or not valid_shards:
        raise RuntimeError(f"{TINYSTORIES_REPO}: expected parquet files under data/ but found {ts_files}")
    for f in (CHAT_TRAIN_FILE, CHAT_TEST_FILE):
        if f not in st_files:
            raise RuntimeError(f"{SMOLTALK_REPO}: expected file {f} is missing")
    out["story_valid"] = valid_shards[0]
    out["story_train"] = train_shards[:num_story_shards] if not smoke else []
    if dailydialog:
        if not api.repo_exists(DAILYDIALOG_REPO, repo_type="dataset"):
            raise RuntimeError(f"dataset {DAILYDIALOG_REPO} does not exist on the Hugging Face Hub")
        dd_files = set(api.list_repo_files(DAILYDIALOG_REPO, repo_type="dataset", revision=DAILYDIALOG_REVISION))
        for f in (DAILYDIALOG_TRAIN, DAILYDIALOG_VAL):
            if f not in dd_files:
                raise RuntimeError(f"{DAILYDIALOG_REPO}@{DAILYDIALOG_REVISION}: expected file {f} is missing; found {sorted(dd_files)}")
        out["dd_train"], out["dd_val"] = DAILYDIALOG_TRAIN, DAILYDIALOG_VAL
        log.info(f"{DAILYDIALOG_REPO}: found train and validation parquet files")
    if nq:
        if not api.repo_exists(NQ_REPO, repo_type="dataset"):
            raise RuntimeError(f"dataset {NQ_REPO} does not exist on the Hugging Face Hub")
        nq_files = set(api.list_repo_files(NQ_REPO, repo_type="dataset"))
        for f in (NQ_TRAIN, NQ_VAL):
            if f not in nq_files:
                raise RuntimeError(f"{NQ_REPO}: expected file {f} is missing; found {sorted(nq_files)[:10]}")
        out["nq_train"], out["nq_val"] = NQ_TRAIN, NQ_VAL
        log.info(f"{NQ_REPO}: found train and validation parquet files")
    out["chat_train"] = CHAT_TRAIN_FILE
    out["chat_test"] = CHAT_TEST_FILE
    log.info(f"{TINYSTORIES_REPO}: found {len(train_shards)} train shard(s), {len(valid_shards)} validation file(s)")
    log.info(f"{SMOLTALK_REPO}: found everyday-conversations train and test files")
    return out


def download(repo, filename, cache_dir, revision=None):
    from huggingface_hub import hf_hub_download

    log = get_logger()
    t0 = time.time()
    path = hf_hub_download(repo, filename, repo_type="dataset", cache_dir=cache_dir, revision=revision)
    log.info(f"downloaded {repo}/{filename} ({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
    return path


def read_story_texts(parquet_path, limit=None):
    """Read the 'text' column of a TinyStories parquet file. Verifies the schema first."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    if "text" not in pf.schema_arrow.names:
        raise RuntimeError(f"{parquet_path}: expected a 'text' column, found {pf.schema_arrow.names}")
    texts = []
    for batch in pf.iter_batches(batch_size=10_000, columns=["text"]):
        for t in batch.column(0).to_pylist():
            if t and t.strip():
                texts.append(t.strip())
                if limit and len(texts) >= limit:
                    return texts
    return texts


def read_conversations(parquet_path):
    """Read SmolTalk conversations as lists of (role, content). Verifies the schema, drops empty
    messages, and drops a trailing user turn so every conversation ends with a bot reply."""
    import pyarrow.parquet as pq

    log = get_logger()
    table = pq.read_table(parquet_path)
    if "messages" not in table.schema.names:
        raise RuntimeError(f"{parquet_path}: expected a 'messages' column, found {table.schema.names}")
    convs = []
    unknown_roles = 0
    trailing_user = 0
    for row in table.column("messages").to_pylist():
        conv = []
        for m in row:
            if set(m) < {"role", "content"}:
                raise RuntimeError(f"{parquet_path}: message missing role/content: {sorted(m)}")
            if m["role"] not in ROLE_TO_MARKER:
                unknown_roles += 1  # e.g. a system prompt; we do not add or keep any
                continue
            content = (m["content"] or "").strip()
            if content:
                conv.append((m["role"], content))
        if conv and conv[-1][0] == "user":
            conv.pop()
            trailing_user += 1
        if conv:
            convs.append(conv)
    log.info(f"{os.path.basename(parquet_path)}: {len(convs)} conversations, "
             f"{unknown_roles} messages with other roles skipped, {trailing_user} trailing user turns dropped")
    return convs


def read_nq_open_questions(parquet_path, limit=None):
    """Questions only (the 'question' column) from NQ-Open: short real search queries such as
    'who sings the song ...'. Verifies the schema first."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["question"]) if "question" in pq.ParquetFile(parquet_path).schema_arrow.names else None
    if table is None:
        raise RuntimeError(f"{parquet_path}: expected a 'question' column")
    qs = [q.strip() for q in table.column("question").to_pylist() if q and q.strip()]
    qs = qs[:limit] if limit else qs
    get_logger().info(f"{os.path.basename(parquet_path)}: {len(qs)} short factual questions (mean {sum(len(q.split()) for q in qs) / len(qs):.1f} words)")
    return qs


def clean_dailydialog(text):
    """DailyDialog is stored pre-tokenized ('Say , Jim , how about it ?'). Undo that so the text looks like
    normal writing, and map curly quotes to plain ones."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"\s+'\s*(s|t|m|re|ve|ll|d)\b", r"'\1", text)      # "Let ' s" -> "Let's"
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)                        # "word ," -> "word,"
    text = re.sub(r"([a-z])([.!?])([A-Z])", r"\1\2 \3", text)         # "right.But" -> "right. But"
    return re.sub(r"\s+", " ", text).strip()


def read_dailydialog(parquet_path, both_ways=True):
    """DailyDialog dialogues as (role, text) lists. Speakers alternate, so the first speaker is 'user' and the
    second is 'assistant'. With both_ways, a dialogue is also added with its first utterance dropped, so the
    bot side learns from both speakers' kinds of replies. Trailing user turns are dropped."""
    import pyarrow.parquet as pq

    log = get_logger()
    table = pq.read_table(parquet_path)
    if "utterances" not in table.schema.names:
        raise RuntimeError(f"{parquet_path}: expected an 'utterances' column, found {table.schema.names}")
    convs, dropped = [], 0
    for utts in table.column("utterances").to_pylist():
        utts = [clean_dailydialog(u) for u in utts]
        utts = [u for u in utts if u]
        variants = [utts] + ([utts[1:]] if both_ways and len(utts) >= 3 else [])
        for v in variants:
            conv = [("user" if i % 2 == 0 else "assistant", u) for i, u in enumerate(v)]
            if conv and conv[-1][0] == "user":
                conv.pop()
                dropped += 1
            if len(conv) >= 2:
                convs.append(conv)
    log.info(f"{os.path.basename(parquet_path)}: {len(convs)} dialogues (both_ways={both_ways}), {dropped} trailing turns dropped")
    return convs


# ---------------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------------
def encode_story(tok, text):
    return tok.encode(text) + [tok.token_id(ENDOFTEXT)]


def encode_conversation(tok, conv):
    """[(role, content), ...] -> ids. <|user|> text <|end|> <|bot|> text <|end|> ... <|endoftext|>"""
    ids = []
    for role, content in conv:
        ids.append(tok.token_id(ROLE_TO_MARKER[role]))
        ids.extend(tok.encode(content))
        ids.append(tok.token_id(END))
    ids.append(tok.token_id(ENDOFTEXT))
    return ids


def encode_conversation_with_mask(tok, conv):
    """Same ids as encode_conversation, plus a 0/1 mask marking the tokens the bot must LEARN to
    produce: the words of each bot reply and the <|end|> that closes it. Role markers, user text and
    <|endoftext|> are 0: the user's words are given, so there is nothing to learn by predicting them."""
    ids, mask = [], []
    for role, content in conv:
        body = tok.encode(content)
        learn = 1 if role == "assistant" else 0
        ids.append(tok.token_id(ROLE_TO_MARKER[role]))
        mask.append(0)
        ids.extend(body)
        mask.extend([learn] * len(body))
        ids.append(tok.token_id(END))
        mask.append(learn)
    ids.append(tok.token_id(ENDOFTEXT))
    mask.append(0)
    return ids, mask


def _write_bin(path, docs_ids, dtype=np.uint16):
    arr = np.fromiter((i for doc in docs_ids for i in doc), dtype=dtype)
    arr.tofile(path)
    return arr


def _stats(name, docs_ids, n_chars):
    lens = np.array([len(d) for d in docs_ids])
    total = int(lens.sum())
    get_logger().info(
        f"{name}: {len(docs_ids)} docs, {total} tokens, tokens/doc mean={lens.mean():.0f} "
        f"p50={np.percentile(lens, 50):.0f} p95={np.percentile(lens, 95):.0f} max={lens.max()}, "
        f"{n_chars / max(total, 1):.2f} chars/token")
    return {"docs": len(docs_ids), "tokens": total}


def write_readable_samples(path, tok, splits, n_chat=150, n_stories=25, seed=0):
    """Plain-text file of random training examples, decoded from the exact token ids the model trains on.
    Open it to judge the data by eye. Role markers show who speaks: <|user|> ... <|bot|> ... <|end|>."""
    import random

    rng = random.Random(seed)
    chat_docs, story_docs = splits["chat_train"][0], splits["stories_train"][0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"RANDOM CHAT EXAMPLES ({n_chat} of {len(chat_docs)} training conversations)\n")
        f.write("Format: <|user|> text <|end|> <|bot|> text <|end|> ... <|endoftext|>\n\n")
        for doc in rng.sample(chat_docs, min(n_chat, len(chat_docs))):
            f.write(tok.decode(doc).replace("<|end|>", "<|end|>\n").replace("<|endoftext|>", "") + "\n" + "-" * 60 + "\n")
        f.write(f"\n\nRANDOM STORY EXAMPLES ({n_stories} of {len(story_docs)} training stories)\n\n")
        for doc in rng.sample(story_docs, min(n_stories, len(story_docs))):
            f.write(tok.decode(doc).replace("<|endoftext|>", "") + "\n" + "-" * 60 + "\n")


def write_dataset(data_dir, tok, story_train, story_val, chat_train, chat_val, sources=None, chat_mode="real"):
    """Encode all four splits, verify the encoding round-trips, write .bin files and meta.json.
    Inputs are lists of strings (stories) and lists of conversations."""
    log = get_logger()
    os.makedirs(data_dir, exist_ok=True)
    if tok.vocab_size > 65535:
        raise ValueError("vocab too large for uint16 storage")
    meta = {"tokenizer_fingerprint": tok.fingerprint(), "vocab_size": tok.vocab_size, "chat_mode": chat_mode,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": sources or {}, "splits": {}}
    splits = {
        "stories_train": ([encode_story(tok, t) for t in story_train], sum(len(t) for t in story_train)),
        "stories_val": ([encode_story(tok, t) for t in story_val], sum(len(t) for t in story_val)),
        "chat_train": ([encode_conversation(tok, c) for c in chat_train], sum(len(x) for c in chat_train for _, x in c)),
        "chat_val": ([encode_conversation(tok, c) for c in chat_val], sum(len(x) for c in chat_val for _, x in c)),
    }
    for name, (docs, n_chars) in splits.items():
        if not docs:
            raise RuntimeError(f"split {name} is empty")
        meta["splits"][name] = _stats(name, docs, n_chars)
        arr = _write_bin(os.path.join(data_dir, name + ".bin"), docs)
        assert len(arr) == meta["splits"][name]["tokens"]
        assert arr.max() < tok.vocab_size
        if name.startswith("chat"):
            pairs = [encode_conversation_with_mask(tok, c) for c in (chat_train if name == "chat_train" else chat_val)]
            assert [p[0] for p in pairs] == docs, "mask encoder disagrees with the plain encoder"
            marr = _write_bin(os.path.join(data_dir, name + ".mask"), [p[1] for p in pairs], dtype=np.uint8)
            assert len(marr) == len(arr)
            meta["splits"][name]["bot_token_fraction"] = float(marr.mean())
            get_logger().info(f"{name}: loss-active (bot) tokens = {marr.mean() * 100:.1f}% of all chat tokens")

    # Sanity check: decoding must give back the original text exactly (byte-level BPE is lossless).
    for text in (story_train[:200] + story_val[:50]):
        assert tok.decode(tok.encode(text)) == text, "tokenizer round-trip failed on a story"
    for conv in (chat_train[:50] + chat_val[:20]):
        got = tok.decode(encode_conversation(tok, conv))
        want = "".join(f"{ROLE_TO_MARKER[r]}{c}{END}" for r, c in conv) + ENDOFTEXT
        assert got == want, "tokenizer round-trip failed on a conversation"
    log.info("round-trip check passed on sample stories and conversations")

    example = encode_conversation(tok, chat_train[0])
    log.info("example conversation as text:\n" + tok.decode(example))
    log.info("example conversation as tokens (first 24): "
             + " ".join(repr(tok.vocab[i].decode("utf-8", "replace")) for i in example[:24]))
    write_readable_samples(os.path.join(data_dir, "readable_samples.txt"), tok, splits)
    meta["chat_tokens_per_conv_p95"] = float(np.percentile([len(d) for d in splits["chat_train"][0]], 95))
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"wrote {len(BIN_FILES)} token files (+ 2 chat masks), tokenizer.json and meta.json to {data_dir}")
    return meta


# ---------------------------------------------------------------------------------------------
# Batch sampling for training
# ---------------------------------------------------------------------------------------------
class DataSource:
    """Memory-mapped token files plus deterministic random batch sampling.

    A training example is a random window of context_len + 1 tokens from a split. Inputs are
    tokens [0:T], targets are tokens [1:T+1]: the model predicts each next token. Each batch is
    drawn from stories and chat according to chat_fraction. Batches are a pure function of
    (seed, step, micro_step), so a resumed run sees exactly the data it would have seen.
    """

    def __init__(self, data_dir, context_len, chat_fraction, seed, chat_loss="bot_only"):
        self.context_len = context_len
        self.chat_fraction = chat_fraction
        self.seed = seed
        self.arrays = {}
        for name in BIN_FILES:
            path = os.path.join(data_dir, name[:-4] + ".bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found; run `python -m tinylm.prepare_data` first")
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            if len(arr) < context_len + 2:
                raise ValueError(f"{name} has only {len(arr)} tokens, fewer than context_len+2")
            self.arrays[name[:-4]] = arr
        # Optional per-token loss masks for chat (1 = bot token the model should learn). With
        # chat_loss="all" they are ignored and every chat token counts.
        self.masks = {}
        if chat_loss == "bot_only":
            for name in ("chat_train", "chat_val"):
                mpath = os.path.join(data_dir, name + ".mask")
                if os.path.exists(mpath):
                    self.masks[name] = np.memmap(mpath, dtype=np.uint8, mode="r")
                    assert len(self.masks[name]) == len(self.arrays[name])
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)

    def _windows(self, name, rng, n):
        arr = self.arrays[name]
        mask = self.masks.get(name)
        T = self.context_len
        starts = rng.integers(0, len(arr) - T - 1, size=n)
        x = np.stack([arr[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([arr[s + 1:s + 1 + T] for s in starts]).astype(np.int64)
        if mask is not None:
            keep = np.stack([mask[s + 1:s + 1 + T] for s in starts]) == 1
            y = np.where(keep, y, -100)   # -100 is ignored by the cross-entropy loss
        return x, y

    def train_batch(self, step, micro, batch_size):
        import torch

        rng = np.random.default_rng([self.seed, step, micro])
        n_chat = int(rng.binomial(batch_size, self.chat_fraction))
        parts = []
        if batch_size - n_chat > 0:
            parts.append(self._windows("stories_train", rng, batch_size - n_chat))
        if n_chat > 0:
            parts.append(self._windows("chat_train", rng, n_chat))
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        return torch.from_numpy(x), torch.from_numpy(y), n_chat

    def val_batch(self, source, i, batch_size):
        """Fixed validation batches (same for every eval) so val loss is comparable across steps."""
        import torch

        rng = np.random.default_rng([987654321, i])
        x, y = self._windows(source + "_val", rng, batch_size)
        return torch.from_numpy(x), torch.from_numpy(y)
