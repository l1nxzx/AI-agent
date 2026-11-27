"""A byte-level BPE tokenizer written from scratch.

How BPE works, in four steps:

1. Start with a vocabulary of the 256 possible bytes. Any text can be written as UTF-8 bytes, so
   nothing is ever "unknown" and encoding is lossless.
2. Split the training text into "pre-tokens" (roughly words with their leading space, digits,
   punctuation). Merges never cross pre-token boundaries, which stops tokens like "dog." from
   gluing a word to punctuation.
3. Repeatedly find the most frequent adjacent pair of tokens across the whole corpus and merge it
   into one new token. Each merge adds one id to the vocabulary. After M merges the vocabulary
   has 256 + M entries. The ordered list of merges IS the tokenizer.
4. To encode new text, split it into pre-tokens, turn each into bytes, then apply the learned
   merges in the order they were learned (earliest merge first) until none applies.

Special tokens (chat markers, end of document) are appended after the merged tokens. They are
never produced by the merge process; the encoder only emits them when explicitly allowed.

Id layout:  [0..255] raw bytes | [256..256+M-1] merges | last N ids: special tokens
"""
import hashlib
import json
import re
from collections import Counter, defaultdict

# Special tokens. The order here fixes their relative ids.
PAD = "<|pad|>"
ENDOFTEXT = "<|endoftext|>"   # separates documents (stories, conversations)
USER = "<|user|>"             # starts a user turn  ("user: ...")
BOT = "<|bot|>"               # starts a bot turn   ("bot: ...")
END = "<|end|>"               # ends a chat turn
SPECIAL_TOKENS = [PAD, ENDOFTEXT, USER, BOT, END]

# Pre-tokenization pattern, in the spirit of GPT-2's: contractions, words, numbers, punctuation,
# whitespace. A word keeps its single leading space, so " dog" and "dog" are different pre-tokens.
PRETOKEN_PATTERN = (
    r"'(?:s|t|re|ve|m|ll|d)"      # common English contractions
    r"| ?[^\W\d_]+"                # optional space + a run of letters
    r"| ?\d+"                      # optional space + a run of digits
    r"| ?(?:[^\s\w]|_)+"           # optional space + a run of punctuation/symbols
    r"|\s+(?!\S)"                  # whitespace that is not followed by a word
    r"|\s+"                        # any remaining whitespace
)


def _merge_pair(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` (left to right) with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None, special_tokens=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.special_tokens = list(SPECIAL_TOKENS if special_tokens is None else special_tokens)
        self._pattern = re.compile(PRETOKEN_PATTERN)
        self._build_tables()

    # ---- tables -------------------------------------------------------------------------------
    def _build_tables(self):
        # rank of each merge: lower rank = learned earlier = applied first when encoding
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        # bytes represented by each id (used by decode)
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.num_base = len(self.vocab)  # 256 + number of merges
        self.special_to_id = {}
        for i, tok in enumerate(self.special_tokens):
            self.special_to_id[tok] = self.num_base + i
            self.vocab.append(tok.encode("utf-8"))
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            if self.special_tokens else None
        )
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def token_id(self, special):
        return self.special_to_id[special]

    # ---- training -----------------------------------------------------------------------------
    @classmethod
    def train(cls, texts, vocab_size, logger=None, special_tokens=None):
        """Learn merges from an iterable of strings until the vocabulary reaches `vocab_size`
        (including the special tokens)."""
        specials = list(SPECIAL_TOKENS if special_tokens is None else special_tokens)
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} is too small: need at least {256 + len(specials)}")
        pattern = re.compile(PRETOKEN_PATTERN)

        # 1) Count pre-tokens. Training on unique words with counts is far faster than on raw text.
        counts = Counter()
        n_texts = 0
        for text in texts:
            counts.update(pattern.findall(text))
            n_texts += 1
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())
        if logger:
            logger.info(f"BPE train: {n_texts} texts, {len(words)} unique pre-tokens, {sum(freqs)} pre-token occurrences")

        # 2) Count adjacent pairs, and remember which words contain each pair.
        pair_counts = defaultdict(int)
        pair_words = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                pair_words[pair].add(wi)

        # 3) Greedy merging.
        merges = []
        for m in range(num_merges):
            if not pair_counts:
                if logger:
                    logger.warning(f"BPE train: ran out of pairs after {m} merges; vocab will be smaller than requested")
                break
            # most frequent pair; ties broken by the pair itself so training is deterministic
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_id = 256 + m
            merges.append(best)
            for wi in list(pair_words[best]):
                w = words[wi]
                f = freqs[wi]
                # remove this word's old pair counts, merge, then add the new pair counts
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                w = _merge_pair(w, best, new_id)
                words[wi] = w
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(wi)
            pair_words.pop(best, None)
            if logger and (m + 1) % 250 == 0:
                logger.info(f"BPE train: {m + 1}/{num_merges} merges")
        tok = cls(merges, specials)
        if logger:
            logger.info(f"BPE train: done, vocab_size={tok.vocab_size} ({len(merges)} merges + 256 bytes + {len(specials)} special)")
        return tok

    # ---- encoding / decoding ------------------------------------------------------------------
    def _encode_pretoken(self, s):
        cached = self._cache.get(s)
        if cached is not None:
            return cached
        ids = list(s.encode("utf-8"))
        while len(ids) >= 2:
            # find the adjacent pair with the lowest merge rank
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = _merge_pair(ids, best_pair, 256 + best_rank)
        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[s] = ids
        return ids

    def encode(self, text, allow_special=False):
        """Text -> list of ids. If allow_special is False (the default, and what the data pipeline
        uses), strings like '<|end|>' inside the text are encoded as ordinary characters, so text
        from a dataset can never inject control tokens."""
        out = []
        if allow_special and self._special_re is not None:
            parts = self._special_re.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_to_id:
                out.append(self.special_to_id[part])
                continue
            for pre in self._pattern.findall(part):
                out.extend(self._encode_pretoken(pre))
        return out

    def decode(self, ids):
        """List of ids -> text. Invalid UTF-8 (possible if a generation is cut mid-character) is replaced."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---- persistence --------------------------------------------------------------------------
    def to_dict(self):
        return {
            "version": 1,
            "pattern": PRETOKEN_PATTERN,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("pattern") != PRETOKEN_PATTERN:
            raise ValueError("tokenizer file was made with a different pre-tokenization pattern")
        return cls(d["merges"], d["special_tokens"])

    def fingerprint(self):
        """Short hash identifying this exact tokenizer; stored in checkpoints to catch mismatches."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
