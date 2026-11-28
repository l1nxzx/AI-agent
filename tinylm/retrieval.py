"""Passage search: a from-scratch BM25 index. No machine learning and no pretrained models involved.

How BM25 works, in one paragraph: split the text into small passages. For a question, take its content words (drop
very common words like "the"). Score every passage: each question word adds points if the passage contains it, more
points when the word is rare in the whole collection (a passage that mentions "mitochondria" is a better match
than one that mentions "the"), more when the word appears several times (with diminishing returns), and a little
less for very long passages. Return the highest-scoring passages. This is a general scoring formula: it has no list
of questions, no answers, and no topic-specific rules, so it works for any question about any text in the collection.

The language model never searches. The program does, using the question's own words as the query, and puts the best
passages in front of the model, which then reads them and writes the answer (see rag.py).

Storage (everything is plain files in one directory):
  passages.txt   one passage per line:  title <TAB> text   (newlines inside text are replaced by spaces)
  offsets.npy    byte offset of each line, so any passage can be read without loading the whole file
  index.npz      the inverted index (for each word, the passages containing it and how often) and passage lengths
  vocab.json     word -> integer id
  meta.json      counts and settings
"""
import json
import os
import re
import time
from array import array
from collections import Counter

import numpy as np

# Very common words carry no information about what a passage is about. This is a general English stop list.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both but
by can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also many much one two may might shall us s t
""".split())

_WORD = re.compile(r"[a-z0-9]+")
_STEM_CACHE = {}


def stem(word):
    """A deliberately simple suffix stripper, so that 'cells' matches 'cell' and 'photosynthesizing' matches
    'photosynthesize'. Used identically for passages and questions."""
    cached = _STEM_CACHE.get(word)
    if cached is not None:
        return cached
    w = word
    if len(w) > 4:
        for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
            if w.endswith(suffix) and not w.endswith("ss") and len(w) - len(suffix) >= 3:
                w = w[: len(w) - len(suffix)] + repl
                break
    elif len(w) == 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    _STEM_CACHE[word] = w
    return w


def tokenize(text):
    """Lowercase, split into words and numbers, drop stop words, stem."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


# ---------------------------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def chunk_text(text, min_words=60, max_words=130):
    """Split a document into passages of roughly min_words..max_words, on paragraph and sentence boundaries."""
    pieces = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            pieces.append(para)
        else:  # a long paragraph: cut it on sentence boundaries
            pieces.extend(s for s in _SENTENCE_END.split(para) if s.strip())
    passages, current, count = [], [], 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            passages.append(" ".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
        if count >= min_words:
            passages.append(" ".join(current))
            current, count = [], 0
    if current and (count >= 20 or not passages):
        passages.append(" ".join(current))
    return passages


def clean_textbook(text):
    """OpenStax text is raw PDF text: hard-wrapped lines, dotted tables of contents, page numbers, running headers.
    Join it back into running prose (the result is then chunked by sentences)."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"[\d\s.]*", s):
            continue                                  # blank, page numbers, dot leaders
        if re.search(r"(\.\s+){4,}", s) or s.count(". .") >= 2:
            continue                                  # table-of-contents lines
        lines.append(s)
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r"(\w)- (\w)", r"\1\2", joined)   # re-join words hyphenated at line ends
    return joined


# ---------------------------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------------------------
class BM25Index:
    K1 = 1.2    # how quickly extra occurrences of a word stop adding score
    B = 0.75    # how strongly long passages are penalized

    def __init__(self, directory):
        self.dir = directory
        self.vocab = None
        self.offsets = None
        self._fh = None

    # ---- build -----------------------------------------------------------------------------------
    @classmethod
    def build(cls, directory, passages, logger=None, title_weight=1):
        """passages: iterable of (title, text). Writes all files into `directory` and returns an opened index.
        title_weight: how many times the title's words are counted as part of the passage (titles say what a
        passage is about, so counting them more helps questions that name the subject)."""
        os.makedirs(directory, exist_ok=True)
        t0 = time.time()
        vocab, terms, docs, tfs = {}, array("I"), array("I"), array("H")
        lengths = array("I")
        offsets = array("Q")
        n_docs = 0
        with open(os.path.join(directory, "passages.txt"), "wb") as out:
            pos = 0
            for title, text in passages:
                line = (title.replace("\t", " ").replace("\n", " ") + "\t" + text.replace("\t", " ").replace("\n", " ") + "\n").encode("utf-8")
                offsets.append(pos)
                out.write(line)
                pos += len(line)
                toks = tokenize(title) * title_weight + tokenize(text)   # the title counts as part of the passage
                lengths.append(len(toks))
                for term, tf in Counter(toks).items():
                    tid = vocab.get(term)
                    if tid is None:
                        tid = vocab[term] = len(vocab)
                    terms.append(tid)
                    docs.append(n_docs)
                    tfs.append(min(tf, 65535))
                n_docs += 1
                if logger and n_docs % 100_000 == 0:
                    logger.info(f"indexed {n_docs:,} passages ({time.time() - t0:.0f}s)")
        terms, docs, tfs = (np.frombuffer(a, dtype=dt) for a, dt in ((terms, np.uint32), (docs, np.uint32), (tfs, np.uint16)))
        order = np.argsort(terms, kind="stable")          # group the postings of each word together
        terms, docs, tfs = terms[order], docs[order], tfs[order]
        starts = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(np.bincount(terms, minlength=len(vocab)), out=starts[1:])
        np.savez(os.path.join(directory, "index.npz"), docs=docs, tfs=tfs, starts=starts,
                 lengths=np.frombuffer(lengths, dtype=np.uint32))
        np.save(os.path.join(directory, "offsets.npy"), np.frombuffer(offsets, dtype=np.uint64))
        with open(os.path.join(directory, "vocab.json"), "w") as f:
            json.dump(vocab, f)
        meta = {"passages": n_docs, "terms": len(vocab), "postings": int(len(docs)), "build_seconds": round(time.time() - t0, 1)}
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)
        if logger:
            logger.info(f"index built: {meta}")
        return cls(directory).load()

    # ---- load / search -----------------------------------------------------------------------------
    def load(self):
        z = np.load(os.path.join(self.dir, "index.npz"))
        self.docs, self.tfs, self.starts, self.lengths = z["docs"], z["tfs"], z["starts"], z["lengths"].astype(np.float32)
        with open(os.path.join(self.dir, "vocab.json")) as f:
            self.vocab = json.load(f)
        self.offsets = np.load(os.path.join(self.dir, "offsets.npy"))
        self.n_docs = len(self.offsets)
        self.avg_len = float(self.lengths.mean())
        self._fh = open(os.path.join(self.dir, "passages.txt"), "rb")
        return self

    def passage(self, doc_id):
        """(title, text) of one passage."""
        self._fh.seek(int(self.offsets[doc_id]))
        title, _, text = self._fh.readline().decode("utf-8").rstrip("\n").partition("\t")
        return title, text

    def search(self, query, k=5):
        """Top-k passages for a question: list of (doc_id, score), best first."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.K1 * (1 - self.B + self.B * self.lengths / self.avg_len)
        seen = set()
        for term in tokenize(query):
            tid = self.vocab.get(term)
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            a, b = self.starts[tid], self.starts[tid + 1]
            docs, tf = self.docs[a:b], self.tfs[a:b].astype(np.float32)
            df = b - a
            idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            scores[docs] += idf * (tf * (self.K1 + 1)) / (tf + norm[docs])
        if not seen:
            return []
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
