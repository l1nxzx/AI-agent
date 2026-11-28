"""Measure the search: for real questions with known answers, how often does one of the top-k retrieved passages
contain the answer text? This is the ceiling for the whole system: the reader model cannot answer from a passage
that was never retrieved.

  python -m tinylm.eval_retrieval --corpus_dir corpus --n 1000

Questions come from two datasets (answers are used ONLY here, to check retrieval, never for training):
  NQ-Open validation  real general-knowledge search questions ('who sings ...'), many are about things a simple
                      encyclopedia does not cover, so this is a hard, honest test of general coverage
  SciQ validation     science questions; the biology/physics/chemistry ones are the focus area
"""
import argparse
import os
import re
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .data import NQ_REPO, NQ_VAL
from .retrieval import BM25Index


def norm(text):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip() + " "


def contains_answer(passage, answers):
    p = norm(passage)
    return any(norm(a) in p for a in answers if a and a.strip())


def load_questions(name, n, cache):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if name == "nq_open":
        t = pq.read_table(hf_hub_download(NQ_REPO, NQ_VAL, repo_type="dataset", cache_dir=cache))
        return [(q, list(a)) for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist())][:n]
    t = pq.read_table(hf_hub_download("allenai/sciq", "data/validation-00000-of-00001.parquet", repo_type="dataset", cache_dir=cache))
    return [(q, [a]) for q, a in zip(t.column("question").to_pylist(), t.column("correct_answer").to_pylist())][:n]


def evaluate(index, questions, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    t0 = time.time()
    for q, answers in questions:
        top = index.search(q, k=max(ks))
        texts = [index.passage(d)[1] for d, _ in top]
        for k in ks:
            if any(contains_answer(t, answers) for t in texts[:k]):
                hits[k] += 1
    return {k: hits[k] / len(questions) for k in ks}, (time.time() - t0) / len(questions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_dir", default="corpus")
    p.add_argument("--n", type=int, default=1000, help="questions per dataset")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args(argv)
    index = BM25Index(args.corpus_dir).load()
    cache = args.cache_dir or os.path.join(args.corpus_dir, "hf_cache")
    print(f"corpus: {index.n_docs:,} passages")
    for name in ("nq_open", "sciq"):
        qs = load_questions(name, args.n, cache)
        recall, latency = evaluate(index, qs)
        print(f"{name:8s} ({len(qs)} questions): answer in top-1 {recall[1]:.1%} | top-3 {recall[3]:.1%} | top-5 {recall[5]:.1%} | {latency * 1000:.0f} ms/question")


if __name__ == "__main__":
    main()
