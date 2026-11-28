"""Build the searchable knowledge corpus: Simple English Wikipedia (general knowledge) plus OpenStax biology
textbooks (the biology focus), split into passages and indexed with BM25.

  python -m tinylm.build_corpus --corpus_dir corpus

Sources (checked on the Hugging Face Hub):
  wikimedia/wikipedia, config 20231101.simple   CC BY-SA 3.0 and GFDL   241,787 articles, 43.6M words
  crumb/openstax-text                           CC BY 4.0               OpenStax textbooks as raw text: the five
                                                biology books (Biology 2e, AP Biology, Concepts of Biology,
                                                Microbiology, Anatomy and Physiology 2e, about 2.7M words) plus,
                                                by default, the other ~70 books (chemistry, physics, astronomy,
                                                history, economics, psychology, ...)
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # see prepare_data.py

from .logutil import setup_logging
from .retrieval import BM25Index, chunk_text, clean_textbook

SIMPLE_WIKI = ("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet")
OPENSTAX_REPO = "crumb/openstax-text"
BIOLOGY_BOOKS = {   # file in the repo -> title shown as the source
    "data/Biology2e-WEB_ICOFkGu.txt": "Biology 2e (OpenStax)",
    "data/APBiology-OP_5meoFaG.txt": "AP Biology (OpenStax)",
    "data/ConceptsBiology-OP_Jyx0DT8.txt": "Concepts of Biology (OpenStax)",
    "data/Microbiology-WEB.txt": "Microbiology (OpenStax)",
    "data/Anatomy_and_Physiology_2e_-_WEB_c9nD9QL.txt": "Anatomy and Physiology 2e (OpenStax)",
}


def iter_passages(cache_dir, max_articles=None, log=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(SIMPLE_WIKI[0], SIMPLE_WIKI[1], repo_type="dataset", cache_dir=cache_dir)
    pf = pq.ParquetFile(path)
    if not {"title", "text"} <= set(pf.schema_arrow.names):
        raise RuntimeError(f"unexpected columns in {path}: {pf.schema_arrow.names}")
    n = 0
    for batch in pf.iter_batches(batch_size=5000, columns=["title", "text"]):
        for title, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if not text or len(text.split()) < 25:
                continue            # stubs and redirects
            for chunk in chunk_text(text):
                yield title, chunk
            n += 1
            if max_articles and n >= max_articles:
                return
    if log:
        log.info(f"Simple Wikipedia: {n:,} articles used")


def book_title(fname):
    """'data/University_Physics_Volume_2-WEB_5eNhMSa.txt' -> 'University Physics Volume 2 (OpenStax)'"""
    import re
    stem_ = os.path.basename(fname)[:-4]
    stem_ = re.sub(r"[-_](WEB|OP|SAMPLE)\b.*$", "", stem_)
    return re.sub(r"[_-]+", " ", stem_).strip() + " (OpenStax)"


def list_books(which):
    """'biology': the five biology books. 'all': every OpenStax book (about 76; adds chemistry, physics, astronomy,
    history, economics, psychology, ...). The biology books are always included."""
    if which == "biology":
        return dict(BIOLOGY_BOOKS)
    from huggingface_hub import HfApi
    names = sorted(f.path for f in HfApi().list_repo_tree(OPENSTAX_REPO, repo_type="dataset", path_in_repo="data")
                   if f.path.endswith(".txt"))
    books = dict(BIOLOGY_BOOKS)
    seen = {book_title(f).lower() for f in books}
    for f in names:
        t = book_title(f)
        if t.lower() not in seen and f not in books:   # skip a second copy of the same book
            books[f] = t
            seen.add(t.lower())
    return books


def iter_book_passages(cache_dir, log=None, which="biology"):
    from huggingface_hub import hf_hub_download

    for fname, title in list_books(which).items():
        path = hf_hub_download(OPENSTAX_REPO, fname, repo_type="dataset", cache_dir=cache_dir)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = clean_textbook(f.read())
        chunks = chunk_text(text, min_words=70, max_words=130)
        if log:
            log.info(f"{title}: {len(text.split()):,} words -> {len(chunks):,} passages")
        for chunk in chunks:
            yield title, chunk


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_dir", default="corpus")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_articles", type=int, default=None, help="limit Simple Wikipedia articles (for quick tests)")
    p.add_argument("--no_books", action="store_true", help="general knowledge only")
    p.add_argument("--books", choices=["biology", "all"], default="all",
                   help="all (default): every OpenStax textbook, about 18M words, biology included; "
                        "biology: only the five biology books (about 2.7M words)")
    p.add_argument("--title_weight", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    log = setup_logging(os.path.join(args.corpus_dir, "build.log"))
    if os.path.exists(os.path.join(args.corpus_dir, "meta.json")) and not args.force:
        log.info("corpus already built (use --force to rebuild)")
        return
    cache = args.cache_dir or os.path.join(args.corpus_dir, "hf_cache")

    def everything():
        yield from iter_passages(cache, args.max_articles, log)
        if not args.no_books:
            yield from iter_book_passages(cache, log, args.books)

    index = BM25Index.build(args.corpus_dir, everything(), log, title_weight=args.title_weight)
    size = sum(os.path.getsize(os.path.join(args.corpus_dir, f)) for f in ("passages.txt", "index.npz", "offsets.npy", "vocab.json"))
    log.info(f"corpus ready in {args.corpus_dir}: {index.n_docs:,} passages, {size / 1e6:.0f} MB on disk")
    for q in ("How do plants make food from sunlight?", "What is the capital of Honduras?", "What do mitochondria do?"):
        top = index.search(q, k=1)
        if top:
            title, text = index.passage(top[0][0])
            log.info(f"sanity: {q!r} -> [{title}] {text[:140]}...")


if __name__ == "__main__":
    main()
