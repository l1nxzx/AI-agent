import pytest

from tinylm.retrieval import BM25Index, chunk_text, clean_textbook, stem, tokenize

PASSAGES = [
    ("Photosynthesis", "Photosynthesis is the process plants use to turn sunlight, water and carbon dioxide into sugar and oxygen. It happens in the chloroplasts of plant cells."),
    ("Mitochondria", "Mitochondria are organelles that make most of the energy a cell needs. They are called the powerhouse of the cell and they have their own DNA."),
    ("Honduras", "Honduras is a country in Central America. Its capital is Tegucigalpa. Most people speak Spanish."),
    ("Dogs", "Dogs are friendly animals. Many people keep dogs as pets and dogs like to play fetch in the park."),
    ("The", "The the the of and is a to in that it."),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx")
    return BM25Index.build(str(d), PASSAGES)


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("What are the cells of the body?") == ["cell", "body"]
    assert stem("cells") == stem("cell") and stem("photosynthesizing") == "photosynthesiz"
    assert tokenize("The of and") == []


def test_search_ranks_the_right_passage_first(index):
    assert index.passage(index.search("what makes the energy in a cell")[0][0])[0] == "Mitochondria"
    assert index.passage(index.search("how do plants turn sunlight into sugar?")[0][0])[0] == "Photosynthesis"
    assert index.passage(index.search("What is the capital of Honduras?")[0][0])[0] == "Honduras"
    assert index.passage(index.search("do people keep dogs as pets")[0][0])[0] == "Dogs"


def test_rare_words_count_more_than_common_ones(index):
    # 'cell' appears in two passages, 'Tegucigalpa' in one: a query with the rare word picks its passage
    assert index.passage(index.search("cell Tegucigalpa")[0][0])[0] == "Honduras"


def test_unknown_and_empty_queries(index):
    assert index.search("zzzzqq xxyyzz") == []
    assert index.search("the of and") == []
    assert index.search("") == []


def test_index_reloads_from_disk(index):
    again = BM25Index(index.dir).load()
    assert again.search("energy cell")[0][0] == index.search("energy cell")[0][0]
    assert again.n_docs == len(PASSAGES)


def test_scores_are_sorted_and_k_limited(index):
    res = index.search("cell plants dogs country", k=3)
    assert len(res) <= 3 and [s for _, s in res] == sorted([s for _, s in res], reverse=True)


def test_chunking_respects_size_and_keeps_all_words():
    text = "\n\n".join(f"This is sentence number {i} of the long document about cells. " * 3 for i in range(30))
    chunks = chunk_text(text, min_words=40, max_words=90)
    assert all(len(c.split()) <= 130 for c in chunks) and len(chunks) > 3
    assert sum(len(c.split()) for c in chunks) >= 0.95 * len(text.split())
    assert chunk_text("Short text here.") == ["Short text here."]


def test_chunking_splits_a_very_long_paragraph_on_sentences():
    para = " ".join(f"Sentence {i} is about the biology of cells and it is quite long indeed." for i in range(60))
    chunks = chunk_text(para, min_words=50, max_words=100)
    assert len(chunks) > 3 and all(c.endswith(".") for c in chunks)


def test_textbook_cleaning():
    raw = "Contents\n1.1 Cells . . . . . . . . . . 12\n12\nCells are the basic unit of\nlife. They are very\nsmall and hyphen-\nated words join.\n\n45\n"
    out = clean_textbook(raw)
    assert "Cells are the basic unit of life." in out and "hyphenated" in out
    assert ". . . ." not in out and not out.strip().endswith("45")
