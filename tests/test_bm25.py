"""BM25 scoring + integration with search_data (B9)."""

import csv

import pytest

from jdatamunch_mcp.bm25 import BM25, tokenize
from jdatamunch_mcp.tools.index_local import index_local
from jdatamunch_mcp.tools.search_data import search_data


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello WORLD_42") == ["hello", "world_42"]


def test_bm25_basic_ranking():
    docs = [
        ["customer", "name", "address"],
        ["product", "id", "price"],
        ["customer", "phone", "number"],
    ]
    bm25 = BM25(docs)
    s0 = bm25.score(["customer"], 0)
    s1 = bm25.score(["customer"], 1)
    s2 = bm25.score(["customer"], 2)
    assert s0 > s1
    assert s2 > s1


def test_bm25_unknown_term_zero():
    bm25 = BM25([["a", "b"], ["b", "c"]])
    assert bm25.score(["zzz"], 0) == 0.0


def test_search_data_uses_bm25_in_all_scope(tmp_path):
    csv_path = tmp_path / "shop.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_email", "product_id", "phone_number"])
        w.writerows([("a@x.com", 1, "+15555550100"), ("b@x.com", 2, "+15555550101")])
    storage = tmp_path / "data-index"
    storage.mkdir()
    index_local(path=str(csv_path), name="shop", storage_path=str(storage))

    res = search_data(dataset="shop", query="customer", storage_path=str(storage))
    assert isinstance(res["result"], list) and res["result"]
    # The top result should reference the customer_email column.
    assert res["result"][0]["name"].lower().startswith("customer")


# --- v1.23.1: Unicode + CJK bigram tokenization (jdocmunch #91 class) -------


def test_tokenize_korean_produces_bigrams():
    # Was [] — the old [A-Za-z0-9_]+ regex dropped all non-ASCII.
    assert tokenize("초과근무 승인") == ["초과", "과근", "근무", "승인"]


def test_tokenize_accented_latin_intact():
    # Was ['caf', 'na', 've'].
    assert tokenize("café naïve") == ["café", "naïve"]


def test_tokenize_mixed_snake_case_and_cjk():
    toks = tokenize("user_id 급여액")
    assert "user_id" in toks  # underscore-joined tokens stay whole
    assert "급여" in toks and "여액" in toks


def test_tokenize_single_cjk_char_kept():
    assert tokenize("車") == ["車"]


def test_bm25_korean_query_scores_korean_doc():
    docs = [tokenize("초과근무 승인 규칙"), tokenize("product id price")]
    bm25 = BM25(docs)
    q = tokenize("초과근무")
    assert bm25.score(q, 0) > 0.0
    assert bm25.score(q, 1) == 0.0
