"""Tiny BM25 implementation for keyword ranking (B9).

Vendored to avoid a runtime dependency. Each column is a "document"
composed of its name, ai_summary, and a flattened list of sample values.
Scoring uses the canonical Okapi BM25 formula with k1=1.5, b=0.75.
"""

from __future__ import annotations

import math
import re
from typing import Iterable


# Unicode word runs, underscore included so snake_case identifiers stay whole.
# (Was [A-Za-z0-9_]+, which silently dropped ALL non-ASCII — CJK column names
# and values produced zero tokens; jdocmunch #91 class.)
_TOKEN_RX = re.compile(r"\w+")
# CJK scripts carry no whitespace word boundaries; runs in these ranges are
# expanded to overlapping character bigrams (same expansion at index and
# query time, so bigram overlap is the match signal).
_CJK_RX = re.compile(
    "[ᄀ-ᇿ"  # Hangul Jamo
    "぀-ヿ"  # Hiragana + Katakana
    "㄰-㆏"  # Hangul Compatibility Jamo
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "㐀-䶿"  # CJK Unified Ideographs Extension A
    "一-鿿"  # CJK Unified Ideographs
    "가-힯"  # Hangul Syllables
    "豈-﫿]+"  # CJK Compatibility Ideographs
)
_K1 = 1.5
_B = 0.75


def _cjk_bigrams(run: str) -> list[str]:
    """Overlapping character bigrams for a CJK run; a lone char passes through."""
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def tokenize(text: str) -> list[str]:
    """Lowercase + split on Unicode word boundaries; CJK runs become bigrams."""
    if not text:
        return []
    # Pad CJK runs with spaces so mixed-script tokens split cleanly.
    text = _CJK_RX.sub(lambda m: " " + m.group(0) + " ", text)
    out: list[str] = []
    for m in _TOKEN_RX.finditer(text):
        tok = m.group(0).lower()
        if _CJK_RX.fullmatch(tok):
            out.extend(_cjk_bigrams(tok))
        else:
            out.append(tok)
    return out


class BM25:
    """In-memory BM25 ranker over a small corpus.

    Designed for the column-search use case (typically <1000 docs).
    """

    __slots__ = ("docs", "doc_lens", "avg_doc_len", "df", "n", "_idf_cache")

    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.doc_lens = [len(d) for d in docs]
        self.n = len(docs)
        self.avg_doc_len = (sum(self.doc_lens) / self.n) if self.n else 0.0
        # Document frequency per term
        self.df: dict[str, int] = {}
        for d in docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1
        self._idf_cache: dict[str, float] = {}

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = self.df.get(term, 0)
        # Add-one smoothing keeps IDF non-negative for rare terms.
        idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = idf
        return idf

    def score(self, query_terms: Iterable[str], doc_idx: int) -> float:
        if doc_idx < 0 or doc_idx >= self.n:
            return 0.0
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        # Term frequency in this document
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = self.doc_lens[doc_idx]
        if self.avg_doc_len <= 0:
            return 0.0
        denom_norm = _K1 * (1.0 - _B + _B * dl / self.avg_doc_len)
        score = 0.0
        for q in query_terms:
            f = tf.get(q, 0)
            if f == 0:
                continue
            score += self._idf(q) * (f * (_K1 + 1.0)) / (f + denom_norm)
        return score
