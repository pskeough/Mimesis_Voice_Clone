"""Regression tests for chunking and extraction.

The chunker case here is the important one. It did not raise, did not warn, and
did not produce a short corpus that looked short: it produced a *plausible* store
that was missing most of the document. A 102,000-word novel ingested to 7 chunks
and reported "Store built" like any other run. Everything downstream -- the
fingerprint, the gate threshold, the anchors -- was then calibrated on the first
few pages of the book.
"""
from __future__ import annotations

from mimesis_voice import ingest


def test_oversized_paragraph_does_not_truncate_the_document():
    """A paragraph over HARD_CAP_WORDS must not end chunking of everything after it.

    The oversized branch sentence-splits the paragraph itself and left ``cur``
    empty. Control then reached the small-tail check, which sees cur_wc == 0,
    treats it as a runt tail, and breaks the OUTER loop. Result: every paragraph
    after the first long one was dropped.
    """
    paragraphs = ["word " * 100] * 3 + ["word " * 400] + ["word " * 100] * 20
    expected = sum(len(p.split()) for p in paragraphs)

    chunks = ingest.chunk_paragraphs(paragraphs)

    assert sum(len(c.split()) for c in chunks) == expected, "content was dropped"
    assert len(chunks) > 20, "chunking stopped early at the oversized paragraph"


def test_content_after_an_oversized_paragraph_survives():
    """Same bug, stated as reachability rather than word count."""
    chunks = ingest.chunk_paragraphs(["alpha " * 100, "beta " * 400, "gamma " * 100])
    joined = " ".join(chunks)
    assert "alpha" in joined
    assert "beta" in joined
    assert "gamma" in joined, "text after the oversized paragraph was lost"


def test_ordinary_corpus_is_unaffected_by_the_fix():
    """The guard must not change behaviour when no paragraph is oversized."""
    paragraphs = ["word " * 100] * 23
    chunks = ingest.chunk_paragraphs(paragraphs)
    assert sum(len(c.split()) for c in chunks) == sum(len(p.split()) for p in paragraphs)


def test_pdf_is_a_supported_extension():
    """Corpora that exist only as PDFs are common enough that dropping them
    silently (find_documents filters on SUPPORTED_EXTS) reads as 'no documents'."""
    assert ".pdf" in ingest.SUPPORTED_EXTS
