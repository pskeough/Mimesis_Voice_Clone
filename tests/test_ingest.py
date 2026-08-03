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


# --- scrubber: the dash rule must follow the author, not the builder ----------

def test_dash_strip_is_disabled_for_an_author_who_uses_dashes():
    """The em-dash strip was unconditional. For a writer whose own corpus runs
    dashes as punctuation that is not de-AI-ing a draft, it is deleting a habit:
    a measured social corpus ran 42/1000 words at p95."""
    from mimesis_voice import scrub

    text = "the room -- and the hall -- went quiet"
    stripped, n = scrub.scalpel(text)
    assert n > 0 and "--" not in stripped, "default behaviour must still strip"

    kept, n_kept = scrub.scalpel(text, allow_dashes=True)
    assert kept == text and n_kept == 0, "an author's own dashes must survive"


def test_author_uses_dashes_follows_the_calibrated_band():
    from mimesis_voice.scrub import ScrubCalibration, author_uses_dashes

    base = dict(banned_words=[], banned_phrases=[], whitelist=[],
                burstiness_floor=5.0, hedge_ceiling=1.0, mean_sentence_len=18.0)
    # A novel with no em-dashes still scores slightly above zero on en-dashes and
    # hyphen runs; that must not license the habit.
    assert not author_uses_dashes(ScrubCalibration(**base, dash_per1k_p95=0.63))
    assert author_uses_dashes(ScrubCalibration(**base, dash_per1k_p95=12.63))
