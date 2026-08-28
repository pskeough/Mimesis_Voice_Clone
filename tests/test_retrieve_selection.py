"""Retrieval selection: per-document cap and move-matched transform demos.

Both were added 2026-08-28 after a measurement pass over the creative profile's
seven frozen briefs found that three novel documents own 49.7% of the chunk pool
and a single stage-play document supplied 11.4% of every anchor slot, 8.97x its
share. Neither is a hypothetical.
"""
from __future__ import annotations

from mimesis_voice import retrieve


def _hit(doc, i):
    return {"id": f"{doc}::{i}", "filename": doc, "text": f"chunk {i} of {doc}"}


def test_cap_limits_chunks_from_one_document():
    hits = [_hit("novel", i) for i in range(5)] + [_hit("letter", 0), _hit("essay", 0)]
    picked = retrieve._cap_per_document(hits, k=3, cap=1)
    assert [h["filename"] for h in picked] == ["novel", "letter", "essay"]


def test_cap_backfills_rather_than_returning_short():
    """A query whose best material genuinely lives in one document must still get k."""
    hits = [_hit("novel", i) for i in range(5)]
    picked = retrieve._cap_per_document(hits, k=3, cap=1)
    assert len(picked) == 3
    assert picked[0]["id"] == "novel::0"


def test_cap_preserves_rank_order_within_the_cap():
    hits = [_hit("a", 0), _hit("b", 0), _hit("a", 1), _hit("c", 0)]
    picked = retrieve._cap_per_document(hits, k=3, cap=1)
    assert [h["filename"] for h in picked] == ["a", "b", "c"]


def test_infer_move_reads_the_rhetorical_slot_from_the_brief():
    assert retrieve.infer_move("write the limitations paragraph for this audit") == "limitations"
    assert retrieve.infer_move("describe what table 4 shows") == "table_description"
    assert retrieve.infer_move("write the abstract") == "abstract"
    assert retrieve.infer_move("a short story about a dog") is None


def test_infer_move_prefers_the_longer_keyword():
    """'methods_procedure' and 'methods' both match a procedure brief; the more
    specific slot has to win or the general one always shadows it."""
    assert retrieve.infer_move("lay out the procedure step by step") == "methods_procedure"
