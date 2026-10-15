"""Two ingest defects found on a second author's corpus, 2026-09-02.

1. A paragraph over HARD_CAP_WORDS that follows a short overlap paragraph made
   chunk_paragraphs fold the short one into the previous chunk and then break
   out of the document. A 124,123-word book came out as 897 words, and the
   existing test for oversized paragraphs did not catch it because there the
   oversized paragraph came first.
2. YAML front matter in a .md export was ingested as body text.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import ingest  # noqa: E402


def _para(n: int, seed: str = "word") -> str:
    return " ".join(f"{seed}{k}" for k in range(n)) + "."


def test_oversized_paragraph_after_a_short_one_does_not_truncate():
    paras = [_para(120, "a"), _para(100, "b"), _para(30, "c"),
             _para(ingest.HARD_CAP_WORDS + 50, "big"),
             _para(150, "d"), _para(150, "e"), _para(150, "f")]
    chunks = ingest.chunk_paragraphs(paras)
    joined = " ".join(chunks)
    for seed in ("big", "d", "e", "f"):
        assert f"{seed}0 " in joined or f"{seed}0." in joined, f"paragraph {seed} was dropped"
    total_in = sum(len(p.split()) for p in paras)
    total_out = sum(len(c.split()) for c in chunks)
    # Overlap repeats a little; truncation loses a lot.
    assert total_out >= total_in


def test_the_book_shape_survives():
    # 35 short paragraphs then a run of 400-word ones, as in the real file.
    paras = [_para(30, f"s{k}_") for k in range(35)] + [_para(400, f"L{k}_") for k in range(20)]
    chunks = ingest.chunk_paragraphs(paras)
    total_in = sum(len(p.split()) for p in paras)
    total_out = sum(len(c.split()) for c in chunks)
    assert total_out >= 0.95 * total_in


def test_front_matter_is_stripped_from_md(tmp_path):
    p = tmp_path / "post_001.md"
    p.write_text("---\ndate: 2026-06-16T10:15:23.083Z\nurl: https://www.linkedin.com/posts/x\nlikes: 3\ntype: post\n---\n"
                 "What if your culture no longer needed people?\n\nBoth cannot be true.\n", encoding="utf-8")
    paras = ingest.extract_paragraphs(p)
    text = "\n".join(paras)
    assert "linkedin.com" not in text
    assert "likes:" not in text
    assert "What if your culture" in text


def test_txt_without_front_matter_is_unchanged(tmp_path):
    p = tmp_path / "essay.txt"
    body = "The second paragraph carries enough words to survive the paragraph filter, which drops very short fragments on ingest. " * 2
    p.write_text("---\nnot front matter, a rule\n\n" + body + "\n", encoding="utf-8")
    paras = ingest.extract_paragraphs(p)
    assert any("second paragraph carries" in x for x in paras)
