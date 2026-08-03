"""Does the StyleDistance backend retrieve *different* anchors than bge-small?

For every brief, retrieve the top-k anchors under the fast (topic-leaning) and
style (content-independent) backends and compare the two sets. Reported: the mean
Jaccard overlap of the retrieved filename sets and the mean count of shared
top-k anchors. Low overlap = the style embedder is pulling genuinely different
passages, which is the literature claim (a style critic anchors on voice, not
subject matter). Offline: no generation, just the two stores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from mimesis_voice import config, retrieve
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mimesis_voice import config, retrieve

EVALS_DIR = Path(__file__).resolve().parent
BRIEFS_PATH = EVALS_DIR / "briefs.json"

PAIRS = [("research", "research", "research-style"), ("creative", "creative", "creative-style")]


def _briefs(voice: str) -> list[dict]:
    data = json.loads(BRIEFS_PATH.read_text(encoding="utf-8"))
    return [b for b in data["briefs"] if b["voice"] == voice]


def _names(profile_slug: str, task: str, k: int) -> list[str]:
    prof = config.resolve_named(profile_slug)
    hits = retrieve.retrieve(task, k, prof)
    return [h["filename"] for h in hits]


def main() -> int:
    k = 5
    out = {"k": k, "voices": {}}
    for voice, fast_slug, style_slug in PAIRS:
        rows = []
        for b in _briefs(voice):
            fast = _names(fast_slug, b["task"], k)
            style = _names(style_slug, b["task"], k)
            sf, ss = set(fast), set(style)
            inter = len(sf & ss)
            union = len(sf | ss) or 1
            rows.append({
                "brief": b["id"],
                "shared": inter,
                "jaccard": round(inter / union, 3),
                "fast": fast,
                "style": style,
            })
        mean_shared = round(sum(r["shared"] for r in rows) / len(rows), 2) if rows else 0
        mean_jac = round(sum(r["jaccard"] for r in rows) / len(rows), 3) if rows else 0
        out["voices"][voice] = {"mean_shared": mean_shared, "mean_jaccard": mean_jac,
                                "top_k": k, "briefs": rows}
        print(f"[{voice}] mean shared top-{k} anchors: {mean_shared}/{k}, mean Jaccard: {mean_jac}")
        for r in rows:
            print(f"  {r['brief']}: shared {r['shared']}/{k}  fast={r['fast']}  style={r['style']}")
    (EVALS_DIR / "scratch").mkdir(parents=True, exist_ok=True)
    (EVALS_DIR / "scratch" / "anchor_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved -> evals/scratch/anchor_overlap.json")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
