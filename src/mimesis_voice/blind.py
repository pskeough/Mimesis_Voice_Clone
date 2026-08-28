"""Blind rating packets: the only mechanism that produces ranked judgments.

The engine has never been validated against a human verdict. When it finally was,
the shipped fingerprint scored 60% pairwise agreement with the author's ranking,
identical to a word-count control. That is not an argument for a better feature.
It is an argument for a feedback loop, because without one there is no way to
tell whether any feature is better.

The author's stored history cannot supply this. It records preferences as rules
("no em-dashes", "too formal", "sounds like AI"), never as "draft A beat draft B".
Pairwise data only exists when someone runs a blind comparison. So the comparison
has to be cheap enough to run routinely rather than as a special occasion.

Two functions:

    packet()  -> unlabeled HTML + a key, order shuffled and VERIFIED non-constant
    ingest()  -> turn the author's ranking back into a Judgment for the store

Blindness is not decoration. The author's own standing instruction is that samples
for comparison must be shown without pre-labeling, because knowing the source
contaminates the read before the first sentence. A packet that leaks its key is
worse than no packet, since it produces confident data pointing the wrong way.
"""
from __future__ import annotations

import hashlib
import html
import json
import random
from dataclasses import dataclass
from pathlib import Path

from . import preference as pref

LETTERS = "ABCDEFGH"


@dataclass
class Packet:
    task_id: str
    voice: str
    html_path: Path
    key_path: Path
    key: dict[str, str]          # letter -> arm
    task_prompt: str | None = None

    def render_key(self) -> str:
        return "\n".join(f"  {k} = {v}" for k, v in sorted(self.key.items()))


def _shuffle(arms: list[str], seed: str) -> list[str]:
    """Deterministic per-task order, verified not to be the identity.

    A hash-derived order silently degenerated once and put the same arm in
    position A for every task in a set, which reads as a position control in the
    writeup while providing none. Reproducibility is worth keeping, so the seed
    is retained and the degenerate case is detected and perturbed rather than
    swapped for unseeded randomness.
    """
    if len(arms) < 2:
        return list(arms)
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    order = list(arms)
    for _ in range(8):
        rng.shuffle(order)
        if order != list(arms):
            return order
    return list(reversed(arms))


def packet(out_dir, task_id: str, candidates: dict[str, str], voice: str,
           task_prompt: str | None = None, title: str | None = None,
           show_word_counts: bool = True) -> Packet:
    """Write an unlabeled comparison packet plus a separate key file.

    ``candidates`` maps arm name -> text. Arm names never reach the HTML.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = _shuffle(list(candidates), seed=task_id)

    key, blocks = {}, []
    for i, arm in enumerate(arms):
        letter = LETTERS[i]
        key[letter] = arm
        text = (candidates[arm] or "").strip()
        wc = (f"<span class='wc'>{len(text.split())} words</span>"
              if show_word_counts else "")
        paras = "".join(f"<p>{html.escape(p.strip())}</p>"
                        for p in text.split("\n") if p.strip())
        blocks.append(f"<section><h3>{letter}{wc}</h3>"
                      f"<div class='prose'>{paras}</div></section>")

    head = html.escape(title or task_id)
    prompt_block = (f"<p class='inst'>{html.escape(task_prompt)}</p>"
                    if task_prompt else "")
    doc = f"""<title>{head}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1a1a17; --mut:#6f6f68; --line:#e3e1d9; --card:#fff; --accent:#8a5a3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#151517; --fg:#e9e8e3; --mut:#98978f; --line:#2b2b30; --card:#1c1c20; --accent:#d0a077; }} }}
:root[data-theme="dark"] {{ --bg:#151517; --fg:#e9e8e3; --mut:#98978f; --line:#2b2b30; --card:#1c1c20; --accent:#d0a077; }}
*{{box-sizing:border-box}}
body {{ background:var(--bg); color:var(--fg); font:17px/1.7 Georgia,'Iowan Old Style',serif;
  max-width:44rem; margin:0 auto; padding:2.5rem 1.25rem 6rem; }}
h1 {{ font-size:1.45rem; font-weight:600; margin:0 0 1.2rem; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:1.4rem 1.6rem; margin:1.5rem 0; }}
h3 {{ font-size:.75rem; font-weight:700; letter-spacing:.14em; color:var(--accent);
  margin:0 0 1rem; display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px solid var(--line); padding-bottom:.5rem; }}
.wc {{ font-weight:400; letter-spacing:.02em; color:var(--mut); font-size:.78rem; }}
.prose p {{ margin:0 0 1rem; }} .prose p:last-child {{ margin-bottom:0; }}
.inst {{ color:var(--mut); font-style:italic; font-size:.9rem;
  border-left:2px solid var(--line); padding-left:1rem; }}
.note {{ background:var(--card); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:6px; padding:1rem 1.25rem;
  font-size:.88rem; color:var(--mut); }}
</style>
<h1>{head}</h1>
<div class="note">Unlabeled, order shuffled. Rank these best to worst and say why.
Your ranking is what the engine gets scored against, so a rough ordering with real
reasons is worth more than a careful one without them.</div>
{prompt_block}
{''.join(blocks)}"""

    html_path = out_dir / f"blind_{task_id}.html"
    key_path = out_dir / f"blind_{task_id}_key.json"
    html_path.write_text(doc, encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")

    return Packet(task_id=task_id, voice=voice, html_path=html_path,
                  key_path=key_path, key=key, task_prompt=task_prompt)


def ingest(profile, pkt: Packet, ranking: list[str],
           grades: dict[str, str] | None = None,
           rationales: dict[str, str] | None = None,
           candidates: dict[str, str] | None = None,
           note: str | None = None) -> pref.Judgment:
    """Turn a letter ranking back into a stored Judgment.

    ``ranking`` is letters best-first, e.g. ["C","A","D","B"]. Letters not
    ranked are dropped rather than assumed worst: silence is not a verdict.
    """
    if candidates is None:
        raise ValueError("candidates required to store the rated texts")
    grades = grades or {}
    rationales = rationales or {}

    items = []
    for rank, letter in enumerate(ranking, start=1):
        arm = pkt.key.get(letter)
        if arm is None:
            raise ValueError(f"letter {letter!r} not in packet key")
        items.append(pref.RatedItem(
            arm=arm, text=candidates[arm], rank=rank,
            grade=grades.get(letter), rationale=rationales.get(letter)))

    j = pref.Judgment(task_id=pkt.task_id, voice=pkt.voice, items=items,
                      blind=True, task_prompt=pkt.task_prompt, note=note)
    pref.record(profile, j)
    return j
