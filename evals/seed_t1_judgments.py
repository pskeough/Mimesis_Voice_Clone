"""Seed the preference store with the 2026-08-15 blind ratings, then benchmark.

These are real verdicts: five chapter openings from one spec, presented blind and
order-shuffled, ranked by the author with written rationale. One of them was his
own prose and he identified it.

The point of loading them is not the record. It is the baseline: running the
SHIPPED fingerprint through benchmark() answers whether the engine's own
selection criterion predicts what the author actually prefers. Every later
proposal is measured against that number.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\AI Coding Projects\Online_AI\Mimesis\src")

from mimesis_voice import cli as up_cli, preference as pref  # noqa: E402
from mimesis_voice.fingerprint import Fingerprint  # noqa: E402

ARMS = r"C:\LocalAI\LLM\eval\basilisk"

# rank, arm, grade, verbatim rationale
RATINGS = [
    (1, "human",        "A",  "good, probably my OG writing. Some errors, but overall fine"),
    (2, "opus_mimesis", "B+", "first paragraph is genuinely pretty good. Transitions are way "
                              "too underdiversified, way too many He the his, in. writing is "
                              "good but transitions are poor. diverted from the spec in a weird way"),
    (3, "qwen_mimesis", "B",  "much better writing than B but has this overdrawn lesser "
                              "offending AI tell. The hallway was empty. the lobby was empty, "
                              "the gutters were clean - all in the same paragraph, its this kind "
                              "of repeating undiversified AI writing. Writing quality is good but "
                              "the structure and delivery is bad"),
    (4, "qwen_bare",    "B-", "way too many comas and extensive, writing is okay, but way too "
                              "run on. too many The, He, his, poor transition diversity. "
                              "'or he did' and 'or the specific quality' are AI tells"),
    (5, "opus_bare",    "F",  "already bad: and, and, the, the, so that, the, for a - all AI "
                              "tells. way too many commas. these punchy short sentences. over "
                              "structuralized. same transitions over and over again. hard reject"),
]

FILES = {
    "human": "T1__HUMAN.json", "opus_bare": "T1__OPUS_bare.json",
    "opus_mimesis": "T1__OPUS_mimesis.json", "qwen_bare": "T1__QWEN_bare.json",
    "qwen_mimesis": "T1__QWEN_mimesis.json",
}


def load_text(arm):
    p = os.path.join(ARMS, FILES[arm])
    return json.load(open(p, encoding="utf-8-sig"))["content"].strip()


def main():
    prof = up_cli._resolve("creative")

    existing = {j.task_id for j in pref.load(prof)}
    if "basilisk_day1_opening_2026-08-15" in existing:
        print("already seeded; skipping the append (store is append-only by design)")
    else:
        j = pref.Judgment(
            task_id="basilisk_day1_opening_2026-08-15", voice="creative", blind=True,
            task_prompt="Write the opening of Day 1 from a spec: student wakes in a wrecked "
                        "dorm, campus emptied, auditorium of the dead, a voice. ~1300 words.",
            note="Blind, order-shuffled. One arm is the author's own prose; he identified it "
                 "and ranked it first. Caveat: the shuffle degenerated so the author's arm "
                 "always sat in position A. Reasons given were craft-specific, but uniform "
                 "position bias cannot be fully excluded.",
            items=[pref.RatedItem(arm=arm, text=load_text(arm), rank=rank,
                                  grade=grade, rationale=why)
                   for rank, arm, grade, why in RATINGS])
        pref.record(prof, j)
        print("seeded 1 judgment")

    print()
    print(pref.summary(prof))
    print()

    # THE baseline. Does the shipped selection criterion predict the author?
    fp = Fingerprint.load(prof.fingerprint_path)
    scorers = {"fingerprint (shipped, lower=better)": fp.distance}

    # A transparently dumb control. If a proposal cannot beat word count, it is
    # not measuring voice.
    scorers["word count (control)"] = lambda t: len(t.split())

    print("BASELINE - how well does each scorer reproduce the author's ranking?")
    print("(50% = chance)")
    for r in pref.compare(prof, scorers):
        print("  " + r.render())


if __name__ == "__main__":
    main()
