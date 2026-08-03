"""Regression tests for the rhetorical-tic detector.

Two things this file guards against, mirroring test_presence.py's discipline:

1. The three false starts this rule went through on 2026-07-28, measured against a
   reference creative corpus (111 real works, held locally):
     v1 (any repeated clause-initial word)          56/111 = 50.5% FPR -- caught the
       author's genuine anaphoric intensification ("the last few days, the last few hours...")
     v2 (discourse connectives incl. not/no/never)  26/111 = 23.4% FPR -- caught the
       author's negation-parallelism, a real device in their existentialist register
     v3 (ordinal/temporal connectives only)          3/111 =  2.7% FPR -- shipped
   Any change to _ESCALATION_OPENERS must re-run the corpus sweep, not just these tests.

2. The three live examples that motivated this module, verbatim from a real generation
   the author read and independently called "terrible" / "super AI heavy" before knowing
   what produced them, plus the real task-leakage failure found the same night in the
   spectrum run's W0_R1_Pdesc cell.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimesis_voice import rhetoric  # noqa: E402

# --- the three lines the author flagged by eye, 2026-07-28 --------------------------

TRIAD_LIVE = "First the language goes, then the reasons, then the wanting."
META_ASIDE_LIVE = "Everything on the plant has a use, which is the part I like."
ABSENCE_PUNCH_LIVE = (
    "He had a list of them; he recited it to himself on the walk in, and for a while "
    "the reciting was enough. That man is gone."
)
LEAKAGE_LIVE = (
    "Okay, here's my task: write a passage in the style of the examples provided. "
    "The topic is \"The Child Who Walks Alone\". Let me try to think through this."
)

# A second triad instance, found independently in a LoRA generation that never saw any
# prompt engineering -- evidence this is a base-model reflex, not an artifact of one
# prompt.
TRIAD_LORA = "as if I were a man in a play, or a woman in a ritual, or a master in a ceremony."


def test_triad_first_then_then():
    assert rhetoric.find_triads(TRIAD_LIVE)


def test_triad_or_or():
    assert rhetoric.find_triads(TRIAD_LORA)


def test_meta_aside_which_is_the_part_i_like():
    assert rhetoric.find_meta_asides(META_ASIDE_LIVE)


def test_absence_punch_that_man_is_gone():
    hits = rhetoric.find_absence_punches(ABSENCE_PUNCH_LIVE)
    assert hits
    assert "gone" in hits[0][1].lower()


def test_leakage_okay_heres_my_task():
    assert rhetoric.find_leakage(LEAKAGE_LIVE)


# --- negative controls: devices this author genuinely uses, must NOT fire ----------

NEGATION_TRIAD_REAL = (
    "No martyr am I, no Christ rose, or Nirvana awoke, no, I live as man, to speak "
    "equally to my fellow men."
)
CONTENT_ANAPHORA_REAL = (
    "This dynamite has built up over the last few days, the last few hours, the last "
    "five seconds."
)


def test_negation_triad_not_flagged():
    """His existentialist register uses not/no parallelism constantly (v2 fired on
    23.4% of his corpus because of exactly this). Must stay unflagged."""
    assert not rhetoric.find_triads(NEGATION_TRIAD_REAL)


def test_content_anaphora_not_flagged():
    """Escalating repetition of a content word (an article, here) is a literary device,
    not the AI tic. v1 fired on 50.5% of his corpus because of exactly this."""
    assert not rhetoric.find_triads(CONTENT_ANAPHORA_REAL)


def test_ordinary_prose_is_clean():
    normal = (
        "The reed breaks at my feet, a soft whisper of green against the earth. I have "
        "been here all morning, bending low, pulling the stalks from the water with "
        "careful hands."
    )
    rep = rhetoric.analyze(normal)
    assert rep.is_clean


def test_full_draft_catches_all_three_live_flags():
    """The actual passage the author read and called terrible, in full. All three
    non-leakage checks must fire; leakage must not (nothing broke character here)."""
    draft = (
        "Morning, and the marsh again. I go out because there is nothing else to go "
        "out for, and I pull the cattails until the ache settles into the small of my "
        "back and stays there.\n\n"
        "Everything on the plant has a use, which is the part I like. Heads to "
        "stuffing, roots to flour, the split stalks twisted into cord that holds "
        "until it doesn't.\n\n"
        "There was a man who came out here with reasons. He had a list of them; he "
        "recited it to himself on the walk in, and for a while the reciting was "
        "enough. That man is gone. What sits here now eats what the water gives and "
        "sleeps when the light goes, and it does not miss him. First the language "
        "goes, then the reasons, then the wanting. Solitude does not cure a man."
    )
    rep = rhetoric.analyze(draft)
    assert rep.triads
    assert rep.meta_asides
    assert rep.absence_punches
    assert not rep.leakage
    assert not rep.is_clean


if __name__ == "__main__":
    import inspect
    fns = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
