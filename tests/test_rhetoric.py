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


# --- self-rating clause (added 2026-08-13) ------------------------------------
#
# The expository sibling of the meta aside. Found live in a paper draft: a finding stated,
# then a clause rating how much the finding matters. Flagged by the author on read as
# "super AI-y". find_meta_asides does not catch it because that rule requires a first-person
# reaction verb, and this construction has none.

SELF_RATING_LIVE = (
    "The raters did not keep the two constructs apart, and this is the strongest "
    "objection available to a reader of our released data."
)

SELF_RATING_VARIANTS = [
    "Its practical consequence is unchanged and is the point worth keeping: it can score 3.",
    "One category is worth separating out because three unrelated measurements agree.",
    "The intervention results are the most encouraging thing here and we say so.",
    "A powered replication is the single most valuable extension of this work.",
    "It is not interchangeable, which is precisely why the arm was set aside.",
    "This is the effect the paper is about, appearing inside its own validation.",
]

# Clauses that back-reference but ADD information rather than rating. These must stay clean:
# banning them would strip the connective tissue that makes expository prose readable.
SELF_RATING_MUST_NOT_FIRE = [
    "The statistics rest on 9 positive cases, which is why the CI spans [0.42, 1.00].",
    "It is the lowest-scoring variant at M=1.42, which is the figure carrying the claim.",
    "Seven of eight do better, which is the opposite of what prompt engineering predicts.",
    "The judge does not emit a verdict directly.",
    "This is a construct problem and not a calibration problem.",
]


def test_self_rating_live_example():
    assert rhetoric.find_self_ratings(SELF_RATING_LIVE)


def test_self_rating_variants_all_caught():
    for s in SELF_RATING_VARIANTS:
        assert rhetoric.find_self_ratings(s), s


def test_self_rating_no_false_positives_on_informative_clauses():
    for s in SELF_RATING_MUST_NOT_FIRE:
        assert not rhetoric.find_self_ratings(s), s


def test_self_rating_rare_in_real_published_prose():
    """Base rate measured over 1.16M words of published papers: 0.032 per 1k words.

    The rule is only safe to ship because real prose almost never does this. If a future
    corpus pushes this past ~0.15/1k the rule is overfiring and should be narrowed.
    """
    import re
    from pathlib import Path
    corpus = Path(r"C:\Research\VoiceModel\literature\txt")
    if not corpus.is_dir():
        return  # corpus not present on this machine
    words = hits = 0
    for f in sorted(corpus.glob("*.txt")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        words += len(re.findall(r"\b[\w']+\b", t))
        hits += len(rhetoric.find_self_ratings(t))
    assert words > 0
    assert 1000 * hits / words < 0.15


# --- the unheeded reversal ----------------------------------------------------

def test_reversal_catches_the_specimens_the_antithesis_detector_missed():
    """Each of these was shown to the author blind; he named all three as tells.

    find_antithesis returned CLEAN on the first two. Its is-not-X-It-is-Y pattern
    requires the restating sentence to open with It/That/This + "is", so a
    reversal that restates by repeating the noun walks through untouched.
    """
    from mimesis_voice.rhetoric import find_reversals

    assert find_reversals("This is not hope. Hope has weight, and this has none.")
    assert find_reversals(
        "The architecture is sound. What it was measuring was too narrow, and "
        "that is a calibration problem, not a design one.")
    assert find_reversals(
        "What troubles me is not that I went quiet. It is that the quiet was comfortable.")


def test_reversal_stays_off_ordinary_negation():
    """A verifier that rejects everything is not a verifier.

    The first version of this detector matched any negation and any But-opener,
    and fired on 71-88% of the reference corpus. Ordinary narrative negates
    constantly and opens with "But" freely; both must pass.
    """
    from mimesis_voice.rhetoric import find_reversals

    assert not find_reversals("It is not raining. We should go to the market before it closes.")
    assert not find_reversals("I have no money. My brother is arriving on Thursday with the car.")
    assert not find_reversals("The coffee had gone cold. But nobody said anything.")
    assert not find_reversals(
        "Sell the cart, and the mare with it. But never the north field.")


def test_reversal_is_a_hard_flag():
    """Presence-based and hard, unlike the rated tics. The author's instruction was
    "make sure it's not gonna happen", not "keep it under a rate"."""
    from mimesis_voice.rhetoric import analyze

    rep = analyze("This is not hope. Hope has weight, and this has none.")
    assert "unheeded-reversal" in rep.hard_flags
    assert not analyze("The kettle takes four minutes. I check my phone.").hard_flags


# --- self-explanation ---------------------------------------------------------

def test_self_explanation_catches_the_hinge_and_is_what_forms():
    """Named in all twelve drafts of the 2026-08-28 blind set, across three voices
    and four generation arms. That it did not vary with the arm is the reason it is
    caught here rather than prompted away."""
    from mimesis_voice.rhetoric import find_self_explanations

    assert find_self_explanations(
        "Each attempt ends there, which means the phone has now asked me a "
        "genuine philosophical question three times.")
    assert find_self_explanations(
        "I have known that since roughly the drive home, which means I have spent "
        "about a hundred and twenty days being correct-adjacent.")
    assert find_self_explanations(
        "Which is the problem, because the contested items are the whole reason "
        "the evaluation exists.")
    assert find_self_explanations("No threshold recovers it. Thresholding is what discards it.")


def test_self_explanation_leaves_his_own_reverse_cleft_alone():
    """A plain-noun subject ("X was what killed it") is ordinary emphatic English
    and appears in real prose. Requiring a NOMINALIZED subject separates it from
    the generated form without needing a per-author rate gate."""
    from mimesis_voice.rhetoric import find_self_explanations

    assert not find_self_explanations("By then, rust was what stopped the engine.")
    assert not find_self_explanations("The ledger was unreadable to everyone who opened it.")
    assert not find_self_explanations("I hope the schedule makes the sequence clearer.")


def test_nominalization_pickup():
    """A verb from the first half returning as an abstract subject in the second."""
    from mimesis_voice.rhetoric import find_nominalization_pickups

    assert find_nominalization_pickups(
        "evidence that an afternoon occurred and was witnessed and that the "
        "witnessing was mine.")
    assert find_nominalization_pickups(
        "Mostly it sits among eleven thousand images and costs me nothing, and the "
        "costing nothing is what makes the decision possible to defer forever.")
    assert not find_nominalization_pickups("She walked to the window and looked out at the rain.")


def test_first_person_meta_is_a_self_rating():
    """"I want to be clear about that" -- flagged as a SUPER BAD tell 2026-08-28.
    _SELF_RATING_RE needs an evaluative head; these need none, the announcement is
    itself the tell."""
    from mimesis_voice.rhetoric import find_self_ratings

    assert find_self_ratings("I want to be clear about that, because there is a version of this.")
    assert find_self_ratings(
        "Additionally, and this is the more consequential point, the contested "
        "items are the entirety of what an evaluation adjudicates.")
    assert not find_self_ratings("I walked to the shop and bought bread.")


def test_it_is_openers_are_rate_gated_not_presence_gated():
    """Profile-dependent by measurement: creative p95 5.45, personal 4.55,
    research 0.00. A global threshold would miss it in research or fire constantly
    on fiction."""
    from mimesis_voice.rhetoric import analyze, RhetoricCalibration

    text = "It is late. The door was open. It was raining. He waited. It is over now."
    assert analyze(text, RhetoricCalibration(it_is_p95=0.0)).it_is_over
    assert not analyze(text, RhetoricCalibration(it_is_p95=90.0)).it_is_over


# --- borrowed simile and declarative rating -----------------------------------

def test_way_simile_fires_on_the_construction_he_flagged_four_times():
    """Measured at up to 3.14/1000w in generated text against 0.02 in his corpus,
    then flagged unprompted in three separate blind packets."""
    from mimesis_voice.rhetoric import find_way_similes

    assert find_way_similes("The mechanism fails quietly, the way a machine refuses.")
    assert find_way_similes("the way a stranger turns a wall that is not his")
    assert find_way_similes("it went off within about three weeks the way anger does")
    assert find_way_similes("the way a tongue returns to a chipped tooth")


def test_way_simile_leaves_the_ordinary_idioms_alone():
    """"the way back", "that is the way it is" and "the way he always walked" are
    ordinary English; matching them was the entire false-positive rate."""
    from mimesis_voice.rhetoric import find_way_similes

    assert not find_way_similes("He walked the way he always walked, north along the river.")
    assert not find_way_similes("I could not find the way back to the car.")
    assert not find_way_similes("That is the way it is.")
    assert not find_way_similes("She showed me the way a second time.")


def test_declarative_rating():
    """_SELF_RATING_RE needs an evaluative head from a fixed list. These rate by
    pointing instead, and measure 0.000 per 1000 words in all three of his corpora."""
    from mimesis_voice.rhetoric import find_declarative_ratings

    assert find_declarative_ratings("That is the part that sat with me.")
    assert find_declarative_ratings("That is a worse position than not knowing.")
    assert find_declarative_ratings("That is what tells me the drift is working.")
    assert not find_declarative_ratings("That is my brother's coat.")
    assert not find_declarative_ratings("It was raining when we left.")


def test_both_are_hard_flags():
    from mimesis_voice.rhetoric import analyze

    assert "way-simile" in analyze("It gave, the way a patient thing gives.").hard_flags
    assert "declarative-rating" in analyze("That is the part that sat with me.").hard_flags
