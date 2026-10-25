"""Minimal-diff editing of text the author wrote.

Every other writing path in this package generates: compose_slate writes new
candidates, gate.compose rewrites a source document into new sentences. That is
the wrong tool for a draft the author already wrote, for two measured reasons.

1. Generation has nothing to add. A voicing pass over an abstract the
   reference author wrote returned 95-99.5% word-identical output.
2. It turns human text into model text. A detection study on research prose
   (Oct 2026, Pangram 4.1, docs/DETECTION.md) found that the detector labels a
   sentence by whether its content came from a model:
   - model polish of human text, by achieved token similarity: >= 0.90 12/12
     Human, 0.85-0.90 14/16, 0.80-0.85 7/9, < 0.75 1/7; at section length
     (about 1,000 words) 12/12 at 0.90;
   - attribution-only fixes at token similarity 0.96-0.996 kept 7/8 Human;
   - on the author's own paper sections, word fixes inside the author's
     sentences scanned 100% human, while restructured sentences plus one new
     sentence scanned 100% AI;
   - one caught sentence can flip about 280 unchanged words around it, because
     the detector labels 512-token windows.

So this module edits inside the author's sentences and nowhere else, and the
limits are enforced here in code rather than requested in a prompt:

- one op replaces one sentence with one sentence: no merges, splits, inserts or
  line changes;
- per sentence, at least ``keep_ratio`` of the old words survive in order and at
  most ``max_added`` words are added (0.6 and 6);
- only sentences the ``scope`` allows may be edited (hard-flagged by default);
- numbers, citations and citation keys cannot change, and dashes cannot be added;
- an op that introduces a hard flag, a banned word or a detection-risk move is
  refused;
- whole-text token similarity stays at or above ``floor`` (default 0.93, never
  below 0.90), and every detector-sized window stays at or above
  ``window_floor`` (0.85, the level at which pooled polish passed 38/40).
  Ops are dropped until both hold.

The precondition matters as much as the limits: polish did not rescue
model-planned text in the study, so this is for text the author wrote.
Similarity is difflib over whitespace tokens with autojunk off, the measure
behind every number above.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

from . import contentrisk
from . import rhetoric_ext as rhetoric_ext_mod
from . import scrub as scrub_mod
from . import segment

DEFAULT_FLOOR = 0.93
MIN_FLOOR = 0.90
WINDOW_FLOOR = 0.85
KEEP_RATIO = 0.6
MAX_ADDED = 6
MAX_ADDED_CAP = 8
WINDOW_WORDS = contentrisk.WINDOW_WORDS

SCOPES = ("hard", "flagged", "goal")

# Any citation command: \cite, \citet, \citep, \parencite, \textcite, \autocite, \footcite ...
_TEX_CITE_KEYS_RE = re.compile(r"\\[A-Za-z]*cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
# A raw sentence boundary, abbreviations included. The segmenter hides boundaries
# after "et al." or "Fig."; an op must not use that blind spot to add a sentence.
_RAW_BOUNDARY_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s+[\"'\u201c\u2018(\[\\]?[A-Z0-9\\]")
_NUM_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def _kept_added(old: str, new: str) -> tuple[int, int, int]:
    o, n = old.split(), new.split()
    sm = difflib.SequenceMatcher(None, o, n, autojunk=False)
    kept = sum(b.size for b in sm.get_matching_blocks())
    return kept, len(n) - kept, len(o)


def _facts(s: str) -> tuple[list[str], set[str], list[str]]:
    """Numbers and citation keys as sorted multisets (dropping one of two equal
    numbers is a change), citations and acronyms as a set."""
    nums = sorted(m.replace(",", "").rstrip("0").rstrip(".") if "." in m else m.replace(",", "")
                  for m in _NUM_TOKEN_RE.findall(s))
    keys = sorted(k.strip() for m in _TEX_CITE_KEYS_RE.findall(s) for k in m.split(","))
    return nums, scrub_mod._citations(s), keys


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# --- locating flags on sentences ------------------------------------------------

# First-family RhetoricReport fields that hold spans, by flag name and tier.
_FIRST_HARD = {"leakage": "task-leakage", "reversals": "unheeded-reversal",
               "self_explanations": "self-explanation", "way_similes": "way-simile",
               "declarative_ratings": "declarative-rating"}
_FIRST_ADVISORY = {"triads": "triad", "meta_asides": "meta-aside", "self_ratings": "self-rating",
                   "absence_punches": "absence-punch", "closing_flourish": "closing-flourish",
                   "nominalization_pickups": "nominalization-pickup", "clefts": "cleft",
                   "antithesis": "antithesis"}
# Second-family fields by the flag name they raise (rhetoric_ext._fired).
_EXT_FIELDS = {"announcers": "announcer", "pointer_verdicts": "pointer-verdict",
               "participial_triads": "participial-triad", "colon_twists": "colon-twist",
               "parallel_address": "parallel-address", "conversions": "conversion",
               "personifications": "personification", "negation_templates": "negation-template",
               "question_fragments": "question-fragment", "linkedin_phrases": "linkedin-phrase",
               "tail_twists": "tail-twist", "does_pairs": "does-pair",
               "generator_phrases": "generator-phrase", "intensifiers": "intensifier"}


@dataclass
class SentenceInfo:
    n: int
    text: str
    hard: list[str] = field(default_factory=list)
    advisory: list[str] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        return self.hard + self.advisory


def _spans(x) -> list[str]:
    out = []
    for s in x or []:
        if isinstance(s, (tuple, list)):
            out.extend(str(t) for t in s)
        else:
            out.append(str(s))
    return [s for s in out if s.strip()]


def _locate(span: str, sents: list[str]) -> list[int]:
    sp = _norm(span)
    if not sp:
        return []
    hits = [i for i, s in enumerate(sents) if s and (sp in s or s in sp)]
    if not hits and len(sp) >= 24:
        hits = [i for i, s in enumerate(sents) if sp[:24] in s or sp[-24:] in s]
    return hits


def flag_sentences(text: str, cal: "scrub_mod.ScrubCalibration",
                   ext: "rhetoric_ext_mod.RhetoricExtCalibration | None" = None,
                   allow_dashes: bool = False) -> list[SentenceInfo]:
    """Per-sentence view of the scrubber: which detector fired on which sentence.

    The detectors run once over the whole text, because some of them need the
    neighbouring sentence (a reversal is two sentences), and their spans are then
    placed on the sentences that contain them.
    """
    sents = segment.sentences(text)
    infos = [SentenceInfo(n=i, text=s) for i, s in enumerate(sents, 1)]
    if not sents:
        return infos
    norm = [_norm(s) for s in sents]
    rep = scrub_mod.analyze(text, cal, ext=ext)

    def put(spans, name: str, hard: bool) -> None:
        for sp in _spans(spans):
            for i in _locate(sp, norm):
                bucket = infos[i].hard if hard else infos[i].advisory
                if name not in bucket:
                    bucket.append(name)

    r = rep.rhetoric
    if r is not None:
        for f, name in _FIRST_HARD.items():
            put(getattr(r, f, []), name, True)
        for f, name in _FIRST_ADVISORY.items():
            put(getattr(r, f, []), name, False)
        if r.it_is_over:
            # The span is just "It is", which would land on every sentence containing it.
            for i, s in enumerate(norm):
                if re.match(r"it (?:is|was)\b", s):
                    infos[i].advisory.append("it-is-opener")
        if r.chain_over:
            put(r.comma_chains, "comma-chain", False)
    x = rep.rhetoric_ext
    if x is not None:
        hard_ext = set(x.hard_flags)
        for f, name in _EXT_FIELDS.items():
            put(getattr(x, f, []), name, name in hard_ext)
        for name, (spans, _rate, _p95) in x.demoted.items():
            put(spans, name, name in hard_ext)
    for i, s in enumerate(sents):
        low = s.lower()
        if not allow_dashes and scrub_mod.count_dashes(s):
            infos[i].hard.append("em-dash")
        for p in cal.banned_phrases:
            if p.lower() in low:
                infos[i].hard.append(f"banned-phrase '{p}'")
        for w in cal.banned_words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                infos[i].advisory.append(f"banned-word '{w}'")
        infos[i].risk = contentrisk.classify(s, i + 1).risk
    return infos


# --- applying edits ---------------------------------------------------------------

@dataclass
class OpResult:
    n: int
    old: str
    new: str
    fixes: str
    status: str = "applied"       # applied | refused
    reason: str = ""
    sim: float = 1.0
    added: int = 0
    flagged: bool = True


@dataclass
class EditResult:
    original: str
    edited: str
    ops: list[OpResult]
    infos: list[SentenceInfo]
    section_sim: float
    min_window_sim: float
    floor: float
    window_floor: float
    scope: str

    @property
    def applied(self) -> list[OpResult]:
        return [o for o in self.ops if o.status == "applied"]

    @property
    def refused(self) -> list[OpResult]:
        return [o for o in self.ops if o.status == "refused"]


_END_RE = re.compile(r"[.!?][\"'”’)\]]*$")


def _introduced(old: SentenceInfo, new: SentenceInfo) -> list[str]:
    out = sorted(set(new.hard) - set(old.hard))
    out += sorted(a for a in set(new.advisory) - set(old.advisory) if a.startswith("banned-word"))
    out += sorted(f"detection-risk move '{r}'" for r in set(new.risk) - set(old.risk))
    return out


def _windows(lengths: list[int]) -> list[tuple[int, int]]:
    """Sentence-index windows of ~WINDOW_WORDS words at half-window stride."""
    n = len(lengths)
    if not n:
        return []
    starts, acc = [0], 0
    for i, L in enumerate(lengths):
        acc += L
        if acc >= WINDOW_WORDS // 2 and i + 1 < n:
            starts.append(i + 1)
            acc = 0
    out = []
    for st in starts:
        j, w = st, 0
        while j < n and w < WINDOW_WORDS:
            w += lengths[j]
            j += 1
        out.append((st, j))
        if j >= n:
            break
    return out


def apply_edits(
    text: str,
    edits: list[dict],
    cal: "scrub_mod.ScrubCalibration",
    ext: "rhetoric_ext_mod.RhetoricExtCalibration | None" = None,
    scope: str = "hard",
    floor: float = DEFAULT_FLOOR,
    window_floor: float = WINDOW_FLOOR,
    keep_ratio: float = KEEP_RATIO,
    max_added: int = MAX_ADDED,
    allow_dashes: bool = False,
    allow_fact_changes: bool = False,
) -> EditResult:
    """Apply sentence-internal ops ``[{"n": 3, "text": "...", "fixes": "..."}]``.

    Limits are clamped, never loosened past the evidence: ``floor`` to
    [0.90, 0.995], ``keep_ratio`` to [0.6, 1.0], ``max_added`` to [0, 8].
    """
    scope = scope if scope in SCOPES else "hard"
    # NaN would pass every comparison below and switch the floor off.
    floor = float(floor) if math.isfinite(float(floor)) else DEFAULT_FLOOR
    window_floor = float(window_floor) if math.isfinite(float(window_floor)) else WINDOW_FLOOR
    floor = min(max(floor, MIN_FLOOR), 0.995)
    window_floor = min(max(window_floor, WINDOW_FLOOR), floor)
    keep_ratio = min(max(float(keep_ratio), KEEP_RATIO), 1.0)
    max_added = min(max(int(max_added), 0), MAX_ADDED_CAP)

    segs = segment.split(text)
    prose_idx = [i for i, s in enumerate(segs) if s.is_prose]
    infos = flag_sentences(text, cal, ext=ext, allow_dashes=allow_dashes)
    ops: list[OpResult] = []
    seen: set[int] = set()

    for e in edits or []:
        if not isinstance(e, dict) or isinstance(e.get("n"), bool):
            continue
        try:
            n = int(e.get("n"))
        except (TypeError, ValueError):
            continue
        new = str(e.get("text") or "").strip()
        fixes = str(e.get("fixes") or e.get("why") or "").strip()
        if not (1 <= n <= len(prose_idx)):
            ops.append(OpResult(n, "", new, fixes, "refused", "no such sentence"))
            continue
        old_full = segs[prose_idx[n - 1]].text
        lead = old_full[: len(old_full) - len(old_full.lstrip())]
        old = old_full.strip()
        o = OpResult(n, old, new, fixes, flagged=bool(infos[n - 1].flags))
        ops.append(o)
        if not new or new == old:
            o.status, o.reason = "refused", "no change"
            continue
        if n in seen:
            o.status, o.reason = "refused", "second op on the same sentence"
            continue
        seen.add(n)
        kept, added, n_old = _kept_added(old, new)
        o.sim, o.added = similarity(old, new), added
        new_segs = segment.split(new)
        if (len([s for s in new_segs if s.is_prose]) != 1
                or any(not s.is_prose and s.text.strip() for s in new_segs)
                # Line breaks stay where they were: a newline is how a heading, a
                # \section line or a % comment would get in, or a heading get deleted.
                or new.count("\n") != old.count("\n")
                or len(_RAW_BOUNDARY_RE.findall(new)) > len(_RAW_BOUNDARY_RE.findall(old))
                or bool(_END_RE.search(old)) != bool(_END_RE.search(new))
                or old[:1].isupper() != new[:1].isupper()):
            o.status, o.reason = "refused", "not one sentence: edits stay inside the sentence (no merge, split, insert or line change)"
        elif scope == "hard" and not infos[n - 1].hard:
            o.status, o.reason = "refused", "sentence has no hard flag (scope=hard)"
        elif scope == "flagged" and not infos[n - 1].flags:
            o.status, o.reason = "refused", "sentence has no flag (scope=flagged)"
        elif scope == "goal" and not o.flagged and not fixes:
            o.status, o.reason = "refused", "unflagged sentence and no stated reason (scope=goal)"
        elif kept < keep_ratio * n_old:
            o.status, o.reason = "refused", f"keeps {kept}/{n_old} words in order, needs {keep_ratio:.0%}"
        elif added > max_added:
            o.status, o.reason = "refused", f"adds {added} words, limit {max_added}"
        elif not allow_dashes and scrub_mod.count_dashes(new) > scrub_mod.count_dashes(old):
            o.status, o.reason = "refused", "adds a dash"
        elif not allow_fact_changes and _facts(old) != _facts(new):
            # Equal, not a subset: adding a citation the author never wrote is a change too.
            o.status, o.reason = "refused", "changes, adds or drops a number or citation"
        else:
            o.new = lead + new if lead else new

    def build(only: "list[OpResult] | None" = None) -> str:
        out = [segment.Segment(s.text, s.tail, s.kind) for s in segs]
        for o in (only if only is not None else ops):
            if o.status == "applied":
                out[prose_idx[o.n - 1]].text = o.new
        return segment.join(out)

    base_hard = set(scrub_mod.analyze(text, cal, ext=ext).hard_flags)

    def global_new_hard(t: str) -> set[str]:
        # Backstop for whole-text flags that are never placed on a sentence
        # (emoji, hashtag clusters, rate-gated openers).
        return set(scrub_mod.analyze(t, cal, ext=ext).hard_flags) - base_hard

    def culprits(live: list[OpResult], bad) -> list[OpResult]:
        """The ops that cause a failure on their own; all of them if none does alone."""
        alone = [o for o in live if bad(build([o]))]
        return alone or live

    orig_sents = [infos[i].text for i in range(len(infos))]
    lengths = [len(s.split()) for s in orig_sents]
    wins = _windows(lengths)

    def window_sims(edited_sents: list[str]) -> list[tuple[float, tuple[int, int]]]:
        return [(similarity(" ".join(orig_sents[a:b]), " ".join(edited_sents[a:b])), (a, b)) for a, b in wins]

    def drop_order(cands: list[OpResult]) -> OpResult:
        # Unflagged (goal-only) ops go first, then the heaviest change.
        return min(cands, key=lambda o: (o.flagged, o.sim))

    for _ in range(len(ops) + 1):
        live = [o for o in ops if o.status == "applied"]
        if not live:
            break
        edited = build()
        # An op may not introduce a hard flag, a banned word or a detection-risk
        # move its sentence did not already carry.
        new_infos = flag_sentences(edited, cal, ext=ext, allow_dashes=allow_dashes)
        if len(new_infos) != len(infos):
            n_sent = len(infos)
            for o in culprits(live, lambda t: len(segment.sentences(t)) != n_sent):
                o.status, o.reason = "refused", "changes where sentences begin or end"
            continue
        extra = global_new_hard(edited)
        if extra:
            for o in culprits(live, lambda t: bool(global_new_hard(t))):
                o.status, o.reason = "refused", "introduces " + ", ".join(sorted(extra))
            continue
        bad = {o.n: _introduced(infos[o.n - 1], new_infos[o.n - 1]) for o in live}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            for o in live:
                if o.n in bad:
                    o.status, o.reason = "refused", "introduces " + ", ".join(bad[o.n])
            continue
        if similarity(text, edited) < floor:
            o = drop_order(live)
            o.status, o.reason = "refused", f"section similarity would fall below {floor:.2f}"
            continue
        ed_sents = list(orig_sents)
        for o in live:
            ed_sents[o.n - 1] = o.new
        ws = window_sims(ed_sents)
        worst = min(ws, key=lambda t: t[0]) if ws else (1.0, (0, 0))
        if worst[0] < window_floor:
            a, b = worst[1]
            inside = [o for o in live if a < o.n <= b]
            o = drop_order(inside or live)
            o.status, o.reason = "refused", f"window {a + 1}-{b} similarity would fall below {window_floor:.2f}"
            continue
        break

    edited = build()
    ed_sents = list(orig_sents)
    for o in ops:
        if o.status == "applied":
            ed_sents[o.n - 1] = o.new
    ws = window_sims(ed_sents)
    return EditResult(
        original=text, edited=edited, ops=ops, infos=infos,
        section_sim=similarity(text, edited),
        min_window_sim=min((w for w, _ in ws), default=1.0),
        floor=floor, window_floor=window_floor, scope=scope,
    )


# --- rendering ----------------------------------------------------------------------

def word_diff(old: str, new: str) -> str:
    o, n = old.split(), new.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(" ".join(o[i1:i2]))
        if tag in ("replace", "delete"):
            out.append("[-" + " ".join(o[i1:i2]) + "-]")
        if tag in ("replace", "insert"):
            out.append("{+" + " ".join(n[j1:j2]) + "+}")
    return " ".join(out)


def _flag_line(info: SentenceInfo) -> str:
    bits = []
    if info.hard:
        bits.append("HARD: " + ", ".join(info.hard))
    if info.advisory:
        bits.append("advisory: " + ", ".join(info.advisory))
    if info.risk:
        bits.append("detection risk: " + ", ".join(info.risk))
    return f" ({'; '.join(bits)})" if bits else ""


def render_kit(text: str, infos: list[SentenceInfo], author: str, slug: str, goal: str | None,
               scope: str, floor: float) -> str:
    n_flag = sum(1 for i in infos if i.flags)
    n_hard = sum(1 for i in infos if i.hard)
    lines = [
        f"# MINIMAL-DIFF EDIT: text {author} wrote ({slug})",
        "",
        f"Use this only on text {author} wrote. Light edits keep human text human (Opus polish at >= 0.90 "
        "token similarity passed Pangram 12/12; edits inside the author's own sentences passed where "
        "rebuilt sentences did not). They do not rescue text a model drafted.",
        "",
        f"GOAL: {goal or '(none given: fix the flagged problems only)'}",
        f"SCOPE: {scope}. {n_hard} sentence(s) carry a hard flag, {n_flag} carry any flag, of {len(infos)}.",
        f"LIMITS, enforced in code by apply_own_edits: one sentence in, one sentence out; at least "
        f"{KEEP_RATIO:.0%} of its words kept in order; at most {MAX_ADDED} words added; numbers, "
        f"citations and dashes unchanged; no new hard flag; whole-text similarity >= {floor:.2f}; every "
        f"~{WINDOW_WORDS}-word window >= {WINDOW_FLOOR:.2f}. Ops that break a limit are refused, not trimmed.",
        "",
        "## SENTENCES",
    ]
    for i in infos:
        lines.append(f"[{i.n}]{_flag_line(i)} {i.text}")
    lines += ["", contentrisk.render(contentrisk.analyze(text)), "", "## HOW TO PRODUCE THE EDITS",
              "1. Edit only sentences the scope allows. With scope=hard (the default) a sentence with no hard flag is not "
              "touched even if you would phrase it differently.",
              "2. Inside a sentence, change the fewest words that remove the flag. Keep the author's own "
              "quirks, word choices and sentence shape. Do not smooth, merge, split, reorder or add a "
              "sentence, and keep its line breaks where they are. Never add a contrast, a summary, a "
              "verdict, a gloss, a stance phrase or a citation.",
              "3. Sentences tagged 'detection risk' carry a move Pangram labels AI whatever the wording. Fix "
              "their mechanical flags if any and otherwise leave them alone: list them for the author to "
              "rewrite in their own words.",
              "4. Return a JSON list, one object per edited sentence: "
              "[{\"n\": 3, \"text\": \"<the whole corrected sentence>\", \"fixes\": \"<flag name or goal reason>\"}].",
              f"5. Call apply_own_edits(text=<the exact text you were given>, edits=<that list>, voice='{slug}', "
              f"scope='{scope}'). Present its diff, the provenance line and the refused ops. Do not re-apply a "
              "refused op by hand."]
    return "\n".join(lines)


def render_result(res: EditResult, author: str) -> str:
    n = len(res.infos)
    touched = len(res.applied)
    lines = [
        f"# MINIMAL-DIFF EDIT RESULT ({author})",
        f"applied {touched} op(s), refused {len(res.refused)}. Sentences: {n - touched}/{n} untouched, "
        f"{touched} edited inside the sentence, 0 added, 0 merged or split.",
        f"whole-text token similarity {res.section_sim:.3f} (floor {res.floor:.2f}); lowest window "
        f"{res.min_window_sim:.3f} (floor {res.window_floor:.2f}); scope {res.scope}.",
        f"PROVENANCE: every sentence is {author}'s; {touched} {'carries' if touched == 1 else 'carry'} an "
        "edit at the per-sentence similarity shown below. No sentence was added by the model.",
    ]
    if res.applied:
        lines += ["", "## DIFF ([-removed-] {+added+})"]
        for o in sorted(res.applied, key=lambda x: x.n):
            lines.append(f"[{o.n}] sim {o.sim:.2f}, +{o.added} words, fixes: {o.fixes or '-'}")
            lines.append(f"    {word_diff(o.old, o.new.strip())}")
    if res.refused:
        lines += ["", "## REFUSED"]
        for o in sorted(res.refused, key=lambda x: x.n):
            lines.append(f"[{o.n}] {o.reason}")
    rest = [i for i in res.infos if i.hard and i.n not in {o.n for o in res.applied}]
    if rest:
        lines += ["", "## HARD FLAGS LEFT IN PLACE (for the author)"]
        for i in rest:
            lines.append(f"[{i.n}] {', '.join(i.hard)}")
    lines += ["", contentrisk.render(contentrisk.analyze(res.edited)), "", "## EDITED TEXT", res.edited]
    return "\n".join(lines)
