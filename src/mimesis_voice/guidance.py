"""Genre guidance: what a passage has to accomplish, separate from how it sounds.

Every other block in the compose kit answers "does this read like the author".
None of them can say what an abstract is for, why a Discussion fails, or in what
order a Related Work paragraph earns its citations. The 2026-09-04 ML4H session
made the gap measurable: a Discussion scored fit 0.201 with zero hard flags and
was rejected in one line ("a bunch of nothing. What are we discussing here?"),
and a Related Work at fit 0.105 was rejected as "an AI report and not a research
paper". The gate saw both as clean. Nothing in the kit was pointed at that axis.

This module supplies that axis as a third block in the kit, and it is built to
lose fights with the voice block on purpose. Two histories govern the design:

* Outline-conditioned and plan-and-write systems reliably buy coherence and pay
  for it in local range (the "over-determination" result), and this system
  already records slate collapse to 16-49% of the author's spread when content
  is fixed to the sentence. A structure block pushes collapse further.
* The old burstiness advisory was "the single most damaging instruction in the
  kit" because a generic writing rule stated forcefully enough overrode the
  corpus. Gopen and Swan's principles are exactly that shape of rule.

So the block is scoped to one rhetorical MOVE, capped in length, says what the
passage must accomplish and what disqualifies it, and never says how to phrase
anything. A card that names words to use is a bug.

Layout on disk:

    src/mimesis_voice/guidance/<genre>/_common.md     principles for every move
    src/mimesis_voice/guidance/<genre>/<move>.md      one card per move
    profiles/<voice>/guidance/_common.md              the author's own rulings
    profiles/<voice>/guidance/<move>.md               ... per move

The genre comes from the profile's ``guidance`` config key ("research"). The
move comes from an explicit ``format=`` on the compose call, else from
``retrieve.infer_move(task)``. The author layer is appended after the shipped
card because it is the more specific of the two, and because rulings mined
from blind rounds outrank a textbook.
"""
from __future__ import annotations

from pathlib import Path

from . import config, retrieve

GUIDANCE_DIR = Path(__file__).parent / "guidance"

# Hard caps. A card longer than this is competing with the voice block rather
# than informing it; trim the card, do not raise the cap.
_CARD_MAX_CHARS = 1400
_COMMON_MAX_CHARS = 1800
_AUTHOR_MAX_CHARS = 1400

HEADER = (
    "## WHAT THIS PASSAGE MUST DO\n"
    "Structure only: what it accomplishes, in what order, and what disqualifies it. "
    "The voice rules above govern every sentence; where the two seem to conflict, the "
    "voice rules win on phrasing and this block wins on what is included and its order. "
    "Never add a sentence to satisfy a slot the task's content does not fill."
)


def _read(path: Path, cap: int) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if not text:
        return ""
    if len(text) > cap:
        # Cut at the last paragraph break under the cap so a card never ends
        # mid-sentence in the kit.
        cut = text.rfind("\n\n", 0, cap)
        text = text[: cut if cut > 0 else cap].rstrip()
    return text


def genre_for(profile: config.Profile) -> str | None:
    g = getattr(profile, "guidance", None)
    if not g:
        return None
    g = str(g).strip().lower()
    return g if (GUIDANCE_DIR / g).is_dir() else None


def available_moves(genre: str) -> list[str]:
    d = GUIDANCE_DIR / genre
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.md") if not p.stem.startswith("_"))


def resolve_move(task: str, fmt: str | None, genre: str) -> str | None:
    """An explicit format naming a move wins; otherwise infer from the brief."""
    moves = set(available_moves(genre))
    if fmt:
        f = fmt.strip().lower().replace("-", "_").replace(" ", "_")
        if f in moves:
            return f
    m = retrieve.infer_move(task)
    return m if m in moves else None


def block(task: str, profile: config.Profile, fmt: str | None = None) -> str:
    """The guidance block for this compose, or "" when the profile has no genre.

    A profile without a ``guidance`` key gets exactly the kit it had before this
    module existed. A genre with no card for the inferred move still gets the
    common principles, because those hold for every passage in the genre.
    """
    genre = genre_for(profile)
    if not genre:
        return ""
    move = resolve_move(task, fmt, genre)
    parts = [HEADER]

    common = _read(GUIDANCE_DIR / genre / "_common.md", _COMMON_MAX_CHARS)
    if common:
        parts.append(common)
    if move:
        card = _read(GUIDANCE_DIR / genre / f"{move}.md", _CARD_MAX_CHARS)
        if card:
            parts.append(f"### This passage is a {move.replace('_', ' ')}\n{card}")

    author_dir = profile.root / "guidance"
    author_common = _read(author_dir / "_common.md", _AUTHOR_MAX_CHARS)
    author_card = _read(author_dir / f"{move}.md", _AUTHOR_MAX_CHARS) if move else ""
    if author_common or author_card:
        parts.append(
            f"### {profile.name}'s own rulings on this kind of passage\n"
            "These come from drafts the author graded blind. They outrank the "
            "textbook guidance above."
        )
        if author_common:
            parts.append(author_common)
        if author_card:
            parts.append(author_card)
    return "\n\n".join(parts)
