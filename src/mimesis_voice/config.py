"""Profile registry, resolution, and mutable state.

A *voice* is a self-contained profile under ``profiles/<slug>/``::

    profiles/
      state.json                 {"version":1,"active":"<slug>","enabled":true}
      <slug>/
        config.json              author_name, embed_backend, gate, whitelist, ...
        VOICE.md                 human-authored voice guide skeleton
        source_documents/        this voice's corpus (docx/txt/md)
        pairs/pairs.jsonl        optional AI->author transform pairs
        data/
          store.sqlite           chunks + FTS5 + vectors
          fingerprint.json       calibrated 13-feature fingerprint
          scrub_calibration.json calibrated banlist/whitelist/burstiness floor

``state.json`` is the single source of truth for which voice is active and
whether voice mode is on. The server resolves the active profile on every call,
so switching voices and toggling on/off take effect with no restart. This is
ported from the v1 Claude_Style_Clone_MCP config, keeping its atomic-write and
resolution patterns and adapting the config schema to the v2 profile model.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "profiles"
STATE_PATH = PROFILES_DIR / "state.json"

# Formats ingest understands.
SUPPORTED_EXTS = (".docx", ".txt", ".md")

# Default gate knobs; a profile's config.json "gate" block overrides per-key.
#
# DEFAULT_GATE is the *legacy* default and stays numeric so profiles written
# before auto-thresholds keep the exact ceiling their reported numbers were
# measured against. NEW_PROFILE_GATE is what `create_profile` writes.
#
# The split exists because 1.1 is not a fact about writing, it is a fact about
# one corpus. On an external long-form author (self-baseline 0.966, p95 fit
# threshold 1.701) a 1.1 ceiling sits BELOW what the author's own prose reliably
# scores: every candidate failed the gate on all six briefs, both rewrite
# iterations burned, and the loop fell back to "closest by RMS-z" every time at
# roughly five times the generation cost. A threshold typed by hand is a bug
# waiting for the second author.
DEFAULT_GATE = {"rmsz_max": 1.1, "slate_size": 4, "max_rewrites": 2}
NEW_PROFILE_GATE = {"rmsz_max": "auto", "slate_size": 4, "max_rewrites": 2}

DEFAULTS = {
    "author_name": "the author",
    "embed_backend": "fast",
    "gate": DEFAULT_GATE,
    "formats": {},
    "anchors": {"exemplars": True, "transform_pairs": None},
    "whitelist": [],
}


# --- low-level helpers --------------------------------------------------------


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:40] or "profile"


def embed_cache_dir() -> Path:
    """Persistent cache dir for the fastembed ONNX model.

    fastembed defaults to ``%TEMP%/fastembed_cache``, which Windows Disk Cleanup
    wipes, breaking retrieval until the model re-downloads. Pin it under
    ``%LOCALAPPDATA%`` (or ``~/.cache`` elsewhere); ``FASTEMBED_CACHE_PATH`` still
    wins so the location stays movable.
    """
    env = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    if env:
        path = Path(env)
    else:
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / ".cache"
        path = base / "fastembed_cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(path)
    return path


# --- profiles -----------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """Resolved, read-only view of a voice profile and its on-disk paths."""

    slug: str
    name: str
    embed_backend: str
    gate: dict
    formats: dict
    anchors: dict
    whitelist: list[str]
    root: Path
    data_dir: Path
    source_dir: Path
    db_path: Path
    fingerprint_path: Path
    scrub_path: Path
    presence_path: Path
    pairs_path: Path | None = field(default=None)
    # "v1" = the 13 order-invariant surface features. "v2" adds the 13 cadence
    # features. Defaults to v1 so profiles calibrated before cadence existed keep
    # scoring on the ruler their thresholds were measured against.
    feature_set: str = "v1"
    # Composite voices: draw calibration and anchors from several corpora with
    # declared shares of influence. See composite.py. Empty = ordinary voice.
    compose_from: list = field(default_factory=list)
    allow_mixed_authors: bool = False

    def format_config(self, fmt: str | None) -> dict:
        """Merge a named format's overrides over the profile defaults."""
        merged = {"gate": dict(self.gate)}
        if fmt and fmt in self.formats:
            override = self.formats[fmt]
            if isinstance(override, dict):
                if isinstance(override.get("gate"), dict):
                    merged["gate"].update(override["gate"])
                for k, v in override.items():
                    if k != "gate":
                        merged[k] = v
        return merged


def profile_dir(slug: str) -> Path:
    return PROFILES_DIR / slug


def _profile_complete(slug: str) -> bool:
    return bool(slug) and (profile_dir(slug) / "config.json").exists()


def load_profile_config(slug: str) -> dict:
    """Config for a profile, merged over DEFAULTS (missing keys filled in)."""
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy of defaults
    raw = _read_json(profile_dir(slug) / "config.json")
    for key, value in raw.items():
        if key == "gate" and isinstance(value, dict):
            cfg["gate"] = {**DEFAULT_GATE, **value}
        elif key == "anchors" and isinstance(value, dict):
            cfg["anchors"] = {**DEFAULTS["anchors"], **value}
        elif value is not None:
            cfg[key] = value
    if not str(cfg.get("author_name", "")).strip():
        cfg["author_name"] = DEFAULTS["author_name"]
    if cfg.get("embed_backend") not in ("fast", "style"):
        cfg["embed_backend"] = "fast"
    return cfg


def _build_profile(slug: str) -> Profile:
    cfg = load_profile_config(slug)
    d = profile_dir(slug)
    data = d / "data"
    pairs_rel = (cfg.get("anchors") or {}).get("transform_pairs")
    pairs_path = (d / pairs_rel) if pairs_rel else None
    return Profile(
        slug=slug,
        name=cfg["author_name"],
        embed_backend=cfg["embed_backend"],
        gate=cfg["gate"],
        formats=cfg.get("formats") or {},
        anchors=cfg.get("anchors") or {},
        whitelist=list(cfg.get("whitelist") or []),
        root=d,
        data_dir=data,
        source_dir=d / "source_documents",
        db_path=data / "store.sqlite",
        fingerprint_path=data / "fingerprint.json",
        scrub_path=data / "scrub_calibration.json",
        presence_path=data / "presence_calibration.json",
        pairs_path=pairs_path,
        feature_set=cfg.get("feature_set", "v1"),
        compose_from=list(cfg.get("compose_from") or []),
        allow_mixed_authors=bool(cfg.get("allow_mixed_authors", False)),
    )


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        p.name
        for p in PROFILES_DIR.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def create_profile(slug: str, name: str, embed_backend: str = "fast") -> Profile:
    """Create an empty profile scaffold and return it."""
    slug = slugify(slug)
    d = profile_dir(slug)
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "source_documents").mkdir(parents=True, exist_ok=True)
    payload = {
        "author_name": (name or "").strip() or DEFAULTS["author_name"],
        "embed_backend": embed_backend if embed_backend in ("fast", "style") else "fast",
        "gate": dict(NEW_PROFILE_GATE),
        "formats": {},
        "anchors": {"exemplars": True, "transform_pairs": None},
        "whitelist": [],
        # v1, deliberately. The order-aware feature set (v2) is a better
        # *measurement* -- it is the only thing that can see rhythm at all -- but
        # as a generation GATE it is not supported by the evidence:
        #
        #   * on a correctly segmented ruler the A/B effect fell from -0.613
        #     (7/7, p=0.016) to -0.174 (5/7, p=0.45);
        #   * blind discrimination showed no difference on one voice (25% vs 25%
        #     fool-rate, n=8) and favoured v1 on another (87.5% vs 37.5%,
        #     Fisher p=0.12).
        #
        # Gating on 26 features instead of 13 plausibly over-constrains selection:
        # candidates that hit more statistical targets are not candidates that
        # read better. Use v2 to diagnose (evals/cadence_audit.py); switch a
        # profile to it only after an A/B on that voice earns it.
        "feature_set": "v1",
    }
    if not (d / "config.json").exists():
        _atomic_write(d / "config.json", json.dumps(payload, indent=2) + "\n")
    if not (d / "VOICE.md").exists():
        _atomic_write(d / "VOICE.md", _voice_md_skeleton(payload["author_name"]))
    return _build_profile(slug)


def _voice_md_skeleton(name: str) -> str:
    return f"""# {name} — Voice Guide

Human-authored notes that ride alongside the measured fingerprint. Keep it short;
the numbers do the enforcing, this file supplies intent the statistics cannot.

## Register
Describe the default register (plain, elevated, technical, wry).

## Cadence
Note sentence-length habits and where the rhythm breaks.

## Diction
Words this author reaches for, and words they never use.

## Do not
Anti-patterns to avoid even when they would pass the fingerprint.
"""


# --- state (active profile + enabled) -----------------------------------------


def read_state() -> dict:
    st = _read_json(STATE_PATH)
    active = st.get("active") if isinstance(st.get("active"), str) else None
    enabled = st.get("enabled") if isinstance(st.get("enabled"), bool) else True
    return {"version": 1, "active": active, "enabled": enabled}


def write_state(active: str | None = None, enabled: bool | None = None) -> None:
    cur = read_state()
    if active is not None:
        cur["active"] = active
    if enabled is not None:
        cur["enabled"] = enabled
    _atomic_write(
        STATE_PATH,
        json.dumps(
            {"version": 1, "active": cur["active"], "enabled": cur["enabled"]},
            indent=2,
        )
        + "\n",
    )


def is_enabled() -> bool:
    return read_state()["enabled"]


def set_enabled(value: bool) -> None:
    write_state(enabled=bool(value))


# --- resolution ---------------------------------------------------------------


def ensure_profiles() -> None:
    """Point ``state.active`` at a real profile if it is unset or stale. Idempotent."""
    st = read_state() if STATE_PATH.exists() else None
    if st and _profile_complete(st.get("active") or ""):
        return
    profs = list_profiles()
    active = "example" if "example" in profs else (profs[0] if profs else None)
    if active and (not st or st.get("active") != active):
        write_state(active=active)


def resolve(slug: str | None = None) -> Profile:
    """Resolve a named profile (for ingest/calibrate) or the active one."""
    if slug:
        return _build_profile(slugify(slug))
    return resolve_active()


def resolve_named(name: str) -> Profile | None:
    """Resolve a specific profile by name/slug for a single call, or None."""
    slug = slugify(name)
    return _build_profile(slug) if _profile_complete(slug) else None


def resolve_active() -> Profile:
    """The voice the server should use right now. Never raises."""
    ensure_profiles()
    slug = read_state().get("active")
    if _profile_complete(slug or ""):
        return _build_profile(slug)
    profs = list_profiles()
    if profs:
        return _build_profile(profs[0])
    # Nothing configured: synthesize a throwaway so callers get a clean error.
    return _build_profile("example")


if __name__ == "__main__":
    p = resolve_active()
    print(
        json.dumps(
            {
                "active": p.slug,
                "name": p.name,
                "embed_backend": p.embed_backend,
                "enabled": is_enabled(),
                "profiles": list_profiles(),
            },
            indent=2,
        )
    )
