"""Binoculars-style machine-text detector (optional extra ``[detect]``).

A clean-room reimplementation of the Binoculars score (Hans et al. 2024,
arXiv:2401.12070; reference code BSD-3-Clause, github.com/ahans30/Binoculars).
No upstream code is copied: the score is a short, well-specified computation and
the algorithm itself is not copyrightable. Two causal LMs over the *same*
tokenizer:

* **observer** M1 (a base model) and **performer** M2 (its instruct sibling).
* numerator ``ppl`` = the performer's perplexity of the text (standard causal
  shift: logits[:-1] predict tokens[1:], mean cross-entropy).
* denominator ``x_ppl`` = the cross-perplexity, i.e. the mean over positions
  (no shift) of ``H(softmax(observer_logits_i), logsoftmax(performer_logits_i))``
  = the observer's next-token distribution scored against the performer's.
* ``score = ppl / x_ppl``. **LOW score => machine-generated, HIGH => human.**

Because the two logit tensors must line up token-for-token and share a vocab
index space, the observer and performer *must* use the identical tokenizer. The
default pair is ``Qwen/Qwen2.5-0.5B`` + ``Qwen/Qwen2.5-0.5B-Instruct`` (tokenizer
verified identical, ~1.9 GB in bf16, fits a 7 GB GPU with headroom). The paper's
0.90/0.85 thresholds are calibrated for the Falcon-7B pair and do NOT transfer;
``calibrate_threshold`` recomputes an operating point from labelled human/AI text
for whatever pair is configured.

Everything imports lazily so the base install never needs torch/transformers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --- configuration ------------------------------------------------------------

_OBSERVER_ENV = "MIMESIS_DETECT_OBSERVER"
_PERFORMER_ENV = "MIMESIS_DETECT_PERFORMER"
_DEVICE_ENV = "MIMESIS_DETECT_DEVICE"
_MAXTOK_ENV = "MIMESIS_DETECT_MAXTOK"

_OBSERVER_DEFAULT = "Qwen/Qwen2.5-0.5B"
_PERFORMER_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
_MAXTOK_DEFAULT = 512
_MIN_TOKENS = 48  # Binoculars is unreliable on very short inputs

_state: dict = {}


def model_pair() -> tuple[str, str]:
    return (
        os.environ.get(_OBSERVER_ENV, _OBSERVER_DEFAULT),
        os.environ.get(_PERFORMER_ENV, _PERFORMER_DEFAULT),
    )


def _device() -> str:
    override = os.environ.get(_DEVICE_ENV, "").strip()
    if override:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _max_tokens() -> int:
    try:
        return int(os.environ.get(_MAXTOK_ENV, _MAXTOK_DEFAULT))
    except ValueError:
        return _MAXTOK_DEFAULT


def _load():
    """Lazily load the observer/performer pair and a shared tokenizer."""
    if _state:
        return _state
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "the detector needs the [detect] extra "
            "(pip install 'mimesis-voice[detect]'); import failed: " + str(exc)
        ) from exc

    observer_name, performer_name = model_pair()
    device = _device()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tok_o = AutoTokenizer.from_pretrained(observer_name)
    tok_p = AutoTokenizer.from_pretrained(performer_name)
    # The cross-perplexity is only meaningful if the two models index tokens the
    # same way. Enforce it, exactly as the reference does.
    if tok_o.get_vocab() != tok_p.get_vocab():
        raise RuntimeError(
            f"observer ({observer_name}) and performer ({performer_name}) do not "
            f"share a tokenizer; cross-perplexity would be meaningless"
        )
    tokenizer = tok_o
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _from_pretrained(name):
        # transformers 5.x renamed torch_dtype -> dtype; support both.
        try:
            return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    observer = _from_pretrained(observer_name).to(device).eval()
    performer = _from_pretrained(performer_name).to(device).eval()
    _state.update(
        observer=observer, performer=performer, tokenizer=tokenizer,
        device=device, observer_name=observer_name, performer_name=performer_name,
    )
    return _state


def _perplexity(performer_logits, input_ids, torch) -> float:
    """Mean token cross-entropy of the text under the performer (causal shift)."""
    shift_logits = performer_logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return float(ce.item())


def _cross_perplexity(observer_logits, performer_logits, torch) -> float:
    """Mean over positions (no shift) of H(observer_dist, performer_logdist).

    Cross-entropy between the observer's next-token *probability* distribution
    and the performer's *log*-distribution: -sum_v p_obs[v] * logsoftmax(perf)[v].
    """
    p_obs = torch.nn.functional.softmax(observer_logits.float(), dim=-1)
    logq_perf = torch.nn.functional.log_softmax(performer_logits.float(), dim=-1)
    h = -(p_obs * logq_perf).sum(dim=-1)  # [B, L]
    return float(h.mean().item())


@dataclass
class DetectResult:
    available: bool
    score: float | None = None
    label: str | None = None          # "machine" | "human" | "uncertain"
    threshold: float | None = None
    calibrated: bool = False
    ppl: float | None = None
    x_ppl: float | None = None
    n_tokens: int = 0
    observer: str | None = None
    performer: str | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "score": None if self.score is None else round(self.score, 5),
            "label": self.label,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "ppl": None if self.ppl is None else round(self.ppl, 5),
            "x_ppl": None if self.x_ppl is None else round(self.x_ppl, 5),
            "n_tokens": self.n_tokens,
            "observer": self.observer,
            "performer": self.performer,
            "detail": self.detail,
        }


def raw_score(text: str) -> DetectResult:
    """Compute the Binoculars ratio for one text. No thresholding."""
    import torch

    st = _load()
    tokenizer = st["tokenizer"]
    device = st["device"]
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=_max_tokens()
    )
    input_ids = enc["input_ids"].to(device)
    n_tokens = int(input_ids.shape[1])
    observer, performer = st["observer"], st["performer"]
    obs_out = None
    with torch.no_grad():
        obs_logits = observer(input_ids).logits
        perf_logits = performer(input_ids).logits
    ppl = _perplexity(perf_logits, input_ids, torch)
    x_ppl = _cross_perplexity(obs_logits, perf_logits, torch)
    del obs_logits, perf_logits, obs_out
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    score = ppl / x_ppl if x_ppl > 1e-9 else float("nan")
    return DetectResult(
        available=True, score=score, ppl=ppl, x_ppl=x_ppl, n_tokens=n_tokens,
        observer=st["observer_name"], performer=st["performer_name"],
    )


def score(
    text: str, threshold: float | None = None, calibrated: bool = False,
    direction: str = "low_is_machine",
) -> dict:
    """Public detector signal: raw Binoculars score plus a label if a threshold is given.

    Returns a plain dict (JSON-friendly). ``direction`` says which side of the
    threshold is machine: the canonical Binoculars convention is
    ``"low_is_machine"`` (score below threshold => machine), but the separating
    direction is pair- and domain-specific, so calibration may report
    ``"high_is_machine"`` instead. ``label`` is "machine"/"human" accordingly, or
    "uncertain" when the text is too short or no threshold is supplied.
    """
    try:
        res = raw_score(text)
    except Exception as exc:  # pragma: no cover
        return DetectResult(available=False, detail=f"detector error: {exc}").as_dict()
    res.threshold = threshold
    res.calibrated = calibrated
    if res.n_tokens < _MIN_TOKENS:
        res.label = "uncertain"
        res.detail = f"only {res.n_tokens} tokens (<{_MIN_TOKENS}); score unreliable"
        return res.as_dict()
    if threshold is not None and res.score == res.score:  # not NaN
        if direction == "high_is_machine":
            res.label = "machine" if res.score > threshold else "human"
        else:
            res.label = "machine" if res.score < threshold else "human"
    else:
        res.label = "uncertain"
    return res.as_dict()


# --- threshold calibration ----------------------------------------------------


def calibrate_threshold(
    human_texts: list[str], ai_texts: list[str], target_fpr: float = 0.01
) -> dict:
    """Pick an operating threshold for the configured pair from labelled text.

    Binoculars scores are pair- and dtype-specific, so the paper's Falcon numbers
    do not transfer. We score known-human (author corpus) and known-AI text, then
    set the cutoff at a low quantile of the *human* distribution so genuine author
    writing is rarely flagged (a low false-positive operating point). Also reports
    an AUROC (AI as positive, using -score since lower = more machine) as a
    threshold-free measure of how well the pair separates in this domain.
    """
    human_scores = [s for s in (_safe_score(t) for t in human_texts) if s == s]
    ai_scores = [s for s in (_safe_score(t) for t in ai_texts) if s == s]
    if len(human_scores) < 3 or len(ai_scores) < 3:
        raise ValueError("need >=3 scorable human and AI texts to calibrate")

    hs = sorted(human_scores)
    # AUROC with AI as positive under the canonical "low = machine" convention.
    auroc_low = _auroc(human_scores, ai_scores)
    # The separating direction is pair/domain-specific; pick the one that actually
    # separates and set the threshold + labelling rule to match, so a weak or
    # inverted pair is handled honestly instead of silently mislabelling.
    if auroc_low >= 0.5:
        direction = "low_is_machine"
        idx = max(0, min(len(hs) - 1, int(round(target_fpr * (len(hs) - 1)))))
        threshold = hs[idx]
        effective_auroc = auroc_low
    else:
        direction = "high_is_machine"
        idx = max(0, min(len(hs) - 1, int(round((1.0 - target_fpr) * (len(hs) - 1)))))
        threshold = hs[idx]
        effective_auroc = 1.0 - auroc_low

    observer_name, performer_name = model_pair()
    import statistics

    return {
        "observer": observer_name,
        "performer": performer_name,
        "threshold": round(threshold, 5),
        "direction": direction,
        "target_fpr": target_fpr,
        "auroc": round(effective_auroc, 4),
        "auroc_canonical_low": round(auroc_low, 4),
        "human_mean": round(statistics.mean(human_scores), 5),
        "human_stdev": round(statistics.stdev(human_scores), 5) if len(human_scores) > 1 else 0.0,
        "ai_mean": round(statistics.mean(ai_scores), 5),
        "ai_stdev": round(statistics.stdev(ai_scores), 5) if len(ai_scores) > 1 else 0.0,
        "n_human": len(human_scores),
        "n_ai": len(ai_scores),
        "max_tokens": _max_tokens(),
    }


def _safe_score(text: str) -> float:
    try:
        r = raw_score(text)
    except Exception:
        return float("nan")
    if r.n_tokens < _MIN_TOKENS:
        return float("nan")
    return r.score if r.score is not None else float("nan")


def _auroc(human_scores: list[float], ai_scores: list[float]) -> float:
    """AUROC with AI as the positive class. Lower Binoculars score => more machine,
    so we rank by -score. Computed as the Mann-Whitney U statistic (P(ai ranks as
    more-machine than human))."""
    # positive = AI (should have LOWER score). count pairs where ai_score < human_score.
    wins = ties = 0
    for a in ai_scores:
        for h in human_scores:
            if a < h:
                wins += 1
            elif a == h:
                ties += 1
    total = len(ai_scores) * len(human_scores)
    return (wins + 0.5 * ties) / total if total else 0.5
