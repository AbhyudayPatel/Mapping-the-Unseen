"""P06 — Positional encoding & long-context recall (needle-in-a-haystack).

Place a factual "needle" at a controlled fractional position inside filler of controlled
length, then ask for it back and score retrieval over a grid of
``(needle_position x context_length)``.  The *shape* of the recall surface is diagnostic:

* **sliding-window cliff** — with the needle at the *start*, recall stays high while the
  context fits the attention window and collapses once the needle falls out of the last
  ``W`` tokens.  The knee of that curve estimates ``context.sliding_window`` (and the
  onset of the drop is ``context.effective_recall_tokens``).  ⇒ ``attn.long_range`` =
  sliding/local.
* **flat, high recall everywhere** ⇒ full attention over the tested range
  (``attn.long_range`` = full_quadratic; ``pos.long_context_ext`` = extended if recall
  survives well past a power of two).
* **lost-in-the-middle dip** — a mid-position trough with high edges (informational only).
* **RoPE-extrapolation collapse** — a sharp drop just past a power-of-two boundary with no
  context-extension (weakly informs ``pos.encoding`` / ``pos.long_context_ext``).

The needle marker and question format below are stable so a device (or the simulator's
responder) can recognise them.
"""
from __future__ import annotations

import math

import numpy as np

from forensics.probes.base import (
    STATISTICAL, WEAK, BudgetExhausted, Estimate, Evidence, Probe, ProbeContext,
    ProbeResult, register,
)
from forensics.core.types import GenRequest
from forensics.stats import knee_point, proportion_ci

_FILLER = "the quiet harbor town kept its ledgers in careful order every season "  # ~13 tok
_NEEDLE = "IMPORTANT FACT: the harbor passcode is {code}."
_QUESTION = "\n\nQuestion: what is the harbor passcode? Reply with only the number."


def _floor(v: float) -> float:
    return max(0.02, float(v))


@register
class NeedleRecallProbe(Probe):
    id = "P06.positional.needle_recall"
    title = "Needle-in-haystack recall vs position and length"
    category = "context"
    kinds = ("generative",)
    informs = {"attn.long_range": 0.5, "pos.encoding": 0.15, "pos.long_context_ext": 0.3}
    estimates = ("context.effective_recall_tokens", "context.sliding_window")
    cost = 27

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        res = self.result()

        max_in = ctx.shared.get("context", {}).get("max_input_tokens") if isinstance(
            ctx.shared.get("context"), dict) else None
        cap = int(ctx.config.get("needle_max_tokens") or (max_in and max_in * 0.85) or 4096)
        cap = max(512, min(cap, 8192))

        n_len = 6 if ctx.fast() else 9
        lengths = sorted({int(v) for v in np.geomspace(128, cap, n_len)})
        positions = [0.0, 0.5, 0.9]

        rows = []
        try:
            for frac in positions:
                for L in lengths:
                    code = f"{int(ctx.rng.integers(10000, 99999))}"
                    prompt = self._make(L, frac, code)
                    resp = await ctx.gen(GenRequest.chat(prompt, max_tokens=24, tag=self.id))
                    ok = 1 if (resp.ok and code in (resp.text or "")) else 0
                    ptoks = resp.usage.prompt_tokens or L
                    rows.append({"position": frac, "target_len": L,
                                 "prompt_tokens": int(ptoks), "recall": ok,
                                 "api_ok": bool(resp.ok)})
        except BudgetExhausted as e:
            res.notes.append(f"budget exhausted: {e}")
            res.status = "partial"
            if not rows:
                res.n_requests = ctx.n_requests
                return res

        res.tables["grid"] = rows

        # ---- per-position curves --------------------------------------------
        for frac in positions:
            pts = [(r["prompt_tokens"], r["recall"]) for r in rows if r["position"] == frac]
            pts.sort()
            if pts:
                res.series[f"recall_pos_{frac}"] = {
                    "x": [p for p, _ in pts], "y": [q for _, q in pts],
                    "xlabel": "prompt tokens", "ylabel": "recall",
                    "kind": "line", "label": f"needle@{frac:g}",
                }

        # ---- sliding-window / cliff from the START-position row -------------
        start = sorted([(r["prompt_tokens"], r["recall"]) for r in rows if r["position"] == 0.0])
        x = np.array([p for p, _ in start], float)
        y = np.array([q for _, q in start], float)
        cliff, window, eff = self._analyse_cliff(x, y)

        # overall recall for a sanity metric
        all_recall = np.array([r["recall"] for r in rows], float)
        k = int(all_recall.sum())
        n = int(all_recall.size)
        rec_lo, rec_hi = proportion_ci(k, n) if n else (0.0, 1.0)

        # ---- attn.long_range evidence ---------------------------------------
        if cliff:
            like = {"full_quadratic": _floor(0.12), "sliding_or_local": 1.0,
                    "linear_recurrent": _floor(0.3), "hybrid": _floor(0.45)}
            summary = f"recall cliff near {window} tokens (needle@start)"
            res.add_estimate(Estimate(
                quantity="context.sliding_window", value=int(window),
                ci=(float(eff["last_hi"]), float(eff["first_lo"])), level="inferred",
                method="needle-recall cliff: geometric midpoint / Kneedle knee",
                identifiability=STATISTICAL, unit="tokens",
                alternatives=["chunked-prefill boundary", "hard context truncation, not a window"],
            ))
        else:
            like = {"full_quadratic": 1.0, "sliding_or_local": _floor(0.2),
                    "linear_recurrent": _floor(0.4), "hybrid": _floor(0.6)}
            summary = f"no recall cliff; recall high across {int(x.max()) if x.size else 0} tokens"

        res.add_evidence(Evidence(
            property="attn.long_range", likelihoods=like, summary=summary,
            level="inferred", strength=0.7,
            stats={"cliff": bool(cliff), "window": window,
                   "recall_rate": k / n if n else None, "recall_ci": [rec_lo, rec_hi],
                   "n": n, "test": "needle-in-haystack"},
            alternatives=["a retrieval/RAG layer would keep recall flat regardless of window",
                          "instruction-following failure could mask true recall"],
        ))

        # effective recall estimate (always)
        eff_val = int(eff["last_hi"]) if cliff else int(x.max()) if x.size else 0
        res.add_estimate(Estimate(
            quantity="context.effective_recall_tokens", value=eff_val,
            # no cliff: recall held up to the longest length tested -> open-ended lower bound
            ci=(float(eff["last_hi"]) if cliff else float(eff_val),
                float(eff["first_lo"]) if cliff else float("inf")),
            level="inferred" if cliff else "estimated",
            method="onset of needle-recall drop" if cliff else "max length with high recall (lower bound)",
            identifiability=STATISTICAL if cliff else WEAK, unit="tokens",
        ))

        # ---- weak long-context-extension evidence ---------------------------
        if x.size:
            survives = (not cliff) and float(x.max()) > 4096
            res.add_evidence(Evidence(
                property="pos.long_context_ext",
                likelihoods={"interpolation_or_yarn": _floor(0.6 if survives else 0.4),
                             "none": _floor(0.4 if survives else 0.6)},
                summary=("recall survives well past 4k" if survives
                         else "recall does not clearly extend past 4k / cliff present"),
                level="inferred", strength=0.25,
                alternatives=["natively long pretraining", "sliding window caps effective length"],
            ))

        # ---- lost-in-the-middle (informational) -----------------------------
        litm = self._litm(rows, lengths)
        if litm is not None:
            res.notes.append(f"lost-in-the-middle dip: middle recall {litm['mid']:.2f} "
                             f"vs edge {litm['edge']:.2f} at ~{litm['len']} tokens")
            res.tables["lost_in_middle"] = [litm]

        res.shared["context_recall"] = {
            "sliding_window": int(window) if cliff else None,
            "effective_recall_tokens": eff_val, "cliff": bool(cliff),
        }
        res.n_requests = ctx.n_requests
        return res

    # ------------------------------------------------------------------ build
    def _make(self, length_units: int, frac: float, code: str) -> str:
        approx = max(1, length_units // 13)          # ~13 tokens per filler unit
        before = int(round(frac * approx))
        after = max(0, approx - before)
        needle = _NEEDLE.format(code=code)
        return (_FILLER * before) + needle + " " + (_FILLER * after) + _QUESTION

    # ---------------------------------------------------------------- analyse
    @staticmethod
    def _analyse_cliff(x: np.ndarray, y: np.ndarray):
        eff = {"last_hi": float(x.max()) if x.size else 0.0,
               "first_lo": float(x.max()) if x.size else 0.0}
        if x.size < 3 or (y.max() - y.min()) < 0.5:
            return False, None, eff
        # needs a high-recall head and a low-recall tail
        hi_x = x[y >= 0.5]
        lo_x = x[y < 0.5]
        if hi_x.size == 0 or lo_x.size == 0:
            return False, None, eff
        last_hi = float(hi_x.max())
        first_lo = float(lo_x[lo_x > last_hi].min()) if np.any(lo_x > last_hi) else float(lo_x.min())
        if first_lo <= last_hi:
            return False, None, eff
        kp = knee_point(x, y)
        geom = math.sqrt(last_hi * first_lo)
        knee = kp["knee_x"] if kp["knee_x"] else geom
        window = int(round((geom + knee) / 2.0))
        eff = {"last_hi": last_hi, "first_lo": first_lo}
        return True, window, eff

    @staticmethod
    def _litm(rows, lengths):
        target = max(lengths)
        edge = [r["recall"] for r in rows if r["position"] in (0.0, 0.9) and r["target_len"] == target]
        mid = [r["recall"] for r in rows if r["position"] == 0.5 and r["target_len"] == target]
        if not edge or not mid:
            return None
        e, m = float(np.mean(edge)), float(np.mean(mid))
        if e - m >= 0.4:
            return {"len": int(target), "edge": e, "mid": m}
        return None
