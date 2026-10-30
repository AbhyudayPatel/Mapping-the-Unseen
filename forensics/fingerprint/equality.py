"""Model-equality test: "are these two endpoints the same model?"

Gao-et-al.-style two-sample test between two adapters over a *fixed shared battery* of
prompts.  For generative endpoints we collect, per prompt, a greedy (temperature-0) draw
plus optional stochastic draws, reading the returned text, prompt-token count and the
first-token top-logprob shape into a feature vector.  We then compute

* a **token-match rate** (do the greedy outputs agree?),
* a **maximum-mean-discrepancy** (MMD, RBF kernel) between the two clouds of feature
  vectors, and
* a **permutation / label-shuffle significance test** (``forensics.stats.tests.permutation_test``)
  giving a p-value for "the two samples come from the same distribution".

Decision (Jev-style) endpoints have no sampling/logprob controls, so we instead compare the
reported answer **probability vectors** over a shared state/question battery with the same
MMD + permutation machinery.

The test is budget-aware (every call is gated by an optional ``Budget``), records through an
optional ``Recorder``, and **never raises** on API errors — a failed response is treated as
missing data.  Insufficient data yields an ``"inconclusive"`` verdict rather than an error.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from forensics.core.types import DecisionQuestion, DecisionRequest, GenRequest
from forensics.stats.tests import permutation_test

__all__ = ["equality_test"]

_LOGPROB_FLOOR = -20.0
_MAX_MMD_ROWS = 80          # per group: cap the permutation working set for speed
_DEFAULT_N_PERM = 300


# --------------------------------------------------------------------------- batteries
_GEN_BATTERY: tuple[str, ...] = (
    "What is the capital of France? Answer with one word.",
    "What is 17 + 25? Reply with only the number.",
    "Write a one-line Python function that returns the square of x.",
    "Translate to Spanish: good morning.",
    "If all cats are animals and Tom is a cat, is Tom an animal? Answer yes or no.",
    "List three primary colors as a comma-separated list.",
    "Complete the sentence with one word: The quick brown fox jumps over the lazy",
    "Name a common English noun. One word only.",
    "What color is a clear daytime sky? One word.",
    "Give the chemical symbol for water.",
    "What is the opposite of 'hot'? One word.",
    "Round 3.14159 to two decimal places.",
    "What planet do we live on? One word.",
    "Spell the word 'cat' in uppercase.",
    "How many days are in a week? Reply with only the number.",
    "Give a synonym for 'happy'. One word.",
    "What is 9 times 6? Number only.",
    "Name the largest ocean on Earth. One word.",
    "Continue: Monday, Tuesday, Wednesday,",
    "What language is primarily spoken in Japan? One word.",
    "Reply with only 'true' or 'false': the sun is a star.",
    "What is the first letter of the alphabet?",
    "Give the past tense of 'run'. One word.",
    "What is 100 divided by 4? Number only.",
    "Name a fruit that is yellow. One word.",
)

_DEC_STATES: tuple[str, ...] = (
    "The customer was billed twice for the same subscription and wants a refund.",
    "The server returned a 500 error during checkout and the order was lost.",
    "Great product, fast shipping, five stars, would buy again!",
    "I have waited three weeks for a reply and I am extremely frustrated.",
    "The mobile app crashes every time I open the settings screen.",
    "Please cancel my account and delete all of my personal data.",
    "The delivery arrived a day early and everything was perfect.",
    "My promo code was rejected at checkout even though it is valid.",
)

_DEC_CHOICE_LABELS = ["billing", "sales", "technical", "other"]
_DEC_SCORE_LEVELS = ["low", "medium", "high", "critical"]


def _dec_questions() -> dict[str, DecisionQuestion]:
    return {
        "route": DecisionQuestion(type="choice", instructions="Which team should handle this?",
                                  criteria={l: "" for l in _DEC_CHOICE_LABELS}),
        "sev": DecisionQuestion(type="score", instructions="How severe is this issue?",
                                criteria=list(_DEC_SCORE_LEVELS)),
        "urgent": DecisionQuestion(type="noul", instructions="Is this urgent?"),
    }


# --------------------------------------------------------------------------- rng / budget
def _as_rng(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _caps(adapter: Any) -> Any:
    try:
        return adapter.static_capabilities()
    except Exception:  # noqa: BLE001
        from forensics.core.types import Capabilities
        return Capabilities(kind=getattr(adapter, "kind", "unknown"))


class _Caller:
    """Budget-aware, recording, never-raising request driver for one test."""

    def __init__(self, budget: Any, recorder: Any) -> None:
        self.budget = budget
        self.recorder = recorder
        self.exhausted = False
        self.n_requests = 0

    def _can(self, req: Any) -> bool:
        if self.exhausted:
            return False
        if self.budget is not None:
            try:
                if not self.budget.can_spend(req):
                    self.exhausted = True
                    return False
            except Exception:  # noqa: BLE001
                return True
        return True

    async def gen(self, adapter: Any, req: GenRequest) -> Any:
        if not self._can(req):
            return None
        try:
            resp = await adapter.generate(req)
        except Exception:  # noqa: BLE001 — API errors are data, not failures
            return None
        self.n_requests += 1
        self._post(req, resp, "generate")
        return resp

    async def decide(self, adapter: Any, req: DecisionRequest) -> Any:
        if not self._can(req):
            return None
        try:
            resp = await adapter.decide(req)
        except Exception:  # noqa: BLE001
            return None
        self.n_requests += 1
        self._post(req, resp, "decide")
        return resp

    def _post(self, req: Any, resp: Any, kind: str) -> None:
        if self.budget is not None and resp is not None:
            try:
                self.budget.charge(req, resp)
            except Exception:  # noqa: BLE001
                pass
        if self.recorder is not None and resp is not None:
            try:
                self.recorder.record(kind, req, resp, probe="equality_test")
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- feature maths
def _gen_feature(resp: Any, k: int) -> list[float] | None:
    if resp is None or not getattr(resp, "ok", False):
        return None
    usage = getattr(resp, "usage", None)
    pt = float(getattr(usage, "prompt_tokens", None) or 0.0)
    text = resp.text or ""
    ot = float(getattr(usage, "completion_tokens", None) or len(text))
    feat = [pt, float(len(text)), ot]
    if k > 0:
        lps: list[float] = []
        toks = getattr(resp, "tokens", None)
        if toks and toks[0].top:
            lps = sorted((float(t.logprob) for t in toks[0].top), reverse=True)[:k]
        lps = (lps + [_LOGPROB_FLOOR] * k)[:k]
        feat.extend(lps)
    return feat


def _dec_feature(resp: Any) -> tuple[list[float], dict[str, Any]] | None:
    """Feature vector + a summary of chosen values for a decision response."""
    if resp is None or not getattr(resp, "ok", False):
        return None
    answers = getattr(resp, "answers", None) or {}
    feat: list[float] = []
    chosen: dict[str, Any] = {}
    # choice: probabilities over the fixed label set
    route = answers.get("route")
    rp = (getattr(route, "probabilities", None) or {}) if route else {}
    feat.extend(float(rp.get(l, 0.0)) for l in _DEC_CHOICE_LABELS)
    if route is not None:
        chosen["route"] = getattr(route, "value", None)
    # score: probabilities over the fixed levels + the scalar value
    sev = answers.get("sev")
    sp = (getattr(sev, "probabilities", None) or {}) if sev else {}
    feat.extend(float(sp.get(str(i), 0.0)) for i in range(len(_DEC_SCORE_LEVELS)))
    if sev is not None:
        v = getattr(sev, "value", None)
        feat.append(float(v) if isinstance(v, (int, float)) else 0.0)
        chosen["sev"] = v
    else:
        feat.append(0.0)
    # noul: probability of true
    urg = answers.get("urgent")
    if urg is not None:
        v = getattr(urg, "value", None)
        feat.append(float(v) if isinstance(v, (int, float)) else 0.0)
        chosen["urgent"] = v
    else:
        feat.append(0.0)
    return feat, chosen


def _standardize(rows: np.ndarray) -> np.ndarray:
    mu = rows.mean(axis=0)
    sd = rows.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (rows - mu) / sd


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Biased MMD^2 estimate with an RBF kernel (>= ~0, 0 iff clouds coincide)."""
    if x.size == 0 or y.size == 0:
        return 0.0
    kxx = float(_rbf_kernel(x, x, gamma).mean())
    kyy = float(_rbf_kernel(y, y, gamma).mean())
    kxy = float(_rbf_kernel(x, y, gamma).mean())
    return max(0.0, kxx + kyy - 2.0 * kxy)


def _median_gamma(z: np.ndarray) -> float:
    n = z.shape[0]
    if n < 2:
        return 1.0
    idx = np.triu_indices(n, k=1)
    d2 = (np.sum(z * z, axis=1)[:, None] + np.sum(z * z, axis=1)[None, :] - 2.0 * z @ z.T)
    d2 = np.maximum(d2[idx], 0.0)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / med          # gamma = 1 / median(||.||^2)


def _mmd_permutation(a_rows: np.ndarray, b_rows: np.ndarray, n_perm: int,
                     rng: np.random.Generator) -> tuple[float, float | None]:
    """Return (observed MMD^2, permutation p-value) using forensics.stats permutation_test."""
    na, nb = a_rows.shape[0], b_rows.shape[0]
    if na < 2 or nb < 2:
        return 0.0, None
    # cap the working set for permutation cost
    if na > _MAX_MMD_ROWS:
        a_rows = a_rows[rng.choice(na, _MAX_MMD_ROWS, replace=False)]
        na = _MAX_MMD_ROWS
    if nb > _MAX_MMD_ROWS:
        b_rows = b_rows[rng.choice(nb, _MAX_MMD_ROWS, replace=False)]
        nb = _MAX_MMD_ROWS
    z = _standardize(np.vstack([a_rows, b_rows]))
    gamma = _median_gamma(z)
    obs = _mmd2(z[:na], z[na:], gamma)

    def stat(ia: np.ndarray, ib: np.ndarray) -> float:
        return _mmd2(z[ia.astype(int)], z[ib.astype(int)], gamma)

    idx_a = np.arange(na, dtype=float)
    idx_b = np.arange(na, na + nb, dtype=float)
    try:
        p = permutation_test(idx_a, idx_b, stat=stat, n=int(n_perm), rng=rng)
    except Exception:  # noqa: BLE001
        p = None
    return obs, p


def _norm_text(t: str | None) -> str:
    return " ".join((t or "").split()).strip().lower()


def _verdict(token_match: float | None, p_value: float | None, mmd: float,
             n_prompts: int) -> tuple[str, list[str]]:
    notes: list[str] = []
    if n_prompts < 3:
        return "inconclusive", ["too few comparable prompts for a reliable verdict"]
    sig_diff = p_value is not None and p_value <= 0.05 and mmd > 1e-6
    not_sig = p_value is None or p_value > 0.05
    tm = token_match if token_match is not None else 0.0
    if tm >= 0.85 and not_sig:
        return "same", notes
    if tm <= 0.5 or sig_diff:
        return "different", notes
    if tm >= 0.7 and not_sig:
        notes.append("high output agreement but weak statistical power")
        return "same", notes
    return "inconclusive", ["signals conflict (partial output agreement, ambiguous MMD)"]


# --------------------------------------------------------------------------- generative path
async def _equality_generative(adapter_a: Any, adapter_b: Any, prompts: list[str],
                               samples: int, caller: _Caller, rng: np.random.Generator,
                               n_perm: int) -> dict[str, Any]:
    ca, cb = _caps(adapter_a), _caps(adapter_b)
    use_lp = bool(ca.logprobs) and bool(cb.logprobs)
    k = min(ca.max_top_logprobs or 0, cb.max_top_logprobs or 0, 5) if use_lp else 0
    can_sample = bool(ca.temperature) and bool(cb.temperature) and samples > 1

    a_rows: list[list[float]] = []
    b_rows: list[list[float]] = []
    per_prompt: list[dict[str, Any]] = []
    matches: list[float] = []

    for i, prompt in enumerate(prompts):
        # greedy reference draw (used for token match + a feature row)
        req0 = GenRequest.chat(prompt, max_tokens=16, temperature=0.0,
                               logprobs=use_lp, top_logprobs=(k or None), seed=0)
        ra = await caller.gen(adapter_a, req0)
        rb = await caller.gen(adapter_b, req0)
        fa, fb = _gen_feature(ra, k), _gen_feature(rb, k)
        ta = _norm_text(ra.text) if ra is not None and ra.ok else None
        tb = _norm_text(rb.text) if rb is not None and rb.ok else None
        if fa:
            a_rows.append(fa)
        if fb:
            b_rows.append(fb)
        if ta is not None and tb is not None and (ta or tb):
            matches.append(1.0 if ta == tb else 0.0)
        per_prompt.append({"prompt": prompt[:48], "text_a": (ta or "")[:32],
                           "text_b": (tb or "")[:32],
                           "match": (None if ta is None or tb is None else ta == tb)})
        # extra stochastic draws feed the distribution comparison
        if can_sample and not caller.exhausted:
            for s in range(1, samples):
                if caller.exhausted:
                    break
                reqs = GenRequest.chat(prompt, max_tokens=16, temperature=1.0,
                                       logprobs=use_lp, top_logprobs=(k or None), seed=s + 1)
                ra2 = await caller.gen(adapter_a, reqs)
                rb2 = await caller.gen(adapter_b, reqs)
                if (f := _gen_feature(ra2, k)):
                    a_rows.append(f)
                if (f := _gen_feature(rb2, k)):
                    b_rows.append(f)

    token_match = float(np.mean(matches)) if matches else None
    A = np.asarray(a_rows, float) if a_rows else np.zeros((0, 0))
    B = np.asarray(b_rows, float) if b_rows else np.zeros((0, 0))
    if A.shape[0] >= 2 and B.shape[0] >= 2 and A.shape[1] == B.shape[1]:
        mmd, p_value = _mmd_permutation(A, B, n_perm, rng)
    else:
        mmd, p_value = 0.0, None

    verdict, notes = _verdict(token_match, p_value, mmd, len(matches))
    if caller.exhausted:
        notes.append("budget exhausted during collection; verdict on partial data")
    notes.append(f"generative equality: {len(a_rows)}+{len(b_rows)} feature rows, "
                 f"logprobs={'on' if k else 'off'} (k={k}), token_match={token_match}")
    return {
        "verdict": verdict, "statistic": round(mmd, 6), "p_value": p_value,
        "token_match_rate": None if token_match is None else round(token_match, 4),
        "kind": "generative", "n_prompts": len(prompts),
        "n_rows_a": len(a_rows), "n_rows_b": len(b_rows),
        "n_requests": caller.n_requests, "per_prompt": per_prompt, "notes": notes,
    }


# --------------------------------------------------------------------------- decision path
async def _equality_decision(adapter_a: Any, adapter_b: Any, states: list[str],
                             caller: _Caller, rng: np.random.Generator,
                             n_perm: int) -> dict[str, Any]:
    a_rows: list[list[float]] = []
    b_rows: list[list[float]] = []
    per_prompt: list[dict[str, Any]] = []
    matches: list[float] = []

    for st in states:
        req = DecisionRequest(state=st, questions=_dec_questions())
        ra = await caller.decide(adapter_a, req)
        rb = await caller.decide(adapter_b, req)
        da, db = _dec_feature(ra), _dec_feature(rb)
        va = da[1] if da else None
        vb = db[1] if db else None
        if da:
            a_rows.append(da[0])
        if db:
            b_rows.append(db[0])
        if va is not None and vb is not None:
            same = (va.get("route") == vb.get("route"))
            matches.append(1.0 if same else 0.0)
        per_prompt.append({"state": st[:48],
                           "route_a": (va or {}).get("route"),
                           "route_b": (vb or {}).get("route"),
                           "match": (None if va is None or vb is None
                                     else va.get("route") == vb.get("route"))})
        if caller.exhausted:
            break

    token_match = float(np.mean(matches)) if matches else None
    A = np.asarray(a_rows, float) if a_rows else np.zeros((0, 0))
    B = np.asarray(b_rows, float) if b_rows else np.zeros((0, 0))
    if A.shape[0] >= 2 and B.shape[0] >= 2 and A.shape[1] == B.shape[1]:
        mmd, p_value = _mmd_permutation(A, B, n_perm, rng)
    else:
        mmd, p_value = 0.0, None

    verdict, notes = _verdict(token_match, p_value, mmd, len(matches))
    if caller.exhausted:
        notes.append("budget exhausted during collection; verdict on partial data")
    notes.append(f"decision equality: {len(a_rows)}+{len(b_rows)} probability vectors, "
                 f"route_match={token_match}")
    return {
        "verdict": verdict, "statistic": round(mmd, 6), "p_value": p_value,
        "token_match_rate": None if token_match is None else round(token_match, 4),
        "kind": "decision", "n_prompts": len(states),
        "n_rows_a": len(a_rows), "n_rows_b": len(b_rows),
        "n_requests": caller.n_requests, "per_prompt": per_prompt, "notes": notes,
    }


# --------------------------------------------------------------------------- public API
async def equality_test(adapter_a: Any, adapter_b: Any, n_prompts: int = 25,
                        samples: int = 10, rng: Any = None, budget: Any = None,
                        recorder: Any = None, n_perm: int = _DEFAULT_N_PERM) -> dict[str, Any]:
    """Test whether two endpoints are the same model over a fixed shared prompt battery.

    Returns a dict with ``verdict`` (``"same" | "different" | "inconclusive"``),
    ``statistic`` (observed MMD^2), ``p_value`` (label-shuffle permutation test),
    ``token_match_rate``, ``per_prompt`` details and ``notes``.  Budget-aware and
    exception-safe: API errors and an exhausted budget degrade to partial data and,
    if necessary, an inconclusive verdict — the function never raises.
    """
    generator = _as_rng(rng)
    caller = _Caller(budget, recorder)
    ca, cb = _caps(adapter_a), _caps(adapter_b)
    kind_a = ca.kind if ca.kind and ca.kind != "unknown" else getattr(adapter_a, "kind", "unknown")
    kind_b = cb.kind if cb.kind and cb.kind != "unknown" else getattr(adapter_b, "kind", "unknown")

    if kind_a != kind_b and "unknown" not in (kind_a, kind_b):
        return {"verdict": "different", "statistic": 1.0, "p_value": 0.0,
                "token_match_rate": 0.0, "kind": "mismatch", "n_prompts": 0,
                "n_rows_a": 0, "n_rows_b": 0, "n_requests": 0, "per_prompt": [],
                "notes": [f"endpoint kinds differ ({kind_a} vs {kind_b}); "
                          "cannot be the same model"]}

    kind = kind_a if kind_a != "unknown" else kind_b
    n = max(1, int(n_prompts))
    if kind == "decision":
        states = [_DEC_STATES[i % len(_DEC_STATES)] for i in range(min(n, len(_DEC_STATES)))]
        return await _equality_decision(adapter_a, adapter_b, states, caller, generator, n_perm)
    prompts = [_GEN_BATTERY[i % len(_GEN_BATTERY)] for i in range(min(n, len(_GEN_BATTERY)))]
    return await _equality_generative(adapter_a, adapter_b, prompts, max(1, int(samples)),
                                      caller, generator, n_perm)
