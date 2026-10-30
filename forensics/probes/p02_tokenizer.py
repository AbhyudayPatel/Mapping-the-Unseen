"""P02 — tokenizer fingerprint.

Method A (primary, needs ``usage.prompt_tokens``): send a battery of crafted discriminator
strings, each as a minimal one-message prompt, and measure the *delta* in prompt tokens
against an empty-message baseline. That delta equals the raw token count of the string
under the target's tokenizer (the constant template overhead cancels). We then compare the
observed delta vector to ``len(encode(s))`` for every currently-loadable tokenizer in the
:class:`TokenizerRegistry` and rank candidates by exact matches, then by L1 distance.

Method B (corroboration, when logprobs return token strings/bytes): read the returned
piece strings and check which candidate reproduces the same segmentation count. Reported in
the table; Method A makes the decision.

Discriminators are chosen so families that share English merges still diverge: single-digit
vs 3-digit chunking, whitespace/newline runs, long letter runs (BPE merge depth), CJK,
Cyrillic, emoji + ZWJ sequences, special-token strings, code and JSON punctuation
(see ``research/jev.md`` token-delta table and ``research/techniques_logits_tokenizer.md`` §9).

Estimates: ``tokenizer.family``, ``tokenizer.type``, ``dims.vocab_size`` (rough, from the
matching families), ``tokenizer.chat_template_overhead`` (the empty-baseline prompt tokens).
Publishes ``shared['tokenizer'] = {family, count_fn, vocab_size, chat_template_overhead}``.
"""
from __future__ import annotations

import json
import math
from typing import Any

from forensics.core.types import DecisionQuestion, DecisionRequest, GenRequest
from forensics.knowledge.tokenizers import TokenizerRegistry
from forensics.probes.base import (
    STATISTICAL, WEAK, BudgetExhausted, Evidence, Estimate, Probe, ProbeContext, ProbeResult,
    register,
)

# Discriminator battery. Non-ASCII written as escapes so the source is encoding-safe.
_CJK1 = "こんにちは世界"                 # こんにちは世界
_CJK2 = "你好，世界。我们今天去公园散步。"  # 你好，世界。我们今天去公园散步。
_CYR1 = "Привет"                       # Привет
_CYR2 = "Привет мир"     # Привет мир
_EMOJI = "\U0001F600"                                               # 😀
_ZWJ = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"  # 👨‍👩‍👧‍👦 family

# (name, string). Core set is used even in fast mode; extended set adds more when not fast.
_CORE: list[tuple[str, str]] = [
    ("hello", "hello"),
    ("hello_world", "hello world"),
    ("lead_space", " hello"),
    ("digits5", "12345"),
    ("digits10", "1234567890"),
    ("pi", "3.14159"),
    ("spaces16", " " * 16),
    ("newlines", "\n\n"),
    ("a4", "a" * 4),
    ("a16", "a" * 16),
    ("a32", "a" * 32),
    ("cjk1", _CJK1),
    ("cjk2", _CJK2),
    ("cyr1", _CYR1),
    ("emoji", _EMOJI),
    ("zwj", _ZWJ),
    ("eot", "<|endoftext|>"),
    ("json", '{"a": 1, "b": [1, 2, 3]}'),
]
_EXTENDED: list[tuple[str, str]] = [
    ("quickfox", "The quick brown fox jumps over the lazy dog"),
    ("spaces5", " " * 5),
    ("a8", "a" * 8),
    ("cyr2", _CYR2),
    ("code", "def foo(x):\n    return x + 1"),
    ("url", "https://www.example.com/path?query=1"),
]


# --------------------------------------------------------------------------- decision battery
# The research/jev.md section-3 token-delta table (plus extra digit / script discriminators).
# FAST subset first (24 strings), the rest only in full mode.
_DEC_FAST: list[tuple[str, str]] = [
    ("hello", "hello"),
    ("lead_space", " hello"),
    ("hello_world", "hello world"),
    ("antidis", "antidisestablishmentarianism"),
    ("solidgold", "SolidGoldMagikarp"),
    ("digits5", "12345"),
    ("digits10", "1234567890"),
    ("pi", "3.14159265358979"),
    ("year_sp", " 2024"),
    ("digits6", "100000"),
    ("spaces16", " " * 16),
    ("code", "def foo(x):\n    return x + 1"),
    ("json", '{"a": 1, "b": [1, 2, 3]}'),
    ("url", "https://www.example.com/path?query=1"),
    ("cyr1", _CYR1),
    ("cyr2", _CYR2),
    ("cjk1", _CJK1),
    ("cjk2", _CJK2),
    ("emoji", _EMOJI),
    ("zwj", _ZWJ),
    ("eot", "<|endoftext|>"),
    ("a8", "a" * 8),
    ("a16", "a" * 16),
    ("a32", "a" * 32),
]
_DEC_FULL_EXTRA: list[tuple[str, str]] = [
    ("quickfox", "The quick brown fox jumps over the lazy dog"),
    ("seven", "7"),
    ("spaces5", " " * 5),
    ("newlines4", "\n\n\n\n"),
    ("tabs2", "\t\t"),
    ("naive", "na\u00efve caf\u00e9"),
    ("umlaut", "\u00c4\u00d6\u00dc\u00df stra\u00dfe"),
    ("cjk_char", "\u4e16"),
    ("a4", "a" * 4),
    ("hello_comma", "hello, world!"),
    ("hello_x50", " ".join(["hello"] * 50)),
    ("num_sp", " 42"),
    ("korean", "\uc548\ub155\ud558\uc138\uc694"),
    ("arabic", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    ("hindi", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"),
]
#: exact-match rate below which no public tokenizer is crowned
_MATCH_RATE = 0.85
_NOUL_Q = "Is this text in English?"


def _json_escaped(s: str) -> str:
    """How the string would look if the service tokenized a JSON-escaped state."""
    return json.dumps(s, ensure_ascii=True)[1:-1]


def _features(deltas: dict[str, int], by_name: dict[str, str]) -> list[dict[str, Any]]:
    """Discriminating features of the observed segmentation (digits, scripts, letter runs)."""
    rows: list[dict[str, Any]] = []

    def per_tok(name: str) -> float | None:
        d = deltas.get(name)
        if not d or name not in by_name:
            return None
        return round(len(by_name[name]) / d, 2)

    d10 = deltas.get("digits10")
    if d10:
        kind = ("1 token per digit" if d10 == 10 else
                "3-digit chunks" if d10 in (4,) else f"{10 / d10:.1f} digits/token")
        rows.append({"feature": "digit splitting", "observed": kind,
                     "evidence": f"'1234567890' -> {d10} tokens"})
    for nm, label in (("cjk1", "Japanese chars/token"), ("cjk2", "Chinese chars/token"),
                      ("cyr1", "Cyrillic chars/token"), ("korean", "Korean chars/token"),
                      ("arabic", "Arabic chars/token"), ("hindi", "Devanagari chars/token")):
        v = per_tok(nm)
        if v is not None:
            rows.append({"feature": label, "observed": v,
                         "evidence": f"{by_name[nm]!r} -> {deltas[nm]} tokens"})
    runs = [(n, deltas[n]) for n in ("a4", "a8", "a16", "a32") if n in deltas]
    if runs:
        rows.append({"feature": "letter-run segmentation ('a' x k)",
                     "observed": ", ".join(f"{n[1:]}->{d}" for n, d in runs),
                     "evidence": "BPE merge depth on repeated letters"})
    if "eot" in deltas:
        rows.append({"feature": "special-token strings",
                     "observed": ("parsed as a special token" if deltas["eot"] == 1
                                  else f"plain text ({deltas['eot']} tokens)"),
                     "evidence": "'<|endoftext|>'"})
    if "spaces16" in deltas:
        rows.append({"feature": "whitespace runs", "observed": f"16 spaces -> {deltas['spaces16']}",
                     "evidence": "long whitespace merges"})
    return rows


def _count(enc: Any, text: str) -> int:
    """Token count of ``text`` as PLAIN text. tiktoken encoders raise on special-token
    strings such as '<|endoftext|>' by default; APIs tokenize user text with special tokens
    disallowed, so retry with ``disallowed_special=()`` instead of dropping the string (which
    would unfairly cost every tiktoken family one exact match)."""
    try:
        return enc.count(text)
    except Exception:  # noqa: BLE001
        fn = getattr(enc, "_encode", None)
        if fn is None:
            raise
        return len(list(fn(text, disallowed_special=())))


_DERIVED_MARKS = ("derived", "mirror", "(", "copy", "alt", "legacy")


def _canonical_rank(fam: str, entry: dict[str, Any] | None) -> int:
    """Lower = more canonical: bundled tiktoken encodings first, then original vocabularies,
    then derived / mirrored / copied ones (used only to order EXACT ties)."""
    e = entry or {}
    if fam.startswith("tiktoken:") or e.get("loads_with_tiktoken_get_encoding"):
        return 0
    low = fam.lower()
    if any(m in low for m in _DERIVED_MARKS) or "derived" in str(e.get("type", "")).lower():
        return 2
    return 1


# Pool for adaptive tie-breaking: diverse strings on which similar vocabularies often differ.
_TIEBREAK_POOL: list[str] = [
    " 2024", " 1000000", "3.14159 2.71828", " 007", "0x7fffffff", "１２３",
    "你好世界", "机器学习模型", "東京都の天気", "こんにちは", "안녕하세요 세계", "مرحبا بالعالم",
    "नमस्ते दुनिया", "Здравствуйте", "Γειά σου κόσμε", "שלום עולם", "สวัสดีครับ",
    "def __init__(self):", "console.log(`${x}`)", "SELECT * FROM t WHERE id=1;", "#include <stdio.h>",
    "    return None", "\t\tindent", "</div><div class=\"x\">", "https://github.com/a/b?c=d",
    "antidisestablishmentarianism", "pneumonoultramicroscopic", "Supercalifragilistic",
    "iPhone", "ChatGPT", "DeepSeek", "Llama", "Qwen", "Anthropic", "HuggingFace",
    "😀😃😄", "🇮🇳🇺🇸", "👍🏽", "café naïve résumé", "ſtraße Œuvre", "——…", "“quoted”",
    "!!!???", "   ", "\n\n\n", "aaaaaaaaaaaaaaaa", "ababababab", "CamelCaseIdentifier",
    "snake_case_name", "UPPERCASE WORDS", "e=mc^2", "∑∫√∞≈", "½ ¾ ⅓", "™ © ®",
]


def _discriminators(reg: TokenizerRegistry, fams: list[str], k: int = 6) -> list[str]:
    """Greedily choose up to k pool strings that separate the most pairs of tied tokenizers
    (offline: uses the local encoders, costs no requests)."""
    encs = {f: reg.get_encoder(f) for f in fams}
    encs = {f: e for f, e in encs.items() if e is not None}
    if len(encs) < 2:
        return []
    counts: dict[str, dict[str, int]] = {}
    for sp in _TIEBREAK_POOL:
        row = {}
        for f, e in encs.items():
            try:
                row[f] = _count(e, sp)
            except Exception:
                row = {}
                break
        if row and len(set(row.values())) > 1:
            counts[sp] = row
    fl = list(encs)
    unresolved = {(a, b) for i, a in enumerate(fl) for b in fl[i + 1:]}
    chosen: list[str] = []
    while unresolved and len(chosen) < k and counts:
        best = max(counts, key=lambda sp: sum(counts[sp][a] != counts[sp][b]
                                              for a, b in unresolved))
        gain = {(a, b) for a, b in unresolved if counts[best][a] != counts[best][b]}
        if not gain:
            break
        chosen.append(best)
        unresolved -= gain
        counts.pop(best)
    return chosen


def _rank_scores(reg: TokenizerRegistry, scores: dict[str, dict[str, Any]]
                 ) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    """Rank by (exact desc, L1 asc); break exact ties canonically. Returns (ranked, tie_set)
    where tie_set lists every family indistinguishable from the winner on this battery."""
    ranked = sorted(scores.items(), key=lambda kv: (
        -kv[1]["exact"], kv[1]["l1"], _canonical_rank(kv[0], reg.get(kv[0])),
        -len((reg.get(kv[0]) or {}).get("used_by") or []), len(kv[0]), kv[0]))
    if not ranked:
        return ranked, []
    w = ranked[0][1]
    ties = [f for f, sc in ranked if sc["exact"] == w["exact"] and sc["l1"] == w["l1"]]
    return ranked, ties


@register
class TokenizerFingerprintProbe(Probe):
    id = "P02.tokenizer.fingerprint"
    title = "Tokenizer family & vocab from prompt-token deltas"
    category = "tokenizer"
    kinds = ("generative", "decision")
    informs = {"tokenizer.family": 0.9, "arch.family": 0.1}
    estimates = ("tokenizer.family", "tokenizer.type", "dims.vocab_size",
                 "tokenizer.chat_template_overhead")
    requires = ("usage_prompt_tokens",)
    cost = 20
    baseline = True

    async def _prompt_tokens(self, ctx: ProbeContext, s: str) -> int | None:
        resp = await ctx.gen(GenRequest.chat(s, max_tokens=1, temperature=0.0))
        if not resp.ok or resp.usage is None or resp.usage.prompt_tokens is None:
            return None
        return int(resp.usage.prompt_tokens)

    @staticmethod
    def _is_decision(ctx: ProbeContext) -> bool:
        kind = ctx.caps.kind if ctx.caps.kind != "unknown" else getattr(ctx.adapter, "kind", "")
        return kind == "decision"

    async def run(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        ctx.current_probe = self.id
        if self._is_decision(ctx):
            return await self._run_decision(ctx)
        res = self.result()
        reg = TokenizerRegistry.load()
        candidates = reg.candidates()
        if not candidates:
            res.status = "skipped"
            res.notes.append("no loadable tokenizers in registry")
            return res

        battery = list(_CORE) if ctx.fast() else list(_CORE) + list(_EXTENDED)

        # ---- baseline (empty message) + deltas --------------------------------
        base = await self._prompt_tokens(ctx, "")
        if base is None:
            res.status = "skipped"
            res.notes.append("usage.prompt_tokens not reported; cannot measure deltas")
            return res

        deltas: dict[str, int] = {}
        for name, s in battery:
            pt = await self._prompt_tokens(ctx, s)
            if pt is not None:
                deltas[name] = pt - base

        if not deltas:
            res.status = "partial"
            res.notes.append("no deltas measured")
            return res

        # ---- Method A: compare delta vector to each candidate -----------------
        by_name = {name: s for name, s in battery}
        rows: list[dict[str, Any]] = []
        scores: dict[str, dict[str, Any]] = {}
        for fam in candidates:
            enc = reg.get_encoder(fam)
            if enc is None:
                continue
            exact = 0
            l1 = 0
            for name, dv in deltas.items():
                try:
                    c = _count(enc, by_name[name])
                except Exception:
                    continue
                l1 += abs(c - dv)
                exact += int(c == dv)
            entry = reg.get(fam) or {}
            scores[fam] = {"exact": exact, "l1": l1,
                           "vocab": entry.get("vocab_size"),
                           "type": entry.get("type")}

        if not scores:
            res.status = "partial"
            res.notes.append("no candidate tokenizer could be scored")
            return res

        ranked, ties = _rank_scores(reg, scores)

        # ---- adaptive tie-break: query only strings on which the tied families disagree
        if len(ties) > 1 and not ctx.config.get("p02_no_tiebreak"):
            extra = _discriminators(reg, ties, k=6)
            for i, sx in enumerate(extra):
                pt = await self._prompt_tokens(ctx, sx)
                if pt is None:
                    continue
                name = f"tiebreak_{i}"
                by_name[name] = sx
                deltas[name] = pt - base
                for fam in scores:
                    enc = reg.get_encoder(fam)
                    try:
                        c = _count(enc, sx)
                    except Exception:
                        continue
                    scores[fam]["l1"] += abs(c - deltas[name])
                    scores[fam]["exact"] += int(c == deltas[name])
            if extra:
                before = list(ties)
                ranked, ties = _rank_scores(reg, scores)
                res.notes.append(f"adaptive tie-break: {len(extra)} discriminating strings "
                                 f"separated {before} -> winner {ranked[0][0]}"
                                 + (f" (still tied with {ties[1:]})" if len(ties) > 1 else ""))

        n = len(deltas)
        for fam, sc in ranked:
            rows.append({"family": fam, "exact": sc["exact"], "n": n,
                         "l1": sc["l1"], "vocab_size": sc["vocab"], "type": sc["type"],
                         "tied_with_winner": fam in ties})
        res.tables["tokenizer_scores"] = rows
        res.tables["deltas"] = [{"probe": k, "delta": v} for k, v in deltas.items()]

        winner, wsc = ranked[0]
        wtype = wsc.get("type") or "bpe"
        wvocab = wsc.get("vocab")

        # ---- likelihoods over families: exp(-L1 / tau) ------------------------
        tau = max(1.0, 0.6 * n)
        lk: dict[str, float] = {}
        for fam, sc in scores.items():
            lk[fam] = max(0.02, math.exp(-sc["l1"] / tau))
        tie_note = ""
        if len(ties) > 1:
            # indistinguishable on this battery: identical likelihood for every tied family
            top_l = max(lk[f] for f in ties)
            for f in ties:
                lk[f] = top_l
            tie_note = f"; exact tie with {len(ties) - 1} other famil(ies): {', '.join(ties[1:6])}"
            res.notes.append(f"tokenizer tie: {', '.join(ties)} give identical counts on this "
                             f"battery; reporting the canonical one ({winner})")
        runner = next((f for f, _ in ranked if f not in ties), "n/a")
        res.add_evidence(Evidence(
            property="tokenizer.family",
            likelihoods=lk,
            summary=(f"best match {winner} (exact {wsc['exact']}/{n}, L1 {wsc['l1']})"
                     f"{tie_note}; runner-up {runner}"),
            level="inferred",
            strength=0.9,
            stats={"exact": wsc["exact"], "n": n, "l1": wsc["l1"],
                   "candidates": len(scores), "tied_with": ties[1:]},
            alternatives=["billing tokenizer differs from the model's tokenizer",
                          "family shares a base vocabulary with another (counts alias)",
                          "hidden variable-length system prompt perturbs deltas"],
        ))

        # weak arch.family nudge: tiktoken/byte-BPE families ride on decoder LMs
        if "wordpiece" in str(wtype).lower():
            arch_lk = {"encoder_only": 1.0, "decoder_only": 0.4}
        else:
            arch_lk = {"decoder_only": 1.0, "encoder_only": 0.6}
        res.add_evidence(Evidence(
            property="arch.family",
            likelihoods={k: max(0.05, v) for k, v in arch_lk.items()},
            summary=f"tokenizer type {wtype!r} weakly consistent with arch family",
            level="speculative", strength=0.2,
            stats={"tokenizer_type": wtype},
            alternatives=["tokenizer reused across architecture families"],
        ))

        # ---- estimates --------------------------------------------------------
        res.add_estimate(Estimate(
            quantity="tokenizer.family", value=winner,
            level="inferred", method="prompt-token delta vector match",
            identifiability=STATISTICAL,
            assumptions=["template overhead is additive/constant across probes",
                         "candidate set limited to currently-loadable tokenizers"],
            alternatives=[r[0] for r in ranked[1:3]],
        ))
        res.add_estimate(Estimate(
            quantity="tokenizer.type", value=wtype,
            level="inferred", method="registry record of best-matching family",
            identifiability=STATISTICAL,
        ))
        # vocab: from families that tie the winner on exact matches
        tied = [sc["vocab"] for fam, sc in scores.items()
                if sc["exact"] == wsc["exact"] and sc["vocab"]]
        if tied:
            lo, hi = float(min(tied)), float(max(tied))
            res.add_estimate(Estimate(
                quantity="dims.vocab_size", value=int(wvocab or min(tied)),
                ci=(lo, hi), level="estimated", unit="tokens",
                method="vocab of best-matching tokenizer family(ies)",
                identifiability=WEAK,
                assumptions=["true vocab equals matched tokenizer's vocab",
                             "counts identify tokenization stack, not model lineage"],
            ))
        res.add_estimate(Estimate(
            quantity="tokenizer.chat_template_overhead", value=int(base),
            level="observed", unit="tokens", identifiability=STATISTICAL,
            method="prompt_tokens of an empty single-message request",
        ))

        res.shared["tokenizer"] = {
            "family": winner, "count_fn": winner, "type": wtype,
            "vocab_size": wvocab, "chat_template_overhead": int(base),
            "tied_with": ties[1:],
        }
        res.notes.append(f"chat_template_overhead (empty baseline) = {base} tokens")
        res.n_requests = ctx.n_requests
        return res

    # ================================================================== decision models
    async def _decision_tokens(self, ctx: ProbeContext, state: str) -> int | None:
        r = await ctx.decide(DecisionRequest(
            state=state, questions={"q": DecisionQuestion(type="noul", instructions=_NOUL_Q)}))
        if not r.ok or r.usage is None or r.usage.prompt_tokens is None:
            return None
        return int(r.usage.prompt_tokens)

    async def _run_decision(self, ctx: ProbeContext) -> ProbeResult:  # noqa: C901
        """Usage-counter token-delta method (research/jev.md sections 3 and 7.1):
        delta_i = input_tokens(state=s_i) - input_tokens(state='') with a fixed noul."""
        res = self.result()
        reg = TokenizerRegistry.load()
        candidates = reg.candidates()
        battery = list(_DEC_FAST) if ctx.fast() else list(_DEC_FAST) + list(_DEC_FULL_EXTRA)
        by_name = dict(battery)

        deltas: dict[str, int] = {}
        base: int | None = None
        try:
            base = await self._decision_tokens(ctx, "")
            if base is None:
                res.status = "skipped"
                res.notes.append("decision usage.input_tokens not reported; cannot measure deltas")
                res.n_requests = ctx.n_requests
                return res
            for name, st in battery:
                pt = await self._decision_tokens(ctx, st)
                if pt is not None:
                    deltas[name] = pt - base
        except BudgetExhausted as e:
            res.status = "partial"
            res.notes.append(f"budget exhausted: {e}")
        res.n_requests = ctx.n_requests
        if base is None or len(deltas) < 4:
            res.status = "partial" if base is not None else "skipped"
            res.notes.append(f"too few deltas measured ({len(deltas)})")
            return res

        n = len(deltas)
        res.tables["deltas"] = [{"probe": k, "string": repr(by_name[k])[:48], "delta": v}
                                for k, v in deltas.items()]

        # ---- compare against every loadable tokenizer (raw and JSON-escaped) ----------
        scores: dict[str, dict[str, Any]] = {}
        per_string: dict[str, dict[str, int]] = {}
        for fam in candidates:
            enc = reg.get_encoder(fam)
            if enc is None:
                continue
            best: dict[str, Any] | None = None
            for variant, tf in (("raw", lambda x: x), ("json_escaped", _json_escaped)):
                exact = l1 = 0
                counts: dict[str, int] = {}
                for name, dv in deltas.items():
                    try:
                        c = _count(enc, tf(by_name[name]))
                    except Exception:  # noqa: BLE001
                        continue
                    counts[name] = c
                    l1 += abs(c - dv)
                    exact += int(c == dv)
                cand = {"exact": exact, "l1": l1, "variant": variant, "counts": counts}
                if best is None or (exact, -l1) > (best["exact"], -best["l1"]):
                    best = cand
            if best is None:
                continue
            entry = reg.get(fam) or {}
            best.update(vocab=entry.get("vocab_size"), type=entry.get("type"))
            scores[fam] = best
            per_string[fam] = best["counts"]
        if not scores:
            res.status = "partial"
            res.notes.append("no candidate tokenizer could be scored")
            return res

        ranked, ties = _rank_scores(reg, scores)
        winner, wsc = ranked[0]
        rate = wsc["exact"] / n
        matched = rate >= _MATCH_RATE
        res.tables["tokenizer_scores"] = [
            {"family": f, "exact": sc["exact"], "n": n, "exact_rate": round(sc["exact"] / n, 3),
             "l1": sc["l1"], "serialization": sc["variant"], "vocab_size": sc["vocab"],
             "type": sc["type"], "tied_with_winner": f in ties} for f, sc in ranked]
        if len(ties) > 1:
            res.notes.append(f"tokenizer tie: {', '.join(ties)} give identical counts on this "
                             f"battery (indistinguishable); canonical: {winner}")
        # side-by-side comparison with the closest few tokenizers
        top3 = [f for f, _ in ranked[:3]]
        res.tables["delta_vs_candidates"] = [
            {"probe": k, "observed": v, **{f: per_string[f].get(k) for f in top3}}
            for k, v in deltas.items()]
        feats = _features(deltas, by_name)
        # the same features under the closest public tokenizer, for contrast
        closest_counts = per_string.get(winner, {})
        for row in feats:
            if row["feature"] == "digit splitting" and "digits10" in closest_counts:
                row[f"under {winner}"] = f"{closest_counts['digits10']} tokens"
        res.tables["features"] = feats
        res.series["token_deltas"] = {
            "x": list(deltas.keys()), "y": list(deltas.values()),
            "xlabel": "discriminator string", "ylabel": "billed token delta", "kind": "bar"}

        n_cand = len(scores)
        common_alts = ["billing tokenizer differs from the model's tokenizer",
                       "fixed template merges at the state boundary perturb some deltas",
                       "candidate set limited to currently-loadable tokenizers "
                       f"({n_cand} scored)"]
        tau = max(1.0, 0.6 * n)
        if matched:
            lk = {fam: max(0.02, math.exp(-sc["l1"] / tau)) for fam, sc in scores.items()}
            for f in ties:                       # tied families are indistinguishable here
                lk[f] = lk[winner]
            lk["unknown_custom"] = 0.05
            res.add_evidence(Evidence(
                property="tokenizer.family", likelihoods=lk,
                summary=(f"billed-token deltas match {winner} ({wsc['exact']}/{n} exact, "
                         f"L1 {wsc['l1']}, {wsc['variant']} serialization)"),
                level="inferred", strength=0.9,
                stats={"exact": wsc["exact"], "n": n, "exact_rate": rate, "l1": wsc["l1"],
                       "candidates": n_cand, "serialization": wsc["variant"],
                       "tied_with": ties[1:]},
                alternatives=common_alts + ["family shares a base vocabulary with another"],
            ))
            res.add_estimate(Estimate(
                quantity="tokenizer.family", value=winner, level="inferred",
                method="usage-counter token-delta vector match (decision state field)",
                identifiability=STATISTICAL,
                assumptions=["the fixed template is additive (empty-state baseline cancels it)",
                             "billing tokenizer == model tokenizer"],
                alternatives=[f for f, _ in ranked[1:3]]))
            if wsc.get("vocab"):
                res.add_estimate(Estimate(
                    quantity="dims.vocab_size", value=int(wsc["vocab"]), level="estimated",
                    unit="tokens", method="vocab of the exactly-matching tokenizer family",
                    identifiability=WEAK,
                    assumptions=["true vocab equals the matched tokenizer's vocab"]))
            if wsc.get("type"):
                res.add_estimate(Estimate(
                    quantity="tokenizer.type", value=wsc["type"], level="inferred",
                    method="registry record of best-matching family", identifiability=STATISTICAL))
            family_out = winner
        else:
            # No public tokenizer reproduces the deltas: spread mass, favour unknown_custom.
            lk = {fam: max(0.03, 0.5 * (sc["exact"] / n) ** 2) for fam, sc in scores.items()}
            lk["unknown_custom"] = 1.0
            res.add_evidence(Evidence(
                property="tokenizer.family", likelihoods=lk,
                summary=(f"no public tokenizer matches the billed-token deltas: closest "
                         f"{winner} only {wsc['exact']}/{n} exact ({rate:.0%}) -> custom or "
                         f"pruned vocabulary"),
                level="inferred", strength=0.7,
                stats={"exact": wsc["exact"], "n": n, "exact_rate": rate, "l1": wsc["l1"],
                       "candidates": n_cand, "closest": winner,
                       "threshold": _MATCH_RATE},
                alternatives=common_alts + ["the true tokenizer exists publicly but is not in "
                                            "the loadable candidate set"],
            ))
            res.add_estimate(Estimate(
                quantity="tokenizer.family", value="unknown_custom", level="inferred",
                method=(f"token-delta vector vs {n_cand} public tokenizers; best exact-match rate "
                        f"{rate:.0%} < {_MATCH_RATE:.0%}"),
                identifiability=WEAK,
                assumptions=["billing tokenizer == model tokenizer",
                             "candidate set covers the likely public tokenizers"],
                alternatives=[f"closest: {f} ({sc['exact']}/{n} exact)" for f, sc in ranked[:3]]))
            res.notes.append(f"custom or pruned vocabulary; closest: {winner} "
                             f"({wsc['exact']}/{n} exact)")
            family_out = "unknown_custom"

        res.add_estimate(Estimate(
            quantity="tokenizer.chat_template_overhead", value=int(base), level="observed",
            unit="tokens", identifiability=STATISTICAL,
            method=f"input_tokens with state='' and one minimal noul ({_NOUL_Q!r})",
            assumptions=["includes the minimal question's own tokens and wrapper"]))

        digit = next((r["observed"] for r in feats if r["feature"] == "digit splitting"), None)
        res.shared["tokenizer"] = {
            "family": family_out, "count_fn": winner if matched else None,
            "closest": winner, "closest_exact": wsc["exact"], "n": n,
            "type": wsc.get("type") if matched else None,
            "vocab_size": wsc.get("vocab") if matched else None,
            "chat_template_overhead": int(base), "digit_splitting": digit,
            "serialization": wsc["variant"], "tied_with": ties[1:],
        }
        res.notes.append(f"template overhead (empty state + minimal noul) = {base} tokens; "
                         f"best public match {winner} {wsc['exact']}/{n} exact "
                         f"({wsc['variant']}); digits: {digit}")
        return res
