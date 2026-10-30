"""Tests for the knowledge loaders (archdb + tokenizers) over the frozen data.

These load the real data files (198 architectures, 65 tokenizers) and index
them.  Tokenizer tests rely only on the tiktoken encodings bundled with the
``tiktoken`` package, so they pass offline; HuggingFace downloads may fail
without network and that is intentionally tolerated.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from forensics.knowledge.archdb import ArchDB
from forensics.knowledge.tokenizers import TokenizerRegistry


# --------------------------------------------------------------------------- archdb
def test_load_has_expected_count():
    db = ArchDB.load()
    entries = db.all()
    assert 190 <= len(entries) <= 205        # ~198 records
    assert len(entries) == 198
    # caching returns the same instance
    assert ArchDB.load() is db


def test_access_helpers():
    db = ArchDB.load()
    llama = db.get("Meta-Llama-3-8B")
    assert llama is not None
    assert llama["d_model"] == 4096
    assert db.get("META-LLAMA-3-8B") is llama         # case-insensitive
    assert db.get("no-such-model") is None
    fam = db.by_family("llama3")
    assert fam and all(e["family"] == "llama3" for e in fam)
    grouped = db.by_family()
    assert isinstance(grouped, dict) and "llama3" in grouped


def test_filter_ranges_and_exact():
    db = ArchDB.load()
    # exact + range constraints, using measured-style names
    res = db.filter(arch_type="decoder", hidden_size=(4000, 4200), vocab_size=128256)
    assert res, "expected at least one decoder with d_model~4096 and vocab 128256"
    assert all(e["arch_type"] == "decoder" for e in res)
    assert all(4000 <= e["d_model"] <= 4200 for e in res)
    names = {e["name"] for e in res}
    assert any("Llama-3" in n for n in names)
    # open-ended upper bound
    big = db.filter(hidden_size=(12000, None))
    assert big and all(e["d_model"] >= 12000 for e in big)


def test_nearest_ranks_llama3_8b_on_top():
    db = ArchDB.load()
    ranked = db.nearest({"hidden_size": 4096, "vocab_size": 128256, "arch_type": "decoder"})
    assert ranked, "nearest returned nothing"
    top_entry, dist, matched, conflicts = ranked[0]
    # A Llama-3-8B-like entry (d=4096, vocab=128256, decoder) should be the top hit.
    assert top_entry["family"] == "llama3"
    assert top_entry["d_model"] == 4096 and top_entry["vocab_size"] == 128256
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert set(matched) >= {"hidden_size", "vocab_size", "arch_type"}
    assert conflicts == []
    # tuple shape contract
    assert len(ranked[0]) == 4


def test_nearest_ignores_none_fields_and_reports_conflicts():
    db = ArchDB.load()
    # vocab None is ignored; a wrong arch_type should surface as a conflict lower down.
    ranked = db.nearest({"hidden_size": 4096, "vocab_size": None,
                         "arch_type": "decoder", "kv_sharing": "gqa"})
    assert ranked
    top = ranked[0][0]
    assert top["d_model"] == 4096
    # find an encoder-only model in the ranking and confirm arch_type conflict recorded
    enc_rows = [r for r in ranked if r[0].get("arch_type") == "encoder"]
    assert enc_rows and all("arch_type" in r[3] for r in enc_rows)


def test_regressions_finite_and_reproduce_reference():
    db = ArchDB.load()
    reg = db.regressions()
    for key in ("layers_vs_dmodel", "head_dim", "ffn_ratio",
                "params_vs_ldv", "moe_active_ratio", "gqa_ratio_by_era"):
        assert key in reg
        assert reg[key]["n"] > 0
        assert "method" in reg[key]

    def all_finite(obj):
        if isinstance(obj, dict):
            return all(all_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(all_finite(v) for v in obj)
        if isinstance(obj, float):
            return math.isfinite(obj)
        return True

    # every fitted numeric coefficient is finite
    assert all_finite(reg["layers_vs_dmodel"]["coef"])
    assert all_finite(reg["params_vs_ldv"]["coef"])
    assert all_finite(reg["moe_active_ratio"]["coef"])

    # reproduce documented reference fits (research/arch_db.md §2)
    lay = reg["layers_vs_dmodel"]
    assert lay["n"] == 91
    assert lay["coef"]["slope"] == pytest.approx(0.568, abs=0.02)
    assert lay["coef"]["intercept"] == pytest.approx(-1.066, abs=0.05)
    assert lay["resid_std"] == pytest.approx(0.262, abs=0.02)

    par = reg["params_vs_ldv"]
    assert par["coef"]["ln_L"] == pytest.approx(1.104, abs=0.05)
    assert par["coef"]["ln_d"] == pytest.approx(1.870, abs=0.05)
    assert par["resid_std"] == pytest.approx(0.226, abs=0.03)

    # head_dim mode is 128
    assert reg["head_dim"]["coef"]["mode"] == 128
    # non-gated FFN ratio is the classic 4.0
    assert reg["ffn_ratio"]["coef"]["dense_nongated"]["median"] == pytest.approx(4.0, abs=0.01)
    # MoE active fraction shrinks with top_k/E (positive slope in log space)
    assert reg["moe_active_ratio"]["coef"]["slope"] > 0.3


def test_predictors_are_sane():
    db = ArchDB.load()
    med, lo, hi = db.predict_n_layers(4096)
    assert lo < med < hi
    assert 20 < med < 60                       # ~40 layers at d=4096 (see arch_db.md)
    pmed, plo, phi = db.predict_total_params_b(32, 4096, 128256, tied=False)
    assert plo < pmed < phi
    assert 4 < pmed < 20                        # Llama-3-8B ~ 8B


# --------------------------------------------------------------------------- tokenizers
def test_registry_loads_and_lists():
    reg = TokenizerRegistry.load()
    assert len(reg.all()) == 65
    assert TokenizerRegistry.load() is reg
    off = reg.offline_families()
    assert len(off) >= 4                        # bundled tiktoken encodings
    assert any("o200k_base" in f for f in off)


def test_get_encoder_tiktoken_offline():
    reg = TokenizerRegistry.load()
    for name in ("o200k_base", "cl100k_base", "r50k_base", "p50k_base"):
        enc = reg.get_encoder(name)
        assert enc is not None, f"{name} should load offline via tiktoken"
        assert enc.kind == "tiktoken"
        assert isinstance(enc.encode("hello world"), list)
    # o200k splits "hello world" into ~2 subtokens
    o200k = reg.get_encoder("o200k_base")
    assert o200k.count("hello world") == 2


def test_get_encoder_by_family_alias():
    reg = TokenizerRegistry.load()
    # resolve by DB family name, by used-by model, and unknown -> None
    assert reg.get_encoder("tiktoken:o200k_base") is not None
    assert reg.get_encoder("GPT-4o") is not None            # via used_by
    assert reg.get_encoder("definitely-not-a-tokenizer") is None


def test_count_all_includes_bundled_tiktoken():
    reg = TokenizerRegistry.load()
    counts = reg.count_all("hello world 12345")
    # at least the 4 guaranteed bundled tiktoken encodings
    bundled = [f for f in counts if f.startswith("tiktoken:")]
    assert len(bundled) >= 4
    assert all(isinstance(v, int) and v > 0 for v in counts.values())
    # o200k should tokenise this short string into a handful of tokens
    o200k = next(f for f in counts if "o200k_base" in f)
    assert 3 <= counts[o200k] <= 8
    # candidates() is consistent with what count_all reported
    assert set(counts).issubset(set(reg.candidates()))
