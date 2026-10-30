"""Certificates never confuse a missing capability with a formal impossibility proof."""
from __future__ import annotations
import asyncio
import json
from dataclasses import replace
from pathlib import Path
import pytest

from forensics.core.adapters.simulated import SimulatedAdapter, sim_decoder_dense
from forensics.core.types import Capabilities, GenRequest
from forensics.inference.certificates import (build_certificates,canonical_hash,
    certify_nonidentifiability,export_certificates,measurement_integrity)
from forensics.inference.run_types import RunResult, IdentifiabilityEntry
from forensics.inference.engine import _Engine
from forensics.probes.base import Evidence,Estimate,ProbeResult,ProbeContext
from forensics.probes.p02_tokenizer import TokenizerFingerprintProbe


def run(**kwargs):
    return RunResult(run_id='test',target={'name':'opaque'},caps=Capabilities(kind='generative'),**kwargs)


def test_missing_capability_is_unresolved_not_proof():
    r=run(identifiability=[IdentifiabilityEntry('dims.hidden_size',5,'legacy unknown','no logits')])
    c=build_certificates(r)[0]
    assert c['status']=='unresolved' and c['level']=='U'
    assert c['nonidentifiability_witness'] is None


def test_witness_requires_every_scope_field():
    with pytest.raises(ValueError):certify_nonidentifiability('dims.n_layers',{'omega_1':'a','omega_2':'b'})
    witness={'omega_1':'f','omega_2':'f o identity','channels':['behavior'],
             'probe_class':'all admissible inputs','different_property_values':[1,2],
             'argument':'identity insertion leaves the complete output law unchanged'}
    cert=certify_nonidentifiability('dims.n_layers',witness)
    assert cert['status']=='non-identifiable' and cert['level']=='L5'


def test_tokenizer_ties_retained_as_a_set():
    rows=[{'family':'o200k_base','exact':18,'n':18},{'family':'o200k_harmony','exact':18,'n':18}]
    r=run(probe_results=[ProbeResult('P02.tokenizer.fingerprint',tables={'tokenizer_scores':rows})],
          estimates=[Estimate('tokenizer.family','o200k_base',level='inferred')])
    cert=next(c for c in build_certificates(r) if c['property']=='tokenizer.family')
    assert cert['candidate_set']==['o200k_base','o200k_harmony']
    assert cert['set_kind']=='finite-reference-match'


def test_unmatched_does_not_assert_custom_tokenizer():
    rows=[{'family':'closest','exact':7,'n':12}]
    r=run(probe_results=[ProbeResult('P02.tokenizer.fingerprint',tables={'tokenizer_scores':rows})],
          estimates=[Estimate('tokenizer.family','unknown_custom')])
    cert=build_certificates(r)[0]
    assert cert['status']=='unresolved'
    assert 'accounting mismatch' in cert['candidate_set']
    assert len(cert['candidate_set'])>1


def test_prior_only_is_not_evidence():
    r=run(posterior={'arch.family':{'decoder_only':.99,'encoder_only':.01}})
    c=build_certificates(r)[0]
    assert c['status']=='unresolved'
    assert c['candidate_set']==['decoder_only','encoder_only']


def test_mechanism_score_remains_suggestive():
    e=Evidence('arch.sparsity',{'moe':1,'dense':.1},'domain effect',probe_id='P11.moe',strength=.9)
    r=run(evidence=[e],posterior={'arch.sparsity':{'moe':.99,'dense':.01}})
    c=build_certificates(r)[0]
    assert c['status']=='suggestive' and c['level']=='L4'
    assert 'dense' in c['candidate_set']
    assert not c['falsifier_cost_verified']


def test_integrity_labels_do_not_attest_provider():
    assert measurement_integrity('P02.tokenizer')['grade']=='I0'
    assert measurement_integrity('P18.decision')['grade']=='I1'
    assert measurement_integrity('P07.timing')['grade']=='mixed'
    assert not measurement_integrity('P02.tokenizer')['attested']


def test_manifest_hash_and_export(tmp_path):
    r=run(config={'seed':17,'planner':'eig'})
    export_certificates(r,tmp_path)
    data=json.loads((tmp_path/'run_manifest.json').read_text())
    assert data['sha256']==canonical_hash(data['manifest'])
    assert data['manifest']['seed']==17 and data['manifest']['region'] is None
    assert (tmp_path/'forensic_certificates.json').exists()


def test_billing_closure_keeps_usage_dataclass_and_disables_p02():
    adapter=SimulatedAdapter(replace(sim_decoder_dense(),expose_usage=False))
    response=asyncio.run(adapter.generate(GenRequest.chat('test',max_tokens=1)))
    assert response.usage is not None and response.usage.prompt_tokens is None
    assert not adapter.static_capabilities().usage_prompt_tokens
    assert not adapter.static_capabilities().usage_cached_tokens
    assert not adapter.static_capabilities().stream_usage
    assert TokenizerFingerprintProbe().applicable(ProbeContext(adapter))[0] is False


def test_planner_seed_and_invalid_policy():
    adapter=SimulatedAdapter(sim_decoder_dense())
    eng=_Engine(adapter,{'seed':100,'planner':'uniform'},None,None,None,7,None)
    assert eng.seed==7 and eng.config['seed']==7
    with pytest.raises(ValueError,match='planner'):
        _Engine(adapter,{'planner':'invalid'},None,None,None,0,None)
