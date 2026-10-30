"""Architecture output keeps candidates and proposed edges, without prior-only recovery."""
from pathlib import Path
import json
from forensics.core.types import Capabilities
from forensics.inference.run_types import RunResult
from forensics.probes.base import Evidence,Estimate
from forensics.inference.architecture import architecture_hypothesis_graph
from forensics.inference.certificates import export_certificates


def test_prior_only_components_not_recovered():
    r=RunResult('t',{},Capabilities(kind='generative'),
        prior={'attn.kv_sharing':{'gqa_or_mqa':.6,'mha':.4}},
        posterior={'attn.kv_sharing':{'gqa_or_mqa':.6,'mha':.4}})
    g=architecture_hypothesis_graph(r)
    n=next(n for n in g['nodes'] if n['id']=='attn.kv_sharing')
    assert n['prior_only'] and n['status']=='unresolved'
    assert n['best_supported']=='gqa_or_mqa'
    assert n['candidate_values']==['gqa_or_mqa','mha']


def test_evidence_nodes_retain_ranked_alternatives():
    r=RunResult('t',{},Capabilities(kind='generative'),
        prior={'arch.sparsity':{'moe':.5,'dense':.5}},
        posterior={'arch.sparsity':{'moe':.82,'dense':.18}},
        evidence=[Evidence('arch.sparsity',{'moe':1,'dense':.3},'routing-compatible',probe_id='P11')])
    n=next(n for n in architecture_hypothesis_graph(r)['nodes'] if n['id']=='arch.sparsity')
    assert n['best_supported']=='moe' and not n['prior_only']
    assert n['status']=='suggestive' and len(n['ranked_candidates'])==2
    assert 'probability' in n['score_semantics']


def test_estimated_interface_boundary_is_reconstruction():
    r=RunResult('t',{},Capabilities(kind='generative'),
        estimates=[Estimate('context.max_input_tokens',16393,ci=(16393,16500),level='estimated')])
    n=next(n for n in architecture_hypothesis_graph(r)['nodes'] if n['id']=='context.max_input_tokens')
    assert n['best_supported']==16393 and n['status']=='estimated'
    assert n['evidence_grade']=='L3' and not n['prior_only']


def test_decision_without_evidence_is_not_reconstructed():
    g=architecture_hypothesis_graph(RunResult('t',{},Capabilities(kind='decision')))
    n=next(n for n in g['nodes'] if n['id']=='arch.family')
    assert n['status']=='unresolved'
    assert all(e['status']=='proposed' for e in g['edges'])


def test_template_edges_are_not_measured_wiring():
    g=architecture_hypothesis_graph(RunResult('t',{},Capabilities(kind='generative')))
    assert all(e['status']=='proposed' for e in g['edges'])
    assert all(not e['probe_ids'] for e in g['edges'])


def test_decision_keeps_sparsity_prior_as_a_candidate_not_measurement():
    r=RunResult('t',{},Capabilities(kind='decision'),
        prior={'arch.sparsity':{'dense':.5,'moe':.5}},
        posterior={'arch.sparsity':{'dense':.5,'moe':.5}})
    n=next(n for n in architecture_hypothesis_graph(r)['nodes'] if n['id']=='arch.sparsity')
    assert n['prior_only'] and n['status']=='unresolved'
    assert {c['value'] for c in n['ranked_candidates']}=={'dense','moe'}


def test_architecture_export_with_certificates(tmp_path):
    r=RunResult('t',{},Capabilities(kind='generative'))
    export_certificates(r,tmp_path)
    path=tmp_path/'architecture_hypothesis.json'
    assert path.exists()
    d=json.loads(path.read_text())
    assert d['objective'].startswith('partial functional architecture')
    assert 'nodes' in d and 'edges' in d
