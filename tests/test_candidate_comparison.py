"""Candidate comparison preserves full distributions and distinct score/value changes."""
from forensics.core.types import Capabilities
from forensics.inference.run_types import RunResult
from forensics.probes.base import Evidence
from forensics.inference.architecture import compare_architecture_candidates
from forensics.core.types import to_jsonable
from forensics.web import app as web
from fastapi.testclient import TestClient
import json


def model(score):
    return RunResult('test',{'name':'opaque'},Capabilities(kind='generative'),
        prior={'arch.family':{'decoder_only':.6,'encoder_decoder':.4}},
        posterior={'arch.family':{'decoder_only':score,'encoder_decoder':1-score}},
        evidence=[Evidence('arch.family',{'decoder_only':1,'encoder_decoder':.2},'generation',probe_id='P19')])


def test_same_leading_not_same_distribution():
    row=next(r for r in compare_architecture_candidates(model(.97),model(.76)) if r['key']=='arch.family')
    assert row['same_leading']
    assert row['a']['leading']=='decoder_only'
    assert row['a']['candidates'][0]['support_score']==.97
    assert row['b']['candidates'][0]['support_score']==.76
    assert len(row['candidates'])==2
    assert abs(sum(c['a_score'] for c in row['candidates'])-1)<1e-10


def test_prior_flag_and_missing_channel_not_zero():
    a=model(.6)
    a.evidence=[]
    b=RunResult('other',{},Capabilities(kind='decision'))
    row=next(r for r in compare_architecture_candidates(a,b) if r['key']=='arch.family')
    assert row['a']['prior_only']
    assert row['b'] is None
    assert row['candidates'][0]['b_score'] is None


def test_web_compares_candidates_and_marks_same_value(tmp_path,monkeypatch):
    monkeypatch.setattr(web,'RUNS_DIR',tmp_path)
    web._SUMMARY_CACHE.clear();web._DETAIL_CACHE.clear()
    for name,r in [('a',model(.97)),('b',model(.76))]:
        p=tmp_path/name;p.mkdir();(p/'run_result.json').write_text(json.dumps(to_jsonable(r)),encoding='utf-8')
    client=TestClient(web.app)
    result=client.get('/api/compare',params={'a':'a','b':'b'}).json()
    old=next(r for r in result['rows'] if r['key']=='arch')
    assert old['same'] and not old['scores_equal']
    row=next(r for r in result['candidate_comparison'] if r['key']=='arch.family')
    assert row['same_leading']
    assert 'not calibrated' in result['score_semantics']
    detail=client.get('/api/runs/a/detail').json()
    assert detail['architecture']['objective'].startswith('partial functional architecture')
    assert 'probabilities' in detail['score_semantics']
