"""Offline test double only; never initializes Gemini/Qdrant."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.benchmark120_hybrid_capture import capture_hybrid


def sample(tmp_path):
    d = tmp_path / 'dataset.jsonl'
    c = tmp_path / 'chunks.jsonl'
    rows = [
        {'question_id': 'X-1', 'question':'Relevant policy?', 'split':'dev', 'answerable':True},
        {'question_id': 'X-2', 'question':'Another policy?', 'split':'dev', 'answerable':True},
        {'question_id': 'X-3', 'question':'Held-out policy?', 'split':'test', 'answerable':True},
    ]
    d.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    c.write_text(''.join(json.dumps({'chunk_id':x,'text':'text'})+'\n' for x in ['c1','c2']))
    return d,c,rows,tmp_path/'checkpoint.jsonl'


def test_preview_does_not_initialize_providers_or_write(tmp_path):
    d,c,qs,p=sample(tmp_path)
    def forbid():
        pytest.fail('Preview must never construct retriever')
    r=capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs, retriever_factory=forbid)
    assert r['pending']==2 and r['newly_saved']==0 and not p.exists()


def test_live_calls_require_explicit_opt_in(tmp_path):
    d,c,qs,p=sample(tmp_path)
    with pytest.raises(ValueError, match='confirm-live'):
        capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1)
    with pytest.raises(ValueError,match='unlock-test'):
        capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,split='test',max_new=1,confirm_live=True)


def test_bounded_resumable_strict_capture(tmp_path):
    d,c,qs,p=sample(tmp_path)
    calls=[]
    def factory():
        calls.append('constructed')
        def retrieve(**kwargs):
            calls.append(kwargs)
            return [{'chunk_id':'c2'},{'chunk_id':'c1'}]
        return SimpleNamespace(retrieve=retrieve,close=lambda: calls.append('closed'))
    first=capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
    assert first['newly_saved']==1 and first['pending']==1
    assert calls[1]['query']=='Relevant policy?' and calls[1]['fused_limit']==20
    assert len(p.read_text().splitlines())==1
    second=capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
    assert second['newly_saved']==1 and second['pending']==0
    assert calls[4]['query']=='Another policy?'
    assert len(p.read_text().splitlines())==2
    third=capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
    assert third['newly_saved']==0 and calls.count('constructed')==2


def test_stale_dataset_and_limits_are_rejected(tmp_path):
    d,c,qs,p=sample(tmp_path)
    factory=lambda: SimpleNamespace(retrieve=lambda **kwargs: [{'chunk_id':'c1'}],close=lambda:None)
    capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
    with pytest.raises(ValueError,match='Stale'):
        capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,fused_limit=5)
    d.write_text(d.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'):
        capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs)


def test_provider_error_preserves_checkpoint(tmp_path):
    d,c,qs,p=sample(tmp_path)
    counter=[0]
    def fail_after_first():
        def retrieve(**kwargs):
            counter[0]+=1
            if counter[0]>1:
                raise RuntimeError('temporary provider failure')
            return [{'chunk_id':'c1'}]
        return SimpleNamespace(retrieve=retrieve,close=lambda:None)
    with pytest.raises(RuntimeError,match='prior checkpoint entries preserved'):
        capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=2,confirm_live=True,retriever_factory=fail_after_first)
    assert len(p.read_text().splitlines())==1
    report=capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs)
    assert report['checkpointed']==1 and report['pending']==1


def test_duplicate_and_unknown_results_rejected(tmp_path):
    d,c,qs,p=sample(tmp_path)
    for bad in ([{'chunk_id':'c1'},{'chunk_id':'c1'}],[{'chunk_id':'absent'}]):
        factory=lambda: SimpleNamespace(retrieve=lambda **kwargs:bad,close=lambda:None)
        with pytest.raises(ValueError,match='Invalid'):
            capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
        assert not p.exists()


def test_checkpoint_scoring_compatible(tmp_path):
    d,c,qs,p=sample(tmp_path)
    factory=lambda: SimpleNamespace(retrieve=lambda **kwargs: [{'chunk_id':'c1'}],close=lambda:None)
    capture_hybrid(dataset_path=d,corpus_path=c,checkpoint_path=p,questions=qs,max_new=1,confirm_live=True,retriever_factory=factory)
    row=json.loads(p.read_text())
    assert row['retrieved_chunk_ids']==['c1'] and row['retrieval_mode']=='strict_hybrid'
    assert row['split']=='dev' and row['question_id']==qs[0]['question_id']
