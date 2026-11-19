"""Offline-only generation replay tests: no real model/network is constructed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.benchmark120_answer_replay import run, report, load_inputs
from src.evaluation.benchmark120_hybrid_capture import _question_digest
from src.evaluation.benchmark120_evidence import sha256


def sample(tmp_path):
    d = tmp_path / 'questions.jsonl'
    c = tmp_path / 'chunks.jsonl'
    r = tmp_path / 'rankings.jsonl'
    a = tmp_path / 'answers.jsonl'
    rows = [
        {'question_id': 'X1', 'split': 'dev', 'answerable': True, 'question': 'Policy one?',
         'expected_documents': [], 'evidence_documents': [], 'expected_facts': ['never put this in prompt']},
        {'question_id': 'X2', 'split': 'dev', 'answerable': False, 'question': 'Unknown rule?',
         'expected_documents': [], 'evidence_documents': [], 'expected_facts': ['not answerable']},
        {'question_id': 'X3', 'split': 'test', 'answerable': True, 'question': 'Heldout?',
         'expected_documents': [], 'evidence_documents': [], 'expected_facts': ['secret']},
    ]
    d.write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')
    c.write_text(''.join(json.dumps({'chunk_id': k, 'text': f'chunk {k}'})+'\n' for k in ('A', 'B')), encoding='utf-8')
    ds, cs = sha256(d), sha256(c)
    r.write_text(''.join(json.dumps({
        'question_id': x['question_id'], 'split': x['split'],
        'question_sha256': _question_digest(x['question']),
        'source_dataset_sha256': ds, 'source_corpus_sha256': cs,
        'retrieval_mode': 'strict_hybrid',
        'retrieval_limits': {'dense_limit': 20, 'sparse_limit': 20, 'fused_limit': 20},
        'retrieved_chunk_ids': ['B', 'A'],
    }) + '\n' for x in rows if x['split'] == 'dev'), encoding='utf-8')
    return dict(dataset_path=d, corpus_path=c, retrieval_path=r, answers_path=a)


class DummyGenerator:
    model_name = 'fake-model'
    def __init__(self, model_name=None):
        self.model_name = model_name or 'fake-model'


def fake_compose(**kwargs):
    assert kwargs['retrieval_source'] == 'saved_hybrid_checkpoint'
    assert kwargs['retrieval_mode'] == 'hybrid'
    assert len(kwargs['candidates']) == 2
    assert kwargs['candidates'][0]['chunk_id'] == 'B'
    assert 'expected_facts' not in kwargs
    evidence = [{'chunk_id': 'B', 'evidence_id': 1, 'text': 'chunk B'}]
    return {
        'retrieval_source': 'saved_hybrid_checkpoint', 'retrieval_mode': 'hybrid',
        'answer': 'answer [Evidence 1]', 'model': kwargs['generator'].model_name,
        'citations': [{'evidence_id': 1, 'chunk_id': 'B'}], 'citation_count': 1,
        'evidence_count': 1, 'evidence_tokens': 2,
        'usage': {}, 'latency': {'retrieval_seconds': None},
        'evaluation_trace': {'selected_evidence': evidence},
    }


def live_kws(opts):
    return {**opts, 'max_new': 1, 'confirm_generation': True,
            'generator_factory': DummyGenerator,
            'context_builder_factory': lambda: object(),
            'citation_builder_factory': lambda: object(), 'compose_fn': fake_compose}


def test_preview_is_offline_and_no_file_write(tmp_path):
    opts = sample(tmp_path)
    def fail(**kw): pytest.fail('Preview must not construct a generator')
    out = run(**opts, generator_factory=fail)
    assert out['retrieval_checkpointed'] == 2 and out['answer_pending'] == 2
    assert not out['live_generation_enabled'] and out['new_embedding_calls'] == 0
    assert not opts['answers_path'].exists()


def test_explicit_opt_in_required(tmp_path):
    opts = sample(tmp_path)
    with pytest.raises(ValueError, match='confirm-generation'):
        run(**opts, max_new=1)
    with pytest.raises(ValueError, match='held out'):
        run(**opts, split='test')


def test_one_answer_capture_and_resume(tmp_path):
    opts = sample(tmp_path)
    res = run(**live_kws(opts))
    assert res['newly_saved'] == 1 and res['answer_pending'] == 1
    assert len(opts['answers_path'].read_text().splitlines()) == 1
    assert run(**opts)['answer_checkpointed'] == 1
    res = run(**live_kws(opts))
    assert res['newly_saved'] == 1 and res['answer_checkpointed'] == 2
    assert run(**opts)['answer_pending'] == 0
    assert run(**live_kws(opts))['newly_saved'] == 0


def test_source_hash_and_question_mismatch_are_rejected(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    opts['corpus_path'].write_text(opts['corpus_path'].read_text() + '\n')
    with pytest.raises(ValueError, match='invalid strict-hybrid checkpoint'):
        run(**opts)


def test_failed_generation_preserves_progress(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    def raise_error(**kw):
        raise RuntimeError('quota')
    with pytest.raises(RuntimeError, match='previous answers preserved'):
        run(**{**live_kws(opts), 'compose_fn': raise_error})
    assert len(opts['answers_path'].read_text().splitlines()) == 1


def test_generation_model_mixing_rejected(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    with pytest.raises(ValueError, match='Different generation model'):
        run(**{**live_kws(opts), 'model_name': 'other-model'})
    assert len(opts['answers_path'].read_text().splitlines()) == 1


def test_export_review_is_unreviewed_and_refuses_overwrite(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    p = tmp_path / 'review.jsonl'
    info = report(**opts, review_export=p)
    assert info['answer_checkpointed'] == 1
    row = json.loads(p.read_text().splitlines()[0])
    assert row['factual_completeness'] == 'unreviewed' and row['groundedness'] == 'unreviewed'
    assert row['generated_answer'] == 'answer [Evidence 1]'
    with pytest.raises(ValueError, match='refusing to overwrite'):
        report(**opts, review_export=p)


def test_reject_tampered_answer_and_unknown_chunk(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    x = json.loads(opts['answers_path'].read_text())
    x['source_retrieval_sha256'] = 'tampered'
    opts['answers_path'].write_text(json.dumps(x) + '\n')
    with pytest.raises(ValueError, match='Stale'):
        run(**opts)


def test_reject_answers_saved_to_retrieval_path(tmp_path):
    opts = sample(tmp_path)
    opts['answers_path'] = opts['retrieval_path']
    with pytest.raises(ValueError, match='separate'):
        run(**opts)


def test_real_production_context_and_composer_with_fake_generator(tmp_path):
    """In the full repository, exercise exact production context+citations offline."""
    ContextBuilder = pytest.importorskip('src.retrieval.context_builder').ContextBuilder
    CitationBuilder = pytest.importorskip('src.generation.citation_builder').CitationBuilder
    compose_answer = pytest.importorskip('src.rag.answer_composer').compose_answer
    opts = sample(tmp_path)
    class StubGenerator:
        model_name = 'stub-only'
        def __init__(self, model_name=None):
            pass
        def generate(self, question, context):
            assert 'chunk B' in context
            assert 'never put this in prompt' not in context
            return {'answer': 'Grounded stub answer [Evidence 1]',
                    'usage': {}, 'model': self.model_name}
    result = run(**{**opts, 'max_new': 1, 'confirm_generation': True,
                    'generator_factory': StubGenerator,
                    'context_builder_factory': ContextBuilder,
                    'citation_builder_factory': CitationBuilder,
                    'compose_fn': compose_answer})
    assert result['newly_saved'] == 1
    row = json.loads(opts['answers_path'].read_text())
    assert row['retrieval_source'] == 'saved_hybrid_checkpoint'
    assert row['selected_evidence'][0]['chunk_id'] == 'B'
    assert row['citations'][0]['chunk_id'] == 'B'
    assert report(**opts)['answers_with_no_citation_metadata'] == 0
