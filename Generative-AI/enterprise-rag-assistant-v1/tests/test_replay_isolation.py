import json
from types import SimpleNamespace

import pytest

from test_benchmark120_answer_replay import sample, live_kws
from src.evaluation.benchmark120_answer_replay import run, report
from src.evaluation.benchmark120_evidence import sha256


def test_resume_rejects_legacy_config_without_mutation(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    row = json.loads(opts['answers_path'].read_text())
    row.pop('experiment')
    row.pop('experiment_sha256')
    opts['answers_path'].write_text(json.dumps(row)+'\n')
    before = opts['answers_path'].read_bytes()
    assert report(**opts)['answer_checkpointed'] == 1
    with pytest.raises(ValueError, match='legacy generation configuration'):
        run(**live_kws(opts))
    assert opts['answers_path'].read_bytes() == before


def test_changed_context_config_refuses_resume(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    with pytest.raises(ValueError, match='configuration'):
        run(**{**live_kws(opts),'context_builder_factory':lambda:SimpleNamespace(max_chunks=2)})


def test_no_overwrite_sources_and_no_competing_writer(tmp_path):
    opts = sample(tmp_path)
    for name in ('dataset_path','corpus_path','retrieval_path'):
        before = opts[name].read_bytes()
        with pytest.raises(ValueError, match='separate'):
            run(**{**live_kws(opts),'answers_path':opts[name]})
        assert opts[name].read_bytes() == before
    lock = opts['answers_path'].with_suffix('.jsonl.lock')
    lock.write_text('another evaluator')
    with pytest.raises(ValueError, match='locked'):
        run(**live_kws(opts))
    assert lock.read_text() == 'another evaluator'


def test_empty_evidence_abstention_roundtrip_uses_zero_requests(tmp_path):
    opts = sample(tmp_path)
    rows = [json.loads(line) for line in opts['retrieval_path'].read_text().splitlines()]
    rows[0]['retrieved_chunk_ids'] = []
    opts['retrieval_path'].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    class NoCall:
        model_name = 'no-call'
        def __init__(self, **kw): pass
        def generate(self, **kw): pytest.fail('Empty evidence must never invoke the provider')
    out = run(**opts,max_new=1,confirm_generation=True,generator_factory=NoCall)
    assert out['newly_saved'] == 1 and out['provider_requests_used'] == 0
    assert report(**opts)['answer_checkpointed'] == 1
    assert json.loads(opts['answers_path'].read_text())['model'] is None


def test_tampered_citation_text_rejected(tmp_path):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    row = json.loads(opts['answers_path'].read_text())
    row['answer'] = 'Answer [Evidence 9]'
    opts['answers_path'].write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='citation syntax'):
        report(**opts)


@pytest.mark.parametrize('field', ['page_start','document_name','section_title'])
def test_tampered_evidence_metadata_rejected(tmp_path,field):
    opts = sample(tmp_path)
    run(**live_kws(opts))
    row = json.loads(opts['answers_path'].read_text())
    row['selected_evidence'][0][field] = 'invented'
    opts['answers_path'].write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='locked corpus'):
        report(**opts)


def test_failed_generation_records_category_and_releases_lock(tmp_path):
    from src.generation.reliability import GenerationError
    opts = sample(tmp_path)
    def fail(**kw): raise GenerationError('quota','Safe quota error')
    with pytest.raises(RuntimeError,match='quota'):
        run(**{**live_kws(opts),'compose_fn':fail})
    assert not opts['answers_path'].exists()
    assert not opts['answers_path'].with_suffix('.jsonl.lock').exists()
    event = json.loads(opts['answers_path'].with_suffix('.jsonl.attempts.jsonl').read_text())
    assert event['category'] == 'quota'


def test_cli_does_not_print_provider_exception_chain(monkeypatch,capsys):
    from scripts import evaluate_benchmark120_answers as cli
    def fail(**kw):
        try: raise ValueError('provider secret-key URL')
        except ValueError as cause: raise RuntimeError('provider secret-key') from cause
    monkeypatch.setattr(cli,'run',fail)
    with pytest.raises(SystemExit) as caught:
        cli.main(['preview','--chunks','unused'])
    assert caught.value.code == 2
    output = capsys.readouterr().err
    assert 'secret-key' not in output and 'Traceback' not in output
