"""Offline request-budget and evaluator-resume regression tests (no provider use)."""
import json

import pytest

from src.evaluation.experiment import append_event, restore_request_budget
from src.evaluation.benchmark120_answer_replay import run
from test_benchmark120_answer_replay import sample, live_kws, fake_compose


def start(path, run_id, *, experiment='abc', question='Q1', attempt=1):
    append_event(path, {
        'event': 'started', 'experiment_sha256': experiment,
        'run_id': run_id, 'question_id': question, 'attempt': attempt,
    })


def test_persistent_cap_counts_completed_and_unfinished_attempts(tmp_path):
    path = tmp_path / 'answers.jsonl.attempts.jsonl'
    args = dict(experiment_sha256='abc', requested_limit=3,
                default_batch_limit=1, saved_provider_answers=0)
    budget, prior, scope = restore_request_budget(path, **args)
    assert (budget.limit, budget.used, budget.remaining, prior, scope) == (3, 0, 3, 0, 'persistent')
    start(path, 'first')
    append_event(path, {'event': 'failed', 'experiment_sha256': 'abc',
                        'run_id': 'first', 'question_id': 'Q1', 'attempt': 1})
    budget, prior, scope = restore_request_budget(path, **args)
    assert (budget.used, budget.remaining, prior) == (1, 2, 1)
    start(path, 'crashed', question='Q2')  # No completion event: still spent.
    budget, prior, scope = restore_request_budget(path, **args)
    assert (budget.used, budget.remaining, prior) == (2, 1, 2)
    start(path, 'last', question='Q3')
    budget, prior, scope = restore_request_budget(path, **args)
    assert (budget.used, budget.remaining, prior) == (3, 0, 3)
    cap = path.with_suffix(path.suffix + '.budget.json')
    assert json.loads(cap.read_text())['limit'] == 3


def test_stored_cap_cannot_be_silently_changed_or_bypassed(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    fields = dict(experiment_sha256='abc', default_batch_limit=1, saved_provider_answers=0)
    restore_request_budget(path, requested_limit=2, **fields)
    start(path, 'first')
    resumed, _, scope = restore_request_budget(path, requested_limit=None, **fields)
    assert (resumed.limit, resumed.used, resumed.remaining, scope) == (2, 1, 1, 'persistent')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='refusing to change'):
        restore_request_budget(path, requested_limit=3, **fields)
    assert path.read_bytes() == before


def test_prior_legacy_usage_counted_when_explicit_cap_is_set(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    start(path, 'legacy')
    budget, _, scope = restore_request_budget(
        path, experiment_sha256='abc', requested_limit=1,
        default_batch_limit=2, saved_provider_answers=1)
    assert scope == 'persistent' and budget.remaining == 0


def test_legacy_default_retains_invocation_semantics(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    start(path, 'legacy')
    budget, _, scope = restore_request_budget(
        path, experiment_sha256='abc', requested_limit=None,
        default_batch_limit=1, saved_provider_answers=1)
    assert (budget.limit, budget.used, budget.remaining, scope) == (2, 1, 1, 'invocation')
    assert not path.with_suffix(path.suffix + '.budget.json').exists()


def test_missing_history_cannot_justify_new_cumulative_cap(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    with pytest.raises(ValueError, match='does not account'):
        restore_request_budget(path, experiment_sha256='abc', requested_limit=3,
                               default_batch_limit=1, saved_provider_answers=1)
    assert not path.with_suffix(path.suffix + '.budget.json').exists()


def test_corrupt_or_cross_experiment_history_fails_closed(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    kwargs = dict(experiment_sha256='abc', requested_limit=2,
                  default_batch_limit=1, saved_provider_answers=0)
    start(path, 'one', experiment='unrelated')
    with pytest.raises(ValueError, match='experiment mismatch'):
        restore_request_budget(path, **kwargs)
    path.write_text('{broken\n')
    with pytest.raises(ValueError, match='Corrupted'):
        restore_request_budget(path, **kwargs)
    path.write_text('')
    start(path, 'dup')
    start(path, 'dup')
    with pytest.raises(ValueError, match='duplicate request start'):
        restore_request_budget(path, **kwargs)


def test_start_count_exceeding_recorded_cap_fails_closed(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    fields = dict(experiment_sha256='abc', requested_limit=1,
                  default_batch_limit=1, saved_provider_answers=0)
    restore_request_budget(path, **fields)
    start(path, 'one')
    start(path, 'two')
    with pytest.raises(ValueError, match='exceed'):
        restore_request_budget(path, **fields)


def test_evaluator_resume_consumes_one_persistent_cap(tmp_path):
    opts = sample(tmp_path)
    calls = []

    def counted_compose(**kwargs):
        calls.append(kwargs['question'])
        gen = kwargs['generator']
        gen.request_budget.consume()
        gen.observer({'event': 'started', 'attempt': 1})
        gen.observer({'event': 'succeeded', 'attempt': 1})
        return fake_compose(**kwargs)

    opts = {**live_kws(opts), 'request_budget': 1, 'compose_fn': counted_compose}
    first = run(**opts)
    assert first['provider_requests_used'] == 1
    assert first['provider_requests_cumulative'] == 1
    assert first['provider_request_budget_scope'] == 'persistent'
    assert first['answer_pending'] == 1 and len(calls) == 1
    resumed = run(**opts)
    assert resumed['newly_saved'] == 0
    assert resumed['provider_requests_used'] == 0
    assert resumed['provider_requests_remaining'] == 0
    assert len(calls) == 1
    implicit = run(**{**opts, 'request_budget': None})
    assert implicit['newly_saved'] == 0 and len(calls) == 1
    with pytest.raises(ValueError, match='refusing to change'):
        run(**{**opts, 'request_budget': 2})
    assert len(calls) == 1


def test_evaluator_inflight_attempt_counts_on_resume(tmp_path):
    opts = sample(tmp_path)
    calls = []

    def metered_compose(**kw):
        calls.append(1)
        kw['generator'].request_budget.consume()
        kw['generator'].observer({'event': 'started', 'attempt': 1})
        return fake_compose(**kw)

    invocation = {**live_kws(opts), 'request_budget': 2, 'compose_fn': metered_compose}
    first = run(**invocation)
    assert first['provider_requests_cumulative'] == 1
    ledger = opts['answers_path'].with_suffix('.jsonl.attempts.jsonl')
    row = json.loads(opts['answers_path'].read_text().splitlines()[0])
    start(ledger, 'interrupted-other-process', experiment=row['experiment_sha256'], question='X2')
    resumed = run(**invocation)
    assert resumed['provider_requests_cumulative'] == 2
    assert resumed['provider_requests_remaining'] == 0
    assert resumed['newly_saved'] == 0 and len(calls) == 1


def test_explicit_budget_extension_preserves_ledger_and_records_history(tmp_path):
    path = tmp_path / 'attempts.jsonl'
    fields = dict(experiment_sha256='abc', default_batch_limit=3, saved_provider_answers=0)
    restore_request_budget(path, requested_limit=5, **fields)
    for n in range(5):
        start(path, f'before-{n}')
    ledger_original = path.read_bytes()
    with pytest.raises(ValueError, match='refusing to change'):
        restore_request_budget(path, requested_limit=8, **fields)
    with pytest.raises(ValueError, match='refusing to change'):
        restore_request_budget(
            path, requested_limit=4, allow_budget_extension=True, **fields
        )
    restored, used, scope = restore_request_budget(
        path, requested_limit=8, allow_budget_extension=True, **fields
    )
    assert (restored.limit, used, restored.remaining, scope) == (8, 5, 3, 'persistent')
    assert path.read_bytes() == ledger_original
    cap_file = path.with_suffix(path.suffix + '.budget.json')
    metadata = json.loads(cap_file.read_text())
    assert metadata['limit'] == 8
    assert [(x['from'], x['to']) for x in metadata['extensions']] == [(5, 8)]
    # Rerunning the exact approved command does not append duplicate revisions.
    restore_request_budget(
        path, requested_limit=8, allow_budget_extension=True, **fields
    )
    assert len(json.loads(cap_file.read_text())['extensions']) == 1
    assert restore_request_budget(path, requested_limit=None, **fields)[0].limit == 8
    with pytest.raises(ValueError, match='requires --request-budget'):
        restore_request_budget(
            path, requested_limit=None, allow_budget_extension=True, **fields
        )


def test_original_evaluation_answers_can_continue_after_explicit_extension(tmp_path):
    opts = sample(tmp_path)
    calls = []

    def compose_with_meter(**kwargs):
        generator = kwargs['generator']
        generator.request_budget.consume()
        generator.observer({'event': 'started', 'attempt': 1})
        generator.observer({'event': 'succeeded', 'attempt': 1})
        calls.append(kwargs['question'])
        return fake_compose(**kwargs)

    invocation = {
        **live_kws(opts), 'request_budget': 1, 'compose_fn': compose_with_meter
    }
    first = run(**invocation)
    assert first['newly_saved'] == 1 and first['provider_requests_cumulative'] == 1
    old_answer = opts['answers_path'].read_text().splitlines()[0]
    with pytest.raises(ValueError, match='refusing to change'):
        run(**{**invocation, 'request_budget': 2})
    second = run(**{
        **invocation, 'request_budget': 2, 'extend_request_budget': True
    })
    assert second['newly_saved'] == 1
    assert second['answer_checkpointed'] == 2
    assert second['provider_requests_cumulative'] == 2
    assert opts['answers_path'].read_text().splitlines()[0] == old_answer
    assert len(calls) == 2


def test_request_spacing_delays_second_generation_without_extra_requests(monkeypatch, tmp_path):
    import src.evaluation.benchmark120_answer_replay as replay
    opts = sample(tmp_path)
    slept = []

    monkeypatch.setattr(replay.time, 'monotonic', lambda: 100.0)
    monkeypatch.setattr(replay.time, 'sleep', lambda seconds: slept.append(seconds))
    result = run(**{
        **live_kws(opts), 'max_new': 2, 'request_spacing_seconds': 65.0,
    })
    assert result['newly_saved'] == 2
    assert slept == [65.0]
    assert result['new_embedding_calls'] == 0


def test_request_spacing_rejects_bad_values_offline(tmp_path):
    opts = sample(tmp_path)
    for bad in (-1, 601, '65'):
        with pytest.raises(ValueError, match='Request spacing'):
            run(**opts, request_spacing_seconds=bad)
