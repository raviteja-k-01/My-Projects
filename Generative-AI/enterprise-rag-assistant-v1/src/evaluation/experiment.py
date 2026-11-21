"""Local experiment isolation, single-writer protection and safe request ledger."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configuration(generator: Any, context_builder: Any) -> dict[str, Any]:
    src = Path(__file__).resolve().parents[1]
    files = ("generation/answer_generator.py", "generation/reliability.py", "generation/citation_builder.py",
             "retrieval/context_builder.py", "rag/answer_composer.py")
    return {
        "schema": 1,
        "generation": getattr(generator, "configuration", {"model": generator.model_name, "injected": True}),
        "context": {name: getattr(context_builder, name, None) for name in
                    ("max_chunks", "max_context_tokens")},
        "source_sha256": {name: hashlib.sha256((src / name).read_bytes()).hexdigest() for name in files},
        "packages": {name: version(name) for name in ("google-genai", "langchain-google-genai", "langchain-core")},
    }


def exclusive_capture(fn):
    """Fail closed on competing evaluators. After a crash, inspect/remove the lock manually."""
    @wraps(fn)
    def wrapped(**kwargs):
        if not kwargs.get("max_new", 0):
            return fn(**kwargs)
        path = kwargs["answers_path"]
        if any(path.resolve() == kwargs[name].resolve() for name in
               ("dataset_path", "corpus_path", "retrieval_path")):
            raise ValueError("Answer output and source inputs must be separate")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(path.suffix + ".lock")
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError("Answer experiment is locked by another or interrupted evaluator") from exc
        try:
            with handle:
                json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, handle)
            return fn(**kwargs)
        finally:
            lock.unlink()
    return wrapped


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Persist before/after requests; unfinished starts remain visible after interruption."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())



def restore_request_budget(
    ledger_path: Path, *, experiment_sha256: str, requested_limit: int | None,
    default_batch_limit: int, saved_provider_answers: int,
    allow_budget_extension: bool = False,
):
    """Restore attempts from the append-only ledger before any provider request.

    An explicit limit is a *cumulative* cap for this answer checkpoint, across
    restarts. Once written, it cannot be silently increased or bypassed by
    omitting --request-budget. Started-but-unfinished attempts count as spent.

    For compatibility, invocations with no explicit/stored cap retain the old
    per-invocation default; this is NOT a persistent authorization limit.
    """
    from src.generation.reliability import RequestBudget

    if not isinstance(experiment_sha256, str) or not experiment_sha256:
        raise ValueError("A valid experiment fingerprint is required")
    if type(default_batch_limit) is not int or default_batch_limit < 0:
        raise ValueError("Invalid default request limit")
    if type(saved_provider_answers) is not int or saved_provider_answers < 0:
        raise ValueError("Invalid saved-answer count")
    if requested_limit is not None and (type(requested_limit) is not int or requested_limit < 0):
        raise ValueError("Cumulative request budget must be a nonnegative integer")
    if type(allow_budget_extension) is not bool:
        raise ValueError("Budget extension flag must be boolean")
    if allow_budget_extension and requested_limit is None:
        raise ValueError("Budget extension requires --request-budget with a cumulative limit")

    # Keep authorization metadata separate from the attempt ledger so its
    # existing reporting/event schema is not changed.
    cap_path = ledger_path.with_suffix(ledger_path.suffix + ".budget.json")
    stored_limit = None
    stored_cap = None
    if cap_path.exists():
        try:
            cap = json.loads(cap_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Unreadable or corrupted persistent budget metadata") from exc
        if (not isinstance(cap, dict) or cap.get("schema") != 1
            or cap.get("experiment_sha256") != experiment_sha256
            or type(cap.get("limit")) is not int or cap["limit"] < 0):
            raise ValueError("Persistent request budget metadata is invalid or from another experiment")
        stored_limit = cap["limit"]
        stored_cap = cap

    seen_starts = set()
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Corrupted request ledger at line {line_no}") from exc
                if not isinstance(event, dict) or event.get("experiment_sha256") != experiment_sha256:
                    raise ValueError(f"Request ledger experiment mismatch at line {line_no}")
                if event.get("event") == "started":
                    key = (event.get("run_id"), event.get("question_id"), event.get("attempt"))
                    if (not all(isinstance(x, str) and x for x in key[:2])
                        or type(key[2]) is not int or key[2] < 1 or key in seen_starts):
                        raise ValueError(f"Invalid or duplicate request start at line {line_no}")
                    seen_starts.add(key)

    prior_used = len(seen_starts)
    if (stored_limit is not None or requested_limit is not None) and prior_used < saved_provider_answers:
        raise ValueError(
            "Request ledger does not account for existing provider-backed answers; "
            "use a fresh experiment output path rather than guessing prior cost"
        )

    if stored_limit is not None:
        if prior_used > stored_limit:
            raise ValueError("Recorded requests exceed persistent request budget; no request sent")
        if requested_limit is not None and requested_limit != stored_limit:
            if requested_limit < stored_limit or not allow_budget_extension:
                raise ValueError(
                    "Persistent request budget already set; refusing to change it. "
                    "To increase it, explicitly provide --extend-request-budget"
                )
            # The enclosing exclusive_capture lock prevents concurrent writers.
            # Persist the increase atomically BEFORE permitting new requests.
            # Never rewrite the attempt ledger or any existing answer capture.
            assert stored_cap is not None
            extensions = stored_cap.get("extensions", [])
            if not isinstance(extensions, list):
                raise ValueError("Corrupt budget extension history")
            updated = {
                **stored_cap,
                "limit": requested_limit,
                "extensions": [
                    *extensions,
                    {"from": stored_limit, "to": requested_limit,
                     "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                ],
            }
            fd, temporary = tempfile.mkstemp(
                prefix=".budget-extension-", suffix=".tmp", dir=cap_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(updated, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, cap_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            stored_limit = requested_limit
        total_limit, scope = stored_limit, "persistent"
    elif requested_limit is not None:
        if requested_limit < prior_used:
            raise ValueError("Requested cumulative limit is lower than already recorded attempts")
        total_limit, scope = requested_limit, "persistent"
        # The enclosing exclusive_capture lock prevents two writers. Persist
        # the cap *before* permitting any new provider request; do not alter
        # the original request-attempt ledger or saved answer checkpoints.
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        with cap_path.open("x", encoding="utf-8") as stream:
            json.dump({"schema": 1, "experiment_sha256": experiment_sha256,
                       "limit": total_limit}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        # Backwards-compatible unbounded-across-runs mode. This intentionally
        # does not assert that max_new was a durable spending authorization.
        total_limit, scope = prior_used + default_batch_limit, "invocation"

    budget = RequestBudget(total_limit, used=prior_used)
    if budget.remaining < 0:
        raise ValueError("Recorded requests exceed persistent request budget; no request sent")
    return budget, prior_used, scope
