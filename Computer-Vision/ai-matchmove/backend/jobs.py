"""Local single-worker jobs with persisted progress and cooperative cancellation."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from time import perf_counter, time_ns
from uuid import uuid4

if __package__:
    from .reconstruction import ReconstructionError
    from .sequence import read_json, write_json
else:
    from reconstruction import ReconstructionError
    from sequence import read_json, write_json

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matchmove")
lock = Lock()
active = {}
logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


@dataclass
class JobControl:
    event: Event
    native_cancel: Callable[[], None] | None = None
    future: Future | None = None
    committed: bool = False


def write_status(target, state):
    temp = target / ".status.tmp"
    write_json(temp, state)
    temp.replace(target / "status.json")


def submit(folder, operation, work, *, parameters=None, retry_of=None):
    with lock:
        if len(active) >= 6:
            raise ValueError("The processing queue is full. Wait for a job to finish.")
        job_id = uuid4().hex
        target = folder / "jobs" / job_id
        target.mkdir(parents=True)
        control = JobControl(Event())
        state = {"job_id": job_id, "operation": operation, "status": "queued", "stage": "Waiting for worker",
                 "created_order": time_ns()}
        if parameters is not None:
            state["parameters"] = parameters
        if retry_of is not None:
            state["retry_of"] = retry_of
        write_json(target / "status.json", state)
        active[job_id] = control
        control.future = executor.submit(_run, target, state, control, work)
        return state.copy()


def _run(target, state, control, work):
    started = perf_counter()

    def publish(**fields):
        with lock:
            if fields.get("status") == "done" and control.event.is_set():
                raise JobCancelled()
            state.update(fields)
            write_status(target, state)

    def update(stage, completed=None, total=None):
        if control.event.is_set():
            raise JobCancelled()
        publish(status="running", stage=stage, completed=completed, total=total)

    def bind_cancel(callback):
        with lock:
            control.native_cancel = callback
            if control.event.is_set():
                callback()

    def commit(action):
        with lock:
            if control.event.is_set():
                raise JobCancelled()
            action()
            control.committed = True

    update.bind_cancel = bind_cancel
    update.commit = commit

    try:
        update("Starting " + state["operation"])
        result = work(target, update)
        if control.event.is_set():
            raise JobCancelled()
        publish(status="done", stage="Complete", completed=None, total=None,
                elapsed_seconds=perf_counter() - started, result=result)
    except JobCancelled:
        publish(status="cancelled", stage="Cancelled", elapsed_seconds=perf_counter() - started)
    except ReconstructionError as exc:
        if control.event.is_set():
            publish(status="cancelled", stage="Cancelled", elapsed_seconds=perf_counter() - started)
        else:
            write_json(target / "diagnostics.json", {"error": str(exc), **exc.statistics})
            publish(status="failed", stage="Failed", failed_stage=exc.statistics.get("stage", state["stage"]), error=str(exc),
                    measurements=exc.statistics, elapsed_seconds=perf_counter() - started)
    except Exception as exc:
        if control.event.is_set():
            publish(status="cancelled", stage="Cancelled", elapsed_seconds=perf_counter() - started)
            return
        detail = getattr(exc, "detail", None)
        if not isinstance(detail, str):
            logger.exception("Job %s failed", state["job_id"])
            detail = "Processing failed. Check the backend log for details."
        publish(status="failed", stage="Failed", error=detail, elapsed_seconds=perf_counter() - started)
    finally:
        with lock:
            active.pop(state["job_id"], None)


def status(target: Path):
    with lock:
        data = read_json(target / "status.json")
        control = active.get(data["job_id"])
        if data["status"] in {"queued", "running"}:
            if control is None:
                data.update(status="failed", stage="Interrupted", error="The backend stopped during processing.")
            elif control.event.is_set():
                data["cancellation_requested"] = True
    return data


def exists(target):
    with lock:
        return (target / "status.json").is_file()


def cancel(target):
    with lock:
        current = read_json(target / "status.json")
        control = active.get(current["job_id"])
        if control is not None and not control.committed and current["status"] in {"queued", "running"}:
            control.event.set()
            if control.future is not None and control.future.cancel():
                current.update(status="cancelled", stage="Cancelled", elapsed_seconds=0.0)
                write_status(target, current)
                active.pop(current["job_id"], None)
                return current
            if control.native_cancel is not None:
                control.native_cancel()
            current["cancellation_requested"] = True
    return current
