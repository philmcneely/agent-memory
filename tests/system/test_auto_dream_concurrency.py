"""Cross-process claim and worker exclusion for automatic Dreaming."""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import os

from agent_memory.adapters import auto_dream, capture, moments
from agent_memory.core import dreaming
from agent_memory.core.clock import FrozenClock
from agent_memory.core.locking import file_lock
from agent_memory.core.store import Store
from agent_memory.core.watermark import Watermark


def _claim(root: str, host: str, output) -> None:
    result = dreaming.claim_due(Store(root), host)
    output.put(result.claim.id if result.claim else "")


def _work(root: str, claim_id: str, output) -> None:
    result = auto_dream.run_worker(Store(root), claim_id, reasoner=None)
    output.put(bool(result["dreamed"]))


def _run_due(root: str, trigger_source: str, ready, output) -> None:
    ready.wait()
    result = auto_dream.run_due(
        Store(root),
        None,
        reasoner_kind="none",
        trigger_source=trigger_source,
    )
    output.put((trigger_source, bool(result["slept"]), str(result["status"])))


def _crash_after_owning_run_lock(root: str, claim_id: str, ready) -> None:
    store = Store(root)
    with file_lock(store.layout.dream_run_lock, timeout=None):
        assert dreaming.current_claim(store, claim_id) is not None
        ready.set()
        os._exit(17)


def _complete(store: Store, session: str) -> None:
    capture.capture(store, session, [f"user: {session}", "assistant: noted"], source="pause")
    mark = Watermark(store.layout).read(session)
    Watermark(store.layout).settle(session, mark.consumed)


def test_two_hosts_can_create_only_one_claim(tmp_path):
    store = Store(tmp_path / "store")
    store.init()
    for number in range(store.config.manage.trigger_min_sessions):
        _complete(store, f"session-{number}")
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [
        context.Process(target=_claim, args=(str(store.root), host, output))
        for host in (moments.HOST_CODEX, moments.HOST_CLAUDE_CODE)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    claims = [output.get(timeout=2) for _ in workers]
    assert len([claim for claim in claims if claim]) == 1


def test_duplicate_workers_execute_one_claim_once_and_exit(tmp_path):
    store = Store(tmp_path / "store")
    store.init()
    for number in range(store.config.manage.trigger_min_sessions):
        _complete(store, f"session-{number}")
    claim = dreaming.claim_due(store, moments.HOST_CODEX).claim
    assert claim is not None
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [
        context.Process(target=_work, args=(str(store.root), claim.id, output))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    assert [output.get(timeout=2) for _ in workers] == [True, True]
    assert len(list(store.layout.dream_reports.glob(f"auto-{claim.id}.md"))) == 1
    assert dreaming.state(store)["status"] == dreaming.STATUS_SUCCEEDED


def test_hook_and_cron_due_runners_execute_manage_once(tmp_path):
    store = Store(tmp_path / "store")
    store.init()
    for number in range(store.config.manage.trigger_min_sessions):
        _complete(store, f"session-{number}")
    context = mp.get_context("spawn")
    ready = context.Event()
    output = context.Queue()
    workers = [
        context.Process(target=_run_due, args=(str(store.root), source, ready, output))
        for source in ("hook", "cron")
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    results = [output.get(timeout=2) for _ in workers]
    assert len([result for result in results if result[1]]) == 1
    assert {result[2] for result in results} == {"skipped", "succeeded"}
    assert len(list(store.layout.dream_reports.glob("auto-*.md"))) == 1
    assert dreaming.state(store)["trigger_source"] in {"hook", "cron"}


def test_abnormally_exited_worker_releases_lock_and_next_event_reclaims(tmp_path):
    clock = FrozenClock(dt.datetime(2026, 1, 15, 9, 0, tzinfo=dt.UTC))
    store = Store(tmp_path / "store", clock=clock)
    store.config.manage.claim_lease_seconds = 1
    store.init()
    for number in range(store.config.manage.trigger_min_sessions):
        _complete(store, f"session-{number}")
    claim = dreaming.claim_due(store, moments.HOST_CODEX).claim
    assert claim is not None
    context = mp.get_context("spawn")
    ready = context.Event()
    worker = context.Process(
        target=_crash_after_owning_run_lock, args=(str(store.root), claim.id, ready)
    )
    worker.start()
    assert ready.wait(5)
    worker.join(5)
    assert worker.exitcode == 17
    assert dreaming.state(store)["status"] == dreaming.STATUS_RUNNING

    clock.advance(seconds=store.config.manage.claim_lease_seconds + 1)
    retried = dreaming.claim_due(store, moments.HOST_CLAUDE_CODE).claim

    assert retried is not None
    assert retried.id == claim.id
    assert retried.attempt == claim.attempt + 1
