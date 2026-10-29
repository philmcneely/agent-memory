"""Authenticated Muse/OpenRouter coverage for the complete native Write path."""

from __future__ import annotations

import os
import shutil
import time

import pytest
from agent_memory.core import distill, reconcile, sessions, toolkit
from agent_memory.core.config import Config, ExecutorConfig
from agent_memory.core.store import Store
from agent_memory.core.watermark import Watermark
from agent_memory.executor import distiller as executor_distiller
from agent_memory.executor.hosts import HOST_MUSE_CODE
from agent_memory.executor.reasoners import ToolHostReasoner


class RecordingMuse:
    def __init__(self, delegate: ToolHostReasoner):
        self.delegate = delegate
        self.record: dict[str, object] | None = None
        self.seconds = 0.0

    def __call__(self, prompt: str) -> str:
        return self.delegate(prompt)

    def converse(self, instructions, material, write_toolkit):
        started = time.monotonic()
        try:
            return self.delegate.converse(instructions, material, write_toolkit)
        finally:
            self.seconds = time.monotonic() - started
            self.record = write_toolkit.record()


def _reasoner() -> RecordingMuse:
    chosen = executor_distiller.distiller(ExecutorConfig(host=HOST_MUSE_CODE))
    assert isinstance(chosen, ToolHostReasoner)
    return RecordingMuse(chosen)


def _messages(store: Store, session: str, lines: list[str]):
    pointer = store.archive.append_session(session, lines)
    return sessions.resolve(store.layout, pointer)


@pytest.mark.skipif(
    shutil.which("muse") is None or os.environ.get("AGENT_MEMORY_LIVE_MUSE_NATIVE") != "1",
    reason="set AGENT_MEMORY_LIVE_MUSE_NATIVE=1 with Muse, pproxy, and OpenRouter configured",
)
def test_live_muse_native_write_matrix(tmp_path):
    store = Store(tmp_path / "store", config=Config.default(), agent="muse-live")
    store.init()

    new_reasoner = _reasoner()
    new_report = distill.distill(
        store,
        "live-new",
        _messages(
            store,
            "live-new",
            [
                "user: For project aurora, the deployment window is Friday at 22:00 UTC. "
                "This is a stable fact worth remembering.",
                "assistant: Understood.",
            ],
        ),
        new_reasoner,
    )
    new_calls = new_reasoner.record["calls"]
    assert [call["name"] for call in new_calls] == ["recall", "submit_operations"]
    assert new_report.batches[0].written
    deployment = store.find("deployment-window")
    assert deployment is not None
    assert deployment.provenance == ["sessions/live-new#0-0"]

    old = store.record(
        name="queue-timeout",
        type="fact",
        fields={"project": "aurora", "subject": "queue timeout"},
        abstract="Project aurora queue timeout is 20 seconds",
        body="The queue timeout is configured to 20 seconds.",
    )
    supersede_reasoner = _reasoner()
    supersede_report = distill.distill(
        store,
        "live-supersede",
        _messages(
            store,
            "live-supersede",
            [
                "user: Project aurora's queue timeout is now 30 seconds, replacing the old "
                "20 second value. This is a durable configuration change.",
                "assistant: Understood.",
            ],
        ),
        supersede_reasoner,
    )
    supersede_calls = supersede_reasoner.record["calls"]
    names = [call["name"] for call in supersede_calls]
    assert names[-1] == "submit_operations"
    assert names.index("recall") < names.index("read") < names.index("submit_operations")
    operations = supersede_reasoner.record["submission"]
    assert any(operation.get("op") == "supersede" for operation in operations)
    assert not store.find(old.name).is_active()
    assert supersede_report.batches[0].written

    before_skip = {record.name for record in store.records(include_invalid=True)}
    store.config.write.event_lane = False
    skip_reasoner = _reasoner()
    skip_report = distill.distill(
        store,
        "live-skip",
        _messages(store, "live-skip", ["user: Thanks.", "assistant: You're welcome."]),
        skip_reasoner,
    )
    skip_operations = skip_reasoner.record["submission"]
    assert len(skip_operations) == 1
    assert skip_operations[0]["op"] == "skip"
    assert skip_report.batches[0].written == ()
    assert {record.name for record in store.records(include_invalid=True)} == before_skip
    assert Watermark(store.layout, store.clock).read("live-skip").distilled == 2

    error_messages = _messages(store, "live-tool-error", ["user: No durable information."])
    sheet = reconcile.build(store, "live-tool-error", error_messages)
    write_toolkit = toolkit.WriteToolkit(store, sheet)
    error_reasoner = _reasoner()
    error_reasoner.converse(
        "This is a transport test. First call read with name missing-memory. After its error, "
        "call submit_operations with exactly one operation: op skip and type fact.",
        "Exercise the Memory tool error round trip now.",
        write_toolkit,
    )
    assert [call.name for call in write_toolkit.calls] == ["read", "submit_operations"]
    assert write_toolkit.calls[0].reply.is_error
    assert not write_toolkit.calls[1].reply.is_error
    assert len(write_toolkit.submission) == 1
    assert write_toolkit.submission[0]["op"] == "skip"
