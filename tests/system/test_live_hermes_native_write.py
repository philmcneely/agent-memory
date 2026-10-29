"""Authenticated Hermes/OpenRouter coverage for the complete native Write path."""

from __future__ import annotations

import json
import os
import shutil
import time

import pytest
from agent_memory.adapters import openrouter
from agent_memory.core import distill, reconcile, sessions, toolkit
from agent_memory.core.config import TOOL_MODE_TEXT, Config, ExecutorConfig
from agent_memory.core.store import Store
from agent_memory.core.watermark import Watermark
from agent_memory.executor import distiller as executor_distiller
from agent_memory.executor.hosts import HOST_HERMES
from agent_memory.executor.reasoners import HostReasoner, ToolHostReasoner

MODEL = "meta/muse-spark-1.3-contributor"


class RecordingHermes:
    def __init__(self, delegate: ToolHostReasoner, instruction_suffix: str = ""):
        self.delegate = delegate
        self.instruction_suffix = instruction_suffix
        self.record: dict[str, object] | None = None
        self.seconds = 0.0

    def __call__(self, prompt: str) -> str:
        return self.delegate(prompt)

    def converse(self, instructions, material, write_toolkit):
        started = time.monotonic()
        try:
            return self.delegate.converse(
                instructions + self.instruction_suffix, material, write_toolkit
            )
        finally:
            self.seconds = time.monotonic() - started
            self.record = write_toolkit.record()


class RecordingText:
    def __init__(self, delegate: HostReasoner):
        self.delegate = delegate
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, prompt: str) -> str:
        started = time.monotonic()
        try:
            self.calls += 1
            return self.delegate(prompt)
        finally:
            self.seconds += time.monotonic() - started


def _reasoner(instruction_suffix: str = "") -> RecordingHermes:
    chosen = executor_distiller.distiller(
        ExecutorConfig(
            host=HOST_HERMES,
            host_provider=openrouter.PROVIDER,
            host_allow_data_training=True,
            host_model=MODEL,
        )
    )
    assert isinstance(chosen, ToolHostReasoner)
    return RecordingHermes(chosen, instruction_suffix)


def _text_reasoner() -> RecordingText:
    chosen = executor_distiller.distiller(
        ExecutorConfig(
            host=HOST_HERMES,
            host_provider=openrouter.PROVIDER,
            host_allow_data_training=True,
            host_model=MODEL,
            tool_mode=TOOL_MODE_TEXT,
        )
    )
    assert isinstance(chosen, HostReasoner)
    assert not isinstance(chosen, ToolHostReasoner)
    return RecordingText(chosen)


def _messages(store: Store, session: str, lines: list[str]):
    pointer = store.archive.append_session(session, lines)
    return sessions.resolve(store.layout, pointer)


def _credential_available() -> bool:
    return openrouter.detect_credential() is not None


@pytest.mark.skipif(
    shutil.which("hermes") is None
    or os.environ.get("AGENT_MEMORY_LIVE_HERMES_NATIVE") != "1"
    or not _credential_available(),
    reason="set AGENT_MEMORY_LIVE_HERMES_NATIVE=1 with Hermes and OpenRouter configured",
)
def test_live_hermes_native_write_matrix(tmp_path, monkeypatch):
    credential = openrouter.detect_credential()
    assert credential is not None
    monkeypatch.setenv("OPENROUTER_API_KEY", credential.value)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)

    store = Store(tmp_path / "store", config=Config.default(), agent="hermes-live")
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
    supersede_reasoner = _reasoner(
        "\n\nLive validation requirement: call recall for the old queue timeout, then call "
        "read on the matching handle before submit_operations."
    )
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

    timings = [
        new_reasoner.seconds,
        supersede_reasoner.seconds,
        skip_reasoner.seconds,
        error_reasoner.seconds,
    ]
    assert all(seconds > 0 for seconds in timings)
    print(
        json.dumps(
            {
                "host": "hermes",
                "model": MODEL,
                "seconds": [round(seconds, 3) for seconds in timings],
                "tool_calls": [
                    [call["name"] for call in reasoner.record["calls"]]
                    for reasoner in (
                        new_reasoner,
                        supersede_reasoner,
                        skip_reasoner,
                        error_reasoner,
                    )
                ],
            },
            sort_keys=True,
        )
    )


@pytest.mark.skipif(
    shutil.which("hermes") is None
    or os.environ.get("AGENT_MEMORY_LIVE_HERMES_NATIVE") != "1"
    or not _credential_available(),
    reason="set AGENT_MEMORY_LIVE_HERMES_NATIVE=1 with Hermes and OpenRouter configured",
)
def test_live_hermes_native_vs_text_processes(tmp_path, monkeypatch):
    credential = openrouter.detect_credential()
    assert credential is not None
    monkeypatch.setenv("OPENROUTER_API_KEY", credential.value)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)

    line = "user: Project comet's durable release channel is stable."
    results = {}
    for mode, reasoner in (("native", _reasoner()), ("text", _text_reasoner())):
        store = Store(tmp_path / mode, config=Config.default(), agent=f"hermes-live-{mode}")
        store.init()
        report = distill.distill(
            store,
            f"live-{mode}",
            _messages(store, f"live-{mode}", [line]),
            reasoner,
        )
        assert report.batches[0].written
        record = store.find(report.batches[0].written[0])
        assert record is not None
        assert record.provenance == [f"sessions/live-{mode}#0-0"]
        results[mode] = {
            "seconds": round(reasoner.seconds, 3),
            "host_processes": 1 if mode == "native" else reasoner.calls,
            "library_rounds": report.batches[0].rounds,
        }

    assert results["native"]["host_processes"] == 1
    assert results["text"]["host_processes"] >= 1
    print(json.dumps({"comparison": results, "model": MODEL}, sort_keys=True))
