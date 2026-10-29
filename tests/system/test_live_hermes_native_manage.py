"""Authenticated Hermes/OpenRouter coverage for the native Manage review path."""

from __future__ import annotations

import json
import os
import pathlib
import shutil

import pytest
from agent_memory.adapters import openrouter
from agent_memory.cli.main import main
from agent_memory.core import sessions
from agent_memory.core.config import OperationExecutorConfig
from agent_memory.core.ledger import LEDGER_FILENAME
from agent_memory.core.manage_toolkit import TOOL_LIST, TOOL_READ, TOOL_SUBMIT
from agent_memory.core.store import Store

LIVE = os.environ.get("AGENT_MEMORY_LIVE_HERMES_MANAGE") == "1"
MODEL = "meta/muse-spark-1.3-contributor"

pytestmark = pytest.mark.skipif(
    shutil.which("hermes") is None or not LIVE or openrouter.detect_credential() is None,
    reason="set AGENT_MEMORY_LIVE_HERMES_MANAGE=1 with Hermes and OpenRouter configured",
)


def _store(tmp_path: pathlib.Path, name: str, *, raw: bool = False) -> Store:
    store = Store(tmp_path / name, agent="live-hermes-manage-seed")
    store.init()
    store.config.manage.git_commit = False
    store.config.manage.review_raw_session_evidence = raw
    store.config.executor.manage = OperationExecutorConfig(
        reasoner="host",
        host="hermes",
        host_provider=openrouter.PROVIDER,
        host_allow_data_training=True,
        host_model=MODEL,
        tool_mode="native",
        timeout_seconds=180.0,
    )
    store.config.save(store.root)
    return store


def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    credential = openrouter.detect_credential()
    assert credential is not None
    monkeypatch.setenv("OPENROUTER_API_KEY", credential.value)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)


def _sleep(store: Store, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    assert main(["--json", "--store", str(store.root), "sleep"]) == 0
    payload = json.loads(capsys.readouterr().out)
    review = payload["review"]
    trace = [item["name"] for item in review["tool_trace"]]
    assert review["mode"] == "native"
    assert review["complete"] is True
    assert trace[0] == TOOL_LIST
    assert TOOL_READ in trace
    assert trace[-1] == TOOL_SUBMIT
    assert payload["decisions"]
    assert pathlib.Path(payload["path"]).is_file()
    assert (store.layout.dream_reports / LEDGER_FILENAME).is_file()
    return payload


def _print_result(case: str, payload: dict[str, object]) -> None:
    review = payload["review"]
    print(
        json.dumps(
            {
                "case": case,
                "host_process": "hermes chat --oneshot",
                "model": MODEL,
                "tool_trace": review["tool_trace"],
                "evidence_reads": review["evidence_reads"],
                "provenance_reads": review["provenance_reads"],
                "verdicts": [decision["verdict"] for decision in payload["decisions"]],
                "duration_seconds": review["duration_seconds"],
                "input_tokens": review["input_tokens"],
                "cached_input_tokens": review["cached_input_tokens"],
                "output_tokens": review["output_tokens"],
            },
            sort_keys=True,
        )
    )


def test_live_hermes_reads_memory_evidence_and_submits_verdicts(tmp_path, monkeypatch, capsys):
    _credential(monkeypatch)
    store = _store(tmp_path, "basic")
    store.record(
        type="fact",
        name="harbor-window-old",
        abstract="Harbor queue drain window closes before worker lease",
        body="The earlier fictional note says the drain window is shorter than the lease.",
    )
    store.record(
        type="fact",
        name="harbor-window-current",
        abstract="Harbor queue drain window closes before worker lease on fictional staging",
        body=(
            "In fictional staging the drain window is 20 seconds and the worker lease is 30 "
            "seconds. Raise drain_timeout above the lease TTL."
        ),
    )

    payload = _sleep(store, capsys)

    assert len(payload["review"]["evidence_reads"]) >= 2
    _print_result("supersede-memory-read", payload)


def test_live_hermes_reads_only_target_bound_raw_session_evidence(tmp_path, monkeypatch, capsys):
    _credential(monkeypatch)
    store = _store(tmp_path, "raw-provenance", raw=True)
    guess = store.archive.append_session(
        "hermes-fictional-guess",
        [
            {
                "role": "assistant",
                "text": "I am guessing the fictional Lyra quota may be 500; not confirmed.",
                "at": "2026-10-21T08:00:00+00:00",
            }
        ],
    )
    confirmed = store.archive.append_session(
        "hermes-fictional-confirmed",
        [
            {
                "role": "user",
                "text": "For this fictional test, I confirm the Lyra quota is 700.",
                "at": "2026-10-21T09:00:00+00:00",
            }
        ],
    )
    assert guess is not None and confirmed is not None
    store.record(
        type="fact",
        name="lyra-quota-old",
        abstract="Fictional Lyra quota has a conflicting recorded value of 500",
        body="The fictional quota is 500; inspect the cited Session to classify its authority.",
        provenance=[sessions.render_pointer(guess)],
        valid_from="2026-10-21T08:00:00+00:00",
    )
    store.record(
        type="fact",
        name="lyra-quota-confirmed",
        abstract="Fictional Lyra quota has a conflicting recorded value of 700",
        body="The fictional quota is 700; inspect the cited Session to classify its authority.",
        provenance=[sessions.render_pointer(confirmed)],
        valid_from="2026-10-21T08:00:00+00:00",
    )

    payload = _sleep(store, capsys)
    pointers = {item[2] for item in payload["review"]["provenance_reads"]}

    assert sessions.render_pointer(guess) in pointers
    assert sessions.render_pointer(confirmed) in pointers
    _print_result("raw-session-provenance", payload)
