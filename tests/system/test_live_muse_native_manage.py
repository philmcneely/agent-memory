"""Authenticated Muse coverage for the complete native Manage review path."""

import json
import os
import pathlib

import pytest
from agent_memory.cli.main import main
from agent_memory.core import sessions
from agent_memory.core.config import OperationExecutorConfig
from agent_memory.core.ledger import LEDGER_FILENAME
from agent_memory.core.manage_toolkit import TOOL_LIST, TOOL_READ, TOOL_SUBMIT
from agent_memory.core.store import Store

LIVE = os.environ.get("AGENT_MEMORY_LIVE_MUSE_MANAGE") == "1"
MODEL = os.environ.get("AGENT_MEMORY_LIVE_MUSE_MODEL", "")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="set AGENT_MEMORY_LIVE_MUSE_MANAGE=1 for authenticated Muse Manage coverage",
)


def _store(tmp_path: pathlib.Path, name: str, *, raw: bool = False) -> Store:
    store = Store(tmp_path / name, agent="live-muse-manage-seed")
    store.init()
    store.config.manage.git_commit = False
    store.config.manage.review_raw_session_evidence = raw
    store.config.executor.manage = OperationExecutorConfig(
        reasoner="host",
        host="muse-code",
        host_model=MODEL,
        tool_mode="native",
        timeout_seconds=180.0,
    )
    store.config.save(store.root)
    return store


def _sleep(store: Store, capsys) -> dict[str, object]:
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


def test_live_muse_reviews_multiple_sources_and_submits_verdicts(tmp_path, capsys):
    store = _store(tmp_path, "basic")
    store.record(
        type="fact",
        name="queue-window-old",
        abstract="Queue drain window closes before worker lease",
        body="The original note says the drain window is shorter than the worker lease.",
    )
    store.record(
        type="fact",
        name="queue-window-current",
        abstract="Queue drain window closes before worker lease on staging",
        body=(
            "On staging the queue drain window is 20 seconds and the worker lease is 30 seconds. "
            "Raise drain_timeout above the lease TTL."
        ),
    )

    payload = _sleep(store, capsys)

    assert len(payload["review"]["evidence_reads"]) >= 2
    _print_result("supersede-multi-read", payload)


def test_live_muse_reads_target_bound_raw_session_evidence(tmp_path, capsys):
    store = _store(tmp_path, "raw-provenance", raw=True)
    guess = store.archive.append_session(
        "muse-guess",
        [
            {
                "role": "assistant",
                "text": "I am guessing the Vega quota may be 500; this is not confirmed.",
                "at": "2026-10-20T08:00:00+00:00",
            }
        ],
    )
    confirmed = store.archive.append_session(
        "muse-confirmed",
        [
            {
                "role": "user",
                "text": "I confirmed with the vendor that the Vega quota is 700.",
                "at": "2026-10-20T09:00:00+00:00",
            }
        ],
    )
    assert guess is not None and confirmed is not None
    store.record(
        type="fact",
        name="vega-quota-old",
        abstract="Vega account quota has a conflicting recorded value of 500",
        body=(
            "The Vega quota is 500. This Memory does not preserve whether its source was a "
            "user confirmation or an assistant guess; check the cited raw Session."
        ),
        provenance=[sessions.render_pointer(guess)],
        valid_from="2026-10-20T08:00:00+00:00",
    )
    store.record(
        type="fact",
        name="vega-quota-confirmed",
        abstract="Vega account quota has a conflicting recorded value of 700",
        body=(
            "The Vega quota is 700. This Memory does not preserve whether its source was a "
            "user confirmation or an assistant guess; check the cited raw Session."
        ),
        provenance=[sessions.render_pointer(confirmed)],
        valid_from="2026-10-20T08:00:00+00:00",
    )

    payload = _sleep(store, capsys)
    pointers = {item[2] for item in payload["review"]["provenance_reads"]}

    assert sessions.render_pointer(guess) in pointers
    assert sessions.render_pointer(confirmed) in pointers
    _print_result("raw-session-provenance", payload)
