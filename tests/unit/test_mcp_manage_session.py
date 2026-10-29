"""The MCP adapter serves only one run-scoped Manage review toolkit."""

import io
import json

from agent_memory.core import reasoning, sessions
from agent_memory.core.manage import Manage
from agent_memory.core.manage_toolkit import (
    ARG_NAME,
    ARG_POINTER,
    ARG_PROPOSAL,
    ARG_VERDICTS,
    TOOL_READ,
    TOOL_SUBMIT,
    ManageReviewToolkit,
)
from agent_memory.mcp import manage_session


def _batch(store, tmp_path, *, raw_session_evidence=False):
    for name, abstract, body in (
        (
            "lease-first",
            "Worker lease outlasts the queue drain timeout",
            "The lease is 30 seconds while the drain is 20 seconds.",
        ),
        (
            "lease-second",
            "Worker lease outlasts the queue drain timeout on staging",
            "Staging uses a 30 second lease and a 20 second drain.",
        ),
    ):
        appended = store.archive.append_session(
            name,
            [{"role": "user", "text": body, "at": "2026-01-15T09:00:00+00:00"}],
        )
        assert appended is not None
        store.record(
            type="experience",
            name=name,
            abstract=abstract,
            body=body,
            provenance=[sessions.render_pointer(appended)],
        )
    proposals = Manage(store).proposals()
    request = reasoning.prepare(
        proposals,
        store.records(),
        reasoning.ReviewLimits(
            8, 24, 120.0, 8000, 32000, raw_session_evidence=raw_session_evidence
        ),
    )
    kit = ManageReviewToolkit(store, request)
    directory = tmp_path / "manage-review"
    directory.mkdir()
    kit.save_seed(directory)
    return kit, directory, proposals[0]


def _rpc(request_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def _call(request_id, name, arguments):
    return _rpc(request_id, "tools/call", {"name": name, "arguments": arguments})


def _serve(directory, *lines):
    stream_out = io.StringIO()
    manage_session.serve(directory, io.StringIO("\n".join(lines) + "\n"), stream_out)
    return [json.loads(line) for line in stream_out.getvalue().splitlines()]


def test_adapter_lists_exactly_the_manage_review_catalogue(store, tmp_path):
    kit, directory, _ = _batch(store, tmp_path)
    (listing,) = _serve(directory, _rpc(1, "tools/list"))
    offered = {tool["name"]: tool for tool in listing["result"]["tools"]}
    assert set(offered) == {spec.name for spec in kit.catalogue()}
    assert "submit_operations" not in offered


def test_real_mcp_calls_read_evidence_and_return_submission_without_writing(store, tmp_path):
    kit, directory, proposal = _batch(store, tmp_path)
    before = {record.name: record.to_text() for record in store.records()}
    read, submitted = _serve(
        directory,
        _call(
            1,
            TOOL_READ,
            {ARG_PROPOSAL: proposal.id, ARG_NAME: proposal.targets[0]},
        ),
        _call(
            2,
            TOOL_SUBMIT,
            {ARG_VERDICTS: [{"proposal": proposal.id, "verdict": "reject"}]},
        ),
    )
    assert read["result"]["isError"] is False
    assert "lease is 30 seconds" in read["result"]["content"][0]["text"]
    assert submitted["result"]["isError"] is False
    kit.absorb_directory(directory)
    assert kit.outcome().complete
    assert {record.name: record.to_text() for record in store.records()} == before


def test_real_mcp_call_returns_only_enabled_target_provenance(store, tmp_path):
    kit, directory, proposal = _batch(store, tmp_path, raw_session_evidence=True)
    source = kit.request.proposals[0].sources[0]
    pointer = source.provenance[0]

    (read,) = _serve(
        directory,
        _call(
            1,
            TOOL_READ,
            {
                ARG_PROPOSAL: proposal.id,
                ARG_NAME: source.name,
                ARG_POINTER: pointer,
            },
        ),
    )

    assert read["result"]["isError"] is False
    payload = json.loads(read["result"]["content"][0]["text"])
    assert payload["evidence_kind"] == "raw_session"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["message_at"] == "2026-01-15T09:00:00+00:00"
    kit.absorb_directory(directory)
    assert kit.outcome().provenance_reads == ((proposal.id, source.name, pointer),)
