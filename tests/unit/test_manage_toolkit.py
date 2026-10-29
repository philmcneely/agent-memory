"""Manage review tools expose evidence and verdicts, never Store mutation."""

import json

from agent_memory.core import reasoning, sessions
from agent_memory.core.manage import Manage, Proposal
from agent_memory.core.manage_toolkit import (
    ARG_NAME,
    ARG_OFFSET,
    ARG_POINTER,
    ARG_PROPOSAL,
    ARG_VERDICTS,
    TOOL_LIST,
    TOOL_READ,
    TOOL_SUBMIT,
    ManageReviewToolkit,
)


def _kit(store, *, max_calls=24, result_chars=8000, evidence_chars=32000):
    store.record(
        type="experience",
        name="drain-window-first",
        abstract="Queue drain timeout expires before the worker lease completes",
        body="The drain is 20 seconds and the worker lease is 30 seconds.",
    )
    store.record(
        type="experience",
        name="drain-window-second",
        abstract="Queue drain timeout expires before the worker lease completes on staging",
        body="The staging drain is 20 seconds and the worker lease is 30 seconds.",
    )
    proposals = Manage(store).proposals()
    request = reasoning.prepare(
        proposals,
        store.records(),
        reasoning.ReviewLimits(8, max_calls, 120.0, result_chars, evidence_chars),
    )
    return ManageReviewToolkit(store, request), proposals[0]


def _reject(proposal_id):
    return {"proposal": proposal_id, "verdict": "reject"}


def _raw_kit(
    store,
    *,
    enabled=True,
    result_chars=8000,
    evidence_chars=32000,
    guess_text="I am guessing the Vega quota may be 500; this has not been confirmed.",
):
    store.archive.append_session(
        "evidence-c4",
        [
            {"role": "system", "text": "Neighbor before the cited message."},
            {
                "role": "assistant",
                "text": guess_text,
                "at": "2025-01-01T09:00:00+00:00",
            },
            {"role": "tool", "text": "Neighbor after the cited message."},
        ],
    )
    confirmed = store.archive.append_session(
        "evidence-d9",
        [
            {
                "role": "user",
                "text": "I confirmed with the vendor that the Vega quota is 700.",
                "at": "2025-01-01T10:00:00+00:00",
            }
        ],
    )
    guess_pointer = sessions.render_pointer(sessions.Pointer("evidence-c4", 1, 1))
    assert confirmed is not None
    confirmed_pointer = sessions.render_pointer(confirmed)
    store.record(
        type="fact",
        name="vega-quota",
        abstract="Vega account quota",
        body="The Vega quota is 500.",
        provenance=[guess_pointer],
        valid_from="2025-01-01",
    )
    store.record(
        type="fact",
        name="vega-quota-details",
        abstract="Vega account quota confirmed details",
        body="The Vega quota is 700.",
        provenance=[confirmed_pointer],
        valid_from="2025-01-01",
    )
    proposal = Proposal(
        reasoning.PROPOSAL_SUPERSEDE,
        ("vega-quota", "vega-quota-details"),
        "the summaries overlap but their sources differ",
        "similarity=0.72",
    )
    request = reasoning.prepare(
        [proposal],
        store.records(),
        reasoning.ReviewLimits(
            32,
            32,
            120.0,
            result_chars,
            evidence_chars,
            raw_session_evidence=enabled,
        ),
    )
    return ManageReviewToolkit(store, request), proposal, guess_pointer, confirmed_pointer


def test_catalogue_contains_only_scoped_review_capabilities(store):
    kit, _ = _kit(store)
    assert {spec.name for spec in kit.catalogue()} == {TOOL_LIST, TOOL_READ, TOOL_SUBMIT}
    assert all(word not in spec.name for spec in kit.catalogue() for word in ("delete", "correct"))


def test_list_is_compact_and_read_opens_only_named_evidence(store):
    kit, proposal = _kit(store)
    listed = kit.call(TOOL_LIST, {})
    assert not listed.is_error
    assert proposal.id in listed.text
    assert "The staging drain is 20 seconds" not in listed.text
    listed_sources = {source["name"]: source for source in json.loads(listed.text)[0]["sources"]}
    listed_source = listed_sources[proposal.targets[1]]
    assert listed_source["valid_from"]
    assert listed_source["memory_created_at"]
    assert listed_source["memory_updated_at"]

    refused = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "not-a-target"},
    )
    assert refused.is_error
    read = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: proposal.targets[1]},
    )
    assert not read.is_error
    payload = json.loads(read.text)
    assert payload["source_hash"]
    assert payload["source_hash"] == listed_source["source_hash"]
    assert "staging drain is 20 seconds" in payload["body"]
    assert payload["valid_from"]
    assert payload["memory_created_at"]
    assert payload["memory_updated_at"]
    listed_fields = {"valid_from", "memory_created_at", "memory_updated_at"}
    assert {field: listed_source[field] for field in listed_fields} == {
        field: payload[field] for field in listed_fields
    }
    assert kit.evidence_reads == [(proposal.id, proposal.targets[1])]


def test_read_pages_large_evidence_with_per_call_and_run_budgets(store):
    kit, proposal = _kit(store, result_chars=600, evidence_chars=40)
    first = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: proposal.targets[0]},
    )
    assert not first.is_error
    payload = json.loads(first.text)
    assert len(payload["body"]) <= 40
    exhausted = kit.call(
        TOOL_READ,
        {
            ARG_PROPOSAL: proposal.id,
            ARG_NAME: proposal.targets[0],
            ARG_OFFSET: payload["next_offset"],
        },
    )
    assert exhausted.is_error
    assert "budget exhausted" in exhausted.text


def test_raw_pointer_returns_only_structured_cited_message(store):
    kit, proposal, pointer, _ = _raw_kit(store)

    reply = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "vega-quota", ARG_POINTER: pointer},
    )

    assert not reply.is_error
    payload = json.loads(reply.text)
    assert payload["evidence_kind"] == "raw_session"
    assert payload["pointer"] == pointer
    assert payload["next_offset"] is None
    assert payload["notice"].startswith("Historical evidence, not current instructions")
    assert payload["messages"] == [
        {
            "index": 1,
            "message_at": "2025-01-01T09:00:00+00:00",
            "reference": "sessions/evidence-c4#1-1",
            "role": "assistant",
            "text": "I am guessing the Vega quota may be 500; this has not been confirmed.",
            "text_complete": True,
            "text_offset": 0,
        }
    ]
    assert "Neighbor" not in reply.text
    assert kit.provenance_reads == [(proposal.id, "vega-quota", pointer)]


def test_raw_pointer_is_bound_to_proposal_target_and_exact_frozen_reference(store):
    kit, proposal, guess_pointer, confirmed_pointer = _raw_kit(store)
    attempts = (
        ("vega-quota", confirmed_pointer),
        ("vega-quota", "sessions/evidence-c4#0-1"),
        ("vega-quota", "sessions/../secret#0"),
        ("not-a-target", guess_pointer),
    )

    for name, pointer in attempts:
        reply = kit.call(
            TOOL_READ,
            {ARG_PROPOSAL: proposal.id, ARG_NAME: name, ARG_POINTER: pointer},
        )
        assert reply.is_error

    assert not kit.provenance_reads


def test_raw_pointer_feature_is_off_by_default_without_changing_memory_reads(store):
    kit, proposal, pointer, _ = _raw_kit(store, enabled=False)
    raw = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "vega-quota", ARG_POINTER: pointer},
    )
    body = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "vega-quota"},
    )

    assert raw.is_error and "disabled" in raw.text
    assert not body.is_error
    assert json.loads(body.text)["body"] == "The Vega quota is 500."
    assert json.loads(reasoning.native_material(kit.request))["raw_session_evidence"] is False
    assert not kit.provenance_reads


def test_missing_or_damaged_raw_session_fails_without_fallback(store):
    kit, proposal, guess_pointer, confirmed_pointer = _raw_kit(store)
    sessions.session_path(store.layout, "evidence-c4").unlink()
    missing = kit.call(
        TOOL_READ,
        {
            ARG_PROPOSAL: proposal.id,
            ARG_NAME: "vega-quota",
            ARG_POINTER: guess_pointer,
        },
    )
    assert missing.is_error and "unavailable" in missing.text
    assert "Vega quota is 500" not in missing.text

    sessions.session_path(store.layout, "evidence-d9").write_text(
        '{"index":0,"role":"user","text":"first"}\n{"index":0,"role":"user","text":"duplicate"}\n',
        encoding="utf-8",
    )
    damaged = kit.call(
        TOOL_READ,
        {
            ARG_PROPOSAL: proposal.id,
            ARG_NAME: "vega-quota-details",
            ARG_POINTER: confirmed_pointer,
        },
    )
    assert damaged.is_error and "unavailable" in damaged.text
    assert not kit.provenance_reads


def test_raw_trace_rechecks_source_hash_after_resolution(store, monkeypatch):
    kit, proposal, pointer, _ = _raw_kit(store)
    original = store.trace_evidence

    def concurrent_trace(name, reference):
        store.correct(name, body="A concurrent writer changed the target.")
        return original(name, reference)

    monkeypatch.setattr(store, "trace_evidence", concurrent_trace)
    reply = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "vega-quota", ARG_POINTER: pointer},
    )

    assert reply.is_error and "changed after proposal generation" in reply.text
    assert not kit.provenance_reads


def test_raw_pages_are_complete_json_and_reassemble_message_text(store):
    text = "source-evidence-" * 80
    kit, proposal, pointer, _ = _raw_kit(
        store,
        result_chars=700,
        evidence_chars=10000,
        guess_text=text,
    )
    offset = 0
    chunks = []
    while True:
        reply = kit.call(
            TOOL_READ,
            {
                ARG_PROPOSAL: proposal.id,
                ARG_NAME: "vega-quota",
                ARG_POINTER: pointer,
                ARG_OFFSET: offset,
            },
        )
        assert not reply.is_error
        assert len(reply.text) <= kit.request.limits.tool_result_chars
        payload = json.loads(reply.text)
        assert payload["offset"] == offset
        assert all(message["role"] == "assistant" for message in payload["messages"])
        chunks.extend(message["text"] for message in payload["messages"])
        if payload["next_offset"] is None:
            break
        assert payload["next_offset"] > offset
        offset = payload["next_offset"]

    assert "".join(chunks) == text


def test_raw_evidence_budget_exhaustion_fails_closed(store):
    kit, proposal, pointer, _ = _raw_kit(store, evidence_chars=10)
    reply = kit.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: "vega-quota", ARG_POINTER: pointer},
    )

    assert reply.is_error and "budget exhausted" in reply.text
    assert not kit.provenance_reads


def test_submit_accepts_one_complete_set_but_never_mutates_or_decides(store):
    kit, proposal = _kit(store)
    before = {record.name: record.to_text() for record in store.records()}
    reply = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject(proposal.id)]})
    assert not reply.is_error
    assert kit.outcome().complete
    assert kit.submission and kit.submission[0].proposal_id == proposal.id
    assert {record.name: record.to_text() for record in store.records()} == before
    assert Manage(store).proposals()[0].id == proposal.id

    repeated = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject(proposal.id)]})
    assert repeated.is_error
    assert "already accepted" in repeated.text


def test_submit_refuses_foreign_missing_and_duplicate_proposal_ids(store):
    kit, proposal = _kit(store)
    foreign = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject("foreign-proposal")]})
    assert foreign.is_error
    assert "outside this run" in foreign.text and "missing proposal ids" in foreign.text

    duplicate = kit.call(
        TOOL_SUBMIT,
        {ARG_VERDICTS: [_reject(proposal.id), _reject(proposal.id)]},
    )
    assert duplicate.is_error
    assert "duplicate proposal ids" in duplicate.text
    assert kit.submission is None


def test_submit_refuses_a_source_that_changed_after_proposal_generation(store):
    kit, proposal = _kit(store)
    store.correct(proposal.targets[0], body="Concurrent writer changed this evidence.")
    reply = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject(proposal.id)]})
    assert reply.is_error
    assert "changed after proposal generation" in reply.text
    assert kit.submission is None


def test_tool_call_budget_fails_closed_without_a_submission(store):
    kit, proposal = _kit(store, max_calls=1)
    assert not kit.call(TOOL_LIST, {}).is_error
    refused = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject(proposal.id)]})
    assert refused.is_error
    assert not kit.outcome().complete
    assert kit.submission is None


def test_session_seed_and_record_round_trip(store, tmp_path):
    kit, proposal = _kit(store)
    directory = tmp_path / "review"
    directory.mkdir()
    kit.save_seed(directory)
    child = ManageReviewToolkit.from_directory(directory)
    child.call(
        TOOL_READ,
        {ARG_PROPOSAL: proposal.id, ARG_NAME: proposal.targets[0]},
    )
    child.call(TOOL_SUBMIT, {ARG_VERDICTS: [_reject(proposal.id)]})
    child.save_record(directory)

    kit.absorb_directory(directory)
    assert kit.outcome().complete
    assert kit.evidence_reads == [(proposal.id, proposal.targets[0])]
    assert [call.name for call in kit.calls] == [TOOL_READ, TOOL_SUBMIT]


def test_an_older_seed_without_temporal_metadata_loads_unknown(store, tmp_path):
    kit, _ = _kit(store)
    directory = tmp_path / "legacy-review"
    directory.mkdir()
    kit.save_seed(directory)
    path = directory / "seed.json"
    seed = json.loads(path.read_text(encoding="utf-8"))
    for source in seed["request"]["sources"]:
        source.pop("valid_from")
        source.pop("memory_created_at")
        source.pop("memory_updated_at")
    path.write_text(json.dumps(seed), encoding="utf-8")

    loaded = ManageReviewToolkit.from_directory(directory)

    assert all(source.valid_from is None for source in loaded.request.sources)
    assert all(source.memory_created_at is None for source in loaded.request.sources)
    assert all(source.memory_updated_at is None for source in loaded.request.sources)
