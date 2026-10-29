"""The Write toolkit: what a reasoner may do during a batch, the same through every transport."""

import json

from agent_memory.core import reconcile, sessions, toolkit

LINES = [
    "user: Our queue timeout is 30 seconds now, we raised it from 20 last Tuesday.",
    "assistant: Noted.",
]


def _messages(store, session="boundary"):
    pointer = store.archive.append_session(session, LINES)
    return sessions.resolve(store.layout, pointer)


def _existing(store, **overrides):
    spec = {
        "type": "fact",
        "fields": {"project": "pipeline", "subject": "queue timeout"},
        "abstract": "Queue timeout is 20 seconds",
        "body": "Set in infra/queue.yaml.",
    }
    return store.record(**{**spec, **overrides})


def _kit(store):
    messages = _messages(store)
    sheet = reconcile.build(store, "boundary", messages)
    empty = reconcile.Sheet(
        sheet.session, sheet.pointer, (), sheet.menus, sheet.profile, sheet.slots, sheet.messages
    )
    return toolkit.WriteToolkit(store, empty)


def _supersede(handle):
    return {
        "type": "fact",
        "op": "supersede",
        "handle": handle,
        "fields": {"project": "pipeline", "subject": "queue timeout 30s"},
        "abstract": "Queue timeout is 30 seconds",
        "provenance": ["0"],
    }


def test_the_catalogue_offers_recall_read_and_submission_with_their_schemas(store):
    catalogue = {spec.name: spec for spec in _kit(store).catalogue()}
    assert set(catalogue) == {toolkit.TOOL_RECALL, toolkit.TOOL_READ, toolkit.TOOL_SUBMIT}
    assert catalogue[toolkit.TOOL_RECALL].schema["required"] == [toolkit.ARG_QUERY]
    assert catalogue[toolkit.TOOL_READ].schema["required"] == [toolkit.ARG_NAME]
    assert catalogue[toolkit.TOOL_SUBMIT].schema["required"] == [toolkit.ARG_OPERATIONS]
    assert all(spec.description for spec in catalogue.values())


def test_recall_lists_matches_by_handle_and_makes_them_nameable(store):
    old = _existing(store)
    kit = _kit(store)
    assert old.name not in kit.sheet.handle_names()
    reply = kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    assert not reply.is_error
    assert old.name in reply.text
    assert old.name in kit.sheet.handle_names()


def test_recall_honours_a_requested_limit_and_defaults_to_the_reconcile_size(store):
    for index in range(4):
        _existing(store, fields={"project": "pipeline", "subject": f"queue timeout {index}"})
    kit = _kit(store)
    limited = kit.call(
        toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout", toolkit.ARG_LIMIT: 1}
    )
    assert _listed(limited.text) == 1
    default = kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    assert 1 < _listed(default.text) <= store.config.write.reconcile_entries


def _listed(text):
    return sum(line.startswith("- ") for line in text.splitlines())


def test_recall_without_a_query_is_an_error_reply(store):
    reply = _kit(store).call(toolkit.TOOL_RECALL, {})
    assert reply.is_error


def test_read_returns_the_full_body_of_an_active_memory_and_makes_it_nameable(store):
    old = _existing(store)
    kit = _kit(store)
    reply = kit.call(toolkit.TOOL_READ, {toolkit.ARG_NAME: old.name})
    assert not reply.is_error
    assert "infra/queue.yaml" in reply.text
    assert old.name in kit.sheet.handle_names()


def test_reading_a_missing_or_superseded_memory_is_an_error_reply(store):
    old = _existing(store)
    store.record(
        type="fact",
        fields={"project": "pipeline", "subject": "queue timeout 25s"},
        abstract="Queue timeout is 25 seconds",
        supersedes=old.name,
    )
    kit = _kit(store)
    assert kit.call(toolkit.TOOL_READ, {toolkit.ARG_NAME: old.name}).is_error
    assert kit.call(toolkit.TOOL_READ, {toolkit.ARG_NAME: "no-such-memory"}).is_error
    assert old.name not in kit.sheet.handle_names()


def test_an_unknown_tool_is_an_error_reply_naming_the_tools(store):
    reply = _kit(store).call("delete", {toolkit.ARG_NAME: "x"})
    assert reply.is_error
    assert all(name in reply.text for name in (toolkit.TOOL_RECALL, toolkit.TOOL_SUBMIT))


def test_arguments_that_are_not_an_object_are_read_as_empty(store):
    reply = _kit(store).call(toolkit.TOOL_RECALL, ["queue"])
    assert reply.is_error


def test_a_submission_naming_an_unseen_handle_is_refused_whole_with_its_index(store):
    old = _existing(store)
    kit = _kit(store)
    skip = {"type": "fact", "op": "skip"}
    reply = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [skip, _supersede(old.name)]})
    assert reply.is_error
    assert "1" in reply.text and "handle" in reply.text
    assert kit.submission is None


def test_a_refused_submission_can_be_corrected_in_the_same_session(store):
    old = _existing(store)
    kit = _kit(store)
    kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [_supersede(old.name)]})
    kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    reply = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [_supersede(old.name)]})
    assert not reply.is_error
    assert kit.submission == [_supersede(old.name)]
    assert reconcile.check(kit.submission[0], kit.sheet) == []


def test_the_latest_accepted_submission_is_the_answer(store):
    kit = _kit(store)
    first = {"type": "fact", "op": "skip"}
    second = {"type": "fact", "op": "new", "abstract": "Queue timeout is 30 seconds"}
    kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [first]})
    kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [second]})
    assert kit.submission == [second]


def test_an_empty_or_malformed_submission_is_refused(store):
    kit = _kit(store)
    for operations in ([], "not a list", [1, "two"], None):
        reply = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: operations})
        assert reply.is_error, operations
    assert kit.submission is None


def test_a_submission_the_store_would_refuse_is_refused_in_the_run_with_its_reason(store):
    old = _existing(store)
    kit = _kit(store)
    kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    rewrite = {
        "type": "fact",
        "op": "update",
        "handle": old.name,
        "fields": {"project": "pipeline", "subject": "queue timeout"},
        "abstract": "Queue timeout is 30 seconds",
        "body": "Raised to 30 seconds.",
    }
    reply = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [rewrite]})
    assert reply.is_error
    assert "operation 0" in reply.text and "supersedes" in reply.text
    assert kit.submission is None
    assert store.find(old.name).is_active()


def test_a_submission_dated_after_its_evidence_is_refused_then_corrected(store):
    old = _existing(store)
    kit = _kit(store)
    kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    future = {**_supersede(old.name), "valid_from": "2099-01-01"}
    refused = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [future]})
    assert refused.is_error and "valid_from" in refused.text
    accepted = kit.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [_supersede(old.name)]})
    assert not accepted.is_error
    assert kit.submission == [_supersede(old.name)]


def test_every_call_is_kept_in_order(store):
    kit = _kit(store)
    kit.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue"})
    kit.call("nope", {})
    assert [call.name for call in kit.calls] == [toolkit.TOOL_RECALL, "nope"]
    assert "nope" in kit.transcript()


def test_a_toolkit_rebuilt_from_its_seed_agrees_and_its_record_flows_back(store):
    old = _existing(store)
    original = _kit(store)
    seed = json.loads(json.dumps(original.seed()))
    remote = toolkit.WriteToolkit.from_seed(store, seed)
    assert remote.sheet.handle_names() == original.sheet.handle_names()
    assert remote.sheet.render() == original.sheet.render()
    assert remote.sheet.messages == original.sheet.messages

    remote.call(toolkit.TOOL_RECALL, {toolkit.ARG_QUERY: "queue timeout"})
    remote.call(toolkit.TOOL_SUBMIT, {toolkit.ARG_OPERATIONS: [_supersede(old.name)]})
    original.absorb(json.loads(json.dumps(remote.record())))
    assert original.submission == remote.submission
    assert original.sheet.handle_names() == remote.sheet.handle_names()
    assert [call.name for call in original.calls] == [call.name for call in remote.calls]
    assert reconcile.check(original.submission[0], original.sheet) == []
