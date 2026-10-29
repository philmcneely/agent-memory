"""Native proposal review preserves every existing Manage authority boundary."""

import json
import pathlib

import pytest
from agent_memory.core import reasoning
from agent_memory.core.errors import ReasonerUnavailableError, ValidationError
from agent_memory.core.manage import (
    PROPOSAL_ABSTRACT_REVIEW,
    PROPOSAL_DELETE,
    PROPOSAL_MERGE,
    PROPOSAL_SPLIT,
    PROPOSAL_SUPERSEDE,
    Manage,
)
from agent_memory.core.manage_toolkit import ARG_VERDICTS, TOOL_SUBMIT

ALL_KINDS = {
    PROPOSAL_MERGE,
    PROPOSAL_SUPERSEDE,
    PROPOSAL_SPLIT,
    PROPOSAL_ABSTRACT_REVIEW,
    PROPOSAL_DELETE,
}


class NativeScript:
    def __init__(self, verdict, *, mutate=None, fail=False):
        self.verdict = verdict
        self.mutate = mutate
        self.fail = fail
        self.proposals = ()
        self.calls = 0

    def review_converse(self, instructions, material, kit):
        self.calls += 1
        self.proposals = kit.request.proposals
        if self.fail:
            raise ReasonerUnavailableError("native review exited abnormally")
        verdicts = [self.verdict(proposal) for proposal in self.proposals]
        reply = kit.call(TOOL_SUBMIT, {ARG_VERDICTS: verdicts})
        assert not reply.is_error
        if self.mutate is not None:
            self.mutate()
        return reasoning.ReviewRuntime(duration_seconds=0.1)


class TextReject:
    def __call__(self, prompt):
        ids = [line.split()[1] for line in prompt.splitlines() if line.startswith("- ")]
        return "\n".join(
            json.dumps({"proposal": proposal_id, "verdict": "reject"}) for proposal_id in ids
        )


def _seed_every_kind(store):
    store.config.manage.merge_proposal_similarity = 0.5
    store.record(
        type="experience",
        name="merge-alpha",
        abstract="Alpha queue drain timeout worker lease deploy failure",
        body="Deploy fails because the queue drain is shorter than the lease.",
    )
    store.record(
        type="experience",
        name="merge-beta",
        abstract="Alpha queue drain timeout worker lease staging incident",
        body="The staging incident also records the exact lease timings.",
    )
    store.record(
        type="fact",
        name="supersede-old",
        abstract="Beta release window starts Friday evening",
        body="The old note only says Friday evening.",
    )
    store.record(
        type="fact",
        name="supersede-new",
        abstract="Beta release window starts Friday evening at nine",
        body="The current note says Friday at 21:00 UTC.",
    )
    store.record(
        type="experience",
        name="compound-memory",
        abstract="Gamma incidents need separate lifecycle notes",
        body="## First\nOne fact.\n## Second\nAnother fact.\n## Third\nA third fact.",
        provenance=["sessions/first#0-0", "sessions/second#0-0"],
    )
    store.record(
        type="fact",
        name="thin-memory",
        abstract="Thin",
        body="Delta deployments require signed release tags.",
    )
    store.record(
        type="fact",
        name="unused-memory",
        abstract="Epsilon obsolete setting is no longer consulted",
        body="This setting has never been recalled.",
        weight=store.config.weight.floor,
    )


def _reject(proposal):
    return {"proposal": proposal.id, "verdict": "reject"}


def test_every_proposal_kind_enters_one_native_review_without_new_authority(store):
    _seed_every_kind(store)
    reviewer = NativeScript(_reject)
    report = Manage(store).sleep(reasoner=reviewer)

    assert {proposal.kind for proposal in reviewer.proposals} >= ALL_KINDS
    assert {decision.proposal_id for decision in report.decisions} == {
        proposal.id for proposal in reviewer.proposals
    }
    assert all(record.is_active() for record in store.records())
    rendered = pathlib.Path(report.path).read_text(encoding="utf-8")
    assert not any(line.endswith(" ") for line in rendered.splitlines())


def test_text_and_native_rejections_leave_identical_t0_store_state(tmp_path, config, clock):
    from agent_memory.core.store import Store

    stores = []
    for mode in ("text", "native"):
        candidate = Store(tmp_path / mode, config=config, clock=clock, agent="paired")
        candidate.init()
        _seed_every_kind(candidate)
        stores.append(candidate)
    text, native = stores

    text_report = Manage(text).sleep(reasoner=TextReject())
    native_report = Manage(native).sleep(reasoner=NativeScript(_reject))

    text_records = {record.name: record.to_text() for record in text.records(True)}
    native_records = {record.name: record.to_text() for record in native.records(True)}
    assert native_records == text_records
    assert [action.as_dict() for action in native_report.actions] == [
        action.as_dict() for action in text_report.actions
    ]


def test_native_accept_still_obeys_the_existing_per_kind_cap(store):
    store.config.manage.max_supersedes_per_sleep = 0
    store.record(
        type="fact",
        name="old-window",
        abstract="Release window starts Friday evening",
    )
    store.record(
        type="fact",
        name="new-window",
        abstract="Release window starts Friday evening nightly",
    )
    reviewer = NativeScript(lambda proposal: {"proposal": proposal.id, "verdict": "accept"})
    report = Manage(store).sleep(reasoner=reviewer)

    proposal = next(item for item in reviewer.proposals if item.kind == PROPOSAL_SUPERSEDE)
    assert proposal.id in report.withheld
    assert store.find("old-window").is_active()
    assert store.find("new-window").is_active()


def test_concurrent_write_after_submission_makes_the_proposal_stale(store):
    first = store.record(
        type="fact",
        name="old-window",
        abstract="Release window starts Friday evening",
        body="Old body.",
    )
    store.record(
        type="fact",
        name="new-window",
        abstract="Release window starts Friday evening nightly",
        body="New body.",
    )
    reviewer = NativeScript(
        lambda proposal: {"proposal": proposal.id, "verdict": "accept"},
        mutate=lambda: store.correct(first.name, body="Concurrent writer won."),
    )
    report = Manage(store).sleep(reasoner=reviewer)

    proposal = next(item for item in reviewer.proposals if item.kind == PROPOSAL_SUPERSEDE)
    assert proposal.id in report.withheld
    assert not report.decisions
    assert store.find(first.name).body == "Concurrent writer won."
    assert store.find(first.name).is_active()


def test_native_host_failure_creates_no_success_report_or_decision(store):
    store.record(type="fact", name="thin-memory", abstract="Thin", body="Useful body.")
    with pytest.raises(ReasonerUnavailableError, match="exited abnormally"):
        Manage(store).sleep(reasoner=NativeScript(_reject, fail=True))
    assert not list(store.layout.dream_reports.glob("*.md"))
    assert Manage(store).proposals()


def test_store_lock_rechecks_expected_hash_for_destructive_operations(store):
    old = store.record(type="fact", name="old", abstract="Old current fact")
    new = store.record(type="fact", name="new", abstract="New current fact")
    old_hash = old.source_hash
    new_hash = new.source_hash
    store.correct(old.name, body="A concurrent edit")

    with pytest.raises(ValidationError, match="changed since proposal generation"):
        store.delete(old.name, expected_source_hash=old_hash)
    with pytest.raises(ValidationError, match="changed since proposal generation"):
        store.supersede(
            old.name,
            new.name,
            expected_source_hash=old_hash,
            expected_successor_hash=new_hash,
        )
    with pytest.raises(ValidationError, match="changed since proposal generation"):
        store.merge(
            [old.name, new.name],
            "Merged abstract",
            "Merged body",
            expected_source_hashes={old.name: old_hash, new.name: new_hash},
        )
    assert store.find(old.name).is_active()
    assert store.find(new.name).is_active()
