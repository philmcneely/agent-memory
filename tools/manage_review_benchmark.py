#!/usr/bin/env python3
"""Paired real-Codex benchmark for text and native Manage proposal review."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import statistics
import time

from agent_memory.core import reasoning
from agent_memory.core.config import Config
from agent_memory.core.ledger import LEDGER_FILENAME
from agent_memory.core.manage import Manage
from agent_memory.core.store import Store
from agent_memory.executor import hosts
from agent_memory.executor.reasoners import HostReasoner, ToolHostReasoner

MODEL = "gpt-5.6-sol"
HOST = hosts.HOST_CODEX
TIMEOUT_SECONDS = 240.0
NATIVE = "native"
TEXT = "text"
ARMS = (TEXT, NATIVE)


@dataclasses.dataclass(frozen=True)
class Case:
    id: str
    batch: str
    kind: str
    targets: tuple[str, ...]
    expected: str
    description: str


CASES = (
    Case(
        "merge-complementary",
        "b1",
        reasoning.PROPOSAL_MERGE,
        ("billing-retry-east", "billing-retry-west"),
        "accepted",
        "same retry failure with complementary regional specifics",
    ),
    Case(
        "similar-different-occasion",
        "b1",
        reasoning.PROPOSAL_MERGE,
        ("deploy-incident-2025", "deploy-incident-2026"),
        "rejected",
        "lexically similar incidents from different years",
    ),
    Case(
        "supersede-current-value",
        "b1",
        reasoning.PROPOSAL_SUPERSEDE,
        ("retention-old", "retention-current"),
        "accepted",
        "current entry fully replaces the older retention note",
    ),
    Case(
        "split-independent-facts",
        "b2",
        reasoning.PROPOSAL_SPLIT,
        ("compound-alerting",),
        "accepted",
        "sections from different sessions expire independently",
    ),
    Case(
        "abstract-grounded",
        "b2",
        reasoning.PROPOSAL_ABSTRACT_REVIEW,
        ("thin-signing",),
        "accepted",
        "thin abstract can be rewritten from its body",
    ),
    Case(
        "delete-critical-unused",
        "b2",
        reasoning.PROPOSAL_DELETE,
        ("critical-unused",),
        "rejected",
        "unused does not mean safe to forget when the body is critical",
    ),
    Case(
        "merge-conflicting-values",
        "b3",
        reasoning.PROPOSAL_MERGE,
        ("api-timeout-thirty", "api-timeout-sixty"),
        "rejected",
        "similar records conflict on the active timeout value",
    ),
    Case(
        "supersede-different-scope",
        "b3",
        reasoning.PROPOSAL_SUPERSEDE,
        ("backup-window", "backup-window-eu"),
        "rejected",
        "the fuller wording applies only to another region",
    ),
    Case(
        "split-atomic-procedure",
        "b3",
        reasoning.PROPOSAL_SPLIT,
        ("atomic-rollout",),
        "rejected",
        "sections are ordered steps of one atomic procedure",
    ),
    Case(
        "merge-shared-policy",
        "b4",
        reasoning.PROPOSAL_MERGE,
        ("release-policy-a", "release-policy-b"),
        "accepted",
        "two records hold complementary parts of one release policy",
    ),
    Case(
        "abstract-searchability",
        "b4",
        reasoning.PROPOSAL_ABSTRACT_REVIEW,
        ("thin-cache",),
        "accepted",
        "body contains a precise searchable cache rule",
    ),
    Case(
        "delete-obsolete-toggle",
        "b4",
        reasoning.PROPOSAL_DELETE,
        ("obsolete-toggle",),
        "accepted",
        "explicitly removed toggle has no future use",
    ),
)


class MeasuredText:
    def __init__(self, delegate: HostReasoner):
        self.delegate = delegate
        self.prompt_chars = 0

    def __call__(self, prompt: str) -> str:
        self.prompt_chars = len(prompt)
        return self.delegate(prompt)


def _record(store: Store, **values) -> None:
    store.record(**values)


def _seed(store: Store, batch: str) -> None:
    if batch == "b1":
        _record(
            store,
            type="experience",
            name="billing-retry-east",
            abstract="Billing retry queue timeout drops invoices in east region",
            body="East drops invoices after three retries; alert finance and replay the queue.",
        )
        _record(
            store,
            type="experience",
            name="billing-retry-west",
            abstract="Billing retry queue timeout drops invoices in west region",
            body="West uses the same retry failure; reset the consumer before replay.",
        )
        _record(
            store,
            type="experience",
            name="deploy-incident-2025",
            abstract="Orion deploy queue lock caused checkout outage in 2025",
            body="The 2025 outage was caused by a stale queue lock.",
        )
        _record(
            store,
            type="experience",
            name="deploy-incident-2026",
            abstract="Orion deploy queue lock caused checkout outage in 2026",
            body="The 2026 outage was caused by a new lock regression after a migration.",
        )
        _record(
            store,
            type="fact",
            name="retention-old",
            abstract="Kappa audit retention lasts ninety days",
            body="The earlier policy retained audit events for ninety days.",
        )
        _record(
            store,
            type="fact",
            name="retention-current",
            abstract="Kappa audit retention lasts ninety days currently",
            body="The current signed policy retains audit events for ninety days.",
        )
    elif batch == "b2":
        _record(
            store,
            type="experience",
            name="compound-alerting",
            abstract="Lambda alerting notes combine unrelated operational facts",
            body=(
                "## Pager\nPager rotation changes weekly.\n"
                "## Email\nEmail digests run at 08:00 UTC.\n"
                "## Escalation\nSecurity escalation uses a separate hotline."
            ),
            provenance=["sessions/pager#0-0", "sessions/digest#0-0"],
        )
        _record(
            store,
            type="fact",
            name="thin-signing",
            abstract="Thin",
            body="Mu production releases require a signed annotated Git tag.",
        )
        _record(
            store,
            type="fact",
            name="critical-unused",
            abstract="Nu disaster recovery key is stored in the offline safe",
            body="Critical recovery instruction; preserve even when it has never been recalled.",
            weight=store.config.weight.floor,
        )
    elif batch == "b3":
        _record(
            store,
            type="fact",
            name="api-timeout-thirty",
            abstract="Omicron partner API timeout is thirty seconds production",
            body="The active production timeout is 30 seconds.",
        )
        _record(
            store,
            type="fact",
            name="api-timeout-sixty",
            abstract="Omicron partner API timeout is sixty seconds production",
            body="A conflicting note claims the active production timeout is 60 seconds.",
        )
        _record(
            store,
            type="fact",
            name="backup-window",
            abstract="Pi backup window begins Saturday midnight",
            body="This note applies to the US production region.",
        )
        _record(
            store,
            type="fact",
            name="backup-window-eu",
            abstract="Pi backup window begins Saturday midnight Europe",
            body="This fuller-looking note applies only to the EU staging region.",
        )
        _record(
            store,
            type="experience",
            name="atomic-rollout",
            abstract="Rho rollout procedure must remain one ordered transaction",
            body=(
                "## Prepare\nFreeze writes.\n"
                "## Switch\nMove traffic while writes remain frozen.\n"
                "## Verify\nUnfreeze only after the same transaction verifies."
            ),
            provenance=["sessions/rollout-a#0-1", "sessions/rollout-b#0-1"],
        )
    elif batch == "b4":
        _record(
            store,
            type="decision",
            name="release-policy-a",
            abstract="Sigma release policy requires signed tags and approval",
            body="Every release needs a signed tag and one maintainer approval.",
        )
        _record(
            store,
            type="decision",
            name="release-policy-b",
            abstract="Sigma release policy requires signed tags and rollback",
            body="The same release policy also requires a tested rollback plan.",
        )
        _record(
            store,
            type="fact",
            name="thin-cache",
            abstract="Cache",
            body="Tau search cache entries expire after fifteen minutes unless pinned.",
        )
        _record(
            store,
            type="fact",
            name="obsolete-toggle",
            abstract="Upsilon legacy checkout toggle was permanently removed",
            body="The code and configuration were deleted; the toggle can never be used again.",
            weight=store.config.weight.floor,
        )
    else:
        raise ValueError(batch)


def _config() -> Config:
    config = Config.default()
    config.manage.git_commit = False
    config.manage.merge_proposal_similarity = 0.5
    config.manage.review_max_steps = 64
    config.manage.review_max_tool_calls = 64
    config.manage.review_evidence_chars = 100000
    config.executor.timeout_seconds = TIMEOUT_SECONDS
    return config


def _reasoner(arm: str):
    spec = hosts.HostSpec(
        name=HOST,
        binary=hosts.BINARIES[HOST][0],
        model=MODEL,
        timeout_seconds=TIMEOUT_SECONDS,
        attempts=1,
    )
    host = hosts.Host(spec)
    return ToolHostReasoner(host=host) if arm == NATIVE else MeasuredText(HostReasoner(host=host))


def _proposal_key(proposal) -> tuple[str, frozenset[str]]:
    return proposal.kind, frozenset(proposal.targets)


def _snapshot(store: Store) -> dict[str, str]:
    return {record.name: record.to_text() for record in store.records(include_invalid=True)}


def _run(root: pathlib.Path, batch: str, arm: str) -> dict[str, object]:
    store = Store(root, agent=f"benchmark-{arm}")
    proposals = Manage(store).proposals()
    by_key = {_proposal_key(proposal): proposal for proposal in proposals}
    cases = [case for case in CASES if case.batch == batch]
    missing = [case.id for case in cases if (case.kind, frozenset(case.targets)) not in by_key]
    if missing:
        raise RuntimeError(f"missing expected proposals: {missing}")
    expected = {by_key[(case.kind, frozenset(case.targets))].id: case for case in cases}
    before = _snapshot(store)
    reasoner = _reasoner(arm)
    started = time.monotonic()
    error = ""
    report = None
    try:
        report = Manage(store).sleep(reasoner=reasoner)
    except Exception as raised:  # benchmark records failures instead of hiding them
        error = f"{type(raised).__name__}: {raised}"
    elapsed = time.monotonic() - started
    after = _snapshot(store)
    decisions = (
        {decision.proposal_id: decision.verdict for decision in report.decisions} if report else {}
    )
    reads = set(report.review.evidence_reads) if report and report.review else set()
    scored = []
    for proposal_id, case in expected.items():
        actual = decisions.get(proposal_id, "missing")
        scored.append(
            {
                **dataclasses.asdict(case),
                "proposal": proposal_id,
                "actual": actual,
                "correct": actual == case.expected,
                "evidence_read": (
                    all((proposal_id, target) in reads for target in case.targets)
                    if arm == NATIVE
                    else True
                ),
            }
        )
    review = report.review if report else None
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    return {
        "batch": batch,
        "arm": arm,
        "model": MODEL,
        "error": error,
        "cases": scored,
        "correct": sum(bool(item["correct"]) for item in scored),
        "total": len(scored),
        "evidence_complete": sum(bool(item["evidence_read"]) for item in scored),
        "tool_calls": review.tool_calls if review else 0,
        "agent_turns": review.agent_turns if review else None,
        "model_requests": review.model_requests if review else None,
        "duration_seconds": review.duration_seconds if review else elapsed,
        "wall_seconds": elapsed,
        "startup_seconds": review.startup_seconds if review else None,
        "model_wait_seconds": review.model_wait_seconds if review else None,
        "mcp_seconds": review.mcp_seconds if review else None,
        "shutdown_seconds": review.shutdown_seconds if review else None,
        "apply_seconds": review.apply_seconds if review else None,
        "input_tokens": review.input_tokens if review else None,
        "cached_input_tokens": review.cached_input_tokens if review else None,
        "output_tokens": review.output_tokens if review else None,
        "prompt_chars": getattr(reasoner, "prompt_chars", None),
        "changed_memories": changed,
        "ledger": (store.layout.dream_reports / LEDGER_FILENAME).is_file(),
        "report": bool(report and pathlib.Path(report.path).is_file()),
        "tool_trace": [list(item) for item in review.tool_trace] if review else [],
    }


def _summary(runs: list[dict[str, object]]) -> dict[str, object]:
    arms = {}
    for arm in ARMS:
        selected = [run for run in runs if run["arm"] == arm]
        cases = [case for run in selected for case in run["cases"]]
        durations = [float(run["wall_seconds"]) for run in selected]

        def total(key: str, selected_runs=selected) -> float | None:
            values = [run[key] for run in selected_runs]
            return (
                sum(float(value) for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            )

        arms[arm] = {
            "correct": sum(bool(case["correct"]) for case in cases),
            "total": len(cases),
            "errors": sum(bool(run["error"]) for run in selected),
            "evidence_complete": sum(bool(case["evidence_read"]) for case in cases),
            "tool_calls": sum(int(run["tool_calls"]) for run in selected),
            "wall_seconds": sum(durations),
            "median_batch_seconds": statistics.median(durations),
            "startup_seconds": total("startup_seconds"),
            "model_wait_seconds": total("model_wait_seconds"),
            "mcp_seconds": total("mcp_seconds"),
            "shutdown_seconds": total("shutdown_seconds"),
            "apply_seconds": total("apply_seconds"),
            "input_tokens": sum(int(run["input_tokens"] or 0) for run in selected) or None,
            "cached_input_tokens": sum(int(run["cached_input_tokens"] or 0) for run in selected)
            or None,
            "output_tokens": sum(int(run["output_tokens"] or 0) for run in selected) or None,
            "prompt_chars": sum(int(run["prompt_chars"] or 0) for run in selected) or None,
            "ledger_complete": all(bool(run["ledger"]) for run in selected),
            "report_complete": all(bool(run["report"]) for run in selected),
        }
    return {
        "model": MODEL,
        "cases": sum(int(run["total"]) for run in runs if run["arm"] == ARMS[0]),
        "batches": len({str(run["batch"]) for run in runs}),
        "arms": arms,
    }


def _markdown(summary: dict[str, object]) -> str:
    arms = summary["arms"]
    lines = [
        "# Manage native review paired experiment",
        "",
        (
            f"Model: `{summary['model']}`. Fixed cases: {summary['cases']} "
            f"in {summary['batches']} batches."
        ),
        "",
        (
            "| Arm | Correct | Evidence complete | Tool calls | Wall s | "
            "Median batch s | Startup/model/MCP/shutdown/apply s | Tokens in/cache/out |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        result = arms[arm]
        lines.append(
            f"| {arm} | {result['correct']}/{result['total']} | "
            f"{result['evidence_complete']}/{result['total']} | {result['tool_calls']} | "
            f"{result['wall_seconds']:.3f} | {result['median_batch_seconds']:.3f} | "
            f"{_seconds(result['startup_seconds'])}/{_seconds(result['model_wait_seconds'])}/"
            f"{_seconds(result['mcp_seconds'])}/{_seconds(result['shutdown_seconds'])}/"
            f"{_seconds(result['apply_seconds'])} | "
            f"{result['input_tokens']}/{result['cached_input_tokens']}/{result['output_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Text receives every target body in one prompt; its token totals are unavailable",
            "from the plain Codex compatibility invocation. Native receives summaries first",
            "and records actual MCP reads plus Codex JSONL usage. This is an integration sample,",
            "not a benchmark score. Native startup/model/MCP phases partition Host duration;",
            "apply is measured separately in Manage.",
            "",
        ]
    )
    return "\n".join(lines)


def _seconds(value: object) -> str:
    return f"{float(value):.3f}" if value is not None else "n/a"


def main() -> int:
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--batches",
        nargs="+",
        choices=("b1", "b2", "b3", "b4"),
        default=("b1", "b2", "b3", "b4"),
        help="run selected three-case batches (use two batches for a six-case diagnostic)",
    )
    args = parser.parse_args()
    MODEL = args.model
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = args.out / "seeds"
    stores = args.out / "stores"
    seeds.mkdir()
    stores.mkdir()
    runs = []
    for number, batch in enumerate(args.batches):
        seed_root = seeds / batch
        seed = Store(seed_root, config=_config(), agent="benchmark-seed")
        seed.init()
        _seed(seed, batch)
        seed.config.save(seed.root)
        order = ARMS if number % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            target = stores / batch / arm
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seed_root, target)
            run = _run(target, batch, arm)
            runs.append(run)
            keys = ("batch", "arm", "correct", "total", "wall_seconds", "error")
            print(json.dumps({key: run[key] for key in keys}), flush=True)
    summary = _summary(runs)
    (args.out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
