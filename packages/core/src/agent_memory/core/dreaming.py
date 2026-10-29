"""Persistent, event-driven automatic Dream claims.

There is deliberately no scheduler here. A completed boundary distillation calls ``claim_due``;
the caller may then spawn one short-lived worker. State and advisory locks make that hand-off
recoverable across processes without a resident service.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import tempfile

from . import timestamp
from .config import DREAM_CLAIM_ID_LENGTH, DREAM_ERROR_MAX_CHARS, STATE_JSON_INDENT
from .locking import file_lock, lock_is_held
from .manage import Manage
from .pending import Pending
from .store import Store
from .watermark import Mark, Watermark

STATE_VERSION = 1
STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_SUCCEEDED = "succeeded"


@dataclasses.dataclass(frozen=True)
class DreamClaim:
    id: str
    host: str
    sessions: tuple[str, ...]
    attempt: int
    trigger_source: str = ""
    reasoner: str = ""
    model: str = ""


@dataclasses.dataclass(frozen=True)
class ClaimResult:
    claim: DreamClaim | None
    reason: str
    sessions_since: int = 0
    due: bool = False


def completed_sessions(store: Store) -> dict[str, Mark]:
    """Distinct durable sessions whose captured Write work has completely settled."""
    watermark = Watermark(store.layout, store.clock)
    pending = Pending(store.layout)
    blocked = set(pending.sessions()) | set(pending.redistill_sessions())
    found: dict[str, Mark] = {}
    for session in watermark.sessions():
        mark = watermark.read(session)
        if mark.consumed > 0 and mark.distilled >= mark.consumed and session not in blocked:
            found[session] = mark
    return found


def write_is_settled(store: Store) -> bool:
    watermark = Watermark(store.layout, store.clock)
    if Pending(store.layout).sessions() or Pending(store.layout).redistill_sessions():
        return False
    return all(watermark.read(session).backlog() == 0 for session in watermark.sessions())


def claim_due(
    store: Store,
    host: str,
    *,
    trigger_source: str = "hook",
    reasoner: str = "host",
    model: str = "",
) -> ClaimResult:
    """Atomically claim one due Dream, or explain why no worker should be launched."""
    if not store.config.manage.auto_dream:
        return ClaimResult(None, "automatic dreaming disabled")
    store.layout.ensure()
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        state, existed = _load(store)
        state = _recover_report(store, state)
        now = store.clock.now()
        if state.get("status") == STATUS_RUNNING:
            expires = _instant(state.get("lease_expires_at"))
            if expires is not None and now < expires:
                _save(store, state)
                return ClaimResult(None, "dream claim active", due=True)
            if lock_is_held(store.layout.dream_run_lock):
                state["lease_expires_at"] = _lease_until(store, now)
                _save(store, state)
                return ClaimResult(None, "dream worker still running", due=True)
            state["status"] = STATUS_FAILED
            state["last_error"] = "claim lease expired before completion"

        complete = completed_sessions(store)
        if not existed and Manage(store).last_sleep() is not None:
            last = Manage(store).last_sleep()
            assert last is not None
            state["covered_sessions"] = sorted(
                session
                for session, mark in complete.items()
                if (_instant(mark.updated_at) or dt.datetime.min.replace(tzinfo=dt.UTC)) <= last
            )
        covered = set(_strings(state.get("covered_sessions")))
        sessions = tuple(sorted(set(complete) - covered))
        if not write_is_settled(store):
            _save(store, state)
            return ClaimResult(None, "write pipeline not settled", len(sessions))
        if not Manage(store).due(len(sessions)):
            _save(store, state)
            return ClaimResult(None, "trigger conditions not met", len(sessions))

        claim_id = _claim_id(Manage(store).last_sleep(), sessions)
        previous_id = str(state.get("claim_id") or "")
        attempt = _integer(state.get("attempt"), 0) + 1 if previous_id == claim_id else 1
        state.update(
            {
                "version": STATE_VERSION,
                "status": STATUS_RUNNING,
                "claim_id": claim_id,
                "host": host,
                "trigger_source": trigger_source,
                "reasoner": reasoner,
                "model": model,
                "sessions": list(sessions),
                "attempt": attempt,
                "claimed_at": now.isoformat(),
                "lease_expires_at": _lease_until(store, now),
                "last_error": "",
            }
        )
        _save(store, state)
        return ClaimResult(
            DreamClaim(
                claim_id,
                host,
                sessions,
                attempt,
                trigger_source=trigger_source,
                reasoner=reasoner,
                model=model,
            ),
            "claimed",
            len(sessions),
            due=True,
        )


def current_claim(store: Store, claim_id: str) -> DreamClaim | None:
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        state = _recover_report(store, _load(store)[0])
        _save(store, state)
        if state.get("status") != STATUS_RUNNING or state.get("claim_id") != claim_id:
            return None
        return DreamClaim(
            id=claim_id,
            host=str(state.get("host") or ""),
            sessions=tuple(_strings(state.get("sessions"))),
            attempt=_integer(state.get("attempt"), 1),
            trigger_source=str(state.get("trigger_source") or ""),
            reasoner=str(state.get("reasoner") or ""),
            model=str(state.get("model") or ""),
        )


def mark_succeeded(store: Store, claim: DreamClaim, report: str) -> bool:
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        state = _load(store)[0]
        if state.get("claim_id") != claim.id:
            return False
        covered = set(_strings(state.get("covered_sessions")))
        covered.update(claim.sessions)
        state.update(
            {
                "status": STATUS_SUCCEEDED,
                "covered_sessions": sorted(covered),
                "last_completed_at": store.clock.now().isoformat(),
                "last_report": report,
                "last_error": "",
                "lease_expires_at": "",
            }
        )
        _save(store, state)
        return True


def mark_failed(store: Store, claim_id: str, detail: str) -> bool:
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        state = _load(store)[0]
        if state.get("claim_id") != claim_id or state.get("status") != STATUS_RUNNING:
            return False
        state.update(
            {
                "status": STATUS_FAILED,
                "last_error": detail.strip()[:DREAM_ERROR_MAX_CHARS],
                "lease_expires_at": "",
            }
        )
        _save(store, state)
        return True


def record_manual_success(store: Store, report: str) -> None:
    """A manual sleep resets the same durable session baseline as an automatic one."""
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        state = _load(store)[0]
        state.update(
            {
                "version": STATE_VERSION,
                "status": STATUS_SUCCEEDED,
                "claim_id": "",
                "host": "manual",
                "trigger_source": "manual",
                "reasoner": "",
                "model": "",
                "sessions": [],
                "covered_sessions": sorted(completed_sessions(store)),
                "last_completed_at": store.clock.now().isoformat(),
                "last_report": report,
                "last_error": "",
                "lease_expires_at": "",
            }
        )
        _save(store, state)


def report_path(store: Store, claim_id: str) -> pathlib.Path:
    return store.layout.dream_reports / f"auto-{claim_id}.md"


def distill_lock_path(store: Store, session: str) -> pathlib.Path:
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()
    return store.layout.distill_locks / f"{digest}.lock"


def state(store: Store) -> dict[str, object]:
    with file_lock(
        store.layout.dream_claim_lock,
        timeout=store.config.storage.lock_timeout_seconds,
    ):
        current = _recover_report(store, _load(store)[0])
        _save(store, current)
        return current


def _recover_report(store: Store, state: dict[str, object]) -> dict[str, object]:
    claim_id = str(state.get("claim_id") or "")
    if not claim_id or state.get("status") == STATUS_SUCCEEDED:
        return state
    report = report_path(store, claim_id)
    if not report.is_file():
        return state
    covered = set(_strings(state.get("covered_sessions")))
    covered.update(_strings(state.get("sessions")))
    state.update(
        {
            "status": STATUS_SUCCEEDED,
            "covered_sessions": sorted(covered),
            "last_completed_at": _report_at(report) or store.clock.now().isoformat(),
            "last_report": str(report),
            "last_error": "",
            "lease_expires_at": "",
        }
    )
    return state


def _load(store: Store) -> tuple[dict[str, object], bool]:
    path = store.layout.dream_state
    if not path.is_file():
        return _empty(), False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**_empty(), "status": STATUS_FAILED, "last_error": "invalid dream state"}, True
    if not isinstance(payload, dict):
        return {**_empty(), "status": STATUS_FAILED, "last_error": "invalid dream state"}, True
    return {**_empty(), **payload}, True


def _empty() -> dict[str, object]:
    return {
        "version": STATE_VERSION,
        "status": STATUS_IDLE,
        "claim_id": "",
        "host": "",
        "trigger_source": "",
        "reasoner": "",
        "model": "",
        "sessions": [],
        "attempt": 0,
        "claimed_at": "",
        "lease_expires_at": "",
        "covered_sessions": [],
        "last_completed_at": "",
        "last_report": "",
        "last_error": "",
    }


def _save(store: Store, payload: dict[str, object]) -> None:
    path = store.layout.dream_state
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=STATE_JSON_INDENT, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(rendered)
        staged = pathlib.Path(tmp.name)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _claim_id(last_sleep: dt.datetime | None, sessions: tuple[str, ...]) -> str:
    material = (last_sleep.isoformat() if last_sleep else "first") + "|" + "|".join(sessions)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:DREAM_CLAIM_ID_LENGTH]


def _lease_until(store: Store, now: dt.datetime) -> str:
    return (now + dt.timedelta(seconds=store.config.manage.claim_lease_seconds)).isoformat()


def _instant(value: object) -> dt.datetime | None:
    if not str(value or "").strip():
        return None
    try:
        return timestamp.parse(str(value))
    except ValueError:
        return None


def _report_at(path: pathlib.Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return ""
    value = first.removeprefix("# dream report ").strip()
    return value if _instant(value) is not None else ""


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _integer(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default
