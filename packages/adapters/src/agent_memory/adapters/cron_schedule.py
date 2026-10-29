"""Opt-in user crontab integration for due-aware Dream wake-ups."""

from __future__ import annotations

import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

from agent_memory.core.config import STORE_ENV_VAR
from agent_memory.core.errors import FieldError, MemoryStoreError, ValidationError
from agent_memory.core.store import Store

DEFAULT_INTERVAL_MINUTES = 60
SUPPORTED_INTERVAL_MINUTES = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
LOG_FILENAME = "cron.log"
MARKER_PREFIX = "agent-memory dream"


def install(
    store: Store,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    mem_path: str | None = None,
    log_path: str | None = None,
    env_file: str | None = None,
) -> dict[str, object]:
    """Install or replace this Store's one managed crontab block."""
    if not (store.root / "config.toml").is_file():
        raise ValidationError(
            [FieldError("store", f"not initialized; run mem --store {store.root} init first")]
        )
    _validate_value("store", str(store.root))
    crontab = _require_crontab()
    interval = _validate_interval(interval_minutes)
    executable = _resolve_mem(mem_path)
    log = _resolve_log(store, log_path)
    environment = _resolve_env_file(env_file)
    home = _required_environment("HOME")
    path = os.environ.get("PATH") or os.defpath
    _validate_value("PATH", path)

    schedule = _schedule(interval)
    prefix = (
        f"HOME={_quote(home)}",
        f"PATH={_quote(path)}",
        f"{STORE_ENV_VAR}={_quote(str(store.root))}",
    )
    invocation = (
        _env_file_invocation(environment, executable)
        if environment is not None
        else f"{_quote(str(executable))} --json sleep --if-due --trigger-source cron"
    )
    command = " ".join((*prefix, invocation, f">> {_quote(str(log))} 2>&1"))
    identity = _identity(store)
    begin, end = _markers(identity)
    block = [begin, f"{schedule} {command}", end]
    current = _read_crontab(crontab)
    retained, found = _without_block(current, begin, end)
    updated = _append_block(retained, block)
    changed = updated != current
    if changed:
        log.parent.mkdir(parents=True, exist_ok=True)
        _write_crontab(crontab, updated)
    return _payload(
        store,
        identity,
        installed=True,
        changed=changed,
        action="updated" if found and changed else "installed" if changed else "unchanged",
        schedule=schedule,
        interval_minutes=interval,
        command=command,
        log=log,
        env_file=environment,
        crontab=crontab,
    )


def status(store: Store) -> dict[str, object]:
    """Report this Store's managed block and whether the scheduler is running."""
    crontab = _require_crontab()
    identity = _identity(store)
    begin, end = _markers(identity)
    current = _read_crontab(crontab)
    block = _find_block(current, begin, end)
    entry = next((line for line in block if line and not line.startswith("#")), "")
    return _payload(
        store,
        identity,
        installed=bool(block),
        changed=False,
        action="installed" if block else "not-installed",
        entry=entry,
        crontab=crontab,
    )


def remove(store: Store) -> dict[str, object]:
    """Remove only this Store's managed block, preserving every other crontab line."""
    crontab = _require_crontab()
    identity = _identity(store)
    begin, end = _markers(identity)
    current = _read_crontab(crontab)
    updated, found = _without_block(current, begin, end)
    if found:
        _write_crontab(crontab, updated)
    return _payload(
        store,
        identity,
        installed=False,
        changed=found,
        action="removed" if found else "not-installed",
        crontab=crontab,
    )


def _payload(
    store: Store,
    identity: str,
    *,
    installed: bool,
    changed: bool,
    action: str,
    crontab: str,
    schedule: str = "",
    interval_minutes: int | None = None,
    command: str = "",
    log: pathlib.Path | None = None,
    env_file: pathlib.Path | None = None,
    entry: str = "",
) -> dict[str, object]:
    wsl = _is_wsl()
    running = _scheduler_running()
    payload: dict[str, object] = {
        "status": action,
        "installed": installed,
        "changed": changed,
        "store": str(store.root),
        "id": identity,
        "platform": "wsl" if wsl else sys.platform,
        "crontab": crontab,
        "scheduler_running": running,
    }
    if schedule:
        payload["schedule"] = schedule
    if interval_minutes is not None:
        payload["interval_minutes"] = interval_minutes
    if command:
        payload["command"] = command
    if log is not None:
        payload["log"] = str(log)
    if env_file is not None:
        payload["env_file"] = str(env_file)
    if entry:
        payload["entry"] = entry
    if wsl and running is not True:
        payload["warning"] = (
            "cron is not running in WSL; the entry is configured but cannot wake Memory "
            "until the distribution's cron service is running"
        )
    elif running is False:
        payload["warning"] = "cron is not running; the configured entry will not execute"
    elif running is None:
        payload["warning"] = "could not verify whether the cron scheduler is running"
    return payload


def _resolve_mem(explicit: str | None) -> pathlib.Path:
    candidate = explicit
    if candidate is None:
        invoked = pathlib.Path(sys.argv[0]).expanduser()
        if invoked.name == "mem" and invoked.exists():
            candidate = str(invoked)
        else:
            candidate = shutil.which("mem")
    if not candidate:
        raise ValidationError(
            [FieldError("mem-path", "mem was not found; pass its absolute executable path")]
        )
    _validate_value("mem-path", candidate)
    path = pathlib.Path(candidate).expanduser().resolve()
    _validate_value("mem-path", str(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValidationError([FieldError("mem-path", f"not an executable file: {path}")])
    return path


def _resolve_log(store: Store, explicit: str | None) -> pathlib.Path:
    raw = explicit or str(store.layout.state_dir / LOG_FILENAME)
    _validate_value("log", raw)
    path = pathlib.Path(raw).expanduser().resolve()
    _validate_value("log", str(path))
    return path


def _resolve_env_file(explicit: str | None) -> pathlib.Path | None:
    if explicit is None:
        return None
    _validate_value("env-file", explicit)
    path = pathlib.Path(explicit).expanduser().resolve()
    _validate_value("env-file", str(path))
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValidationError([FieldError("env-file", f"not a readable file: {path}")])
    if path.stat().st_mode & 0o077:
        raise ValidationError(
            [FieldError("env-file", "must not be readable or writable by group or others")]
        )
    return path


def _env_file_invocation(environment: pathlib.Path, executable: pathlib.Path) -> str:
    script = 'set -a; . "$1"; set +a; exec "$2" --json sleep --if-due --trigger-source cron'
    return " ".join(
        (
            "/bin/sh",
            "-c",
            _quote(script),
            "agent-memory-cron",
            _quote(str(environment)),
            _quote(str(executable)),
        )
    )


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValidationError([FieldError(name, f"{name} must be set when installing cron")])
    _validate_value(name, value)
    return value


def _validate_value(field: str, value: str) -> None:
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValidationError([FieldError(field, "must not contain a newline or NUL")])


def _validate_interval(value: int) -> int:
    if value not in SUPPORTED_INTERVAL_MINUTES:
        rendered = ", ".join(str(item) for item in SUPPORTED_INTERVAL_MINUTES)
        raise ValidationError(
            [
                FieldError(
                    "interval-minutes",
                    f"must evenly divide an hour; choose one of: {rendered}",
                )
            ]
        )
    return value


def _schedule(interval_minutes: int) -> str:
    return "0 * * * *" if interval_minutes == 60 else f"*/{interval_minutes} * * * *"


def _quote(value: str) -> str:
    # Vixie cron treats an unescaped percent as a newline before invoking the shell.
    return shlex.quote(value).replace("%", r"\%")


def _identity(store: Store) -> str:
    return hashlib.sha256(str(store.root).encode("utf-8")).hexdigest()[:16]


def _markers(identity: str) -> tuple[str, str]:
    return (
        f"# BEGIN {MARKER_PREFIX} {identity}",
        f"# END {MARKER_PREFIX} {identity}",
    )


def _append_block(crontab: str, block: list[str]) -> str:
    lines = crontab.splitlines()
    if lines and lines[-1]:
        lines.append("")
    lines.extend(block)
    return "\n".join(lines) + "\n"


def _find_block(crontab: str, begin: str, end: str) -> list[str]:
    lines = crontab.splitlines()
    for index, line in enumerate(lines):
        if line != begin:
            continue
        try:
            finish = lines.index(end, index + 1)
        except ValueError as error:
            message = f"managed cron block is missing its end marker: {begin}"
            raise MemoryStoreError(message) from error
        return lines[index : finish + 1]
    return []


def _without_block(crontab: str, begin: str, end: str) -> tuple[str, bool]:
    lines = crontab.splitlines()
    retained: list[str] = []
    found = False
    index = 0
    while index < len(lines):
        if lines[index] != begin:
            retained.append(lines[index])
            index += 1
            continue
        found = True
        try:
            finish = lines.index(end, index + 1)
        except ValueError as error:
            message = f"managed cron block is missing its end marker: {begin}"
            raise MemoryStoreError(message) from error
        index = finish + 1
        if retained and not retained[-1] and index == len(lines):
            retained.pop()
    return ("\n".join(retained) + ("\n" if retained else ""), found)


def _require_crontab() -> str:
    if os.name != "posix":
        raise MemoryStoreError("cron integration is supported only on POSIX platforms")
    binary = shutil.which("crontab")
    if binary is None:
        raise MemoryStoreError("crontab is not installed on this platform")
    return binary


def _read_crontab(binary: str) -> str:
    result = subprocess.run(
        [binary, "-l"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    if result.returncode == 1 and "no crontab" in result.stderr.lower():
        return ""
    detail = result.stderr.strip() or result.returncode
    raise MemoryStoreError(f"could not read user crontab: {detail}")


def _write_crontab(binary: str, content: str) -> None:
    result = subprocess.run(
        [binary, "-"],
        input=content,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MemoryStoreError(
            f"could not update user crontab: {result.stderr.strip() or result.returncode}"
        )


def _is_wsl() -> bool:
    try:
        release = pathlib.Path("/proc/sys/kernel/osrelease").read_text("utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower()


def _scheduler_running() -> bool | None:
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return None
    for name in ("cron", "crond"):
        result = subprocess.run(
            [pgrep, "-x", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        if result.returncode not in (1,):
            return None
    return False
