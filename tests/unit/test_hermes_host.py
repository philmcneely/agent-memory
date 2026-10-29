"""Hermes native Write runs in an isolated profile over one exact MCP toolset."""

from __future__ import annotations

import json
import pathlib
import subprocess

from agent_memory.executor import hosts


def _host(*, timeout_seconds=5.0):
    return hosts.Host(
        hosts.HostSpec(
            name=hosts.HOST_HERMES,
            binary="hermes",
            model="meta/muse-spark-1.3-contributor",
            provider="openrouter",
            allow_data_training=True,
            timeout_seconds=timeout_seconds,
            attempts=1,
        )
    )


def _server(tmp_path):
    return hosts.ToolServer(
        name="memory",
        command="/venv/bin/python",
        args=("-m", "agent_memory.mcp.write_session", str(tmp_path / "batch")),
        tools=("recall", "read", "submit_operations"),
    )


def _manage_server(tmp_path):
    return hosts.ToolServer(
        name="memory_review",
        command="/venv/bin/python",
        args=("-m", "agent_memory.mcp.manage_session", str(tmp_path / "review")),
        tools=("list_proposals", "read_evidence", "submit_verdicts"),
        host_analytics_enabled=False,
    )


def test_hermes_native_write_stages_only_the_batch_server(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "must-not-trigger-gcloud")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-root"))

    def run(command, **kwargs):
        environment = kwargs["env"]
        home = pathlib.Path(environment["HERMES_HOME"])
        observed["command"] = command
        observed["environment"] = environment
        observed["config"] = json.loads((home / "config.yaml").read_text(encoding="utf-8"))
        observed["prompt"] = pathlib.Path(command[command.index("--query-file") + 1]).read_text(
            encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(hosts.subprocess, "run", run)
    result = _host().converse(
        "session evidence",
        system_prompt="distill with submit_operations",
        server=_server(tmp_path),
    )

    assert result.ok
    assert result.text == "done"
    assert pathlib.Path(observed["environment"]["HERMES_HOME"]).parent == (
        tmp_path / "hermes-root" / "profiles"
    )
    assert not pathlib.Path(observed["environment"]["HERMES_HOME"]).exists()
    assert observed["environment"]["HERMES_DISABLE_LAZY_INSTALLS"] == "1"
    assert observed["environment"]["OPENROUTER_API_KEY"] == "test-openrouter-key"
    assert observed["prompt"] == "distill with submit_operations\n\nsession evidence"
    assert set(observed["config"]) == {"mcp_servers", "security"}
    assert observed["config"]["security"] == {"allow_data_training_tiers_noninteractive": True}
    assert set(observed["config"]["mcp_servers"]) == {hosts.HERMES_WRITE_SERVER_NAME}
    configured = observed["config"]["mcp_servers"][hosts.HERMES_WRITE_SERVER_NAME]
    assert configured["command"] == "/venv/bin/python"
    assert configured["tools"] == {
        "include": ["recall", "read", "submit_operations"],
        "prompts": False,
        "resources": False,
    }


def test_hermes_native_manage_stages_only_the_review_server(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-root"))

    def run(command, **kwargs):
        home = pathlib.Path(kwargs["env"]["HERMES_HOME"])
        observed["command"] = command
        observed["config"] = json.loads((home / "config.yaml").read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0, stdout="review submitted\n", stderr="")

    monkeypatch.setattr(hosts.subprocess, "run", run)
    result = _host().converse(
        "proposal summaries",
        system_prompt="review with submit_verdicts",
        server=_manage_server(tmp_path),
    )

    assert result.ok
    assert observed["command"][observed["command"].index("--toolsets") + 1] == (
        hosts.HERMES_MANAGE_SERVER_NAME
    )
    assert "terminal" not in observed["command"]
    assert set(observed["config"]["mcp_servers"]) == {hosts.HERMES_MANAGE_SERVER_NAME}
    configured = observed["config"]["mcp_servers"][hosts.HERMES_MANAGE_SERVER_NAME]
    assert configured["args"][:2] == ["-m", "agent_memory.mcp.manage_session"]
    assert configured["tools"] == {
        "include": ["list_proposals", "read_evidence", "submit_verdicts"],
        "prompts": False,
        "resources": False,
    }
    assert observed["config"]["security"] == {"allow_data_training_tiers_noninteractive": True}


def test_hermes_native_write_timeout_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-root"))

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("hermes", 0.01)

    monkeypatch.setattr(hosts.subprocess, "run", timeout)
    result = _host(timeout_seconds=0.01).converse(
        "material", system_prompt="instructions", server=_server(tmp_path)
    )

    assert not result.ok
    assert result.error == "timeout"


def test_hermes_text_fallback_uses_the_same_isolated_profile(monkeypatch, tmp_path):
    observed = {}
    root = tmp_path / "hermes-root"
    monkeypatch.setenv("HERMES_HOME", str(root))

    def run(command, **kwargs):
        home = pathlib.Path(kwargs["env"]["HERMES_HOME"])
        observed["home"] = home
        observed["config"] = json.loads((home / "config.yaml").read_text(encoding="utf-8"))
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="answer\n", stderr="")

    monkeypatch.setattr(hosts.subprocess, "run", run)
    result = _host().run("prompt")

    assert result.ok
    assert observed["home"].parent == root / "profiles"
    assert not observed["home"].exists()
    assert observed["config"] == {"security": {"allow_data_training_tiers_noninteractive": True}}
    assert "--ignore-user-config" not in observed["command"]
    assert observed["command"][observed["command"].index("-t") + 1] == ""


def test_old_hermes_native_cli_fails_explicitly_without_text_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-root"))
    monkeypatch.setattr(
        hosts.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, stdout="", stderr="error: unrecognized arguments: --query-file"
        ),
    )

    result = _host().converse("material", system_prompt="instructions", server=_server(tmp_path))

    assert not result.ok
    assert result.error == hosts.HERMES_NATIVE_TOOL_REQUIREMENT
