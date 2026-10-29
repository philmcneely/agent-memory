"""Compatibility bridge for Muse's namespaced MCP tools over OpenRouter.

Muse sends Responses API namespace tools to the model, but Muse Code 1.4.9 dispatches a
returned call by its scalar name. OpenRouter preserves the namespace separately, so a correct
``name=recall, namespace=mcp__memory`` call is rejected as an unknown ``recall`` tool. The
bridge expands namespaces in the outbound request. The model then returns the canonical
``mcp__memory__recall`` name Muse already knows how to dispatch.

This is a transport shim only. It never executes a tool, holds a model credential, or runs an
agent loop. Authorization passes from Muse through the shim to the already-running pproxy.
"""

from __future__ import annotations

import contextlib
import copy
import http.client
import json
import pathlib
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
HOP_HEADERS = {
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def flatten_namespaced_tools(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a request whose namespace children use Muse's canonical scalar ids."""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload
    flattened: list[object] = []
    changed = False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            flattened.append(tool)
            continue
        namespace = tool.get("name")
        children = tool.get("tools")
        if not isinstance(namespace, str) or not isinstance(children, list):
            flattened.append(tool)
            continue
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("name"), str):
                continue
            expanded = copy.deepcopy(child)
            expanded["type"] = "function"
            expanded["name"] = f"{namespace}__{child['name']}"
            flattened.append(expanded)
        changed = True
    if not changed:
        return payload
    result = dict(payload)
    result["tools"] = flattened
    return result


class Bridge:
    """A loopback HTTP forwarder with one request-body transformation."""

    def __init__(self, upstream: str):
        self.upstream = upstream.rstrip("/")
        handler = _handler(self.upstream)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> Bridge:
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextlib.contextmanager
def for_muse_settings(settings_path: pathlib.Path):
    """Wrap a local pproxy route and point the isolated settings at the bridge."""
    upstream = _pproxy_url(settings_path)
    if upstream is None:
        yield None
        return
    original = settings_path.read_bytes()
    bridge = Bridge(upstream).start()
    try:
        _replace_base_url(settings_path, bridge.url)
        yield bridge
    finally:
        bridge.close()
        settings_path.write_bytes(original)


def _pproxy_url(settings_path: pathlib.Path) -> str | None:
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    transport = settings.get("endpoint_transport") if isinstance(settings, dict) else None
    base_url = transport.get("base_url") if isinstance(transport, dict) else None
    if not isinstance(base_url, str):
        return None
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK:
        return None
    request = urllib.request.Request(base_url.rstrip("/") + "/healthz")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=2
        ) as response:
            health = json.loads(response.read())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(health, dict):
        return None
    upstream = health.get("upstream")
    signature = {"auth", "has_key", "model_filter", "ok", "upstream"}
    return (
        base_url
        if signature <= set(health) and health.get("ok") is True and isinstance(upstream, str)
        else None
    )


def _replace_base_url(settings_path: pathlib.Path, base_url: str) -> None:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    transport = settings.get("endpoint_transport")
    if not isinstance(transport, dict):
        return
    transport["base_url"] = base_url
    settings_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _handler(upstream: str) -> type[BaseHTTPRequestHandler]:
    target = urllib.parse.urlsplit(upstream)
    hostname = target.hostname or ""
    if not hostname:
        raise ValueError("namespace bridge upstream needs a hostname")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._forward()

        def do_POST(self) -> None:  # noqa: N802
            self._forward()

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _forward(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length) if length else b""
            if self.command == "POST" and self.path.rstrip("/") == "/responses":
                body = _flatten_body(body)
            connection_type = (
                http.client.HTTPSConnection
                if target.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_type(hostname, target.port, timeout=600)
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS
            }
            path = (target.path.rstrip("/") + "/" + self.path.lstrip("/")).rstrip("/")
            try:
                connection.request(self.command, path or "/", body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                self.close_connection = True
                connection.close()

    return Handler


def _flatten_body(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if not isinstance(payload, dict):
        return body
    return json.dumps(flatten_namespaced_tools(payload), separators=(",", ":")).encode()
