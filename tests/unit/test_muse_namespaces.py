import json

from agent_memory.executor import muse_namespaces


def test_namespaced_mcp_tools_are_flattened_to_muse_canonical_ids():
    payload = {
        "model": "muse-spark",
        "tools": [
            {
                "type": "namespace",
                "name": "mcp__memory",
                "description": "Memory tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "recall",
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "submit_operations",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            {"type": "function", "name": "ordinary", "parameters": {}},
        ],
    }

    flattened = muse_namespaces.flatten_namespaced_tools(payload)

    assert [tool["name"] for tool in flattened["tools"]] == [
        "mcp__memory__recall",
        "mcp__memory__submit_operations",
        "ordinary",
    ]
    assert payload["tools"][0]["type"] == "namespace"
    assert flattened["tools"][0]["description"] == "Search"


def test_muse_settings_route_through_a_scoped_bridge(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "endpoint_transport": {
                    "auth": "bearer",
                    "base_url": "http://127.0.0.1:8817",
                }
            }
        ),
        encoding="utf-8",
    )
    events = []

    class FakeBridge:
        url = "http://127.0.0.1:43210"

        def __init__(self, upstream):
            events.append(("init", upstream))

        def start(self):
            events.append(("start", self.url))
            return self

        def close(self):
            events.append(("close", self.url))

    monkeypatch.setattr(muse_namespaces, "_pproxy_url", lambda _path: "http://127.0.0.1:8817")
    monkeypatch.setattr(muse_namespaces, "Bridge", FakeBridge)

    with muse_namespaces.for_muse_settings(settings) as bridge:
        routed = json.loads(settings.read_text(encoding="utf-8"))
        assert bridge.url == "http://127.0.0.1:43210"
        assert routed["endpoint_transport"]["base_url"] == bridge.url

    assert events == [
        ("init", "http://127.0.0.1:8817"),
        ("start", "http://127.0.0.1:43210"),
        ("close", "http://127.0.0.1:43210"),
    ]
    restored = json.loads(settings.read_text(encoding="utf-8"))
    assert restored["endpoint_transport"]["base_url"] == "http://127.0.0.1:8817"
