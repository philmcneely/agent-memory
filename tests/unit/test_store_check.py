"""Checking a batch is writing it without the write: same verdicts, nothing on disk."""

import hashlib
import pathlib

from agent_memory.core import sessions
from agent_memory.core.store import Store


def _fingerprint(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(pathlib.Path(root).rglob("*"))
        if path.is_file() and ".index" not in path.parts
    }


def _seed(store):
    store.record(
        type="fact",
        fields={"project": "pipeline", "subject": "queue timeout"},
        abstract="Queue timeout is 20 seconds",
        body="Set in infra/queue.yaml.",
    )
    pointer = store.archive.append_session("boundary", ["user: queue timeout is 30 seconds now"])
    return sessions.resolve(store.layout, pointer)[0]


def _specs(message):
    cited = f"sessions/boundary#{message.index}-{message.index}"
    return [
        {
            "type": "fact",
            "fields": {"project": "pipeline", "subject": "queue timeout 30s"},
            "abstract": "Queue timeout is 30 seconds",
            "provenance": [cited],
        },
        {"type": "fact", "abstract": "Named from its abstract"},
        {
            "type": "fact",
            "fields": {"project": "pipeline", "subject": "future"},
            "abstract": "Dated after its evidence",
            "valid_from": "2099-01-01",
            "provenance": [cited],
        },
        {
            "type": "fact",
            "name": "queue-timeout",
            "fields": {"project": "pipeline", "subject": "queue timeout"},
            "abstract": "Queue timeout is 20 seconds",
            "body": "A different fact in place.",
        },
        {"type": "no-such-type", "abstract": "Unknown type"},
        "not an object",
        {
            "type": "fact",
            "fields": {"project": "pipeline", "subject": "excerpt"},
            "abstract": "Cites an excerpt",
            "provenance": ["an excerpt that would be archived"],
        },
    ]


def _verdicts(rejected):
    return {item.index: sorted(error.field for error in item.errors) for item in rejected}


def test_checking_gives_the_verdicts_writing_would(tmp_path, config, clock):
    checked = Store(tmp_path / "checked", config=config, clock=clock, agent="t")
    written = Store(tmp_path / "written", config=config, clock=clock, agent="t")
    for store in (checked, written):
        store.init()
    specs_checked = _specs(_seed(checked))
    specs_written = _specs(_seed(written))
    verdicts = _verdicts(checked.check_many(specs_checked))
    assert verdicts == _verdicts(written.record_many(specs_written).rejected)
    assert set(verdicts) == {2, 3, 4, 5}


def test_checking_writes_nothing(store):
    message = _seed(store)
    before = _fingerprint(store.root)
    store.check_many(_specs(message))
    assert _fingerprint(store.root) == before


def test_an_empty_check_is_not_an_error(store):
    assert store.check_many([]) == []
