"""The Write toolkit: everything a reasoner may do while it distills one batch.

Recall and read let it look; submitting operations is how it answers. The toolkit is the
only definition of those tools — the endpoint loop calls it in process and the MCP adapter
serves it to a host — so the same call yields the same result through every transport
(Invariant 8). Tools never write the Store: an accepted submission is handed back to the
library, which applies it through the single write path (Invariant 2).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import tempfile

from . import reconcile
from .recall import Recall
from .sessions import Message, Pointer
from .store import Store

TOOL_RECALL = "recall"
TOOL_READ = "read"
TOOL_SUBMIT = "submit_operations"
TOOLS = (TOOL_RECALL, TOOL_READ, TOOL_SUBMIT)
ARG_QUERY = "query"
ARG_LIMIT = "limit"
ARG_NAME = "name"
ARG_OPERATIONS = "operations"
SEED_FILENAME = "seed.json"
RECORD_FILENAME = "record.json"
NOTHING_FOUND = "(nothing found)"

OPERATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(reconcile.OPS)},
        "type": {"type": "string", "description": "A type from the type table."},
        "fields": {"type": "object", "additionalProperties": {"type": "string"}},
        "abstract": {"type": "string", "description": "The one line that states the fact."},
        "body": {"type": "string", "description": "Markdown with the specifics in full."},
        "handle": {"type": "string", "description": "The memory an update or supersede names."},
        "valid_from": {"type": "string", "description": "When the fact starts to hold."},
        "create_group": {"type": "boolean"},
        "provenance": {
            "type": "array",
            "items": {"type": "string"},
            "description": 'Message ranges the memory comes from, such as "3-5" or "7".',
        },
    },
    "required": ["op"],
}


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, object]


@dataclasses.dataclass(frozen=True)
class ToolReply:
    text: str
    is_error: bool = False


@dataclasses.dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]
    reply: ToolReply


CATALOGUE = (
    ToolSpec(
        TOOL_RECALL,
        "Search current memory. Lists matching memories by handle; a listed handle may then be "
        "named in update and supersede.",
        {
            "type": "object",
            "properties": {
                ARG_QUERY: {"type": "string", "description": "Words the memory would contain."},
                ARG_LIMIT: {"type": "integer", "minimum": 1},
            },
            "required": [ARG_QUERY],
        },
    ),
    ToolSpec(
        TOOL_READ,
        "Open one active memory in full by its handle.",
        {
            "type": "object",
            "properties": {ARG_NAME: {"type": "string"}},
            "required": [ARG_NAME],
        },
    ),
    ToolSpec(
        TOOL_SUBMIT,
        "Submit every operation for this conversation. Refused operations come back with "
        "reasons; submit the corrected full list again. The latest accepted submission is the "
        "answer. Use op skip when the conversation holds nothing worth keeping.",
        {
            "type": "object",
            "properties": {
                ARG_OPERATIONS: {"type": "array", "items": OPERATION_SCHEMA, "minItems": 1}
            },
            "required": [ARG_OPERATIONS],
        },
    ),
)


class WriteToolkit:
    def __init__(self, store: Store, sheet: reconcile.Sheet):
        self.store = store
        self.sheet = sheet
        self.submission: list[dict[str, object]] | None = None
        self.calls: list[ToolCall] = []

    def catalogue(self) -> tuple[ToolSpec, ...]:
        return CATALOGUE

    def call(self, name: str, arguments: object) -> ToolReply:
        given = arguments if isinstance(arguments, dict) else {}
        if name == TOOL_RECALL:
            reply = self._recall(given)
        elif name == TOOL_READ:
            reply = self._read(given)
        elif name == TOOL_SUBMIT:
            reply = self._submit(given)
        else:
            reply = ToolReply(f"unknown tool; the tools are {', '.join(TOOLS)}", True)
        self.calls.append(ToolCall(name, given, reply))
        return reply

    def transcript(self) -> str:
        return "\n\n".join(
            f"[{call.name}] {json.dumps(call.arguments, sort_keys=True)}\n{call.reply.text}"
            for call in self.calls
        )

    def _recall(self, arguments: dict[str, object]) -> ToolReply:
        query = arguments.get(ARG_QUERY)
        if not isinstance(query, str) or not query.strip():
            return ToolReply("recall needs a query", True)
        limit = arguments.get(ARG_LIMIT)
        count = limit if isinstance(limit, int) and limit > 0 else None
        hits = Recall(self.store).recall(
            query.strip(), limit=count or self.store.config.write.reconcile_entries, log=False
        )
        found = tuple(
            reconcile.Handle(hit.name, hit.type, hit.status, hit.updated, hit.abstract)
            for hit in hits
        )
        self._extend(found)
        return ToolReply("\n".join(handle.render() for handle in found) or NOTHING_FOUND)

    def _read(self, arguments: dict[str, object]) -> ToolReply:
        name = arguments.get(ARG_NAME)
        record = self.store.find(name.strip()) if isinstance(name, str) and name.strip() else None
        if record is None or not record.is_active():
            return ToolReply(f"no active memory named {name}", True)
        handle = reconcile.Handle(
            record.name, record.type, record.status, record.valid_from or "", record.abstract
        )
        self._extend((handle,))
        return ToolReply(f"{handle.render()}\n\n{record.body.strip()}")

    def _submit(self, arguments: dict[str, object]) -> ToolReply:
        operations = arguments.get(ARG_OPERATIONS)
        if not isinstance(operations, list) or not operations:
            return ToolReply(
                "submit one or more operations as a list; use op skip when the conversation "
                "holds nothing worth keeping",
                True,
            )
        problems = []
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                problems.append(f"operation {index}: must be an object")
                continue
            problems.extend(
                f"operation {index}: {error.field}: {error.reason}"
                for error in reconcile.check(operation, self.sheet)
            )
        if not problems:
            problems = self._store_refusals(operations)
        if problems:
            return ToolReply("\n".join(["Nothing was accepted.", *problems]), True)
        self.submission = list(operations)
        return ToolReply(f"accepted {len(operations)} operations")

    def _store_refusals(self, operations: list[dict[str, object]]) -> list[str]:
        writes = [
            (index, record_spec)
            for index, operation in enumerate(operations)
            if (record_spec := reconcile.to_record_spec(operation, self.sheet)) is not None
        ]
        refused = self.store.check_many([record_spec for _, record_spec in writes])
        return [
            f"operation {writes[item.index][0]}: {error.field}: {error.reason}"
            for item in refused
            for error in item.errors
        ]

    def _extend(self, found: tuple[reconcile.Handle, ...]) -> None:
        known = self.sheet.handle_names()
        fresh = tuple(handle for handle in found if handle.name not in known)
        if fresh:
            self.sheet = dataclasses.replace(self.sheet, handles=self.sheet.handles + fresh)

    def seed(self) -> dict[str, object]:
        sheet = self.sheet
        return {
            "store": str(self.store.root),
            "session": sheet.session,
            "pointer": [sheet.pointer.session, sheet.pointer.start, sheet.pointer.end],
            "handles": [dataclasses.asdict(handle) for handle in sheet.handles],
            "menus": {key: list(values) for key, values in sheet.menus.items()},
            "profile": list(sheet.profile),
            "messages": [message.as_dict() for message in sheet.messages],
        }

    @classmethod
    def from_seed(cls, store: Store, seed: dict[str, object]) -> WriteToolkit:
        session, start, end = _list(seed["pointer"])
        sheet = reconcile.Sheet(
            session=str(seed["session"]),
            pointer=Pointer(str(session), int(start), int(end)),
            handles=tuple(reconcile.Handle(**handle) for handle in _list(seed["handles"])),
            menus={key: tuple(values) for key, values in _dict(seed["menus"]).items()},
            profile=tuple(_list(seed["profile"])),
            slots=tuple(store.schemas.all()),
            messages=tuple(Message(**message) for message in _list(seed["messages"])),
        )
        return cls(store, sheet)

    def record(self) -> dict[str, object]:
        return {
            "submission": self.submission,
            "handles": [dataclasses.asdict(handle) for handle in self.sheet.handles],
            "calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "text": call.reply.text,
                    "is_error": call.reply.is_error,
                }
                for call in self.calls
            ],
        }

    def absorb(self, record: dict[str, object]) -> None:
        submission = record.get("submission")
        self.submission = list(submission) if isinstance(submission, list) else None
        self._extend(tuple(reconcile.Handle(**handle) for handle in _list(record["handles"])))
        self.calls.extend(
            ToolCall(
                str(call["name"]),
                _dict(call["arguments"]),
                ToolReply(str(call["text"]), bool(call["is_error"])),
            )
            for call in _list(record["calls"])
        )

    def save_seed(self, directory: pathlib.Path) -> None:
        _write_json(directory / SEED_FILENAME, self.seed())

    def save_record(self, directory: pathlib.Path) -> None:
        _write_json(directory / RECORD_FILENAME, self.record())

    def absorb_directory(self, directory: pathlib.Path) -> None:
        path = directory / RECORD_FILENAME
        if path.exists():
            self.absorb(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_directory(cls, directory: pathlib.Path) -> WriteToolkit:
        seed = json.loads((directory / SEED_FILENAME).read_text(encoding="utf-8"))
        return cls.from_seed(Store(str(seed["store"]), agent="write-toolkit"), seed)


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)
