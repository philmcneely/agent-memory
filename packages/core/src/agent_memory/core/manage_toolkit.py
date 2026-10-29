"""Run-scoped tools for reviewing Manage proposals without mutation authority."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile

from . import reasoning, sessions, trace
from .errors import MemoryStoreError
from .store import Store
from .toolkit import ToolCall, ToolReply, ToolSpec

TOOL_LIST = "list_proposals"
TOOL_READ = "read_evidence"
TOOL_SUBMIT = "submit_verdicts"
TOOLS = (TOOL_LIST, TOOL_READ, TOOL_SUBMIT)
ARG_PROPOSAL = "proposal"
ARG_NAME = "name"
ARG_OFFSET = "offset"
ARG_POINTER = "pointer"
ARG_VERDICTS = "verdicts"
SEED_FILENAME = "seed.json"
RECORD_FILENAME = "record.json"
MIN_SPLIT_PARTS = 2
BISECTION_DIVISOR = 2
EVIDENCE_READ_FIELDS = 2
PROVENANCE_READ_FIELDS = 3
PROVENANCE_POINTER_INDEX = 2

VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        reasoning.FIELD_PROPOSAL: {"type": "string"},
        reasoning.FIELD_VERDICT: {
            "type": "string",
            "enum": [reasoning.VERDICT_ACCEPT, reasoning.VERDICT_REJECT],
        },
        reasoning.FIELD_TEXT: {"type": "string"},
        reasoning.FIELD_ABSTRACT: {"type": "string"},
        reasoning.FIELD_BODY: {"type": "string"},
        reasoning.FIELD_PARTS: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    reasoning.FIELD_ABSTRACT: {"type": "string"},
                    reasoning.FIELD_BODY: {"type": "string"},
                    reasoning.FIELD_PROVENANCE: {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [reasoning.FIELD_ABSTRACT, reasoning.FIELD_BODY],
            },
        },
    },
    "required": [reasoning.FIELD_PROPOSAL, reasoning.FIELD_VERDICT],
}

CATALOGUE = (
    ToolSpec(
        TOOL_LIST,
        "List every proposal in this Manage review run and its allowed evidence handles. "
        "This tool reads no Store content and cannot change memory.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        TOOL_READ,
        "Read a bounded page of evidence for one proposal target. Only targets listed for this "
        "run are readable. Use offset from the prior result to continue a long body. When this "
        "run explicitly enables raw Session evidence, pointer may name one complete Session "
        "pointer frozen on that same target; arbitrary, neighboring, and legacy sources are "
        "refused.",
        {
            "type": "object",
            "properties": {
                ARG_PROPOSAL: {"type": "string"},
                ARG_NAME: {"type": "string"},
                ARG_OFFSET: {"type": "integer", "minimum": 0},
                ARG_POINTER: {"type": "string"},
            },
            "required": [ARG_PROPOSAL, ARG_NAME],
        },
    ),
    ToolSpec(
        TOOL_SUBMIT,
        "Submit exactly one verdict for every proposal in this run. Invalid, duplicate, foreign, "
        "or stale proposals are refused. This tool never applies a verdict or writes the Store.",
        {
            "type": "object",
            "properties": {
                ARG_VERDICTS: {
                    "type": "array",
                    "items": VERDICT_SCHEMA,
                    "minItems": 1,
                }
            },
            "required": [ARG_VERDICTS],
        },
    ),
)


class ManageReviewToolkit:
    """The complete capability surface available to one native Manage reviewer."""

    def __init__(self, store: Store, request: reasoning.ReviewRequest):
        self.store = store
        self.request = request
        self.submission: tuple[reasoning.Verdict, ...] | None = None
        self.calls: list[ToolCall] = []
        self.evidence_chars = 0
        self.evidence_reads: list[tuple[str, str]] = []
        self.provenance_reads: list[tuple[str, str, str]] = []

    def catalogue(self) -> tuple[ToolSpec, ...]:
        return CATALOGUE

    def call(self, name: str, arguments: object) -> ToolReply:
        given = arguments if isinstance(arguments, dict) else {}
        budget = min(self.request.limits.max_tool_calls, self.request.limits.max_steps)
        if len(self.calls) >= budget:
            reply = ToolReply("review tool-call budget exhausted", True)
        elif name == TOOL_LIST:
            reply = self._list(given)
        elif name == TOOL_READ:
            reply = self._read(given)
        elif name == TOOL_SUBMIT:
            reply = self._submit(given)
        else:
            reply = ToolReply(f"unknown tool; the tools are {', '.join(TOOLS)}", True)
        reply = self._bounded(reply)
        self.calls.append(ToolCall(name, given, reply))
        return reply

    def outcome(self) -> reasoning.ReviewOutcome:
        submitted = self.submission is not None
        return reasoning.ReviewOutcome(
            verdicts=self.submission or (),
            submitted=submitted,
            complete=submitted,
            mode="native",
            rounds=min(len(self.calls) + 1, self.request.limits.max_steps),
            tool_calls=len(self.calls),
            evidence_reads=tuple(self.evidence_reads),
            provenance_reads=tuple(self.provenance_reads),
            tool_trace=tuple((call.name, call.reply.is_error) for call in self.calls),
            error="" if submitted else "native reviewer submitted no complete verdict set",
        )

    def _list(self, arguments: dict[str, object]) -> ToolReply:
        if arguments:
            return ToolReply("list_proposals takes no arguments", True)
        payload = [proposal.as_dict() for proposal in self.request.proposals]
        return ToolReply(_json(payload))

    def _read(self, arguments: dict[str, object]) -> ToolReply:
        proposal_id = arguments.get(ARG_PROPOSAL)
        name = arguments.get(ARG_NAME)
        offset = arguments.get(ARG_OFFSET, 0)
        raw_pointer = arguments.get(ARG_POINTER)
        if not isinstance(proposal_id, str) or not isinstance(name, str):
            return ToolReply("read_evidence needs proposal and name", True)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return ToolReply("read_evidence offset must be a non-negative integer", True)
        if ARG_POINTER in arguments and not isinstance(raw_pointer, str):
            return ToolReply("read_evidence pointer must be a string", True)
        proposal = self._proposal(proposal_id)
        if proposal is None:
            return ToolReply(f"proposal {proposal_id!r} is not in this review run", True)
        source = next((item for item in proposal.sources if item.name == name), None)
        if source is None:
            return ToolReply(f"memory {name!r} is not evidence for proposal {proposal_id}", True)
        stale = self._stale(source)
        if stale:
            return ToolReply(stale, True)
        if isinstance(raw_pointer, str):
            return self._read_provenance(proposal_id, source, raw_pointer, offset)
        return self._read_memory(proposal_id, source, offset)

    def _read_memory(
        self, proposal_id: str, source: reasoning.ReviewSource, offset: int
    ) -> ToolReply:
        remaining = self.request.limits.evidence_chars - self.evidence_chars
        if remaining <= 0:
            return ToolReply("review evidence budget exhausted", True)
        body = source.body
        if offset > len(body):
            return ToolReply("read_evidence offset is beyond the evidence body", True)
        maximum = min(len(body) - offset, remaining)
        chunk_size = self._memory_chunk_size(proposal_id, source, offset, maximum)
        if chunk_size < 0 or (maximum > 0 and chunk_size == 0):
            return ToolReply("review tool-result budget is too small for evidence metadata", True)
        chunk = body[offset : offset + chunk_size]
        self.evidence_chars += len(chunk)
        self.evidence_reads.append((proposal_id, source.name))
        next_offset = offset + len(chunk)
        return ToolReply(
            _json(self._memory_payload(proposal_id, source, offset, chunk, next_offset))
        )

    def _read_provenance(
        self,
        proposal_id: str,
        source: reasoning.ReviewSource,
        raw_pointer: str,
        offset: int,
    ) -> ToolReply:
        if not self.request.limits.raw_session_evidence:
            return ToolReply("raw Session evidence is disabled for this review run", True)
        pointer = raw_pointer.strip()
        if pointer not in source.provenance:
            return ToolReply(
                f"pointer {pointer!r} is not frozen evidence for memory {source.name!r}", True
            )
        parsed = sessions.parse_pointer(pointer)
        if parsed is None:
            return ToolReply("read_evidence pointer must be a raw Session pointer", True)
        try:
            result = self.store.trace_evidence(source.name, pointer)
        except MemoryStoreError as error:
            return ToolReply(f"raw Session evidence is unavailable: {error}", True)
        current_hash = (
            result.record.source_hash
            or hashlib.sha256(result.record.to_text().encode("utf-8")).hexdigest()
        )
        if not result.record.is_active() or current_hash != source.source_hash:
            return ToolReply(f"evidence {source.name!r} changed after proposal generation", True)
        if len(result.evidence) != 1 or result.evidence[0].reference != pointer:
            return ToolReply("raw Session evidence did not resolve to the requested pointer", True)
        messages = result.evidence[0].messages
        reply, spent = self._provenance_page(
            proposal_id, source, pointer, parsed.session, messages, offset
        )
        if reply.is_error:
            return reply
        self.evidence_chars += spent
        self.evidence_reads.append((proposal_id, source.name))
        self.provenance_reads.append((proposal_id, source.name, pointer))
        return reply

    def _memory_chunk_size(
        self,
        proposal_id: str,
        source: reasoning.ReviewSource,
        offset: int,
        maximum: int,
    ) -> int:
        empty = self._memory_payload(proposal_id, source, offset, "", offset)
        if len(_json(empty)) > self.request.limits.tool_result_chars:
            return -1
        low, high, best = 0, maximum, 0
        while low <= high:
            middle = (low + high) // BISECTION_DIVISOR
            end = offset + middle
            payload = self._memory_payload(
                proposal_id, source, offset, source.body[offset:end], end
            )
            if len(_json(payload)) <= self.request.limits.tool_result_chars:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _memory_payload(
        proposal_id: str,
        source: reasoning.ReviewSource,
        offset: int,
        body: str,
        end: int,
    ) -> dict[str, object]:
        return {
            "proposal": proposal_id,
            "evidence_kind": "memory",
            **source.as_dict(),
            "body": body,
            "offset": offset,
            "next_offset": end if end < len(source.body) else None,
        }

    def _provenance_page(
        self,
        proposal_id: str,
        source: reasoning.ReviewSource,
        pointer: str,
        session: str,
        messages: tuple[sessions.Message, ...],
        offset: int,
    ) -> tuple[ToolReply, int]:
        widths = [max(1, len(message.text)) for message in messages]
        total = sum(widths)
        if offset > total:
            return ToolReply("read_evidence offset is beyond the raw Session evidence", True), 0
        remaining = self.request.limits.evidence_chars - self.evidence_chars
        if remaining <= 0:
            return ToolReply("review evidence budget exhausted", True), 0
        fragments: list[dict[str, object]] = []
        position = offset
        spent = 0
        while position < total:
            message_number, local_offset = _message_position(widths, position)
            message = messages[message_number]
            reference = sessions.render_pointer(
                sessions.Pointer(session, message.index, message.index)
            )
            metadata_chars = len(reference) + len(message.role) + len(message.at)
            budget_left = remaining - spent - metadata_chars
            if budget_left < 0:
                break
            available = len(message.text) - local_offset
            maximum = min(available, budget_left)
            if not message.text:
                maximum = 0
            take = self._raw_fragment_size(
                proposal_id,
                source,
                pointer,
                fragments,
                message,
                reference,
                local_offset,
                maximum,
                offset,
                position,
                total,
            )
            if take < 0:
                break
            advance = take if message.text else 1
            fragment = _message_fragment(message, reference, local_offset, take)
            fragments.append(fragment)
            position += advance
            spent += metadata_chars + take
            if message.text and take < available:
                break
        if position < total and not fragments:
            return ToolReply(
                "review evidence budget exhausted or page metadata is too large", True
            ), 0
        payload = self._raw_payload(
            proposal_id,
            source,
            pointer,
            fragments,
            offset,
            position if position < total else None,
            total,
        )
        text = _json(payload)
        if len(text) > self.request.limits.tool_result_chars:
            return ToolReply("review tool-result budget is too small for raw evidence", True), 0
        return ToolReply(text), spent

    def _raw_fragment_size(
        self,
        proposal_id: str,
        source: reasoning.ReviewSource,
        pointer: str,
        fragments: list[dict[str, object]],
        message: sessions.Message,
        reference: str,
        local_offset: int,
        maximum: int,
        page_offset: int,
        position: int,
        total: int,
    ) -> int:
        if message.text and maximum <= 0:
            return -1
        low = 1 if message.text else 0
        high = maximum
        best = -1
        while low <= high:
            middle = (low + high) // BISECTION_DIVISOR
            candidate = [
                *fragments,
                _message_fragment(message, reference, local_offset, middle),
            ]
            end = position + (middle if message.text else 1)
            payload = self._raw_payload(
                proposal_id,
                source,
                pointer,
                candidate,
                page_offset,
                end if end < total else None,
                total,
            )
            if len(_json(payload)) <= self.request.limits.tool_result_chars:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _raw_payload(
        proposal_id: str,
        source: reasoning.ReviewSource,
        pointer: str,
        messages: list[dict[str, object]],
        offset: int,
        next_offset: int | None,
        total: int,
    ) -> dict[str, object]:
        return {
            "proposal": proposal_id,
            "name": source.name,
            "source_hash": source.source_hash,
            "evidence_kind": "raw_session",
            "pointer": pointer,
            "notice": trace.EVIDENCE_NOTICE,
            "messages": messages,
            "offset": offset,
            "next_offset": next_offset,
            "total_chars": total,
        }

    def _submit(self, arguments: dict[str, object]) -> ToolReply:
        if self.submission is not None:
            return ToolReply("a complete verdict set was already accepted", True)
        raw = arguments.get(ARG_VERDICTS)
        if not isinstance(raw, list) or not raw:
            return ToolReply("submit_verdicts needs a non-empty verdicts list", True)
        verdicts: list[reasoning.Verdict] = []
        problems: list[str] = []
        for index, payload in enumerate(raw):
            verdict = reasoning.verdict_from_payload(payload)
            if verdict is None:
                problems.append(f"verdict {index}: invalid verdict object")
            else:
                verdicts.append(verdict)
        allowed = {proposal.id: proposal for proposal in self.request.proposals}
        given_ids = [verdict.proposal_id for verdict in verdicts]
        duplicates = sorted({item for item in given_ids if given_ids.count(item) > 1})
        foreign = sorted(set(given_ids) - set(allowed))
        missing = sorted(set(allowed) - set(given_ids))
        if duplicates:
            problems.append("duplicate proposal ids: " + ", ".join(duplicates))
        if foreign:
            problems.append("proposal ids outside this run: " + ", ".join(foreign))
        if missing:
            problems.append("missing proposal ids: " + ", ".join(missing))
        for verdict in verdicts:
            proposal = allowed.get(verdict.proposal_id)
            if proposal is None:
                continue
            problems.extend(self._verdict_problems(proposal, verdict))
            problems.extend(filter(None, (self._stale(source) for source in proposal.sources)))
        if problems:
            return ToolReply("Nothing was accepted.\n" + "\n".join(problems), True)
        self.submission = tuple(verdicts)
        return ToolReply(f"accepted {len(verdicts)} verdicts for Manage validation")

    @staticmethod
    def _verdict_problems(
        proposal: reasoning.ReviewProposal, verdict: reasoning.Verdict
    ) -> list[str]:
        if not verdict.accept:
            return []
        if proposal.kind == reasoning.PROPOSAL_MERGE and (
            not verdict.abstract.strip() or not verdict.body.strip()
        ):
            return [f"proposal {proposal.id}: accepted merge needs abstract and body"]
        if proposal.kind == reasoning.PROPOSAL_SPLIT:
            complete = [
                part for part in verdict.parts if part.abstract.strip() and part.body.strip()
            ]
            if len(complete) < MIN_SPLIT_PARTS:
                return [f"proposal {proposal.id}: accepted split needs at least two complete parts"]
        if proposal.kind == reasoning.PROPOSAL_ABSTRACT_REVIEW and not verdict.text.strip():
            return [f"proposal {proposal.id}: accepted abstract review needs text"]
        return []

    def _proposal(self, proposal_id: str) -> reasoning.ReviewProposal | None:
        return next((item for item in self.request.proposals if item.id == proposal_id), None)

    def _stale(self, source: reasoning.ReviewSource) -> str:
        current = self.store.find(source.name)
        if current is None or not current.is_active():
            return f"evidence {source.name!r} is no longer active"
        current_hash = (
            current.source_hash or hashlib.sha256(current.to_text().encode("utf-8")).hexdigest()
        )
        if current_hash != source.source_hash:
            return f"evidence {source.name!r} changed after proposal generation"
        return ""

    def _bounded(self, reply: ToolReply) -> ToolReply:
        limit = self.request.limits.tool_result_chars
        if len(reply.text) <= limit:
            return reply
        if reply.is_error:
            suffix = "\n(truncated)"
            if limit <= len(suffix):
                return ToolReply(suffix[:limit], True)
            return ToolReply(reply.text[: limit - len(suffix)] + suffix, True)
        message = "tool result exceeded the configured character limit"
        return ToolReply(message[:limit], True)

    def seed(self) -> dict[str, object]:
        return {"store": str(self.store.root), "request": dataclasses.asdict(self.request)}

    def record(self) -> dict[str, object]:
        return {
            "submission": (
                [reasoning.verdict_as_dict(verdict) for verdict in self.submission]
                if self.submission is not None
                else None
            ),
            "evidence_chars": self.evidence_chars,
            "evidence_reads": [list(item) for item in self.evidence_reads],
            "provenance_reads": [list(item) for item in self.provenance_reads],
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
        if isinstance(submission, list):
            verdicts = [reasoning.verdict_from_payload(item) for item in submission]
            if all(verdict is not None for verdict in verdicts):
                self.submission = tuple(verdict for verdict in verdicts if verdict is not None)
        self.evidence_chars = _int(record.get("evidence_chars"))
        self.evidence_reads = [
            (str(item[0]), str(item[1]))
            for item in _list(record.get("evidence_reads"))
            if isinstance(item, list) and len(item) == EVIDENCE_READ_FIELDS
        ]
        self.provenance_reads = [
            (str(item[0]), str(item[1]), str(item[PROVENANCE_POINTER_INDEX]))
            for item in _list(record.get("provenance_reads"))
            if isinstance(item, list) and len(item) == PROVENANCE_READ_FIELDS
        ]
        self.calls.extend(
            ToolCall(
                str(call.get("name") or ""),
                _dict(call.get("arguments")),
                ToolReply(str(call.get("text") or ""), bool(call.get("is_error"))),
            )
            for call in _dicts(record.get("calls"))
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
    def from_directory(cls, directory: pathlib.Path) -> ManageReviewToolkit:
        seed = json.loads((directory / SEED_FILENAME).read_text(encoding="utf-8"))
        return cls(
            Store(str(seed["store"]), agent="manage-review-toolkit"),
            _request(_dict(seed.get("request"))),
        )


def _request(raw: dict[str, object]) -> reasoning.ReviewRequest:
    proposals = []
    all_sources = tuple(_source(item) for item in _dicts(raw.get("sources")))
    by_name = {source.name: source for source in all_sources}
    for item in _dicts(raw.get("proposals")):
        source_names = [str(source.get("name") or "") for source in _dicts(item.get("sources"))]
        proposals.append(
            reasoning.ReviewProposal(
                id=str(item.get("id") or ""),
                kind=str(item.get("kind") or ""),
                targets=tuple(str(value) for value in _list(item.get("targets"))),
                reason=str(item.get("reason") or ""),
                evidence=str(item.get("evidence") or ""),
                sources=tuple(by_name[name] for name in source_names if name in by_name),
            )
        )
    limits = _dict(raw.get("limits"))
    return reasoning.ReviewRequest(
        proposals=tuple(proposals),
        sources=all_sources,
        limits=reasoning.ReviewLimits(
            max_steps=int(limits.get("max_steps") or 0),
            max_tool_calls=int(limits.get("max_tool_calls") or 0),
            timeout_seconds=float(limits.get("timeout_seconds") or 0),
            tool_result_chars=int(limits.get("tool_result_chars") or 0),
            evidence_chars=int(limits.get("evidence_chars") or 0),
            raw_session_evidence=limits.get("raw_session_evidence") is True,
        ),
    )


def _source(raw: dict[str, object]) -> reasoning.ReviewSource:
    return reasoning.ReviewSource(
        name=str(raw.get("name") or ""),
        source_hash=str(raw.get("source_hash") or ""),
        type=str(raw.get("type") or ""),
        abstract=str(raw.get("abstract") or ""),
        body=str(raw.get("body") or ""),
        provenance=tuple(str(value) for value in _list(raw.get("provenance"))),
        valid_from=_optional_string(raw.get("valid_from")),
        memory_created_at=_optional_string(raw.get("memory_created_at")),
        memory_updated_at=_optional_string(raw.get("memory_updated_at")),
    )


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _dicts(value: object) -> list[dict]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_string(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _message_position(widths: list[int], offset: int) -> tuple[int, int]:
    consumed = 0
    for index, width in enumerate(widths):
        if offset < consumed + width:
            return index, offset - consumed
        consumed += width
    raise ValueError("raw evidence offset is outside the message stream")


def _message_fragment(
    message: sessions.Message, reference: str, text_offset: int, size: int
) -> dict[str, object]:
    return {
        "reference": reference,
        "index": message.index,
        "role": message.role or None,
        "message_at": message.at or None,
        "text": message.text[text_offset : text_offset + size],
        "text_offset": text_offset,
        "text_complete": text_offset + size >= len(message.text),
    }


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)
