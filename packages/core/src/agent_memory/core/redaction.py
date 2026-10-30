"""Secret shapes that never enter stored or projected text."""

from __future__ import annotations

import re

PREFIXED = re.compile(r"(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,\"']+")
BARE = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}"),
)
REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    cleaned = PREFIXED.sub(rf"\1{REDACTED}", text)
    for pattern in BARE:
        cleaned = pattern.sub(REDACTED, cleaned)
    return cleaned
