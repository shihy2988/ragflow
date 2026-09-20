from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Experience:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("experience text must not be empty")


@dataclass(frozen=True)
class SimilarExperience:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    action: str  # discard | merge | insert
    similarity: float
    existing_id: str | None = None
    experience: str | None = None
    reason: str = ""
