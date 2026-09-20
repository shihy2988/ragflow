from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    qdrant_url: str
    qdrant_api_key: str | None
    collection: str
    high_threshold: float
    suspect_threshold: float
    search_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embedding_base_url=os.environ["EMBEDDING_BASE_URL"],
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
            embedding_model=os.environ["EMBEDDING_MODEL"],
            llm_base_url=os.environ["LLM_BASE_URL"],
            llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            llm_model=os.environ["LLM_MODEL"],
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection=os.getenv("QDRANT_COLLECTION", "maintenance_experiences"),
            high_threshold=float(os.getenv("HIGH_DUPLICATE_THRESHOLD", "0.95")),
            suspect_threshold=float(os.getenv("SUSPECT_DUPLICATE_THRESHOLD", "0.85")),
            search_limit=int(os.getenv("SEARCH_LIMIT", "5")),
        )
