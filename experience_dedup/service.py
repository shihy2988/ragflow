from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    action: str  # discard | merge | insert
    similarity: float
    existing_id: str | None = None
    experience: str | None = None
    reason: str = ""


class ExperienceService:
    """Experience store with conservative, three-level duplicate handling."""

    def __init__(self) -> None:
        self.embedding = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"),
            base_url=os.environ["EMBEDDING_BASE_URL"],
        )
        self.embedding_model = os.environ["EMBEDDING_MODEL"]
        self.llm = OpenAI(
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            base_url=os.environ["LLM_BASE_URL"],
        )
        self.llm_model = os.environ["LLM_MODEL"]
        self.qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        self.collection = os.getenv("QDRANT_COLLECTION", "maintenance_experiences")
        self.high_threshold = float(os.getenv("HIGH_DUPLICATE_THRESHOLD", "0.95"))
        self.suspect_threshold = float(os.getenv("SUSPECT_DUPLICATE_THRESHOLD", "0.85"))
        self.search_limit = int(os.getenv("SEARCH_LIMIT", "5"))

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("experience text must not be empty")
        response = self.embedding.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        return list(response.data[0].embedding)

    def _ensure_collection(self, dimension: int) -> None:
        names = {item.name for item in self.qdrant.get_collections().collections}
        if self.collection not in names:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def _payload(self, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"experience": text, "metadata": metadata or {}}

    def _llm_judge(self, new_text: str, old_text: str) -> tuple[str, str, str]:
        prompt = f"""你是维修经验知识库审核员。请比较下面两条经验。

经验A（新经验）：
{new_text}

经验B（库中经验）：
{old_text}

请严格只返回 JSON，不要 Markdown：
{{
  "same_experience": true或false,
  "merge": true或false,
  "merged_experience": "如果 merge=true，给出合并后的完整经验；否则为空字符串",
  "reason": "简短说明"
}}

判定规则：
1. 只有描述完全相同、维修对象/故障现象/关键步骤/结果都相同，same_experience 才为 true。
2. 如果是同一故障现象，但维修方案不同（例如临时修复与彻底更换），merge 必须为 true，并在 merged_experience 的维修方案中列出多种方案及适用条件，不能丢弃任一方案。
3. 如果关键步骤或结论不同，保留两条经验，merge=false。
"""
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "你只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON; keeping new experience")
            return "insert", "", "invalid LLM JSON"
        same = bool(data.get("same_experience", False))
        merge = bool(data.get("merge", False))
        merged = str(data.get("merged_experience", "")).strip()
        reason = str(data.get("reason", ""))
        if merge and merged:
            return "merge", merged, reason
        if same:
            return "discard", "", reason
        return "insert", "", reason

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> Decision:
        vector = self.embed(text)
        self._ensure_collection(len(vector))
        hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=self.search_limit,
            with_payload=True,
        ).points
        best = hits[0] if hits else None
        similarity = float(best.score) if best else 0.0

        if best and similarity > self.high_threshold:
            return Decision("discard", similarity, str(best.id), reason="highly duplicate")

        if best and similarity > self.suspect_threshold:
            old = str((best.payload or {}).get("experience", ""))
            action, merged, reason = self._llm_judge(text, old)
            if action == "discard":
                return Decision(action, similarity, str(best.id), reason=reason)
            if action == "merge":
                merged_vector = self.embed(merged)
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=[models.PointStruct(
                        id=best.id,
                        vector=merged_vector,
                        payload=self._payload(merged, metadata),
                    )],
                )
                return Decision(action, similarity, str(best.id), merged, reason)

        point_id = str(uuid.uuid4())
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(
                id=point_id,
                vector=vector,
                payload=self._payload(text, metadata),
            )],
        )
        return Decision("insert", similarity, point_id, text, "new experience")
