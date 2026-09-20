from __future__ import annotations

import uuid

from .config import Settings
from .embedding import VLLMEmbedder
from .llm_judge import LLMJudge
from .models import Decision, Experience
from .vector_store import QdrantStore


class ExperiencePipeline:
    """Step 4/4: orchestrate thresholds, LLM arbitration, and persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: VLLMEmbedder | None = None,
        store: QdrantStore | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        self.settings = settings
        self.embedder = embedder or VLLMEmbedder(settings)
        self.store = store or QdrantStore(settings)
        self.judge = judge or LLMJudge(settings)

    def process(self, experience: Experience) -> Decision:
        vector = self.embedder.embed(experience)
        self.store.ensure_collection(len(vector))
        matches = self.store.search(vector)
        best = matches[0] if matches else None
        score = best.score if best else 0.0

        if best and score > self.settings.high_threshold:
            return Decision("discard", score, best.id, reason="highly duplicate")

        if best and score > self.settings.suspect_threshold:
            action, merged_text, reason = self.judge.judge(experience.text, best)
            if action == "discard":
                return Decision("discard", score, best.id, reason=reason)
            if action == "merge":
                merged = Experience(merged_text, {**best.metadata, **experience.metadata})
                merged_vector = self.embedder.embed(merged)
                self.store.upsert(best.id, merged_vector, merged)
                return Decision("merge", score, best.id, merged.text, reason)

        point_id = str(uuid.uuid4())
        self.store.upsert(point_id, vector, experience)
        return Decision("insert", score, point_id, experience.text, "new experience")
