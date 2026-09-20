from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from .config import Settings
from .models import Experience, SimilarExperience


class QdrantStore:
    """Step 2/4: create/search/upsert experiences in Qdrant."""

    def __init__(self, settings: Settings) -> None:
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection = settings.collection
        self.search_limit = settings.search_limit

    def ensure_collection(self, dimension: int) -> None:
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    def search(self, vector: list[float]) -> list[SimilarExperience]:
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=self.search_limit,
            with_payload=True,
        ).points
        results: list[SimilarExperience] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                SimilarExperience(
                    id=str(point.id),
                    score=float(point.score),
                    text=str(payload.get("experience", "")),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return results

    def upsert(self, point_id: str, vector: list[float], experience: Experience) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "experience": experience.text,
                        "metadata": experience.metadata,
                    },
                )
            ],
        )
