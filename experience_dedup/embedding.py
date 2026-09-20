from __future__ import annotations

from openai import OpenAI

from .config import Settings
from .models import Experience


class VLLMEmbedder:
    """Step 1/4: call the vLLM OpenAI-compatible embedding endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
        self.model = settings.embedding_model

    def embed(self, experience: Experience | str) -> list[float]:
        text = experience.text if isinstance(experience, Experience) else experience
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return list(response.data[0].embedding)
