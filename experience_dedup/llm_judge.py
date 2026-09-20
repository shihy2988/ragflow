from __future__ import annotations

import json
import logging

from openai import OpenAI

from .config import Settings
from .models import SimilarExperience

logger = logging.getLogger(__name__)


class LLMJudge:
    """Step 3/4: use the LLM to decide whether a suspicious match is mergeable."""

    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model

    def judge(self, new_text: str, old: SimilarExperience) -> tuple[str, str, str]:
        prompt = f"""你是维修经验知识库审核员。请比较下面两条经验。

经验A（新经验）：
{new_text}

经验B（库中经验）：
{old.text}

请严格只返回 JSON，不要 Markdown：
{{
  "same_experience": true���false,
  "merge": true或false,
  "merged_experience": "如果 merge=true，给出合并后的完整经验；否则为空字符串",
  "reason": "简短说明"
}}

规则：
1. 只有故障现象、维修对象、关键步骤和结果都相同，same_experience 才为 true。
2. 同一故障但维修方案不同（如临时修复与彻底更换）时，merge 必须为 true，并在合并经验中列出多种方案及适用条件。
3. 如果关键步骤或结论不同，保留两条经验，merge=false。
"""
        response = self.client.chat.completions.create(
            model=self.model,
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
            logger.warning("LLM returned invalid JSON; keep the new experience")
            return "insert", "", "invalid LLM JSON"

        merged = str(data.get("merged_experience", "")).strip()
        reason = str(data.get("reason", ""))
        if bool(data.get("merge")) and merged:
            return "merge", merged, reason
        if bool(data.get("same_experience")):
            return "discard", "", reason
        return "insert", "", reason
