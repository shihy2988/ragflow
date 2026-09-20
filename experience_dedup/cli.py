from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .service import ExperienceService


def main() -> None:
    parser = argparse.ArgumentParser(description="维修经验向量去重与合并")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="新增一条经验")
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    args = parser.parse_args()

    if args.text is not None:
        text, metadata = args.text, {}
    else:
        data = json.loads(args.file.read_text(encoding="utf-8"))
        text = data.get("experience") or data.get("text")
        metadata = data.get("metadata") or {}
        if not text:
            raise SystemExit("JSON 必须包含 experience 或 text 字段")

    result = ExperienceService().add(text, metadata)
    print(json.dumps({
        "action": result.action,
        "similarity": result.similarity,
        "existing_id": result.existing_id,
        "experience": result.experience,
        "reason": result.reason,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
