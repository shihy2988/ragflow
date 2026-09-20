# experience_dedup

基于 vLLM OpenAI 兼容接口 + Qdrant 的维修经验去重与合并示例。

## 功能

- 长文本通过 Embedding 模型转为向量并写入 Qdrant。
- 新经验先进行余弦相似度检索。
- `similarity > 0.95`：高度重复，直接丢弃。
- `0.85 < similarity <= 0.95`：疑似重复，调用 LLM 二次裁决。
- `similarity <= 0.85`：作为新经验入库。
- 二次裁决支持合并经验：同一故障但维修方案不同，不删除新经验，而是合并方案后更新库中记录。

## 快速开始

```bash
cd experience_dedup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`：

```dotenv
EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
EMBEDDING_API_KEY=EMPTY
EMBEDDING_MODEL=BAAI/bge-m3
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=EMPTY
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=maintenance_experiences
```

vLLM 示例：

```bash
vllm serve BAAI/bge-m3 --port 8000
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
```

启动 Qdrant：

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:latest
```

## 命令行

新增经验：

```bash
python -m experience_dedup.cli add --text "设备报警 E101，重启后短暂恢复，最终更换温度传感器后故障消失。"
```

指定 JSON 文件：

```bash
python -m experience_dedup.cli add --file ./experience.json
```

JSON 文件至少包含 `experience` 或 `text` 字段，也可以包含 `metadata`：

```json
{
  "experience": "设备报警 E101，检查发现温度传感器漂移，更换传感器后恢复。",
  "metadata": {"device": "冷却机组", "source": "工单-1001"}
}
```

## 设计说明

疑似重复时，系统把新经验和最相似的旧经验发给 LLM，并要求返回 JSON：

```json
{
  "same_experience": true,
  "merge": true,
  "merged_experience": "...",
  "reason": "..."
}
```

LLM 失败、返回非法 JSON 或合并文本为空时，默认采取保守策略：保留新经验并入库，不误杀。

> 生产环境建议为 Qdrant 配置鉴权、为写入增加业务唯一键，并对 LLM 返回的合并结果进行人工抽样审核。
