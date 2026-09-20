# 代码结构

每个步骤独立为一个 Python 文件：

1. `config.py`：读取环境变量和阈值配置。
2. `models.py`：定义经验、相似结果和裁决结果的数据结构。
3. `embedding.py`：调用 vLLM OpenAI 兼容 Embedding 接口。
4. `vector_store.py`：负责 Qdrant 建库、相似度查询和向量写入。
5. `llm_judge.py`：负责疑似重复经验的 LLM 二次裁决。
6. `pipeline.py`：按阈值串联所有步骤，并处理插入、丢弃和合并。
7. `cli.py`：命令行入口。

运行：

```bash
python -m experience_dedup.cli add --text "设备报警 E101，重启后短暂恢复，最终更换温度传感器后故障消失。"
```

也可以直接在业务代码中调用：

```python
from experience_dedup.models import Experience
from experience_dedup.pipeline import ExperiencePipeline

result = ExperiencePipeline().process(
    Experience("设备报警 E101，检查发现温度传感器漂移，更换传感器后恢复。")
)
print(result.action, result.similarity)
```
