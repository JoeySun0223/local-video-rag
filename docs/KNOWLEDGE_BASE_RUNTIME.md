# 已处理知识库的后续运行要求

本文件只讨论视频已经完成ASR、时间对齐、分章和Chunk以后，如何继续检索、生成回答并播放对应视频。

## 当前资产

知识资产统一位于 `knowledge_base/`。正式主副本为 `knowledge_base/data/metadata.db`，受管视频位于 `knowledge_base/media/`，FAISS索引位于 `knowledge_base/data/indexes/`。数据库中的媒体引用是相对 `knowledge_base/` 的路径，因此整体迁移后无需改盘符。

## 当前检索与回答模型

| 环节 | 当前模型/工具 | 当前配置 |
|---|---|---|
| 稠密召回 | `BAAI/bge-large-zh-v1.5` | 1024维、512 tokens、L2归一化、余弦/IP、CPU |
| 关键词召回 | SQLite FTS5 + jieba | 与稠密召回混合 |
| 重排 | `BAAI/bge-reranker-base` | CrossEncoder、最大512 tokens、CPU |
| 回答/标题 | `Qwen3.5-4B-Q4_K_M.gguf` | 独立llama-server标识符`qwen3.5-4b`、上下文32000、temperature=0、reasoning off |
| 视频片段 | FFmpeg 7.1 | 根据章节毫秒范围从受管视频生成 |

## 已处理资产不再需要的模型

只进行问答时，不需要加载或调用：

- `Qwen/Qwen3-ASR-1.7B-hf`
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`
- SeACo Paraformer、FSMN-VAD、CT-Transformer标点模型

它们只在新视频入库或重新ASR时需要。

## 不能缺少的组件

现有FAISS已经保存文档向量，但新问题仍必须由同一个BGE模型编码到相同向量空间。因此不能只复制FAISS而不保留Embedding模型。若更换Embedding模型，必须对所有Chunk全量重建向量。Reranker可以更换，但排序阈值需要重新评测；回答LLM可以更换，只要支持OpenAI兼容Chat Completions和结构化JSON输出。

播放章节必须保留 `knowledge_base/media/` 和FFmpeg。`data/clips/`只是缓存，删除不影响知识库正确性。

## 最小迁移集合

若目标电脑已有相同代码和模型，只需复制整个 `knowledge_base/`。若目标电脑什么都没有，需要复制整个项目（建议不复制旧`venv`，而是在目标机重建环境）、`models/`及`tools/`；回答模型和llama-server均已位于项目内。
