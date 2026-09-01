# 已处理知识资产的运行要求

这个目录包含回答和播放所需的知识资产：原视频、逐句原文/人工校订、章节与 Chunk 范围、SQLite FTS、FAISS 索引和 Build 来源记录。处理完成后，仅做检索、问答和章节播放时，不再需要 ASR、VAD、标点模型或 ForcedAligner；它们只在导入新视频或重新转写时使用。

## 当前检索与回答配置

- 查询 Embedding：`BAAI/bge-large-zh-v1.5`，1024 维，最大 512 tokens，L2 归一化；当前本地目录为项目相对路径 `models/embedding-v1.5`，设备为 CPU。
- 词法检索：SQLite FTS5 + jieba，数据位于 `data/metadata.db`。
- 向量检索：FAISS `IndexFlatIP`，当前索引和 Chunk 映射位于 `data/indexes/`；归一化向量的内积等价于余弦相似度。
- 重排：`BAAI/bge-reranker-base`，当前本地目录为 `models/reranker`，设备为 CPU；当前最低分初值为 `0.50`，需要用正式知识库问题集校准。
- 证据回答：项目内独立llama-server加载`qwen3.5-4b`（Q4_K_M GGUF），OpenAI兼容接口`http://127.0.0.1:1234/v1`，实际上下文32000，temperature 0，关闭reasoning；最多返回3个按相关度排序的章节。
- 视频片段：FFmpeg 7.1 或兼容版本；程序按章节 `start_ms/end_ms` 从 `media/` 原视频生成 `data/clips/` 缓存。
- 当前 Python 主版本：3.11；精确包版本在项目 `requirements-local-rag.txt`。

## 迁移后必须保持的兼容性

1. 把本目录作为一个整体复制，不能只复制 SQLite 而漏掉 `media/`；否则文字问答仍可能工作，但章节视频无法播放。
2. 数据库内视频路径相对此目录保存，不依赖原电脑盘符。新电脑的项目 `config.yaml` 应把 `project.assets_dir` 指向本目录。
3. 继续使用当前 FAISS 时，必须使用相同的 BGE 模型、1024 维和归一化方式生成查询向量。更换 Embedding 模型后必须全量重建 FAISS，不能混用旧索引。
4. 更换 reranker 或回答 Qwen 不需要重做 ASR，也不需要重建 FAISS，但应重新校准拒答阈值并跑问答评测。
5. `data/cache/`、`data/clips/` 可删除并再生；`data/metadata.db`、`media/`、`glossaries/` 和当前 Build 检查材料不应当作普通缓存删除。
6. 复制前先停止网页和模型，避免复制正在写入的 SQLite WAL 或半切换索引。可用项目命令 `rag.py backup` 生成整个目录的 ZIP 快照。

如果只有这批资产、没有原 ASR 模型，仍可直接检索、生成回答和播放章节；前提是另有兼容的项目程序、Python 环境、当前 Embedding、reranker、回答 LLM 与 FFmpeg。若只保留可读 JSON 而没有 SQLite/FAISS，仍可人工检查，但不能直接获得当前速度和排序结果。
