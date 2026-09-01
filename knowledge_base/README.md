# 可迁移知识资产目录

该目录是知识库的完整资产根目录。停止服务后可整体复制；数据库中的视频路径使用相对此目录的路径，不依赖原电脑盘符或用户名。

主要内容：

- `media/`：受管原视频，是播放章节和重新生成片段的依据。
- `data/metadata.db`：SQLite正式主副本，保存来源、Build、原始/校订句子、章节、Chunk、FTS和查询日志。
- `data/builds/`：逐Build检查材料，包括句子、章节、Chunk、模型来源和人工校订审计。
- `data/indexes/`：当前及上一代FAISS索引和Chunk映射，可由数据库与Embedding模型重建。
- `data/cache/`：音频、ASR检查点等可再生材料；迁移后若只问答可不依赖ASR缓存。
- `data/clips/`：播放时生成的章节片段缓存，可删除并按需重建。
- `data/tasks/`：网页任务状态和校订审计的工作记录。
- `glossaries/`：全局、分类和来源级人工受控术语表。

仅复制本目录即可保留知识内容和现有索引，但要继续运行问答还需要项目代码、Python环境、FFmpeg、Embedding、Reranker及回答LLM。随资产迁移的完整要求见 [`RUNTIME_REQUIREMENTS.md`](RUNTIME_REQUIREMENTS.md)；工程内的扩展说明见 [`../docs/KNOWLEDGE_BASE_RUNTIME.md`](../docs/KNOWLEDGE_BASE_RUNTIME.md)。

复制前先运行 `关闭本地视频知识库.cmd`，避免复制SQLite WAL或正在写入的索引。
