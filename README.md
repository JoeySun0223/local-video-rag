# 本地视频知识库

本项目在本机完成视频转写、章节划分、Embedding、检索、重排和 Qwen 问答。程序默认只监听
`127.0.0.1`，不会把视频、转写或问题发送给云端模型。电脑可以联网，但运行知识库不依赖云端推理。

## 一键运行

- 双击 `启动本地视频知识库.cmd`：启动项目内独立 `llama-server`，按 `config.yaml` 加载指定 Qwen，
  启动网页，预热 BGE embedding/reranker，最后打开浏览器。
- 双击 `关闭本地视频知识库.cmd`：如果有入库任务，可选择等待或安全取消；随后关闭网页、卸载
  BGE 与 reranker 所在进程，并停止 `llama-server` 以完整卸载 Qwen。

等价命令：

```powershell
.\venv\Scripts\python.exe .\rag.py start
.\venv\Scripts\python.exe .\rag.py stop
```

默认网页为 <http://127.0.0.1:8765>。要更换 Qwen，修改 `config.yaml` 的
`llm.model_path` 与 `llm.model`；启动器会直接加载项目内 GGUF，并验证实际上下文长度。

## 处理和检索方法

- ASR：默认 `Qwen3-ASR-1.7B-hf`，再用独立 `Qwen3-ForcedAligner-0.6B-hf` 生成真实词/字时间戳。
  项目遵循官方时间戳模式：最多 180 秒低能量切块，每块转写和对齐后直接拼接全文并恢复全片时间
  偏移，最后只在合并后的全文上分句。模型切块边界不再被当作句子边界，Qwen 路径不调用 CT-Punc。
  旧 SeACo-Paraformer + FSMN-VAD + CT-Punc 作为可手动切换的 Provider 保留，
  不会静默回退。
- 父章节：BGE 左右窗口主题变化为主信号，结合本视频相对静音和话语标记，再用动态规划约束
  章节时长。Qwen 只能在确定性候选附近复核，并生成标题、摘要和关键词。
- 子 Chunk：只在完整句之间切，正文不重叠；Embedding 可携带一条只读邻句，并对最终输入执行
  512-token 硬校验。
- 召回：`bge-large-zh-v1.5` + SQLite FTS；CPU `bge-reranker-base` 重排；按章节聚合后最多
  返回 3 个相关章节。
- 回答：Qwen 只看到带明确 chunk/section ID 的检索证据。召回、重排或生成证据门任一失败时，
  明确回答“当前知识库中没有足够信息”。当前阈值是保守初值，需用本人的库内/库外问题评测后校准。

## 数据保存结构

```text
knowledge_base/
  manifest.json                       # 资产格式和相对路径约定
  data/
    metadata.db                       # 唯一正式主数据
    indexes/                          # 可重建FAISS；保留当前和上一版本
    builds/<source_id>/<build_id>/    # 分层人工检查JSON
      source.json
      sentences.json                 # 唯一完整正文快照：raw/approved/时间戳
      sections.json                  # 章节范围、标题、摘要、边界依据
      chunks.json                    # Chunk范围和token统计
      corrections.json               # 术语纠错候选
      build.json                     # 配置、ASR来源、模型、词表、降级状态
      asr_raw.json                   # 原始转写、分片、对齐器时间戳和资源记录
      asr_preflight.json             # 首/中/尾三段预检、耗时估算和内存峰值
    cache/<source_id>/audio.wav       # 可再生ASR音频
    clips/                            # 可再生章节视频缓存
    tasks/<task_id>/status.json       # 任务历史，不是当前知识库真相
    exports/.../sections_expanded.json # 按需生成的完整章节检查导出
  media/                              # 受管视频；播放和重建依赖它
  glossaries/global.json
  glossaries/categories/<类别>.json
  glossaries/sources/<source_id>.json # 人工受控术语映射
```

`knowledge_base/` 是可整体复制的知识资产目录，数据库中的视频引用使用相对此目录的路径。

网页“检查数据”可以查看原始/批准后的句子、每个章节全文、Chunk范围、边界依据和 Build 历史；
“导出完整章节”按需生成 expanded JSON。检索不读取检查 JSON，而读取 SQLite 当前
`current_build_id` 和对应 FAISS 版本。

正文只以 `sentences.raw_text` 为不可变原文。人工批准的确定性术语修正写到 `approved_text`；
章节和 Chunk 只保存句子范围，检索物化文档是可重建派生数据。摘要仅用于导航和回答辅助，
不替代原文，也不是 Embedding 的事实来源。

## 入库与术语

网页选择视频后可不填类别，默认“未分类”。页面持续显示阶段、百分比、消息、阶段耗时和绝对
保存路径；只有出现“入库完成 / 新 Build 已原子切换并可检索”才算完成。失败或安全取消不替换
原来可用的 Build。

Qwen3-ASR 入库会先实测开头、中段、结尾各 60 秒，然后暂停并显示预计全片耗时和内存峰值。
只有点击“确认并运行完整ASR”才继续。ASR 阶段会暂停问答并暂时卸载 BGE、reranker 和 llama-server
Qwen；ASR/对齐器结束后程序先重新预热问答模型，才会显示入库完成，所以之后的提问不承担这次
重新加载成本。

```powershell
.\venv\Scripts\python.exe .\rag.py ingest ".\incoming\example.mp4" --category "前端技术"
.\venv\Scripts\python.exe .\rag.py ingest ".\incoming\example.mp4" --yes
.\venv\Scripts\python.exe .\rag.py ingest .\video.mp4 --sentences-json .\sentences.json --no-llm
```

外部句子 JSON 必须是连续 ID、非空完整句、单调且不重叠的时间范围。程序不截断句子。

术语表格式：

```json
{
  "TCP/IP": ["Pcpip", "TCPIP"],
  "DNS": ["d ns"]
}
```

替换最长词优先，ASCII 术语检查词边界；原始 ASR 永不覆盖。空 `{}` 表示没有已批准的领域映射，
不是模型漏生成了术语表。术语映射是人工受控数据，不由 Qwen 自由改写。

数据检查页面支持逐句编辑人工校订文本。点击“预览全部修改”后会列出每个句子的修改前后内容和
所有差异片段；可明确勾选其中哪些“错词 → 规范词”进入来源术语表。点击“一键修改并同步全部
资产”后，系统使用原时间戳生成替换Build，并同步章节、Chunk、FTS、Embedding和FAISS；原始
ASR文本不覆盖，失败不会切换当前Build。

## ASR Provider 与模型来源

在 `config.yaml` 中修改 `asr.primary`：

```yaml
asr:
  primary: qwen3_asr  # 手动切回旧模型时改为 funasr
```

Qwen3-ASR 和 ForcedAligner 优先从 ModelScope 官方 `Qwen` 命名空间下载到
`models/qwen3-asr-1.7b-hf` 与 `models/qwen3-forced-aligner-0.6b-hf`。模型来源、revision 和关键
文件哈希写入 `models/manifest.json` 及各 Build 的 ASR provenance；运行时始终
`local_files_only=True`，不会临时访问云端。官方 BF16 模型与旧 FunASR 模型均保留在本机。

## 自检与维护

```powershell
.\venv\Scripts\python.exe .\rag.py doctor
.\venv\Scripts\python.exe .\rag.py doctor --hashes
.\venv\Scripts\python.exe .\rag.py status
.\venv\Scripts\python.exe .\rag.py rebuild-index
.\venv\Scripts\python.exe .\rag.py export-sections <source_id>
.\venv\Scripts\python.exe .\rag.py backup
```

新 Build 先暂存，完整新 FAISS 写好后才在一次数据库事务中切换来源和索引指针。启动/doctor
核对 DB chunk ID、mapping、FAISS 数量和哈希，不一致时从 SQLite 重建。删除来源默认保留视频。

提问返回检索重排、Qwen 回答和总耗时。首次提问还可能包含模型预热；以后通常主要耗时在 CPU
reranker 和 Qwen 生成，实际以页面分阶段数据为准。

当前不做 OCR：音频型 MP4 没有画面；有画面的视频也不应盲目逐帧 OCR。以后添加时先用场景
变化和固定稀疏采样找关键帧，把 OCR 作为可选派生证据，不改写 ASR 原文。

## 工程与迁移文档

- `docs/KNOWLEDGE_BASE_RUNTIME.md`：处理完成后的检索、回答与视频播放还需要什么。
- `docs/SETUP_AND_MIGRATION.md`：完整环境、模型、GitHub整理和跨电脑迁移。
- `docs/GPU_ACCELERATION.md`：CPU现状及NVIDIA CUDA配置。
- `docs/CLOUD_MODEL_MIGRATION.md`：把ASR、Embedding、Reranker和回答LLM替换为云Provider。
