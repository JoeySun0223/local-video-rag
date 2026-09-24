# 视频知识数据流水线

本项目负责把视频处理为后续 Embedding、检索和问答可直接使用的结构化数据。目前流程到语义片段为止，尚未建立向量索引。

首次部署与开发交接请先阅读 [HANDOFF.md](HANDOFF.md)。仓库包含代码、测试和默认配置；视频、模型、运行环境、审核历史、业务词表及 API 密钥在本地准备，不随 Git 分发。

## 工程结构

```text
D:\asr_test\
├─ video_pipeline\       # 处理代码与配置
│  ├─ asr\               # 本地 ASR
│  ├─ cleanup\           # GLM 清洗与纠错审核逻辑
│  ├─ segments\          # 一次性语义边界规划与片段标签生成
│  ├─ validation\        # 跨阶段数据校验
│  ├─ services\          # 目录、队列、编辑、词表和事务服务
│  ├─ shared\            # 配置、文件 I/O、GLM 客户端
│  └─ config.json
├─ beginner_webui\       # 当前唯一 Web UI（端口 8876）
├─ data\                 # 正式知识数据
│  ├─ videos\
│  ├─ asr_raw\
│  ├─ cleaned_asr\
│  ├─ semantic_segments\      # 可编辑的结构化 JSON
│  └─ semantic_markdown\      # 与 JSON 同步的完整 Markdown
├─ history\              # 当前清洗审核记录和处理审计
├─ work\                 # ASR checkpoint 和临时运行状态
├─ tests\                # 自动化测试
├─ resources\            # 合并后的业务词表等资源
├─ models\               # 当前流程使用的本地模型
├─ tools\                # 本地 FFmpeg 可执行文件（不进 Git）
└─ venv\                 # Python 虚拟环境
```

代码、正式数据、审核状态、历史档案和运行缓存相互独立。修改或重装 `video_pipeline` 不需要移动 `data`。

## 运行命令

所有命令均在项目根目录下执行，不要求固定安装在 `D:\asr_test`。默认分区和云端模型参数由 `video_pipeline/config.json` 配置。

### Web 可视化操作

双击 `启动视频处理界面.cmd`，浏览器会打开 <http://127.0.0.1:8876>。页面支持：

- 单个或批量上传视频；
- 勾选视频后一键完成本地 ASR、GLM 清洗和语义章节生成；
- 查看视频、章节时间点、章节截图和处理进度，并可终止任务；
- 编辑章节标题、摘要、关键词和正文，实时预览 Markdown；
- 批准、拒绝或自定义修改已归档的 AI 清洗建议；
- 基于当前章节正文生成标题、摘要和关键词，确认前不会直接写盘；
- 从人工修改中提取词表候选，并按视频或整个资料库搜索、选择性应用；
- 编辑完成后确认入库，或删除明确选中的视频及其项目内派生物。

人工修改会保留原句 ID 与时间，更新当前分区的 `cleaned_asr` 和 `semantic_segments` JSON，并同步生成 `semantic_markdown`。保存前会校验句子覆盖、章节连续性和字段长度；过期页面保存返回 409。正在处理队列中的视频暂不允许同时人工修改。

章节编辑中的云端生成只把结果回填到当前页面，不会立即写入文件。可以继续人工修改模型给出的标题、摘要和关键词，最后保存章节。云端调用沿用 `video_pipeline/config.json` 的模型设置和项目内连接状态。

双击 `关闭视频处理界面.cmd` 或点击页面右上角“关闭服务”即可关闭。为避免中断本地模型或写入过程，存在运行中或排队任务时会拒绝关闭。

也可以使用命令行启动且不自动打开浏览器：

```powershell
.\venv\Scripts\python.exe -m beginner_webui --no-browser
```

上传但尚未生成 ASR 的视频保存在 `work/uploads`；生成 ASR 后，现有 ASR 流程会把视频复制到正式 `data/videos`。

### 内部阶段入口（供开发与运维排错）

正式操作走 WebUI。当前任务 worker 为隔离耗时模型进程，调用以下内部入口；这些命令会直接写项目文件，不能作为独立的产品操作流程或权限边界。

```powershell
# 1. 视频 -> 本地原始 ASR
.\venv\Scripts\python.exe -m video_pipeline.asr <视频或目录> --partition 2026-09

# 2. 原始 ASR -> GLM 清洗文本
.\venv\Scripts\python.exe -m video_pipeline.cleanup --partition 2026-09

# 3. 清洗文本 -> 可检索、可播放的语义片段
.\venv\Scripts\python.exe -m video_pipeline.segments --partition 2026-09

# 4. 不调用模型的全流程数据校验
.\venv\Scripts\python.exe -m video_pipeline.validation --partition 2026-09
```

云端连接优先使用 WebUI 保存的项目内连接状态，未保存时可使用环境变量 `ZHIPUAI_API_KEY`。各阶段默认跳过已有结果；显式 `--overwrite` 也不能覆盖已有下游、审核或人工确认结果。

新版清洗不会暂停等待逐句审核。模型候选会自动采用并写入 `cleaned_asr`，完整记录归档到 `history/cleanup/<分区>/<video_id>.json`，供章节编辑时复核。

前后端调用字段、状态和时间轴规则见 [video_pipeline/INTERFACES.md](video_pipeline/INTERFACES.md)。

未来的 Embedding 和索引建议分别加入 `data/embeddings`、`data/indexes`，实现代码则放入 `video_pipeline/embedding` 和 `video_pipeline/retrieval`。旧版 RAG 和 8765 Web UI 已退出当前运行路径；本版本不依赖开发者电脑上的归档目录。

运行自动化测试：

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\venv\Scripts\python.exe -m unittest discover -s beginner_webui/tests -p "test_*.py"
```
