# 开发交接与部署

本版本是 Windows 本机视频处理工具：视频 → 本地 ASR → 云端清洗 → 语义分章 → 人工编辑与确认。尚无向量索引、检索问答和多用户权限体系。旧版 RAG 和 8765 界面不在当前运行路径中。

## 环境准备

使用 Python 3.11，在克隆后的项目根目录执行：

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
Copy-Item resources/glossary.example.json resources/glossary.json
```

词表复制仅用于首次部署；已有 `resources/glossary.json` 时不要覆盖。依赖版本来自现有环境，换机部署还需要在目标硬件上验收完整 ASR。无需安装已移除下载脚本使用的 ModelScope SDK。

### 本地模型

自行获取以下两套完整的 Hugging Face 格式模型，下载工具和方式不限：

| 模型标识 | 默认目录 |
| --- | --- |
| `Qwen/Qwen3-ASR-1.7B-hf` | `models/qwen3-asr-1.7b-hf` |
| `Qwen/Qwen3-ForcedAligner-0.6B-hf` | `models/qwen3-forced-aligner-0.6b-hf` |

需要完整权重、配置、tokenizer 和 processor 文件，不能只放单独的权重文件。当前适配器检查 `model.safetensors`；不要直接替换为其他格式或其他版本模型。模型加载使用 `local_files_only=True`，运行时不会自动下载。代码中的下载溯源文件是可选元数据，不需要专用下载脚本生成。

配置入口为 `video_pipeline/config.json` 中的 `asr.providers.qwen3_asr`。默认使用 CPU 和 bfloat16，其他硬件配置需实际验证。

### FFmpeg

本项目实际调用项目目录中的程序，默认路径是：

```text
tools/ffmpeg/bin/ffmpeg.exe
```

FFmpeg 负责提取音频、切分音频和生成章节截图，必须保留。词句时间戳由 Qwen3 Forced Aligner 生成，不依赖 FFprobe。`ffprobe.exe` 是可选项，只用于读取视频容器的完整时长；当前本机未安装，代码会退回使用最后一句 ASR 的结束时间，这不一定包含结尾静音。无需为本次交付额外安装 FFprobe；如自行配备，应与 FFmpeg 放在同一目录。可以修改配置中的 `ffmpeg` 为其他实际文件路径。

模型、FFmpeg 二进制和虚拟环境均不进 Git，也不提供自动下载脚本。

### 配置和启动

1. 根据资料批次修改 `video_pipeline/config.json` 的 `partition`，默认仍为 `2026-09`。分区不会自动按月份切换。
2. 双击 `启动视频处理界面.cmd`，或运行 `.\venv\Scripts\python.exe -m beginner_webui --no-browser`。
3. 浏览器访问 `http://127.0.0.1:8876`，在“API连接”中配置云端连接，也可使用 `ZHIPUAI_API_KEY` 环境变量。不要把真实密钥写入 Git。
4. 用短视频验收上传、ASR、清洗、分章、播放截图、编辑、确认和取消流程。

默认配置中的相对路径以项目根目录为基准；使用自定义配置文件时，相对路径以该配置文件目录为基准。更新代码后需要重启服务；先完成当前任务并保存页面中的编辑内容。

## 代码入口

| 职责 | 文件/目录 |
| --- | --- |
| 前端页面与交互 | `beginner_webui/templates/index.html`、`static/app.js`、`static/style.css` |
| HTTP 接口 | `beginner_webui/app.py` |
| 队列与模型子进程 | `video_pipeline/services/jobs.py` |
| 编辑、词表、文件事务 | `video_pipeline/services/editing.py`、`glossary.py`、`storage.py` |
| ASR、清洗与分章 | `video_pipeline/asr`、`cleanup`、`segments` |
| 字段、时间轴、版本与状态契约 | `video_pipeline/INTERFACES.md` |
| 模型提示词说明 | `PIPELINE_PROMPTS.md`（实际行为以代码为准） |

## Git 与本地资料的边界

提交代码、测试、默认无密钥配置、说明和空词表示例。以下内容保留本地，不随仓库分发：

- `data/`：视频和正式处理结果。
- `history/`：审核历史及历史比较材料；不能当缓存批量删除。
- `work/`：上传、检查点、事务、任务状态及缓存；存在未结束任务或未恢复事务时不可清空。
- `models/`、`venv/`、`tools/ffmpeg/`：部署依赖。
- `resources/glossary.json`、`resources/cloud_credentials.json`：业务词表和云端密钥。
- `backups/`：本地归档，不随 Git 分发。2026-09-15 留下的业务标注样例（4 个视频、26 章）本次压缩归档并移出活动目录。当前前端没有该功能页面；服务层仍兼容读取 `business_knowledge_library/`，目录不存在不影响主流程。需要恢复时，从压缩包解压该目录至项目根目录。
- `视频知识切片人工操作指南.docx`：本地人工操作文档，包含截图占位，未作为本次代码交付文档发布。

本次清理移除无运行引用的 Qwen 下载脚本、过时的 `data/manifest.json`，以及旧 `beginner_webui/cache` 截图缓存。原视频、处理结果和审核历史不在清理范围内。

## 验证与已知待办

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\venv\Scripts\python.exe -m unittest discover -s beginner_webui/tests -p "test_*.py"
.\venv\Scripts\python.exe -m video_pipeline.validation --partition 2026-09
```

取消恢复测试会创建并终止自己的测试子进程，运行环境需允许 `taskkill` 操作这些子进程。空仓库的数据校验显示 0 个视频，不能代替短视频端到端验收。

已修复词表并发覆盖、人工英文正文重建和取消任务残留事务问题。以下已识别问题尚未在本次交接中修复：

1. 章节保存与词表保存不是同一事务，词表失败时章节可能已保存。
2. 确认入库、复核建议接口尚未校验页面版本。
3. ASR 检查点依赖临时音频修改时间，重试复用不可靠，且未逐块保存。

多人或远程部署需另行增加认证授权、数据隔离和相应的运维措施；现有本机服务不能直接视为完整的多人产品。
