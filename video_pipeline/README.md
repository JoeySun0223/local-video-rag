# Pipeline 模块边界

- `asr/engine.py`：本地 Qwen3-ASR 与 Forced Aligner 模型适配。
- `asr/media.py`：视频发现、FFmpeg 音频提取和时长读取。
- `asr/cli.py`：视频入库和 `asr_raw` 生成。
- `cleanup/prompts.py`：GLM 清洗提示词。
- `cleanup/core.py`：清洗结果验证与安全采用规则。
- `cleanup/service.py`：逐视频云端清洗流程。
- `cleanup/cli.py`：不暂停的清洗命令行入口和审核记录归档。
- `cleanup/terms.py`：从全局和视频词表读取已确认术语。
- `segments/core.py`：一次性语义边界规划、标签和正文生成逻辑。
- `segments/cli.py`：普通语义片段生成入口。
- `validation/rules.py`：不调用模型的纯校验规则。
- `validation/cli.py`：生产数据校验入口。
- `shared/`：跨阶段共用的配置、文件 I/O 和 GLM HTTP 客户端。
- `services/`：与 Web 框架分离的目录、任务、章节编辑、词表和安全写入服务。

字段、状态、函数和 HTTP 示例见 [INTERFACES.md](INTERFACES.md)。

本地 ASR 会从 `resources/glossary.json` 自动加载全局热词和以视频哈希标识的来源热词；命令行 `--hotword` 可继续追加临时热词。

正式数据不存放在代码包中。默认路径由 `config.json` 指向项目根目录的 `data`、`history` 和 `work`。通过 `--config` 指定其他配置文件时，相对路径以该配置文件所在目录为准。

新版清洗自动发布最终文本到 `data/cleaned_asr`，并把完整审核记录写入 `history/cleanup`，供章节编辑时复核。

当前 Web UI 位于项目根目录的 `beginner_webui`，可通过 `启动视频处理界面.cmd` 和 `关闭视频处理界面.cmd` 启停，默认地址为 <http://127.0.0.1:8876>。耗时步骤仍调用本目录各阶段命令，不复制模型处理逻辑。
