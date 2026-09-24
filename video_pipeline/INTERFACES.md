# 视频处理与界面调用契约

## 数据与时间轴

`video_id` 是视频内容 SHA-256。一个视频在当前分区下有 `asr_raw`、`cleaned_asr`、`semantic_segments` 和派生的 `semantic_markdown`。正式 JSON/Markdown 位于 `data`，清洗建议记录位于 `history/cleanup`，队列快照、锁和事务记录位于 `work`。路径统一通过 `video_pipeline.services.catalog.PipelinePaths` 解析。

原始句子 `id/start_ms/end_ms/raw_text` 对应音频。清洗句子 `sentence_id/start_ms/end_ms/text` 保持同一 ID 和时间锚点；`text` 可以大幅改写或为空。空文本行不能删除，因为后续章节范围和未来的边界调整仍需稳定引用。时间是原句的大致音频位置，并非改写后每个字的精确对齐。章节的 `start_sentence_id/end_sentence_id` 是连续句子范围，`start_ms/end_ms` 从范围内有效句子的时间生成；当前 UI 仅用于跳转，不修改边界。

`semantic_segments.review_status` 为 `pending_review` 或 `confirmed`；没有该字段的旧文件按待审核处理。`revision` 是递增整数，旧文件视为 `0`。每次人工保存、清洗句子派生更新、复核建议或确认都会递增。已确认文件再次编辑会回到待审核。`semantic_markdown` 始终由章节 JSON 生成，不能单独修改文件来反向更新 JSON。

## 服务模块

| 模块 | 调用入口 | 作用 |
| --- | --- | --- |
| `asr`、`cleanup`、`segments` | `python -m video_pipeline.<阶段>` | 生成原始 ASR、自动清洗和章节；已有结果默认跳过 |
| `validation.rules` | `validate_cleaned(raw, cleaned)`、`validate_segments(cleaned, segments)` | 检查 ID、时间、章节连续性及字段规则；保存和确认前由服务调用 |
| `services.catalog` | `PipelinePaths.from_config`、`dashboard`、`video_detail`、`ingest_upload` | 路径、列表、详情、与 HTTP 无关的视频流接收 |
| `services.jobs.JobManager` | `create(stage, video_ids)`、`snapshot`、`list`、`cancel` | 单 worker 队列；重启后旧任务标为 `interrupted`，不会自动重试 |
| `services.editing` | `save_chapter_fields`、`save_chapter_markdown`、`decide_archived_suggestion`、`confirm_segments` | 章节字段、旧版 Markdown 兼容、逐项复核及整视频确认 |
| `services.glossary` | `glossary_candidates`、`save_glossary_terms`、`search_glossary_occurrences`、`apply_glossary_occurrences` | 术语候选、词表与跨视频选择性替换 |
| `services.storage` | `video_lock`、`commit_artifacts`、`recover_video_transactions`、`recover_transactions` | 同视频互斥、多文件备份与失败恢复；服务启动、worker 退出和后续写入前恢复未结束事务 |

## 当前 HTTP 契约

`GET /api/videos/{video_id}` 获取视频详情和 `artifacts.segments.revision`。`POST /api/jobs` 提交 `{ "stage": "pipeline", "video_ids": ["<64位哈希>"] }`；`GET /api/jobs` 查看状态。状态有 `queued/running/completed/failed/cancelled/interrupted`。`pipeline` 依次执行 ASR、清洗和章节生成，清洗中途不等待人工确认。

章节编辑调用 `PUT /api/videos/{video_id}/segments/{segment_no}`：

```json
{
  "title": "章节标题",
  "summary": "章节摘要。",
  "content": "章节正文。",
  "keywords": ["关键词"],
  "glossary_terms": [],
  "expected_revision": 0
}
```

服务在锁内检查版本，保留清洗句子的 ID/时间，将正文按旧句边界映射回原句，并一同更新清洗 JSON、章节 JSON 和 Markdown。章节正文可完全重写，无需与 ASR 内容相似；保存时只校验时间范围、结构和正文同步结果。成功返回最新视频详情；版本过期返回 HTTP 409，前端应刷新后让用户重新核对。新 UI 使用结构化字段接口和 `expected_revision`。旧页面的 `PUT .../markdown` 可继续提交 `expected_updated_at`；首次编辑尚无版本的文件也可保存，之后必须带有效版本或时间戳。`GET .../markdown` 和 AI 标题/词表候选接口仅用于预览与候选，不单独改正式数据。

`POST /api/videos/{video_id}/suggestions/{sentence_id}` 接受 `approved/rejected`，同步清洗句子和派生章节。清洗阶段暂用 AI 建议供后续分章，但其决定仍为 `pending`。`POST /api/videos/{video_id}/confirm` 将当前章节设为 `confirmed`，把仍待复核的建议标记为 `confirmed_by_video`；没有中途确认步骤。未点击确认则维持待审核。旧档案的 `auto_approved` 也按待确认处理。

词表搜索返回的 `hit_id` 包含命中句子的内容指纹。应用时按视频加锁，在锁内重新核对命中、读取最新清洗稿并保存；命中句子变化返回 HTTP 409，提示重新搜索。其他句子的已保存修改会保留。跨视频应用逐视频提交，不是整个资料库的单一事务。

人工章节正文映射回原句后，英文词可能跨越句子边界；后续复核或词表替换按该章节原有的拼接方式重建，保留句子边界上的空格。旧文件按逐句换行拼接的正文继续兼容，不批量改写历史数据。

取消任务时，worker 在子进程退出后先恢复该视频的未完成事务，再释放任务占用。已持久化完成标记的事务只清理备份，不回滚；未完成事务恢复到提交前状态。恢复失败的任务标为 `failed` 并保留错误，后续编辑、确认和删除返回 HTTP 409，直到残留事务可恢复。事务清理先移除日志再删除备份，避免清理中断留下指向缺失备份的有效日志。

命令行 `--overwrite` 不能覆盖已有下游文件、清洗审核记录或人工编辑/已确认章节。需要从旧数据重新生成时，先设计独立版本及迁移流程，不能就地覆盖。未来如开放章节起止时间编辑，应先改句子范围并验证所有章节连续、无重叠、无遗漏，再从句子时间重算起止；不要直接改 `start_ms/end_ms`。

命令行仅是本地 worker 的内部进程入口，不作为前端调用方式。前端调用 HTTP，HTTP 与内部入口最终必须遵守同一份数据规则。当前项目是本机文件式应用；如果改成多用户或远程服务，还需增加身份认证、授权、数据库事务和跨机器任务队列，不能把本地脚本直接当成产品后端部署。
