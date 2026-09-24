# 视频数据处理 WebUI

这是项目当前唯一的 Web 界面，默认使用端口 `8876`。原 8765 界面不参与当前运行。

页面代码和服务 PID 状态位于本目录，章节截图缓存位于 `work/cache/thumbnails`。界面读取 `video_pipeline/config.json`，并把用户主动处理或编辑的正式结果写入项目的 `data`、`work` 和 `history`。

流程为：上传视频 → 一键处理到语义章节 → Markdown 编辑 → 确认入库。清洗阶段不会停在逐句人工审核；不确定项会自动采用模型候选并保留到清洗历史中，供后续章节编辑复核。

“确认入库”目前表示在 `data/semantic_segments` 对应 JSON 中写入 `review_status=confirmed`。本项目还没有 Embedding/索引模块，因此此按钮暂时不会生成向量索引。人工正文映射回清洗句子时保留原 ID 和时间，章节用 `manual_content_override=true` 标记；这些时间锚点不代表重写后的文字已经逐字对齐音频。

双击项目根目录的 `启动视频处理界面.cmd` 启动，浏览器地址为 <http://127.0.0.1:8876>。
