"""FastAPI application for the local video pipeline dashboard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from video_pipeline.asr.media import VIDEO_EXTENSIONS
from video_pipeline.shared.config import CONFIG_PATH, load_config, project_path
from video_pipeline.shared.cloud import cloud_status, connect_cloud, disconnect_cloud
from video_pipeline.services.catalog import PipelinePaths, dashboard, delete_video, ingest_upload, video_detail
from video_pipeline.services.editing import (
    EditConflictError, chapter_markdown, confirm_segments, decide_archived_suggestion,
    generate_chapter_metadata, markdown_chapter_content,
    mark_custom_glossary_reviewed, save_chapter_fields, save_chapter_markdown,
)
from video_pipeline.services.glossary import (
    apply_glossary_occurrences, glossary_candidates, save_glossary_terms,
    search_glossary_occurrences,
)
from video_pipeline.services.jobs import JobManager
from video_pipeline.services.storage import TransactionRecoveryError, recover_transactions


WEB_ROOT = Path(__file__).resolve().parent


class JobRequest(BaseModel):
    stage: Literal["pipeline", "asr", "cleanup", "segments"]
    video_ids: list[str] = Field(min_length=1)


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    approved_text: str | None = None


class GlossaryEntry(BaseModel):
    term: str = Field(min_length=1, max_length=40)
    aliases: list[str] = Field(default_factory=list)
    scope: Literal["global", "video"]


class GlossarySearchRequest(BaseModel):
    source_video_id: str
    terms: list[GlossaryEntry] = Field(min_length=1)
    scope: Literal["database", "video"]


class GlossaryApplyRequest(GlossarySearchRequest):
    selected_hit_ids: list[str] = Field(default_factory=list)


class MarkdownEditRequest(BaseModel):
    markdown: str = Field(min_length=1)
    glossary_terms: list[GlossaryEntry] = Field(default_factory=list)
    expected_updated_at: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)


class ChapterEditRequest(BaseModel):
    title: str = Field(min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    glossary_terms: list[GlossaryEntry] = Field(default_factory=list)
    expected_revision: int = Field(ge=0)


class CloudConnectRequest(BaseModel):
    api_key: str | None = None
    model: str = Field(min_length=1, max_length=100)
    api_url: str = Field(min_length=1, max_length=500)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, (EditConflictError, TransactionRecoveryError)):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (ValueError, KeyError, FileNotFoundError)):
        return HTTPException(status_code=404 if isinstance(error, (KeyError, FileNotFoundError)) else 400, detail=str(error))
    return HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")


def create_app(config_path: Path | None = None, partition: str | None = None) -> FastAPI:
    resolved_config = (config_path or CONFIG_PATH).resolve()
    config = load_config(resolved_config if config_path is not None else None)
    paths = PipelinePaths.from_config(config, partition)
    recover_transactions(paths)
    jobs = JobManager(paths, resolved_config)
    app = FastAPI(title="视频数据处理", docs_url="/api/docs", redoc_url=None)
    app.state.paths = paths
    app.state.config_path = resolved_config
    app.state.jobs = jobs
    app.state.stop_callback = None
    app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "templates" / "index.html")

    @app.get("/videos/{video_id}", include_in_schema=False)
    @app.get("/videos/{video_id}/edit", include_in_schema=False)
    def video_page(video_id: str) -> FileResponse:
        return FileResponse(WEB_ROOT / "templates" / "index.html")

    @app.get("/api/dashboard")
    def get_dashboard() -> dict[str, Any]:
        return dashboard(paths)

    @app.get("/api/cloud/status")
    def get_cloud_status() -> dict[str, Any]:
        try:
            return cloud_status(config)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/cloud/connect")
    def connect_cloud_api(value: CloudConnectRequest) -> dict[str, Any]:
        try:
            return connect_cloud(
                config, api_url=value.api_url, api_key=value.api_key, model=value.model
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.delete("/api/cloud/connect")
    def disconnect_cloud_api() -> dict[str, Any]:
        try:
            return disconnect_cloud(config)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        try:
            return video_detail(paths, video_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/uploads")
    async def upload_videos(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                for candidate in files:
                    await candidate.close()
                raise HTTPException(status_code=400, detail=f"不支持的视频格式：{file.filename}")
        accepted = []
        for file in files:
            async def chunks():
                while chunk := await file.read(1024 * 1024):
                    yield chunk
            try:
                accepted.append(await ingest_upload(chunks(), file.filename, paths))
            finally:
                await file.close()
        return {"uploads": accepted}

    @app.post("/api/jobs")
    def create_job(value: JobRequest) -> dict[str, Any]:
        try:
            return jobs.create(value.stage, value.video_ids)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": jobs.list()}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/videos/{video_id}/suggestions/{sentence_id}")
    def decide_suggestion(
        video_id: str, sentence_id: int, value: ReviewDecision
    ) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，任务结束后才能复核 AI 建议")
        try:
            decide_archived_suggestion(
                paths, video_id, sentence_id, value.decision, value.approved_text
            )
            return video_detail(paths, video_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/videos/{video_id}/segments/{segment_no}/markdown")
    def get_chapter_markdown(video_id: str, segment_no: int) -> dict[str, str]:
        try:
            detail = video_detail(paths, video_id)
            rows = detail.get("artifacts", {}).get("segments", {}).get("segments", [])
            if segment_no < 1 or segment_no > len(rows):
                raise KeyError(f"找不到第 {segment_no} 章")
            return {"markdown": chapter_markdown(rows[segment_no - 1])}
        except Exception as error:
            raise _http_error(error) from error

    @app.put("/api/videos/{video_id}/segments/{segment_no}/markdown")
    def edit_chapter_markdown(
        video_id: str, segment_no: int, value: MarkdownEditRequest
    ) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，任务结束后才能编辑")
        try:
            save_chapter_markdown(
                paths, video_id, segment_no, value.markdown,
                value.expected_updated_at, value.expected_revision,
            )
            save_glossary_terms(
                config, video_id, [entry.model_dump() for entry in value.glossary_terms]
            )
            mark_custom_glossary_reviewed(paths, video_id, segment_no)
            return video_detail(paths, video_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.put("/api/videos/{video_id}/segments/{segment_no}")
    def edit_chapter_fields(
        video_id: str, segment_no: int, value: ChapterEditRequest
    ) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，任务结束后才能编辑")
        try:
            save_chapter_fields(
                paths, video_id, segment_no,
                title=value.title, summary=value.summary,
                content=value.content, keywords=value.keywords,
                expected_revision=value.expected_revision,
            )
            save_glossary_terms(
                config, video_id, [entry.model_dump() for entry in value.glossary_terms]
            )
            mark_custom_glossary_reviewed(paths, video_id, segment_no)
            return video_detail(paths, video_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/videos/{video_id}/segments/{segment_no}/glossary-candidates")
    def detect_glossary_candidates(
        video_id: str, segment_no: int, value: MarkdownEditRequest
    ) -> dict[str, Any]:
        try:
            detail = video_detail(paths, video_id)
            rows = detail.get("artifacts", {}).get("segments", {}).get("segments", [])
            if segment_no < 1 or segment_no > len(rows):
                raise KeyError(f"找不到第 {segment_no} 章")
            original = str(rows[segment_no - 1].get("content") or "")
            edited = markdown_chapter_content(value.markdown)
            candidates = glossary_candidates(config, video_id, original, edited)
            row = rows[segment_no - 1]
            start_id = int(row.get("start_sentence_id", -1))
            end_id = int(row.get("end_sentence_id", -1))
            history = detail.get("artifacts", {}).get("history", {})
            for item in history.get("changes", []):
                if not isinstance(item, dict):
                    continue
                sentence_id = int(item.get("sentence_id", -1))
                approved = str(item.get("approved_text") or "").strip()
                proposed = str(item.get("proposed_text") or item.get("raw_text") or "").strip()
                is_custom = item.get("decision") == "approved" and approved and approved != proposed
                if not (start_id <= sentence_id <= end_id and is_custom):
                    continue
                if item.get("glossary_review_pending") is False:
                    continue
                extra = glossary_candidates(
                    config, video_id, proposed, approved
                )
                for candidate in extra:
                    existing = next((row for row in candidates if row["term"] == candidate["term"]), None)
                    if existing is None:
                        candidates.append(candidate)
                    else:
                        existing["aliases"] = list(dict.fromkeys([
                            *existing.get("aliases", []), *candidate.get("aliases", []),
                        ]))
            return {"candidates": candidates}
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/glossary/search")
    def search_glossary(value: GlossarySearchRequest) -> dict[str, Any]:
        try:
            return {"hits": search_glossary_occurrences(
                paths,
                value.source_video_id,
                [entry.model_dump() for entry in value.terms],
                value.scope,
            )}
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/glossary/apply")
    def apply_glossary(value: GlossaryApplyRequest) -> dict[str, Any]:
        if jobs.is_busy():
            raise HTTPException(status_code=409, detail="有视频仍在处理，任务结束后才能批量应用词表")
        try:
            return apply_glossary_occurrences(
                paths,
                value.source_video_id,
                [entry.model_dump() for entry in value.terms],
                value.scope,
                value.selected_hit_ids,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/videos/{video_id}/segments/{segment_no}/generate-metadata")
    def generate_metadata(
        video_id: str, segment_no: int, value: MarkdownEditRequest
    ) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，任务结束后才能调用模型")
        try:
            return generate_chapter_metadata(
                paths, video_id, segment_no, value.markdown, config
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/videos/{video_id}/confirm")
    def confirm_video(video_id: str) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，任务结束后才能确认入库")
        try:
            confirm_segments(paths, video_id)
            return video_detail(paths, video_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.delete("/api/videos/{video_id}")
    def remove_video(video_id: str) -> dict[str, Any]:
        if jobs.is_video_busy(video_id):
            raise HTTPException(status_code=409, detail="该视频仍在处理，不能删除")
        try:
            return {"removed": delete_video(paths, video_id)}
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/videos/{video_id}/media")
    def media(video_id: str) -> FileResponse:
        try:
            path = paths.artifact(video_id, "video") or paths.artifact(video_id, "upload")
            if not path or not path.is_file():
                raise FileNotFoundError("找不到对应视频")
            return FileResponse(path)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/videos/{video_id}/thumbnail")
    def thumbnail(video_id: str, ms: int = Query(default=0, ge=0)) -> FileResponse:
        try:
            source = paths.artifact(video_id, "video") or paths.artifact(video_id, "upload")
            if not source or not source.is_file():
                raise FileNotFoundError("找不到对应视频")
            cache = paths.work_root / "cache" / "thumbnails" / paths.partition / video_id
            cache.mkdir(parents=True, exist_ok=True)
            destination = cache / f"{ms}.jpg"
            if not destination.is_file():
                configured = str(config.get("ffmpeg", "ffmpeg"))
                ffmpeg = project_path(config, configured) if configured != "ffmpeg" else Path("ffmpeg")
                result = subprocess.run(
                    [str(ffmpeg), "-y", "-ss", f"{ms / 1000:.3f}", "-i", str(source),
                     "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(destination)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
                )
                if result.returncode != 0 or not destination.is_file():
                    raise RuntimeError("章节截图生成失败")
            return FileResponse(destination, media_type="image/jpeg")
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/api/system/shutdown")
    def shutdown(request: Request) -> dict[str, str]:
        if request.client and request.client.host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            raise HTTPException(status_code=403, detail="关闭服务只允许从本机访问")
        callback = app.state.stop_callback
        if callback is None:
            raise HTTPException(status_code=409, detail="当前服务器不支持远程关闭")
        if jobs.is_busy():
            raise HTTPException(status_code=409, detail="仍有任务正在运行或排队，请等待任务结束后再关闭")
        callback()
        return {"status": "stopping"}

    return app
