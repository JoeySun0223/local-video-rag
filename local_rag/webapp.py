from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .config import database_path, ensure_project_dirs, index_dir, project_path, resolve_asset_path
from .corrections import preview_sentence_edits
from .db import Database, now_iso
from .pipeline import Ingestor, export_expanded_sections, get_embedding_engine, prune_index_versions, reconcile_vector_index, sha256_file, write_index_manifest, write_vector_index
from .search import RAGService
from .runtime import FileLock


def create_app(config: dict[str, Any]) -> FastAPI:
    ensure_project_dirs(config); db=Database(database_path(config));db.prune_logs(int(config["web"].get("query_log_retention_days",90)))
    app=FastAPI(title="本地视频RAG",version="0.2.0");root=Path(config["_root"]);app.mount("/static",StaticFiles(directory=root/"web"/"static"),name="static");templates=Jinja2Templates(directory=root/"web"/"templates")
    task_lock=threading.Lock();ask_lock=threading.Lock();build_lock=threading.Lock();clip_lock=threading.Lock();tasks:dict[str,dict[str,Any]]={};cancel_events:dict[str,threading.Event]={};continue_events:dict[str,threading.Event]={};qa_paused=threading.Event()
    data_dir=project_path(config,config["project"]["data_dir"]);tasks_root=data_dir/"tasks";tasks_root.mkdir(parents=True,exist_ok=True)

    def status_path(task_id):return tasks_root/task_id/"status.json"
    def persist(task_id):
        path=status_path(task_id);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".json.tmp");tmp.write_text(json.dumps(tasks[task_id],ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)
    def update(task_id,**values):
        with task_lock:
            item=dict(tasks.get(task_id,{}));item.update(values);item["task_id"]=task_id;item.setdefault("created_at",now_iso());item["updated_at"]=now_iso();tasks[task_id]=item;persist(task_id);return dict(item)

    def activated_task_result(item:dict[str,Any])->dict[str,Any]|None:
        """Recognize a Build that committed before a later maintenance/UI error."""
        build_entry=next((x for x in item.get("storage",[]) if x.get("key")=="build"),None)
        if not build_entry:return None
        try:
            build=json.loads(Path(build_entry["path"]).read_text(encoding="utf-8"));build_id=build["id"]
            source_entry=next(x for x in item["storage"] if x.get("key")=="source")
            source_file=json.loads(Path(source_entry["path"]).read_text(encoding="utf-8"));source=db.source(source_file["id"])
            if not source or source.get("current_build_id")!=build_id:return None
            current=next(x for x in db.list_sources() if x["id"]==source["id"])
            return {"source_id":source["id"],"build_id":build_id,"category":source["category"],"title":source["title"],"sentences":current["sentence_count"],"sections":current["section_count"],"chunks":current["chunk_count"],"degraded":["任务在Build激活后的维护阶段出现警告"],"index":db.meta("active_index"),"storage":item.get("storage",[])}
        except Exception:return None
    for file in tasks_root.glob("*/status.json"):
        try:
            item=json.loads(file.read_text(encoding="utf-8"));task_id=file.parent.name
            if item.get("status") in {"queued","running","awaiting_confirmation"}:item.update({"status":"failed","stage":"interrupted","message":"服务曾在Build完成前停止；当前知识库未切换到该Build。","error":"服务重启中断"})
            tasks[task_id]=item;persist(task_id)
        except Exception:pass
    for task_id,item in list(tasks.items()):
        recovered=activated_task_result(item)
        if recovered and item.get("status")!="completed":
            update(task_id,status="completed",stage="completed_with_warning",stage_label="入库完成（已恢复状态）",progress=100,message="Build与FAISS已实际切换成功；先前错误发生在激活后的维护阶段。",result=recovered,storage=recovered["storage"],error=None)

    @app.get("/")
    def home(request:Request):return templates.TemplateResponse(request,"index.html",{})

    @app.get("/api/health")
    def health():
        from .lmstudio import LocalLLMClient
        report=db.integrity();report["web_pid"]=os.getpid();report["llm"]=LocalLLMClient(config).health();report["ok"]=report["sqlite_ok"] and not report["foreign_key_errors"] and report["llm"].get("available",False)
        return report

    @app.post("/api/runtime/prewarm")
    def prewarm():
        service=RAGService(config,db);service.prewarm();check=reconcile_vector_index(db,service.retriever.embedding,config,True);return {"ok":True,"index":check}

    @app.get("/api/sources")
    def sources():return db.list_sources()

    @app.patch("/api/sources/{source_id}")
    def update_source(source_id:str,payload:dict[str,Any]):
        source=db.update_source_category(source_id,str(payload.get("category","未分类")))
        if not source:raise HTTPException(404,"来源不存在")
        return source

    @app.get("/api/sources/{source_id}/inspection")
    def inspection(source_id:str):
        value=db.source_inspection(source_id)
        if not value:raise HTTPException(404,"来源不存在")
        for section in value["sections"]:
            full=db.section_text(section["id"]);section["text"]=full["text"] if full else "";section["keywords"]=json.loads(section.pop("keywords_json","[]"));section["boundary_meta"]=json.loads(section.pop("boundary_meta_json","{}"))
        return value

    @app.get("/api/sources/{source_id}/export")
    def export_source(source_id:str):
        source=db.source(source_id)
        if not source:raise HTTPException(404,"来源不存在")
        output=data_dir/"exports"/source_id/source["current_build_id"]/"sections_expanded.json";export_expanded_sections(db,source_id,output)
        return FileResponse(output,media_type="application/json",filename=f"{source_id}-sections-expanded.json")

    @app.get("/api/query-logs")
    def logs(limit:int=200):return db.list_query_logs(limit)
    @app.get("/api/query-logs/export")
    def export_logs():return Response(json.dumps(db.list_query_logs(2000),ensure_ascii=False,indent=2),media_type="application/json; charset=utf-8",headers={"Content-Disposition":'attachment; filename="query-logs.json"'})
    @app.delete("/api/query-logs")
    def clear_logs():return {"deleted":db.clear_query_logs()}
    @app.get("/api/corrections")
    def corrections(status:str|None="pending"):return db.list_correction_candidates(status)

    @app.post("/api/corrections/{candidate_id}/review")
    def review(candidate_id:str,payload:dict[str,Any]):
        # Approval is recorded locally. A replacement Build is required before it may affect retrieval.
        candidate=db.review_correction(candidate_id,bool(payload.get("approve")))
        if not candidate:raise HTTPException(404,"纠错候选不存在")
        return {"id":candidate_id,"status":candidate["status"],"requires_rebuild":candidate["status"]=="approved"}

    @app.post("/api/sources/{source_id}/sentence-edits/preview")
    def preview_edits(source_id:str,payload:dict[str,Any]):
        inspection=db.source_inspection(source_id)
        if not inspection:raise HTTPException(404,"来源不存在")
        expected=str(payload.get("expected_build_id") or "")
        if expected and expected!=inspection["source"]["current_build_id"]:
            raise HTTPException(409,"当前Build已经变化，请重新打开数据检查页面")
        try:result=preview_sentence_edits(inspection["sentences"],payload.get("edits") or [])
        except (TypeError,ValueError) as error:raise HTTPException(400,str(error))
        return {"source_id":source_id,"build_id":inspection["source"]["current_build_id"],**result}

    @app.post("/api/sources/{source_id}/sentence-edits/apply")
    def apply_edits(source_id:str,payload:dict[str,Any],background:BackgroundTasks):
        inspection=db.source_inspection(source_id)
        if not inspection:raise HTTPException(404,"来源不存在")
        source=inspection["source"];expected=str(payload.get("expected_build_id") or "")
        if expected!=source["current_build_id"]:raise HTTPException(409,"当前Build已经变化，请重新预览校订")
        try:preview=preview_sentence_edits(inspection["sentences"],payload.get("edits") or [])
        except (TypeError,ValueError) as error:raise HTTPException(400,str(error))
        available={item["id"]:item for item in preview["glossary_candidates"]}
        selected_ids=list(dict.fromkeys(str(x) for x in (payload.get("glossary_candidate_ids") or [])))
        if any(item not in available for item in selected_ids):raise HTTPException(400,"提交了不属于本次文字差异的术语候选")
        selected=[available[item] for item in selected_ids]
        edits={int(item["sentence_id"]):str(item["approved_text"]).strip() for item in payload.get("edits") or []}
        rows=[]
        for sentence in inspection["sentences"]:
            row={"id":sentence["id"],"text":sentence["raw_text"],"start":sentence["start"],"end":sentence["end"]}
            approved=edits.get(int(sentence["id"]),sentence.get("approved_text"))
            if approved:row["approved_text"]=approved
            rows.append(row)
        task_id=uuid.uuid4().hex;directory=tasks_root/task_id;directory.mkdir(parents=True,exist_ok=True)
        sentences_json=directory/"sentences-input.json";sentences_json.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        review_path=directory/"manual_corrections.json";review_payload={"source_id":source_id,"parent_build_id":expected,"changes":preview["changes"],"selected_glossary_candidates":selected,"created_at":now_iso()};review_path.write_text(json.dumps(review_payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        cancel_events[task_id]=threading.Event();continue_events[task_id]=threading.Event()
        storage=[{"key":"manual_corrections","label":"人工校订审计","path":str(review_path.resolve()),"description":"修改前后差异及明确选择加入来源术语表的项目"}]
        update(task_id,status="queued",stage="correction_queued",stage_label="校订已确认",progress=0,message=f"将同步重建{len(preview['changes'])}句及所有派生资产",category=source["category"],title=source["title"],storage=storage)
        terms=[{"variant":item["variant"],"canonical":item["canonical"]} for item in selected]
        background.add_task(run_ingest,task_id,resolve_asset_path(config,source["video_path"]),source["category"],source["title"],sentences_json,False,terms,review_path)
        return {"task_id":task_id,"changes":len(preview["changes"]),"glossary_terms":len(terms)}

    @app.post("/api/ask")
    def ask(payload:dict[str,Any]):
        query=str(payload.get("query","")).strip()
        if not query:raise HTTPException(400,"问题不能为空")
        if qa_paused.is_set():raise HTTPException(503,"ASR正在独占本机内存；预检取消或入库完成后会自动恢复并预热问答模型")
        if not ask_lock.acquire(False):raise HTTPException(429,"已有问题正在处理")
        try:return RAGService(config,db).ask(query,payload.get("categories") or None,payload.get("log_query"))
        finally:ask_lock.release()

    def run_ingest(task_id:str,path:Path,category:str,title:str|None,sentences_json:Path|None=None,delete_input_on_success:bool=True,glossary_terms:list[dict[str,str]]|None=None,correction_review:Path|None=None,preflight_report:dict[str,Any]|None=None):
        if not build_lock.acquire(False):update(task_id,status="failed",stage="busy",message="已有Build正在运行",error="busy");return
        event=cancel_events[task_id]
        glossary_path=project_path(config,config["project"]["glossaries_dir"])/"sources"/f"{sha256_file(path.resolve())}.json" if glossary_terms else None
        glossary_backup=glossary_path.read_bytes() if glossary_path and glossary_path.is_file() else None
        keep_glossary=False
        stage_clock=time.perf_counter();last_stage=None;stage_timings:dict[str,int]={}
        status_entry={"key":"task_status","label":"任务状态","path":str(status_path(task_id).resolve()),"description":"本次历史任务状态；当前知识库状态以数据库current_build_id为准"}
        def report(stage,percent,message,storage):
            nonlocal stage_clock,last_stage
            now=time.perf_counter()
            if last_stage is not None:stage_timings[last_stage]=stage_timings.get(last_stage,0)+round((now-stage_clock)*1000)
            last_stage=stage;stage_clock=now
            update(task_id,status="running",stage=stage,stage_label=stage,progress=percent,message=message,storage=[status_entry,*storage],stage_timings=dict(stage_timings))
        def confirm_preflight(report:dict[str,Any])->bool:
            event=continue_events[task_id]
            update(task_id,status="awaiting_confirmation",stage="awaiting_confirmation",stage_label="ASR预检完成，等待确认",progress=24,message=f"三段共{report['sampled_seconds']:.0f}秒耗时{report['elapsed_seconds']:.1f}秒；预计全片约{report['estimated_full_seconds']:.0f}秒。确认后才运行完整ASR。",preflight={k:v for k,v in report.items() if k!="raw"},storage=tasks[task_id].get("storage",[]),stage_timings=dict(stage_timings))
            while not event.wait(.5):
                if cancel_events[task_id].is_set():return False
            return not cancel_events[task_id].is_set()
        try:
            if glossary_path and glossary_terms:
                glossary_path.parent.mkdir(parents=True,exist_ok=True)
                try:glossary_data=json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.is_file() else {}
                except Exception as error:raise ValueError(f"来源术语表不是有效JSON：{error}")
                for term in glossary_terms:
                    canonical,variant=term["canonical"],term["variant"]
                    glossary_data[canonical]=list(dict.fromkeys([*glossary_data.get(canonical,[]),variant]))
                temporary=glossary_path.with_suffix(".tmp");temporary.write_text(json.dumps(glossary_data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");temporary.replace(glossary_path)
            if sentences_json is None:
                qa_paused.set()
                # Let an already-running answer finish before any model is unloaded.
                with ask_lock:pass
            result=Ingestor(config,db).ingest(path,category,title,sentences_json=sentences_json,progress=report,cancelled=event.is_set,confirm_preflight=confirm_preflight,preflight_report=preflight_report)
            if correction_review and correction_review.is_file():
                build_entry=next((x for x in result["storage"] if x.get("key")=="build"),None)
                if build_entry:
                    destination=Path(build_entry["path"]).parent/"manual_corrections.json";shutil.copy2(correction_review,destination);result["storage"].append({"key":"manual_corrections","label":"manual_corrections.json","path":str(destination.resolve()),"description":"本Build逐句人工校订及术语表选择审计"})
            result["storage"]=[status_entry,*result["storage"]];keep_glossary=True
            if last_stage is not None:stage_timings[last_stage]=stage_timings.get(last_stage,0)+round((time.perf_counter()-stage_clock)*1000)
            update(task_id,status="completed",stage="completed",stage_label="入库完成",progress=100,message="新Build已原子切换并可检索",result=result,storage=result["storage"],stage_timings=stage_timings)
            # Upload staging is redundant after a verified managed copy exists.
            if delete_input_on_success and path.is_file():path.unlink()
        except InterruptedError as error:update(task_id,status="cancelled",stage="cancelled",stage_label="已安全取消",message=str(error),error=str(error))
        except Exception as error:
            recovered=activated_task_result(tasks.get(task_id,{}))
            if recovered:
                keep_glossary=True
                update(task_id,status="completed",stage="completed_with_warning",stage_label="入库完成（有维护警告）",progress=100,message=f"Build已原子切换；激活后的维护步骤出现警告：{error}",result=recovered,storage=recovered["storage"],error=None)
                if delete_input_on_success and path.is_file():path.unlink()
            else:update(task_id,status="failed",stage="failed",stage_label="入库失败",message=f"处理失败：{error}",error=str(error))
        finally:
            if glossary_path and not keep_glossary:
                if glossary_backup is None:
                    if glossary_path.exists():glossary_path.unlink()
                else:glossary_path.write_bytes(glossary_backup)
            qa_paused.clear();build_lock.release();cancel_events.pop(task_id,None);continue_events.pop(task_id,None)

    @app.post("/api/ingest")
    async def ingest(background:BackgroundTasks,video:UploadFile=File(...),category:str=Form("未分类"),title:str|None=Form(None)):
        task_id=uuid.uuid4().hex;directory=tasks_root/task_id;directory.mkdir(parents=True,exist_ok=True);path=directory/Path(video.filename or "video.mp4").name
        with path.open("wb") as out:shutil.copyfileobj(video.file,out)
        cancel_events[task_id]=threading.Event();continue_events[task_id]=threading.Event();update(task_id,status="queued",stage="uploaded",stage_label="上传完成",progress=0,message="等待本地Build",filename=path.name,category=category or "未分类",title=title or path.stem,storage=[{"key":"upload","label":"上传暂存","path":str(path.resolve()),"description":"成功切换Build后自动删除"}]);background.add_task(run_ingest,task_id,path,category or "未分类",title,None,True);return {"task_id":task_id}

    @app.post("/api/sources/{source_id}/glossary")
    def add_glossary_term(source_id:str,payload:dict[str,Any],background:BackgroundTasks):
        source=db.source(source_id)
        if not source:raise HTTPException(404,"来源不存在")
        canonical=str(payload.get("canonical","")).strip();variants=[str(x).strip() for x in payload.get("variants",[]) if str(x).strip()]
        if not canonical or not variants:raise HTTPException(400,"规范术语和至少一个错词/别名不能为空")
        glossary=project_path(config,config["project"]["glossaries_dir"])/"sources"/f"{source_id}.json";glossary.parent.mkdir(parents=True,exist_ok=True)
        try:data=json.loads(glossary.read_text(encoding="utf-8")) if glossary.is_file() else {}
        except Exception as error:raise HTTPException(400,f"来源术语表不是有效JSON：{error}")
        data[canonical]=list(dict.fromkeys([*data.get(canonical,[]),*variants]));tmp=glossary.with_suffix(".tmp");tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(glossary)
        inspection=db.source_inspection(source_id);task_id=uuid.uuid4().hex;directory=tasks_root/task_id;directory.mkdir(parents=True,exist_ok=True);sentences_json=directory/"sentences-input.json"
        rows=[{"id":s["id"],"text":s["raw_text"],"start":s["start"],"end":s["end"]} for s in inspection["sentences"]];sentences_json.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        cancel_events[task_id]=threading.Event();continue_events[task_id]=threading.Event();update(task_id,status="queued",stage="glossary_saved",stage_label="术语已保存",progress=0,message="将从原始句子构建校订视图并原子替换Build",category=source["category"],title=source["title"],storage=[{"key":"glossary","label":"来源术语表","path":str(glossary.resolve()),"description":"人工批准的规范术语与错词/别名"}]);background.add_task(run_ingest,task_id,resolve_asset_path(config,source["video_path"]),source["category"],source["title"],sentences_json,False);return {"task_id":task_id,"glossary":str(glossary)}

    @app.post("/api/tasks/{task_id}/continue")
    def continue_task(task_id:str):
        event=continue_events.get(task_id);item=tasks.get(task_id)
        if not event or not item or item.get("status")!="awaiting_confirmation":raise HTTPException(409,"任务不在等待确认状态")
        event.set();return {"task_id":task_id,"continue_requested":True}

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id:str):
        event=cancel_events.get(task_id)
        if not event:raise HTTPException(409,"任务不在可取消状态")
        event.set();return {"task_id":task_id,"cancel_requested":True}

    @app.post("/api/tasks/{task_id}/retry")
    def retry_task(task_id:str,background:BackgroundTasks):
        item=tasks.get(task_id)
        if not item:raise HTTPException(404,"任务不存在")
        if item.get("status") not in {"failed","cancelled"}:raise HTTPException(409,"只有失败或已取消任务可以重试")
        filename=Path(str(item.get("filename") or "")).name;upload=tasks_root/task_id/filename
        if not filename or not upload.is_file():raise HTTPException(409,"原上传文件已不存在，请重新选择视频")
        if not item.get("preflight"):raise HTTPException(409,"任务没有可复用的预检报告，请重新开始入库")
        cancel_events[task_id]=threading.Event();continue_events[task_id]=threading.Event()
        update(task_id,status="queued",stage="retry_queued",stage_label="复用检查点重试",progress=0,message="保留原上传，复用已确认预检和Qwen转写检查点；仅重跑ForcedAligner及后续资产",error=None,result=None)
        background.add_task(run_ingest,task_id,upload,item.get("category") or "未分类",item.get("title") or upload.stem,None,True,None,None,item["preflight"])
        return {"task_id":task_id,"checkpoint_retry":True}
    @app.get("/api/tasks-active")
    def active_tasks():
        return [dict(item) for item in tasks.values() if item.get("status") in {"queued","running","awaiting_confirmation"}]
    @app.get("/api/tasks/{task_id}")
    def task(task_id:str):
        if task_id not in tasks:raise HTTPException(404,"任务不存在")
        return dict(tasks[task_id])
    @app.get("/api/tasks-latest")
    def latest():
        if not tasks:raise HTTPException(404,"暂无任务")
        return dict(max(tasks.values(),key=lambda x:x.get("updated_at","")))

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id:str):
        if not build_lock.acquire(False):raise HTTPException(409,"Build正在运行")
        try:
            with FileLock(data_dir/"runtime"/"build.lock"):
                source=db.source(source_id)
                if not source:raise HTTPException(404,"来源不存在")
                embedding=get_embedding_engine(project_path(config,config["models"]["embedding"]),int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu")));meta=write_vector_index(db.active_index_documents(exclude_source_id=source_id),embedding,config);deleted=db.delete_source_with_index(source_id,meta);write_index_manifest(config,meta);prune_index_versions(config,meta);return {"deleted":source_id,"video_preserved":deleted["video_path"]}
        finally:build_lock.release()

    @app.get("/api/video/{section_id}")
    def video(section_id:str):
        section=db.section_text(section_id)
        if not section:raise HTTPException(404,"章节不存在")
        path=resolve_asset_path(config,section["video_path"]);sources_root=project_path(config,config["project"]["sources_dir"])
        if not path.is_file() or sources_root not in path.parents:raise HTTPException(404,"视频不存在")
        start,end=int(section["start_ms"]),int(section["end_ms"]);clips=data_dir/"clips";clips.mkdir(parents=True,exist_ok=True);clip=clips/f"{section_id}-{start}-{end}.mp4"
        with clip_lock:
            if not clip.is_file() or clip.stat().st_size<1024:
                temporary=clip.with_suffix(".tmp.mp4");ffmpeg=project_path(config,config["project"]["ffmpeg"]);duration=(end-start)/1000
                command=[str(ffmpeg),"-hide_banner","-loglevel","error","-y","-ss",f"{start/1000:.3f}","-i",str(path),"-t",f"{duration:.3f}","-map","0:v:0?","-map","0:a:0?","-c","copy","-avoid_negative_ts","make_zero","-movflags","+faststart",str(temporary)]
                result=subprocess.run(command,capture_output=True,text=True)
                if result.returncode or not temporary.is_file() or temporary.stat().st_size<1024:
                    if temporary.exists():temporary.unlink()
                    command=[str(ffmpeg),"-hide_banner","-loglevel","error","-y","-ss",f"{start/1000:.3f}","-i",str(path),"-t",f"{duration:.3f}","-map","0:v:0?","-map","0:a:0?","-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(temporary)];result=subprocess.run(command,capture_output=True,text=True)
                if result.returncode or not temporary.is_file():raise HTTPException(500,"生成章节片段失败")
                temporary.replace(clip)
            os.utime(clip,None)
            max_bytes=int(config["web"].get("clip_cache_max_mb",1024))*1024*1024
            cached=sorted((p for p in clips.glob("*.mp4") if p.is_file() and p!=clip),key=lambda p:p.stat().st_mtime)
            total=sum(p.stat().st_size for p in cached)+clip.stat().st_size
            for old in cached:
                if total<=max_bytes:break
                size=old.stat().st_size;old.unlink();total-=size
        return FileResponse(clip,media_type="video/mp4")
    return app
