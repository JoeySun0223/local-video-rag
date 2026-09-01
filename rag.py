from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from local_rag.config import assets_dir, database_path, ensure_project_dirs, load_config, project_path


def parser()->argparse.ArgumentParser:
    root=argparse.ArgumentParser(description="本地视频RAG");root.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml"));root.add_argument("--model")
    commands=root.add_subparsers(dest="command",required=True)
    ingest=commands.add_parser("ingest");ingest.add_argument("input",type=Path);ingest.add_argument("--category",default="未分类");ingest.add_argument("--title");ingest.add_argument("--sentences-json",type=Path);ingest.add_argument("--no-llm",action="store_true");ingest.add_argument("--yes",action="store_true",help="ASR预检成功后无需交互直接继续")
    for name in ("ask","search"):
        item=commands.add_parser(name);item.add_argument("query");item.add_argument("--category",action="append",dest="categories")
    commands.add_parser("status");serve=commands.add_parser("serve");serve.add_argument("--host");serve.add_argument("--port",type=int)
    commands.add_parser("start");commands.add_parser("stop");commands.add_parser("runtime-status")
    delete=commands.add_parser("delete");delete.add_argument("source_id")
    export=commands.add_parser("export-sections");export.add_argument("source_id");export.add_argument("--output",type=Path)
    review_sections=commands.add_parser("review-sections");review_sections.add_argument("source_id");review_sections.add_argument("--output",type=Path)
    curate_review=commands.add_parser("curate-section-review");curate_review.add_argument("preview",type=Path);curate_review.add_argument("--ends",required=True,help="逗号分隔的内部边界句ID");curate_review.add_argument("--output",type=Path,required=True);curate_review.add_argument("--labels",type=Path)
    activate_review=commands.add_parser("activate-section-review");activate_review.add_argument("preview",type=Path)
    commands.add_parser("rebuild-index");doctor=commands.add_parser("doctor");doctor.add_argument("--hashes",action="store_true")
    backup=commands.add_parser("backup");backup.add_argument("--output",type=Path)
    evaluate=commands.add_parser("eval");evaluate.add_argument("file",type=Path)
    return root


def hash_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def doctor(config,db,hashes=False):
    from local_rag.asr import get_asr_provider
    from local_rag.lmstudio import LocalLLMClient
    from local_rag.pipeline import get_embedding_engine,reconcile_vector_index
    manifest_path=project_path(config,config["project"]["models_dir"])/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf-8"));assets=[]
    for name in ("asr","vad","punctuation","embedding","reranker","qwen3_asr","qwen3_forced_aligner","qwen"):
        item=manifest.get(name,{});relative=item.get("weight") or item.get("path","");path=manifest_path.parent/relative if relative else manifest_path.parent/f"__missing_{name}__";value={"name":name,"path":str(path),"exists":bool(relative) and path.exists(),"sha256_ok":None}
        if hashes and path.is_file() and item.get("sha256"):value["sha256_ok"]=hash_file(path).lower()==item["sha256"].lower()
        assets.append(value)
    ffmpeg=project_path(config,config["project"]["ffmpeg"]);ffmpeg_ok=ffmpeg.is_file() and subprocess.run([str(ffmpeg),"-version"],capture_output=True).returncode==0
    integrity=db.integrity();index={"ok":not integrity["active_chunks"],"repaired":False,"error":None}
    embedding_path=project_path(config,config["models"]["embedding"])
    if embedding_path.exists():
        try:index=reconcile_vector_index(db,get_embedding_engine(embedding_path,int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu"))),config,repair=True)
        except Exception as error:index={"ok":False,"repaired":False,"error":str(error)}
    asr=get_asr_provider(config).doctor();required_assets=[x for x in assets if x["name"] in {"embedding","reranker","qwen"}]
    llama_server=project_path(config,config["llm"]["server_path"]);llama_ok=llama_server.is_file()
    report={"assets":assets,"asr":asr,"ffmpeg":{"path":str(ffmpeg),"ok":ffmpeg_ok},"llama_server":{"path":str(llama_server),"exists":llama_ok},"database":integrity,"index":index,"llm":LocalLLMClient(config).health()};report["ok"]=all(x["exists"] and x["sha256_ok"] is not False for x in required_assets) and asr["available"] and ffmpeg_ok and llama_ok and integrity["sqlite_ok"] and not integrity["foreign_key_errors"] and index["ok"] and report["llm"].get("available",False)
    return report


def main()->None:
    args=parser().parse_args();config=load_config(args.config)
    if args.model:config["llm"]["model"]=args.model
    ensure_project_dirs(config)
    if args.command in {"start","stop","runtime-status"}:
        from local_rag.runtime import start_stack,stop_stack,_state_path
        if args.command=="start":result=start_stack(config)
        elif args.command=="stop":result=stop_stack(config)
        else:
            path=_state_path(config);result=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status":"stopped"}
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    from local_rag.db import Database
    db=Database(database_path(config))
    if args.command=="ingest":
        from local_rag.pipeline import Ingestor
        def confirm(report):
            print(json.dumps({k:v for k,v in report.items() if k!="raw"},ensure_ascii=False,indent=2))
            if args.yes:return True
            try:return input("预检完成。输入 Y 运行完整ASR，其他输入取消：[y/N] ").strip().lower()=="y"
            except EOFError:return False
        result=Ingestor(config,db).ingest(args.input,args.category,args.title,args.sentences_json,not args.no_llm,confirm_preflight=confirm)
    elif args.command in {"ask","search"}:
        from local_rag.search import RAGService
        service=RAGService(config,db);result=service.ask(args.query,args.categories) if args.command=="ask" else service.retriever.search([args.query],args.categories)
    elif args.command=="status":result={"sources":db.list_sources(),"integrity":db.integrity(),"active_index":db.meta("active_index")}
    elif args.command=="serve":
        import uvicorn
        from local_rag.webapp import create_app
        uvicorn.run(create_app(config),host=args.host or config["web"]["host"],port=args.port or int(config["web"]["port"]));return
    elif args.command=="delete":
        from local_rag.pipeline import get_embedding_engine,prune_index_versions,write_vector_index,write_index_manifest
        from local_rag.runtime import FileLock
        data=project_path(config,config["project"]["data_dir"])
        with FileLock(data/"runtime"/"build.lock"):
            source=db.source(args.source_id)
            if not source:raise SystemExit("来源不存在")
            embedding=get_embedding_engine(project_path(config,config["models"]["embedding"]),int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu")));meta=write_vector_index(db.active_index_documents(args.source_id),embedding,config);deleted=db.delete_source_with_index(args.source_id,meta);write_index_manifest(config,meta);prune_index_versions(config,meta);result={"deleted":args.source_id,"video_preserved":deleted["video_path"]}
    elif args.command=="export-sections":
        from local_rag.pipeline import export_expanded_sections
        source=db.source(args.source_id)
        if not source:raise SystemExit("来源不存在")
        output=args.output or project_path(config,config["project"]["data_dir"])/"exports"/args.source_id/source["current_build_id"]/"sections_expanded.json";result={"path":str(export_expanded_sections(db,args.source_id,output.resolve()))}
    elif args.command=="review-sections":
        from local_rag.pipeline import preview_section_rebuild
        source=db.source(args.source_id)
        if not source:raise SystemExit("来源不存在")
        output=args.output or project_path(config,config["project"]["data_dir"])/"section_reviews"/args.source_id/f"{source['current_build_id']}.json"
        result=preview_section_rebuild(config,db,args.source_id,output)
    elif args.command=="curate-section-review":
        from local_rag.pipeline import curate_section_preview
        labels=json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
        result=curate_section_preview(db,args.preview,[int(x) for x in args.ends.split(",") if x.strip()],args.output,labels)
    elif args.command=="activate-section-review":
        from local_rag.pipeline import activate_section_preview
        result=activate_section_preview(config,db,args.preview)
    elif args.command=="rebuild-index":
        from local_rag.pipeline import get_embedding_engine,reconcile_vector_index
        result=reconcile_vector_index(db,get_embedding_engine(project_path(config,config["models"]["embedding"]),int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu"))),config,True)
    elif args.command=="doctor":result=doctor(config,db,args.hashes)
    elif args.command=="backup":
        asset_root=assets_dir(config);output=(args.output or Path(config["_root"])/"backups"/"knowledge-base.zip").resolve()
        if output==asset_root or asset_root in output.parents:raise SystemExit("备份文件必须放在 knowledge_base 外，避免把ZIP递归打包进自身")
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(asset_root.rglob("*")):
                if path.is_file():archive.write(path,"knowledge_base/"+str(path.relative_to(asset_root)))
        result={"path":str(output),"sha256":hash_file(output),"assets_root":str(asset_root),"note":"建议先运行 rag.py stop，再制作可迁移快照"}
    elif args.command=="eval":
        from local_rag.search import RAGService
        service=RAGService(config,db);rows=[]
        for case in json.loads(args.file.read_text(encoding="utf-8")):
            answer=service.ask(case["question"],[case["category"]] if case.get("category") else None,False);pred=bool(answer["has_sufficient_evidence"]);rows.append({"question":case["question"],"expected":bool(case["should_answer"]),"predicted":pred,"correct":pred==bool(case["should_answer"])})
        result={"accuracy":sum(x["correct"] for x in rows)/max(1,len(rows)),"cases":rows}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
