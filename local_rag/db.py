from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import jieba

SCHEMA_VERSION = 3
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE sources(id TEXT PRIMARY KEY,video_hash TEXT NOT NULL UNIQUE,category TEXT NOT NULL DEFAULT '未分类',title TEXT NOT NULL,video_path TEXT NOT NULL,media_duration_ms INTEGER NOT NULL DEFAULT 0,current_build_id TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE builds(id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,status TEXT NOT NULL,category TEXT NOT NULL,title TEXT NOT NULL,video_path TEXT NOT NULL,media_duration_ms INTEGER NOT NULL,config_hash TEXT NOT NULL,model_manifest_hash TEXT NOT NULL,glossary_version TEXT NOT NULL,boundary_method TEXT NOT NULL,asr_provenance_json TEXT NOT NULL DEFAULT '{}',asr_preflight_json TEXT NOT NULL DEFAULT '{}',index_version TEXT,sentence_count INTEGER NOT NULL DEFAULT 0,section_count INTEGER NOT NULL DEFAULT 0,chunk_count INTEGER NOT NULL DEFAULT 0,degraded INTEGER NOT NULL DEFAULT 0,degradation_json TEXT NOT NULL DEFAULT '[]',error TEXT,started_at TEXT NOT NULL,finished_at TEXT);
CREATE INDEX builds_source_idx ON builds(source_id,started_at DESC);
CREATE TABLE sentences(build_id TEXT NOT NULL REFERENCES builds(id) ON DELETE CASCADE,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,sentence_id INTEGER NOT NULL,raw_text TEXT NOT NULL,approved_text TEXT,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,PRIMARY KEY(build_id,sentence_id));
CREATE TABLE sections(id TEXT PRIMARY KEY,build_id TEXT NOT NULL REFERENCES builds(id) ON DELETE CASCADE,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,position INTEGER NOT NULL,title TEXT NOT NULL,summary TEXT NOT NULL,keywords_json TEXT NOT NULL,start_sentence_id INTEGER NOT NULL,end_sentence_id INTEGER NOT NULL,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,boundary_meta_json TEXT NOT NULL,UNIQUE(build_id,position));
CREATE TABLE chunks(id TEXT PRIMARY KEY,build_id TEXT NOT NULL REFERENCES builds(id) ON DELETE CASCADE,section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,position INTEGER NOT NULL,start_sentence_id INTEGER NOT NULL,end_sentence_id INTEGER NOT NULL,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,token_count INTEGER NOT NULL,context_before_id INTEGER,context_after_id INTEGER,UNIQUE(section_id,position));
CREATE TABLE chunk_documents(chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,build_id TEXT NOT NULL,source_id TEXT NOT NULL,section_id TEXT NOT NULL,search_text TEXT NOT NULL,embedding_text TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED,build_id UNINDEXED,source_id UNINDEXED,category UNINDEXED,tokens);
CREATE TABLE correction_candidates(id TEXT PRIMARY KEY,build_id TEXT NOT NULL REFERENCES builds(id) ON DELETE CASCADE,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,original TEXT NOT NULL,candidate TEXT NOT NULL,sentence_ids_json TEXT NOT NULL,reason TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',reviewed_at TEXT);
CREATE TABLE glossary_terms(id INTEGER PRIMARY KEY AUTOINCREMENT,scope TEXT NOT NULL,scope_id TEXT NOT NULL,canonical TEXT NOT NULL,variants_json TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(scope,scope_id,canonical));
CREATE TABLE query_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,query TEXT NOT NULL,answer TEXT,sufficient INTEGER NOT NULL,section_ids_json TEXT NOT NULL,elapsed_ms INTEGER NOT NULL,timings_json TEXT NOT NULL);
CREATE TABLE system_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


class LegacyDatabaseError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def lexical_tokens(text: str) -> str:
    words = [word.strip().lower() for word in jieba.cut_for_search(text) if word.strip()]
    ascii_terms: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in "/._-+"):
            current.append(char.lower())
        elif current:
            ascii_terms.append("".join(current)); current = []
    if current:
        ascii_terms.append("".join(current))
    return " ".join(words + ascii_terms)


class Database:
    def __init__(self, path: Path):
        self.path = path.resolve(); self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con:
            existing = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sources'").fetchone()
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            if existing and version == 2:
                con.execute("ALTER TABLE builds ADD COLUMN asr_provenance_json TEXT NOT NULL DEFAULT '{}'")
                con.execute("ALTER TABLE builds ADD COLUMN asr_preflight_json TEXT NOT NULL DEFAULT '{}'")
                con.execute(f"PRAGMA user_version={SCHEMA_VERSION}");con.commit();version=SCHEMA_VERSION
            if existing and version != SCHEMA_VERSION:
                raise LegacyDatabaseError(f"检测到不支持的数据库schema（version={version}）：{self.path}")
            if not existing:
                con.executescript(SCHEMA); con.execute(f"PRAGMA user_version={SCHEMA_VERSION}"); con.commit()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30); con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON"); con.execute("PRAGMA journal_mode=WAL")
        return con

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        con = self.connect()
        try:
            con.execute("BEGIN IMMEDIATE"); yield con; con.commit()
        except Exception:
            con.rollback(); raise
        finally:
            con.close()

    def meta(self, key: str, default: Any = None) -> Any:
        with closing(self.connect()) as con:
            row = con.execute("SELECT value FROM system_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    @staticmethod
    def _set_meta(con: sqlite3.Connection, key: str, value: Any) -> None:
        con.execute("INSERT INTO system_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, ensure_ascii=False, separators=(",", ":"))))

    def stage_bundle(self, bundle: dict[str, Any]) -> None:
        src, build = bundle["source"], bundle["build"]
        with self.transaction() as con:
            con.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,NULL,?,?) ON CONFLICT(id) DO NOTHING", (src["id"],src["video_hash"],src["category"],src["title"],src["video_path"],src["media_duration_ms"],now_iso(),now_iso()))
            con.execute("""INSERT INTO builds(id,source_id,status,category,title,video_path,media_duration_ms,config_hash,model_manifest_hash,glossary_version,boundary_method,asr_provenance_json,asr_preflight_json,index_version,sentence_count,section_count,chunk_count,degraded,degradation_json,error,started_at,finished_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (build["id"],src["id"],"staged",src["category"],src["title"],src["video_path"],src["media_duration_ms"],build["config_hash"],build["model_manifest_hash"],build["glossary_version"],build["boundary_method"],json.dumps(build.get("asr_provenance",{}),ensure_ascii=False),json.dumps(build.get("asr_preflight",{}),ensure_ascii=False),None,len(bundle["sentences"]),len(bundle["sections"]),len(bundle["chunks"]),int(bool(build.get("degraded"))),json.dumps(build.get("degradation",[]),ensure_ascii=False),None,build.get("started_at",now_iso()),None))
            con.executemany("INSERT INTO sentences VALUES (?,?,?,?,?,?,?)", [(build["id"],src["id"],s["id"],s["text"],s.get("approved_text"),s["start"],s["end"]) for s in bundle["sentences"]])
            for pos,s in enumerate(bundle["sections"],1):
                con.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (s["id"],build["id"],src["id"],pos,s["title"],s.get("summary",""),json.dumps(s.get("keywords",[]),ensure_ascii=False),s["start_id"],s["end_id"],s["start"],s["end"],json.dumps(s.get("boundary_meta",{}),ensure_ascii=False)))
            for c in bundle["chunks"]:
                con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (c["id"],build["id"],c["section_id"],src["id"],c["position"],c["start_id"],c["end_id"],c["start"],c["end"],c["token_count"],c.get("context_before_id"),c.get("context_after_id")))
                con.execute("INSERT INTO chunk_documents VALUES (?,?,?,?,?,?)", (c["id"],build["id"],src["id"],c["section_id"],c["search_text"],c["embedding_text"]))
                con.execute("INSERT INTO chunks_fts VALUES (?,?,?,?,?)", (c["id"],build["id"],src["id"],src["category"],lexical_tokens(c["search_text"])))
            for c in bundle.get("correction_candidates",[]):
                con.execute("INSERT INTO correction_candidates VALUES (?,?,?,?,?,?,?,?,NULL)", (c["id"],build["id"],src["id"],c["original"],c["candidate"],json.dumps(c.get("sentence_ids",[])),c.get("reason",""),"pending"))

    def fail_build(self, build_id: str, error: str, status: str = "failed") -> None:
        with self.transaction() as con:
            con.execute("UPDATE builds SET status=?,error=?,finished_at=? WHERE id=? AND status='staged'", (status,error,now_iso(),build_id))

    def index_documents_for_activation(self, source_id: str, build_id: str) -> list[dict[str, Any]]:
        sql = """SELECT d.chunk_id id,d.embedding_text,d.build_id,d.source_id,d.section_id FROM chunk_documents d JOIN sources s ON s.id=d.source_id WHERE (d.source_id<>? AND d.build_id=s.current_build_id) OR (d.source_id=? AND d.build_id=?) ORDER BY d.source_id,d.chunk_id"""
        with closing(self.connect()) as con:
            return [dict(r) for r in con.execute(sql,(source_id,source_id,build_id))]

    def active_index_documents(self, exclude_source_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT d.chunk_id id,d.embedding_text,d.build_id,d.source_id,d.section_id FROM chunk_documents d JOIN sources s ON s.id=d.source_id AND s.current_build_id=d.build_id"
        params: list[Any] = []
        if exclude_source_id: sql += " WHERE d.source_id<>?"; params.append(exclude_source_id)
        sql += " ORDER BY d.source_id,d.chunk_id"
        with closing(self.connect()) as con:
            return [dict(r) for r in con.execute(sql,params)]

    def activate_build(self, bundle: dict[str, Any], index_meta: dict[str, Any]) -> None:
        src, build = bundle["source"], bundle["build"]
        with self.transaction() as con:
            row = con.execute("SELECT status FROM builds WHERE id=?",(build["id"],)).fetchone()
            if not row or row["status"] != "staged": raise RuntimeError("Build不存在或已经结束")
            con.execute("UPDATE sources SET category=?,title=?,video_path=?,media_duration_ms=?,current_build_id=?,updated_at=? WHERE id=?", (src["category"],src["title"],src["video_path"],src["media_duration_ms"],build["id"],now_iso(),src["id"]))
            con.execute("UPDATE builds SET status='ready',index_version=?,finished_at=? WHERE id=?", (index_meta["version"],now_iso(),build["id"]))
            self._set_meta(con,"active_index",index_meta)

    def activate_index_only(self, index_meta: dict[str, Any]) -> None:
        with self.transaction() as con: self._set_meta(con,"active_index",index_meta)

    def delete_source_with_index(self, source_id: str, index_meta: dict[str, Any]) -> dict[str, Any] | None:
        with self.transaction() as con:
            row=con.execute("SELECT * FROM sources WHERE id=?",(source_id,)).fetchone()
            if not row:return None
            con.execute("DELETE FROM chunks_fts WHERE source_id=?",(source_id,)); con.execute("DELETE FROM sources WHERE id=?",(source_id,)); self._set_meta(con,"active_index",index_meta)
            return dict(row)

    def list_sources(self) -> list[dict[str, Any]]:
        sql="SELECT s.*,b.sentence_count,b.section_count,b.chunk_count,b.index_version,b.degraded FROM sources s JOIN builds b ON b.id=s.current_build_id ORDER BY s.updated_at DESC"
        with closing(self.connect()) as con:return [dict(r) for r in con.execute(sql)]

    def source(self, source_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as con:r=con.execute("SELECT * FROM sources WHERE id=?",(source_id,)).fetchone()
        return dict(r) if r else None

    def update_source_category(self,source_id:str,category:str)->dict[str,Any]|None:
        category=(category or "未分类").strip() or "未分类"
        with self.transaction() as con:
            row=con.execute("SELECT current_build_id FROM sources WHERE id=?",(source_id,)).fetchone()
            if not row:return None
            con.execute("UPDATE sources SET category=?,updated_at=? WHERE id=?",(category,now_iso(),source_id))
            if row["current_build_id"]:
                con.execute("UPDATE chunks_fts SET category=? WHERE source_id=? AND build_id=?",(category,source_id,row["current_build_id"]))
        return self.source(source_id)

    def list_builds(self, source_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as con:rows=[dict(r) for r in con.execute("SELECT * FROM builds WHERE source_id=? ORDER BY started_at DESC",(source_id,))]
        for row in rows:
            row["asr_provenance"]=json.loads(row.pop("asr_provenance_json","{}") or "{}")
            row["asr_preflight"]=json.loads(row.pop("asr_preflight_json","{}") or "{}")
        return rows

    def active_chunk_ids(self) -> list[str]:
        with closing(self.connect()) as con:return [r["id"] for r in con.execute("SELECT c.id FROM chunks c JOIN sources s ON s.id=c.source_id AND s.current_build_id=c.build_id ORDER BY c.source_id,c.id")]

    def chunk_map(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:return {}
        sql="""SELECT c.*,d.search_text,d.embedding_text,sec.title section_title,sec.position section_position,sec.summary,sec.keywords_json,sec.start_ms section_start,sec.end_ms section_end,s.category,s.title source_title,s.video_path FROM chunks c JOIN chunk_documents d ON d.chunk_id=c.id JOIN sections sec ON sec.id=c.section_id JOIN sources s ON s.id=c.source_id AND s.current_build_id=c.build_id WHERE c.id IN (%s)""" % ",".join("?" for _ in ids)
        with closing(self.connect()) as con:return {r["id"]:dict(r) for r in con.execute(sql,ids)}

    def section_text(self, section_id: str, include_sentences: bool=False) -> dict[str,Any] | None:
        with closing(self.connect()) as con:
            sec=con.execute("SELECT x.*,s.category,s.title source_title,s.video_path FROM sections x JOIN sources s ON s.id=x.source_id AND s.current_build_id=x.build_id WHERE x.id=?",(section_id,)).fetchone()
            if not sec:return None
            rows=[dict(r) for r in con.execute("SELECT sentence_id,raw_text,approved_text,start_ms,end_ms,COALESCE(approved_text,raw_text) effective_text FROM sentences WHERE build_id=? AND sentence_id BETWEEN ? AND ? ORDER BY sentence_id",(sec["build_id"],sec["start_sentence_id"],sec["end_sentence_id"]))]
        out=dict(sec);out["text"]="".join(r["effective_text"] for r in rows)
        if include_sentences:out["sentences"]=rows
        return out

    def sections_by_ids(self, ids:list[str])->list[dict[str,Any]]:
        return [x for x in (self.section_text(i) for i in ids) if x]

    def source_inspection(self,source_id:str)->dict[str,Any]|None:
        src=self.source(source_id)
        if not src or not src.get("current_build_id"):return None
        bid=src["current_build_id"]
        with closing(self.connect()) as con:
            sentences=[dict(r) for r in con.execute("SELECT sentence_id id,raw_text,approved_text,start_ms start,end_ms end,COALESCE(approved_text,raw_text) effective_text FROM sentences WHERE build_id=? ORDER BY sentence_id",(bid,))]
            sections=[dict(r) for r in con.execute("SELECT * FROM sections WHERE build_id=? ORDER BY position",(bid,))]
            chunks=[dict(r) for r in con.execute("SELECT * FROM chunks WHERE build_id=? ORDER BY section_id,position",(bid,))]
        return {"source":src,"sentences":sentences,"sections":sections,"chunks":chunks,"builds":self.list_builds(source_id)}

    def lexical_search(self,query:str,limit:int,categories:list[str]|None=None)->list[str]:
        tokens=lexical_tokens(query).split()
        if not tokens:return []
        match=" OR ".join(f'"{t.replace(chr(34),"")}"' for t in tokens[:30])
        sql="SELECT f.chunk_id FROM chunks_fts f JOIN sources s ON s.id=f.source_id AND s.current_build_id=f.build_id WHERE chunks_fts MATCH ?";params:list[Any]=[match]
        if categories:sql+=" AND s.category IN (%s)" % ",".join("?" for _ in categories);params.extend(categories)
        sql+=" ORDER BY bm25(chunks_fts) LIMIT ?";params.append(max(1,int(limit)))
        with closing(self.connect()) as con:return [r["chunk_id"] for r in con.execute(sql,params)]

    def list_correction_candidates(self,status:str|None="pending")->list[dict[str,Any]]:
        sql="SELECT c.*,s.title source_title,s.video_path FROM correction_candidates c JOIN sources s ON s.id=c.source_id AND s.current_build_id=c.build_id";params=[]
        if status:sql+=" WHERE c.status=?";params.append(status)
        with closing(self.connect()) as con:rows=con.execute(sql,params).fetchall()
        out=[]
        for r in rows:
            x=dict(r);x["sentence_ids"]=json.loads(x.pop("sentence_ids_json"));out.append(x)
        return out

    def review_correction(self,candidate_id:str,approve:bool)->dict[str,Any]|None:
        with self.transaction() as con:
            row=con.execute("SELECT * FROM correction_candidates WHERE id=?",(candidate_id,)).fetchone()
            if not row:return None
            status="approved" if approve else "rejected";con.execute("UPDATE correction_candidates SET status=?,reviewed_at=? WHERE id=?",(status,now_iso(),candidate_id))
            # A ready Build is immutable. Approval is audit state only; applying it
            # requires constructing and atomically activating a replacement Build.
            return {**dict(row),"status":status}

    def log_query(self,query:str,answer:str|None,sufficient:bool,section_ids:list[str],elapsed_ms:int,timings:dict[str,int]|None=None)->None:
        with closing(self.connect()) as con:con.execute("INSERT INTO query_logs(created_at,query,answer,sufficient,section_ids_json,elapsed_ms,timings_json) VALUES (?,?,?,?,?,?,?)",(now_iso(),query,answer,int(sufficient),json.dumps(section_ids),elapsed_ms,json.dumps(timings or {},ensure_ascii=False)));con.commit()

    def list_query_logs(self,limit:int=200)->list[dict[str,Any]]:
        with closing(self.connect()) as con:rows=con.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT ?",(max(1,min(int(limit),2000)),)).fetchall()
        out=[]
        for r in rows:
            x=dict(r);x["sufficient"]=bool(x["sufficient"]);x["section_ids"]=json.loads(x.pop("section_ids_json"));x["timings"]=json.loads(x.pop("timings_json"));out.append(x)
        return out

    def clear_query_logs(self)->int:
        with closing(self.connect()) as con:cur=con.execute("DELETE FROM query_logs");con.commit();return cur.rowcount

    def prune_logs(self,days:int)->int:
        cutoff=(datetime.now(timezone.utc)-timedelta(days=days)).isoformat()
        with closing(self.connect()) as con:cur=con.execute("DELETE FROM query_logs WHERE created_at<?",(cutoff,));con.commit();return cur.rowcount

    def integrity(self)->dict[str,Any]:
        with closing(self.connect()) as con:
            ok=con.execute("PRAGMA integrity_check").fetchone()[0]=="ok";fk=[tuple(r) for r in con.execute("PRAGMA foreign_key_check")];sources=con.execute("SELECT count(*) FROM sources WHERE current_build_id IS NOT NULL").fetchone()[0];chunks=con.execute("SELECT count(*) FROM chunks c JOIN sources s ON s.id=c.source_id AND s.current_build_id=c.build_id").fetchone()[0]
        return {"schema_version":SCHEMA_VERSION,"sqlite_ok":ok,"foreign_key_errors":fk,"sources":sources,"active_chunks":chunks}
