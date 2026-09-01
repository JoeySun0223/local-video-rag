from __future__ import annotations

import gc
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

import faiss

from .config import index_dir, project_path
from .db import Database
from .lmstudio import LocalLLMClient
from .pipeline import get_embedding_engine, reconcile_vector_index


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "has_sufficient_evidence": {"type": "boolean"},
        "answer": {"type": "string"},
        "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "source_section_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["has_sufficient_evidence", "answer", "evidence_chunk_ids", "source_section_ids"],
    "additionalProperties": False,
}


_HOW_IT_WORKS_PATTERNS = (
    re.compile(r"^(?P<topic>.+?)(?:是)?怎么(?:运作|运行|工作)(?:的)?[？?。.]?$"),
    re.compile(r"^(?P<topic>.+?)(?:是)?如何(?:运作|运行|工作)(?:的)?[？?。.]?$"),
)


def normalize_retrieval_query(query: str) -> str:
    """Make a narrow, intent-preserving retrieval variant for colloquial questions."""
    value = query.strip()
    for pattern in _HOW_IT_WORKS_PATTERNS:
        match = pattern.fullmatch(value)
        if match:
            topic = match.group("topic").strip(" ，,：:；;")
            if topic:
                return f"{topic}的工作原理和具体过程是什么？"
    return value


def aggregate_section_score(scores: list[float]) -> float:
    """Rank a section by its strongest evidence, independent of section length."""
    return max(scores, default=0.0)


class RerankerEngine:
    def __init__(self, path: Path, device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        self.path = path.resolve(); self.device = device
        self.model = CrossEncoder(str(self.path), device=device, local_files_only=True, max_length=512)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        # Request logits explicitly. CrossEncoder otherwise applies the model's
        # configured sigmoid; applying sigmoid twice would make irrelevant text
        # cluster just above 0.5 and defeat the refusal gate.
        import torch
        raw = self.model.predict([(query, text) for text in documents], batch_size=4, show_progress_bar=False, activation_fn=torch.nn.Identity())
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [1.0 / (1.0 + math.exp(-float(value[0] if isinstance(value, list) else value))) for value in values]


_RERANKERS: dict[str, RerankerEngine] = {}
_RERANKER_LOCK = threading.Lock()


def get_reranker_engine(path: Path, device: str = "cpu") -> RerankerEngine:
    key = f"{path.resolve()}|{device}"
    with _RERANKER_LOCK:
        if key not in _RERANKERS:
            _RERANKERS[key] = RerankerEngine(path, device)
        return _RERANKERS[key]


def unload_rerankers() -> None:
    with _RERANKER_LOCK:
        _RERANKERS.clear()
    gc.collect()


class Retriever:
    def __init__(self, config: dict[str, Any], db: Database):
        self.config = config; self.db = db
        self._embedding = None; self._reranker = None
        self._version = ""; self._index = None; self._mapping: list[dict[str, Any]] = []

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_engine(
                project_path(self.config, self.config["models"]["embedding"]),
                int(self.config["models"]["embedding_max_tokens"]),
                str(self.config["models"].get("embedding_device", "cpu")),
            )
        return self._embedding

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker_engine(project_path(self.config, self.config["models"]["reranker"]),str(self.config["models"].get("reranker_device","cpu")))
        return self._reranker

    def prewarm(self) -> None:
        _ = self.embedding; _ = self.reranker

    def _load_index(self) -> None:
        if not self.db.active_chunk_ids():
            self._index = None; self._mapping = []; self._version = ""; return
        check = reconcile_vector_index(self.db, self.embedding, self.config, repair=True)
        meta = check["meta"]
        if meta["version"] != self._version:
            directory = index_dir(self.config)
            self._index = faiss.read_index(str(directory / meta["index"]))
            self._mapping = json.loads((directory / meta["mapping"]).read_text(encoding="utf-8"))
            self._version = meta["version"]

    def search(self, queries: list[str], categories: list[str] | None = None) -> list[dict[str, Any]]:
        self._load_index()
        if self._index is None or not self._mapping:
            return []
        rules = self.config["retrieval"]; allowed = set(categories or [])
        all_ids = [item["id"] for item in self._mapping]; metadata = self.db.chunk_map(all_ids)
        dense_scores: dict[str, float] = {}; dense_ranks: dict[str, int] = {}; lexical_ranks: dict[str, int] = {}
        normalized_queries = [normalize_retrieval_query(query) for query in queries]
        retrieval_queries = list(dict.fromkeys([*queries, *normalized_queries]))
        for query in retrieval_queries:
            vector = self.embedding.encode([query])
            scores, indices = self._index.search(vector, self._index.ntotal)
            rank = 0
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                chunk_id = self._mapping[index]["id"]; item = metadata.get(chunk_id)
                if not item or (allowed and item["category"] not in allowed):
                    continue
                rank += 1; dense_scores[chunk_id] = max(dense_scores.get(chunk_id, -1.0), float(score)); dense_ranks[chunk_id] = min(dense_ranks.get(chunk_id, 10**9), rank)
                if rank >= int(rules["vector_candidates"]):
                    break
            for pos, chunk_id in enumerate(self.db.lexical_search(query, int(rules["lexical_candidates"]), categories), 1):
                lexical_ranks[chunk_id] = min(lexical_ranks.get(chunk_id, 10**9), pos)
        loose_ids = {
            chunk_id for chunk_id, score in dense_scores.items()
            if score >= float(rules["min_dense_cosine"])
        } | set(lexical_ranks)
        if not loose_ids:
            return []
        rrf_k = int(rules["rrf_k"]); fused=[]
        for chunk_id in loose_ids:
            score = 0.0
            if chunk_id in dense_ranks: score += 1/(rrf_k+dense_ranks[chunk_id])
            if chunk_id in lexical_ranks: score += 1/(rrf_k+lexical_ranks[chunk_id])
            fused.append((score,chunk_id))
        fused.sort(reverse=True); candidate_ids=[chunk_id for _,chunk_id in fused[:int(rules["rerank_candidates"])]]
        candidate_meta=self.db.chunk_map(candidate_ids); query=normalized_queries[0]
        rerank_scores=self.reranker.score(query,[candidate_meta[item]["search_text"] for item in candidate_ids])
        reranked=[]
        for chunk_id,score in zip(candidate_ids,rerank_scores):
            if score < float(rules["min_reranker_score"]):
                continue
            item=candidate_meta[chunk_id];item["dense_score"]=dense_scores.get(chunk_id);item["reranker_score"]=score;item["retrieval_score"]=score;reranked.append(item)
        reranked.sort(key=lambda x:x["reranker_score"],reverse=True)
        sections:dict[str,dict[str,Any]]={}
        for item in reranked:
            section=sections.setdefault(item["section_id"],{"score":0.0,"support_scores":[],"chunks":[],**item})
            section["support_scores"].append(item["reranker_score"]);section["chunks"].append(item)
        for section in sections.values():
            values=section.pop("support_scores");section["score"]=aggregate_section_score(values);section["chunk_ids"]=[x["id"] for x in section["chunks"]]
        return sorted(sections.values(),key=lambda x:x["score"],reverse=True)[:int(rules["max_sections"])]


class RAGService:
    def __init__(self, config: dict[str, Any], db: Database):
        self.config=config;self.db=db;self.retriever=Retriever(config,db);self.llm=LocalLLMClient(config)

    def prewarm(self)->None:self.retriever.prewarm()

    @staticmethod
    def _ms(start:float)->int:return int((time.perf_counter()-start)*1000)

    def ask(self,query:str,categories:list[str]|None=None,log_query:bool|None=None)->dict[str,Any]:
        total=time.perf_counter();timings={};stage=time.perf_counter();results=self.retriever.search([query],categories);timings["retrieval_and_rerank_ms"]=self._ms(stage)
        if not results:
            answer="当前知识库中没有足够信息。";elapsed=self._ms(total);self._log(query,answer,False,[],elapsed,timings,log_query)
            return {"query":query,"answer":answer,"has_sufficient_evidence":False,"degraded":False,"message":None,"sources":[],"evidence_chunk_ids":[],"elapsed_ms":elapsed,"timings":timings}
        health=self.llm.health();cards=[]
        for item in results:
            cards.append({"section_id":item["section_id"],"title":item["section_title"],"summary":item.get("summary",""),"category":item["category"],"source_title":item["source_title"],"video_path":item["video_path"],"start":item["section_start"],"end":item["section_end"],"score":item["score"],"reranker_score":max(c["reranker_score"] for c in item["chunks"]),"evidence_chunk_ids":item["chunk_ids"]})
        if not health["available"]:
            elapsed=self._ms(total);self._log(query,None,False,[x["section_id"] for x in cards],elapsed,timings,log_query)
            return {"query":query,"answer":None,"has_sufficient_evidence":False,"degraded":True,"message":"本地生成模型不可用，已返回本地检索章节。","sources":cards,"evidence_chunk_ids":[],"elapsed_ms":elapsed,"timings":timings}
        context=[];allowed_pairs={};budget=self.llm.context_budget(int(self.config["llm"]["context_window"]),int(self.config["llm"].get("max_answer_tokens",512)))
        used=0
        for item in results:
            for chunk in item["chunks"]:
                part=f'[section_id={item["section_id"]} chunk_id={chunk["id"]} sentences={chunk["start_sentence_id"]}-{chunk["end_sentence_id"]}]\n{chunk["search_text"]}'
                count=self.llm.count_tokens(part)
                if context and used+count>budget:continue
                context.append(part);used+=count;allowed_pairs[chunk["id"]]=item["section_id"]
        stage=time.perf_counter();answer=None;sufficient=False;evidence=[];sections=[]
        try:
            result=self.llm.structured([{"role":"system","content":"只能依据输入的本地知识库Chunk回答。证据不足必须拒答，不得使用自身知识。引用ID必须与输入一致。"},{"role":"user","content":f"问题：{query}\n\n"+"\n\n---\n\n".join(context)}],"grounded_answer",ANSWER_SCHEMA,int(self.config["llm"].get("max_answer_tokens",512)))
            evidence=[x for x in result.get("evidence_chunk_ids",[]) if x in allowed_pairs];sections=list(dict.fromkeys(result.get("source_section_ids",[])))
            pair_sections={allowed_pairs[x] for x in evidence}
            if result.get("has_sufficient_evidence") and result.get("answer","").strip() and evidence and set(sections)==pair_sections:
                sufficient=True;answer=result["answer"].strip();cards=[x for x in cards if x["section_id"] in pair_sections][:3]
        except Exception:
            pass
        timings["qwen_answer_ms"]=self._ms(stage)
        if not sufficient:answer="当前知识库中没有足够信息。";cards=[];evidence=[]
        elapsed=self._ms(total);self._log(query,answer,sufficient,[x["section_id"] for x in cards],elapsed,timings,log_query)
        return {"query":query,"answer":answer,"has_sufficient_evidence":sufficient,"degraded":False,"message":None,"sources":cards,"evidence_chunk_ids":evidence,"elapsed_ms":elapsed,"timings":timings}

    def _log(self,query,answer,sufficient,sections,elapsed,timings,choice):
        enabled=self.config["web"].get("query_logging",True) if choice is None else bool(choice)
        if enabled:self.db.log_query(query,answer,sufficient,sections,elapsed,timings)
