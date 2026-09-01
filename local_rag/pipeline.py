from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from .asr import ASRRequest, ASRResult, get_asr_provider
from .config import index_dir, portable_asset_path, project_path
from .db import Database, now_iso
from .lmstudio import LocalLLMClient
from .runtime import FileLock


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 20) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return cleaned[:80] or "untitled"


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Glossary:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.mapping: dict[str, list[str]] = {}
        for path in paths:
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            for canonical, variants in raw.items():
                target = self.mapping.setdefault(str(canonical), [])
                target.extend(str(item) for item in variants if str(item))
        self.mapping = {key: list(dict.fromkeys(value)) for key, value in self.mapping.items()}

    @staticmethod
    def _replace(text: str, variant: str, canonical: str) -> str:
        ascii_word = variant.isascii() and all(char.isalnum() or char in "/._-+ " for char in variant)
        pattern = re.escape(variant)
        if ascii_word:
            pattern = rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])"
        return re.sub(pattern, canonical, text, flags=re.IGNORECASE)

    def normalize(self, text: str) -> str:
        result = text
        matched: list[str] = []
        pairs = sorted(
            ((variant, canonical) for canonical, variants in self.mapping.items() for variant in variants),
            key=lambda item: len(item[0]), reverse=True,
        )
        for variant, canonical in pairs:
            changed = self._replace(result, variant, canonical)
            if changed != result:
                result = changed; matched.append(canonical)
        return result + (("\n术语：" + " ".join(dict.fromkeys(matched))) if matched else "")

    def correct_text(self, text: str) -> str:
        """Apply only human-controlled replacements, without retrieval aliases."""
        result = text
        pairs = sorted(
            ((variant, canonical) for canonical, variants in self.mapping.items() for variant in variants),
            key=lambda item: len(item[0]), reverse=True,
        )
        for variant, canonical in pairs:
            result = self._replace(result, variant, canonical)
        return result

    def hotwords(self) -> str:
        return " ".join(self.mapping)

    def version(self) -> str:
        return json_hash(self.mapping)

    def candidates(self, sentences: list[dict[str, Any]], source_id: str, build_id: str) -> list[dict[str, Any]]:
        # Every mapping loaded here is already human-controlled and is therefore
        # applied as approved_text, not presented again as a pending suggestion.
        # A future non-LLM detector may populate correction_candidates, but it must
        # never guess a canonical term without a controlled vocabulary.
        return []


class EmbeddingEngine:
    def __init__(self, model_path: Path, max_tokens: int = 512, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.path = model_path.resolve(); self.max_tokens = int(max_tokens); self.device = device
        self.model = SentenceTransformer(str(self.path), device=device, local_files_only=True)
        self.tokenizer = self.model.tokenizer

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


_EMBEDDING_ENGINES: dict[str, EmbeddingEngine] = {}
_EMBEDDING_LOCK = threading.Lock()


def get_embedding_engine(model_path: Path, max_tokens: int = 512, device: str = "cpu") -> EmbeddingEngine:
    key = f"{model_path.resolve()}|{device}"
    with _EMBEDDING_LOCK:
        if key not in _EMBEDDING_ENGINES:
            _EMBEDDING_ENGINES[key] = EmbeddingEngine(model_path, max_tokens, device)
        return _EMBEDDING_ENGINES[key]


def unload_embedding_engines() -> None:
    with _EMBEDDING_LOCK:
        _EMBEDDING_ENGINES.clear()
    gc.collect()


def extract_audio(ffmpeg: Path, source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ffmpeg),"-hide_banner","-loglevel","error","-y","-i",str(source),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(output)], check=True)
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("FFmpeg没有生成有效WAV")


def media_duration_ms(ffmpeg: Path, source: Path) -> int:
    ffprobe = ffmpeg.with_name("ffprobe.exe")
    if ffprobe.is_file():
        result = subprocess.run([str(ffprobe),"-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(source)], capture_output=True, text=True, check=True)
        return max(0, round(float(result.stdout.strip()) * 1000))
    return 0


def validate_sentences(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result=[]; previous_end=-1
    for expected,item in enumerate(sentences,1):
        raw = item.get("text", item.get("raw_text"))
        row={"id":int(item["id"]),"text":str(raw).strip(),"start":int(item["start"]),"end":int(item["end"])}
        approved = item.get("approved_text")
        if approved is not None and str(approved).strip():
            row["approved_text"] = str(approved).strip()
        if row["id"] != expected or not row["text"] or row["start"] < 0 or row["end"] <= row["start"] or row["start"] < previous_end:
            raise ValueError(f"句子JSON在第{expected}项不满足连续ID/非空正文/单调时间要求")
        result.append(row); previous_end=row["end"]
    if not result: raise ValueError("没有有效完整句")
    return result


def _robust(values: np.ndarray) -> np.ndarray:
    median=float(np.median(values)); mad=float(np.median(np.abs(values-median)))
    scale=1.4826*mad
    return (values-median)/(scale if scale>1e-6 else 1.0)


DISCOURSE = re.compile(r"^(接下来|下面|然后我们|第二|第三|最后|总结|回到|再来看|现在来看|下一部分)")


def deterministic_section_plan(sentences: list[dict[str,Any]], embedding: EmbeddingEngine, config: dict[str,Any]) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    rules=config["sectioning"]; n=len(sentences)
    if n==1:return [{"start_id":1,"end_id":1,"boundary_meta":{"method":"single"}}],[]
    vectors=embedding.encode([s["text"] for s in sentences]); window=max(2,int(rules.get("window_sentences",4)))
    semantic=np.zeros(n-1,dtype="float32"); gaps=np.zeros(n-1,dtype="float32"); discourse=np.zeros(n-1,dtype="float32")
    for i in range(n-1):
        left=vectors[max(0,i-window+1):i+1].mean(axis=0); right=vectors[i+1:min(n,i+1+window)].mean(axis=0)
        left/=max(np.linalg.norm(left),1e-9);right/=max(np.linalg.norm(right),1e-9)
        semantic[i]=1-float(np.dot(left,right));gaps[i]=math.log1p(max(0,sentences[i+1]["start"]-sentences[i]["end"]));discourse[i]=1 if DISCOURSE.search(sentences[i+1]["text"]) else 0
    if len(semantic)>=3: semantic=np.convolve(np.pad(semantic,(1,1),mode="edge"),np.ones(3)/3,mode="valid")
    sem=np.maximum(_robust(semantic),0); gap=np.maximum(_robust(gaps),0); disc=discourse
    score=float(rules.get("semantic_weight",.65))*sem+float(rules.get("silence_weight",.2))*gap+float(rules.get("discourse_weight",.15))*disc
    # Keep local peaks, but all sentence boundaries remain available to the duration optimizer.
    local=np.array([score[i] if (i==0 or score[i]>=score[i-1]) and (i==n-2 or score[i]>=score[i+1]) else 0 for i in range(n-1)])
    hard_min=int(rules.get("hard_min_ms",60000));target_min=int(rules["target_min_ms"]);ideal_min=int(rules["ideal_min_ms"]);ideal_max=int(rules["ideal_max_ms"]);soft_max=int(rules["target_max_ms"]);hard=int(rules["hard_max_ms"])
    dp=[-1e18]*(n+1);prev=[-1]*(n+1);dp[0]=0
    def duration_reward(ms:int)->float:
        if ideal_min<=ms<=ideal_max:return 1.0
        if ms<target_min:return -3.0*(target_min-ms)/target_min
        if ms<ideal_min:return (ms-target_min)/max(1,ideal_min-target_min)
        if ms<=soft_max:return (soft_max-ms)/max(1,soft_max-ideal_max)
        return -2.0*(ms-soft_max)/max(1,hard-soft_max)
    for end in range(1,n+1):
        for start in range(end-1,-1,-1):
            duration=sentences[end-1]["end"]-sentences[start]["start"]
            if duration>hard: break
            # A greeting, transition or closing sentence is not a navigable
            # parent chapter.  Keep short material inside its neighbouring
            # chapter; the whole video may of course be shorter than this.
            if duration<hard_min and not (start==0 and end==n):continue
            boundary=float(local[end-1]) if end<n else 0.0
            value=dp[start]+duration_reward(duration)+boundary
            if value>dp[end]:dp[end]=value;prev[end]=start
    if prev[n]<0: raise RuntimeError("无法在12分钟硬上限内生成完整句章节")
    ranges=[];end=n
    while end>0:start=prev[end];ranges.append((start,end));end=start
    ranges.reverse(); plan=[]; candidates=[]
    for pos,(start,end) in enumerate(ranges,1):
        boundary_id=sentences[end-1]["id"]
        meta={"method":"embedding_silence_dp","score":float(local[end-1]) if end<n else None,"semantic":float(semantic[end-1]) if end<n else None,"gap_ms":sentences[end]["start"]-sentences[end-1]["end"] if end<n else None,"qwen_reviewed":False}
        plan.append({"start_id":sentences[start]["id"],"end_id":boundary_id,"boundary_meta":meta})
        if end<n:candidates.append({"boundary_id":boundary_id,**meta})
    return plan,candidates


def _boundary_schema(boundary_id:int,radius:int)->dict[str,Any]:
    return {"type":"object","properties":{
        "boundary_id":{"type":"integer","const":boundary_id},
        "action":{"type":"string","enum":["accept","reject","move"]},
        "shift":{"type":"integer","minimum":-radius,"maximum":radius},
    },"required":["boundary_id","action","shift"],"additionalProperties":False}


def _label_schema(positions:list[int])->dict[str,Any]:
    return {"type":"object","properties":{"labels":{"type":"array","minItems":len(positions),"maxItems":len(positions),"items":{"type":"object","properties":{
        "position":{"type":"integer","enum":positions},
        "title":{"type":"string","maxLength":48},
        "summary":{"type":"string","maxLength":240},
        "keywords":{"type":"array","maxItems":8,"items":{"type":"string","maxLength":32}},
    },"required":["position","title","summary","keywords"],"additionalProperties":False}}},"required":["labels"],"additionalProperties":False}


def _replacement_schema(candidate_ids:list[int])->dict[str,Any]:
    return {"type":"object","properties":{"boundary_id":{"type":"integer","enum":[0,*candidate_ids]}},"required":["boundary_id"],"additionalProperties":False}


def _valid_reviewed_ends(ends:list[int],sentences:list[dict[str,Any]],hard_max_ms:int,hard_min_ms:int=0)->tuple[bool,str|None]:
    if ends!=sorted(set(ends)) or any(end<1 or end>=len(sentences) for end in ends):
        return False,"边界不再严格递增或超出句子范围"
    start=1
    for end in ends+[len(sentences)]:
        duration=sentences[end-1]["end"]-sentences[start-1]["start"]
        if duration>hard_max_ms:
            return False,f"句子{start}-{end}超过章节硬上限"
        if len(sentences)>1 and duration<hard_min_ms:
            return False,f"句子{start}-{end}短于章节硬下限"
        start=end+1
    return True,None


def review_and_label(plan:list[dict[str,Any]],sentences:list[dict[str,Any]],client:LocalLLMClient|None,config:dict[str,Any])->tuple[list[dict[str,Any]],list[str],dict[str,Any]]:
    by_id={s["id"]:s for s in sentences}; degraded=[];rules=config["sectioning"]
    review_audit={"strategy":"one_boundary_per_request_with_replacement_scan","boundaries":[],"replacement_searches":[],"labels":[]}
    if client and len(plan)>1:
        radius=int(rules.get("qwen_review_radius",3));originals=[s["end_id"] for s in plan[:-1]];targets={x:x for x in originals}
        attempts=int(rules.get("qwen_boundary_attempts",2));max_tokens=int(rules.get("qwen_boundary_max_tokens",192));hard=int(rules["hard_max_ms"]);hard_min=int(rules.get("hard_min_ms",60000))
        for index,section in enumerate(plan[:-1]):
            bid=section["end_id"];start=max(1,bid-radius+1);end=min(len(sentences),bid+radius)
            next_section=plan[index+1]
            left_ms=by_id[bid]["end"]-by_id[section["start_id"]]["start"]
            right_ms=by_id[next_section["end_id"]]["end"]-by_id[next_section["start_id"]]["start"]
            prompt=(
                f"只判断候选边界 boundary_id={bid}。左侧当前章节约{left_ms/1000:.1f}秒，右侧当前章节约{right_ms/1000:.1f}秒。"
                f"前一候选={originals[index-1] if index else '无'}，后一候选={originals[index+1] if index+1<len(originals) else '无'}。\n"
                "你必须比较四类方案：保留当前边界、向左移动、向右移动、删除并合并左右章节。"
                "边界应放在前一主题完整结束之后、后一主要主题开始之前。"
                "问候/致谢不能单独成章；问题及其回答不能拆开；举例、总结句、代词承接句、转折句、同一功能的继续说明不能被切开。"
                "引出新主题的总起句归入右章；收束旧主题的总结句归入左章。"
                "如果窗口内有更自然的转题句，必须move；如果两侧仍属同一主要主题且合并后不超过8分钟，应reject；"
                "只有当前位置确实是窗口内最强的主题转换点时才accept。"
                "accept表示保留；reject表示合并；move的shift表示新边界句ID减当前boundary_id。只输出JSON，不输出理由。\n\n"
                +"\n".join(f'[{i}] {by_id[i]["text"]}' for i in range(start,end+1))
            )
            attempts_audit=[];record={"boundary_id":bid,"context_start_id":start,"context_end_id":end,"left_duration_ms":left_ms,"right_duration_ms":right_ms,"attempts":attempts_audit}
            try:
                decision=client.structured(
                    [{"role":"system","content":"你是视频知识库的语义章节编辑。你的首要目标是主题完整，不得机械保留算法候选。只能选择提供的完整句边界。"},{"role":"user","content":prompt}],
                    "boundary_review",_boundary_schema(bid,radius),max_tokens,attempts=attempts,audit=attempts_audit,
                )
                action=decision["action"];shift=int(decision["shift"])
                if action in {"accept","reject"} and shift!=0:raise ValueError(f"{action}时shift必须为0")
                if action=="move" and shift==0:raise ValueError("move时shift不能为0")
                proposed=bid if action=="accept" else (None if action=="reject" else bid+shift)
                tentative=dict(targets);tentative[bid]=proposed
                ends=[tentative[x] for x in originals if tentative[x] is not None]
                valid,reason=_valid_reviewed_ends(ends,sentences,hard,hard_min)
                if not valid:raise ValueError(reason)
                targets=tentative;record.update({"success":True,"decision":decision,"applied":True})
            except Exception as error:
                record.update({"success":False,"applied":False,"error":str(error)})
            review_audit["boundaries"].append(record)
        replacement_attempts=int(rules.get("qwen_replacement_attempts",2));replacement_tokens=int(rules.get("qwen_replacement_max_tokens",192))
        for index,section in enumerate(plan[:-1]):
            bid=section["end_id"];record=next(x for x in review_audit["boundaries"] if x["boundary_id"]==bid)
            if not record.get("success") or record.get("decision",{}).get("action")!="reject":continue
            span_start=section["start_id"];span_end=plan[index+1]["end_id"];candidate_ids=[]
            for candidate in range(span_start,span_end):
                left_duration=by_id[candidate]["end"]-by_id[span_start]["start"]
                right_duration=by_id[span_end]["end"]-by_id[candidate+1]["start"]
                if left_duration>=hard_min and right_duration>=hard_min and candidate!=bid:candidate_ids.append(candidate)
            scan={"rejected_boundary_id":bid,"span_start_id":span_start,"span_end_id":span_end,"candidate_count":len(candidate_ids),"attempts":[]}
            if not candidate_ids:
                scan.update({"success":True,"selected_boundary_id":0,"inserted":False});review_audit["replacement_searches"].append(scan);continue
            prompt=(f"候选边界{bid}已判定不合适。重新检查句子{span_start}-{span_end}的完整合并章节。"
                    "如果其中仍有明确的主要主题转换，选择最自然的一个句子ID作为新边界；如果整段应保持一章，返回0。"
                    "问题与回答、同一操作步骤、总起句与其说明不得拆开。只输出JSON。\n\n"+
                    "\n".join(f'[{i}] {by_id[i]["text"]}' for i in range(span_start,span_end+1)))
            try:
                result=client.structured([{"role":"system","content":"你是视频知识库的章节边界搜索器，只能从允许的完整句边界中选择。"},{"role":"user","content":prompt}],"boundary_replacement",_replacement_schema(candidate_ids),replacement_tokens,attempts=replacement_attempts,audit=scan["attempts"])
                selected=int(result["boundary_id"]);tentative=dict(targets);tentative[bid]=selected or None
                ends=[tentative[x] for x in originals if tentative[x] is not None];valid,reason=_valid_reviewed_ends(ends,sentences,hard,hard_min)
                if not valid:raise ValueError(reason)
                targets=tentative;scan.update({"success":True,"selected_boundary_id":selected,"inserted":bool(selected)})
                if selected:record["replacement_boundary_id"]=selected
            except Exception as error:
                scan.update({"success":False,"selected_boundary_id":0,"inserted":False,"error":str(error)})
            review_audit["replacement_searches"].append(scan)
        failed=sum(not x["success"] for x in review_audit["boundaries"])
        if failed:degraded.append(f"Qwen边界复核部分失败：{failed}/{len(originals)}个候选使用确定性边界")
        replacement_failed=sum(not x["success"] for x in review_audit["replacement_searches"])
        if replacement_failed:degraded.append(f"Qwen替代边界搜索部分失败：{replacement_failed}/{len(review_audit['replacement_searches'])}个合并区间保留无边界结果")
        meta_by_end={s["end_id"]:dict(s["boundary_meta"]) for s in plan[:-1]};records={x["boundary_id"]:x for x in review_audit["boundaries"]}
        rebuilt=[];section_start=1
        for original in originals:
            target=targets[original]
            if target is None:continue
            record=records[original];meta=meta_by_end[original]
            meta.update({"qwen_reviewed":bool(record["success"]),"qwen_decision":record.get("decision"),"original_boundary_id":original})
            if target!=original:meta["method"]="qwen_bounded"
            rebuilt.append({"start_id":section_start,"end_id":target,"boundary_meta":meta});section_start=target+1
        rebuilt.append({"start_id":section_start,"end_id":len(sentences),"boundary_meta":dict(plan[-1]["boundary_meta"])})
        plan=rebuilt
    elif not client:
        review_audit["strategy"]="deterministic_only"
    excerpts=[]
    for pos,s in enumerate(plan,1):
        text="".join(by_id[i]["text"] for i in range(s["start_id"],s["end_id"]+1));excerpts.append(f"章节{pos}，句子{s['start_id']}-{s['end_id']}：\n{text[:1200]}")
    if client:
        labels={}
        # A single request containing every chapter can exceed the 4B CPU/GPU
        # generation timeout. Small independent batches preserve valid labels for
        # the other chapters if one batch fails.
        label_batch=max(1,int(rules.get("qwen_label_batch_size",2)))
        label_attempts=int(rules.get("qwen_label_attempts",2));label_max_tokens=int(rules.get("qwen_label_max_tokens",512))
        for offset in range(0,len(excerpts),label_batch):
            positions=list(range(offset+1,min(offset+label_batch,len(excerpts))+1));attempts_audit=[]
            try:
                result=client.structured([{"role":"system","content":"你是忠于原文的视频章节编辑。"},{"role":"user","content":"为固定边界生成具体中文标题、一句摘要和关键词；每个position必须恰好返回一次，不得添加原文没有的事实。\n\n"+"\n\n".join(excerpts[offset:offset+label_batch])}],"section_labels",_label_schema(positions),label_max_tokens,attempts=label_attempts,audit=attempts_audit)
                for item in result.get("labels",[]):
                    if item.get("position") in positions:labels[item["position"]]=item
                if set(positions)-set(labels):raise ValueError(f"缺少章节{sorted(set(positions)-set(labels))}")
                review_audit["labels"].append({"positions":positions,"success":True,"attempts":attempts_audit})
            except Exception as error:
                degraded.append(f"Qwen章节标签批次{offset//label_batch+1}失败：{error}")
                review_audit["labels"].append({"positions":positions,"success":False,"error":str(error),"attempts":attempts_audit})
    else:labels={};degraded.append("本地生成模型不可用，章节使用确定性标题")
    for pos,s in enumerate(plan,1):
        label=labels.get(pos,{});s["title"]=(label.get("title") or by_id[s["start_id"]]["text"][:36]).strip();s["summary"]=label.get("summary","").strip();s["keywords"]=label.get("keywords",[])
    review_audit["summary"]={"boundary_total":len(review_audit["boundaries"]),"boundary_success":sum(x.get("success",False) for x in review_audit["boundaries"]),"replacement_total":len(review_audit["replacement_searches"]),"replacement_success":sum(x.get("success",False) for x in review_audit["replacement_searches"]),"replacement_inserted":sum(x.get("inserted",False) for x in review_audit["replacement_searches"]),"label_batches":len(review_audit["labels"]),"label_batch_success":sum(x.get("success",False) for x in review_audit["labels"])}
    return plan,degraded,review_audit


def build_sections_chunks(sentences:list[dict[str,Any]],plan:list[dict[str,Any]],source_id:str,build_id:str,embedding:EmbeddingEngine,glossary:Glossary,config:dict[str,Any])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    by_id={s["id"]:s for s in sentences};sections=[];chunks=[];rules=config["chunking"];neighbor=int(rules.get("embedding_neighbor_sentences",1))
    effective={i:glossary.normalize(s["approved_text"] if s.get("approved_text") else s["text"]) for i,s in by_id.items()}
    for sp,p in enumerate(plan,1):
        group=[by_id[i] for i in range(p["start_id"],p["end_id"]+1)];sid=stable_id(source_id,build_id,"section",sp)
        section={"id":sid,"start_id":group[0]["id"],"end_id":group[-1]["id"],"start":group[0]["start"],"end":group[-1]["end"],"title":p["title"],"summary":p.get("summary",""),"keywords":p.get("keywords",[]),"boundary_meta":p.get("boundary_meta",{})};sections.append(section)
        groups=[];current=[]
        for item in group:
            candidate="".join(effective[x["id"]] for x in current+[item])
            if current and embedding.token_count(section["title"]+"\n"+candidate)>int(rules["hard_max_tokens"]):groups.append(current);current=[]
            current.append(item);count=embedding.token_count(section["title"]+"\n"+"".join(effective[x["id"]] for x in current))
            if count>=int(rules["target_max_tokens"]):groups.append(current);current=[]
        if current:
            if groups and embedding.token_count(section["title"]+"\n"+"".join(effective[x["id"]] for x in groups[-1]+current))<=int(rules["hard_max_tokens"]):groups[-1]+=current
            else:groups.append(current)
        for pos,items in enumerate(groups,1):
            core="".join(effective[x["id"]] for x in items);before=max(1,items[0]["id"]-neighbor);after=min(len(sentences),items[-1]["id"]+neighbor)
            embedding_text=section["title"]+"\n"+"".join(effective[i] for i in range(before,after+1));tokens=embedding.token_count(embedding_text)
            while tokens>embedding.max_tokens and (before<items[0]["id"] or after>items[-1]["id"]):
                if before<items[0]["id"]:before+=1
                elif after>items[-1]["id"]:after-=1
                embedding_text=section["title"]+"\n"+"".join(effective[i] for i in range(before,after+1));tokens=embedding.token_count(embedding_text)
            if tokens>embedding.max_tokens:raise RuntimeError(f"完整句Chunk超过Embedding {embedding.max_tokens} token硬上限：句子{items[0]['id']}-{items[-1]['id']}")
            chunks.append({"id":stable_id(source_id,build_id,"chunk",sid,pos),"section_id":sid,"position":pos,"start_id":items[0]["id"],"end_id":items[-1]["id"],"start":items[0]["start"],"end":items[-1]["end"],"search_text":core,"embedding_text":embedding_text,"token_count":tokens,"context_before_id":before if before<items[0]["id"] else None,"context_after_id":after if after>items[-1]["id"] else None})
    return sections,chunks


def write_vector_index(documents:list[dict[str,Any]],embedding:EmbeddingEngine,config:dict[str,Any])->dict[str,Any]:
    directory=index_dir(config);directory.mkdir(parents=True,exist_ok=True);version=uuid.uuid4().hex;dimension=int(config["models"]["embedding_dimension"]);index=faiss.IndexFlatIP(dimension)
    if documents:
        vectors=embedding.encode([d["embedding_text"] for d in documents])
        if vectors.shape[1]!=dimension:raise RuntimeError(f"Embedding维度{vectors.shape[1]}与配置{dimension}不符")
        index.add(vectors)
    index_name=f"faiss-{version}.index";mapping_name=f"chunks-{version}.json";index_tmp=directory/f"{index_name}.tmp";faiss.write_index(index,str(index_tmp));index_tmp.replace(directory/index_name)
    mapping=[{k:d[k] for k in ("id","source_id","build_id","section_id")} for d in documents];_write_json(directory/mapping_name,mapping)
    meta={"version":version,"index":index_name,"mapping":mapping_name,"count":len(mapping),"dimension":dimension,"chunk_ids_hash":json_hash([d["id"] for d in documents]),"embedding_model":str(embedding.path),"created_at":now_iso()}
    return meta


def write_index_manifest(config:dict[str,Any],meta:dict[str,Any])->None:_write_json(index_dir(config)/"manifest.json",meta)


def prune_index_versions(config:dict[str,Any],active_meta:dict[str,Any],keep:int=2)->list[str]:
    """Keep the active index and the newest previous complete index pair."""
    directory=index_dir(config).resolve();active=active_meta.get("version");versions:dict[str,list[Path]]={}
    for path in directory.iterdir():
        match=re.fullmatch(r"(?:faiss|chunks)-([0-9a-f]{32})\.(?:index|json)",path.name)
        if match and path.is_file():versions.setdefault(match.group(1),[]).append(path)
    ordered=sorted(versions,key=lambda version:max(p.stat().st_mtime for p in versions[version]),reverse=True)
    protected=[active] if active in versions else []
    previous=[version for version in ordered if version!=active and len(versions[version])==2][:max(0,keep-1)]
    protected.extend(previous)
    removed=[]
    for version,paths in versions.items():
        if version in protected:continue
        for path in paths:
            if path.resolve().parent != directory:raise RuntimeError("索引清理路径越界")
            path.unlink();removed.append(path.name)
    return removed


def reconcile_vector_index(db:Database,embedding:EmbeddingEngine,config:dict[str,Any],repair:bool=True)->dict[str,Any]:
    meta=db.meta("active_index");active=db.active_chunk_ids();valid=False;error=None
    if meta:
        try:
            directory=index_dir(config);mapping=json.loads((directory/meta["mapping"]).read_text(encoding="utf-8"));ids=[x["id"] if isinstance(x,dict) else x for x in mapping];index=faiss.read_index(str(directory/meta["index"]));valid=index.ntotal==len(ids)==len(active) and ids==active and meta.get("chunk_ids_hash")==json_hash(ids)
            if not valid:error="DB、mapping或FAISS数量/顺序不一致"
        except Exception as exc:error=str(exc)
    elif not active:valid=True
    if valid:return {"ok":True,"repaired":False,"meta":meta,"error":None}
    if not repair:return {"ok":False,"repaired":False,"meta":meta,"error":error or "缺少索引"}
    documents=db.active_index_documents();new=write_vector_index(documents,embedding,config);db.activate_index_only(new);write_index_manifest(config,new)
    prune_index_versions(config,new)
    return {"ok":True,"repaired":True,"meta":new,"error":error}


def write_inspection_files(directory:Path,bundle:dict[str,Any])->dict[str,Path]:
    source={**bundle["source"],"build_id":bundle["build"]["id"]};payloads={"source":source,"sentences":[{"id":s["id"],"raw_text":s["text"],"approved_text":s.get("approved_text"),"start":s["start"],"end":s["end"]} for s in bundle["sentences"]],"sections":bundle["sections"],"chunks":[{k:v for k,v in c.items() if k not in {"search_text","embedding_text"}} for c in bundle["chunks"]],"corrections":bundle.get("correction_candidates",[]),"build":bundle["build"]}
    if bundle.get("asr_raw") is not None:payloads["asr_raw"]=bundle["asr_raw"]
    if bundle.get("asr_preflight") is not None:payloads["asr_preflight"]=bundle["asr_preflight"]
    paths={}
    for key,value in payloads.items():paths[key]=directory/f"{key}.json";_write_json(paths[key],value)
    return paths


def export_expanded_sections(db:Database,source_id:str,output:Path)->Path:
    data=db.source_inspection(source_id)
    if not data:raise KeyError(source_id)
    expanded=[]
    for section in data["sections"]:
        full=db.section_text(section["id"],True);expanded.append({**section,"text":full["text"],"sentences":full["sentences"]})
    _write_json(output,expanded);return output


def _sentences_from_inspection(data:dict[str,Any])->list[dict[str,Any]]:
    rows=[]
    for item in data["sentences"]:
        row={"id":item["id"],"text":item["raw_text"],"start":item["start"],"end":item["end"]}
        if item.get("approved_text"):row["approved_text"]=item["approved_text"]
        rows.append(row)
    return validate_sentences(rows)


def _source_glossary(config:dict[str,Any],source:dict[str,Any])->Glossary:
    root=project_path(config,config["project"]["glossaries_dir"])
    return Glossary([root/"global.json",root/"categories"/f"{safe_name(source['category'])}.json",root/"sources"/f"{source['id']}.json"])


def _preserve_complete_parent_labels(plan:list[dict[str,Any]],old_sections:list[dict[str,Any]])->dict[str,Any]:
    """Keep proven labels when a section range is unchanged; only repair missing labels."""
    by_range={(int(x["start_id"]),int(x["end_id"])):x for x in old_sections}
    preserved=[];repaired=[];new_or_moved=[]
    for position,item in enumerate(plan,1):
        item_position=int(item.get("position",position))
        key=(int(item["start_id"]),int(item["end_id"]));old=by_range.get(key)
        if not old:
            new_or_moved.append(item_position);continue
        keywords=old.get("keywords",[])
        if isinstance(keywords,str):
            try:keywords=json.loads(keywords)
            except json.JSONDecodeError:keywords=[]
        if str(old.get("title") or "").strip() and str(old.get("summary") or "").strip() and keywords:
            item["title"]=old["title"];item["summary"]=old["summary"];item["keywords"]=keywords
            preserved.append(item_position)
        else:repaired.append(item_position)
    return {"preserved_parent_positions":preserved,"qwen_repaired_positions":repaired,"new_or_moved_positions":new_or_moved}


def preview_section_rebuild(config:dict[str,Any],db:Database,source_id:str,output:Path)->dict[str,Any]:
    data=db.source_inspection(source_id)
    if not data:raise KeyError(source_id)
    source=data["source"];sentences=_sentences_from_inspection(data)
    embedding=get_embedding_engine(project_path(config,config["models"]["embedding"]),int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu")))
    started=time.perf_counter();plan,boundaries=deterministic_section_plan(sentences,embedding,config);deterministic_ms=round((time.perf_counter()-started)*1000)
    client=LocalLLMClient(config);health=client.health()
    if not health.get("available"):raise RuntimeError(health.get("error","本地生成模型不可用"))
    started=time.perf_counter();plan,degradation,audit=review_and_label(plan,sentences,client,config);qwen_ms=round((time.perf_counter()-started)*1000)
    old_sections=[]
    for item in data["sections"]:
        old_sections.append({"position":item["position"],"start_id":item["start_sentence_id"],"end_id":item["end_sentence_id"],"start":item["start_ms"],"end":item["end_ms"],"title":item["title"],"summary":item["summary"],"keywords":item.get("keywords_json",[])})
    label_preservation=_preserve_complete_parent_labels(plan,old_sections)
    old_ends=[x["end_id"] for x in old_sections[:-1]];new_ends=[x["end_id"] for x in plan[:-1]];changed=sorted(set(old_ends)^set(new_ends))
    contexts=[]
    for boundary_id in changed:
        begin=max(1,boundary_id-3);finish=min(len(sentences),boundary_id+4)
        contexts.append({"boundary_id":boundary_id,"was_boundary":boundary_id in old_ends,"is_boundary":boundary_id in new_ends,"sentences":[{"id":i,"text":sentences[i-1].get("approved_text") or sentences[i-1]["text"],"start":sentences[i-1]["start"],"end":sentences[i-1]["end"]} for i in range(begin,finish+1)]})
    report={
        "schema_version":1,"source_id":source_id,"source_title":source["title"],"parent_build_id":source["current_build_id"],"created_at":now_iso(),
        "method":"embedding_silence_dp+qwen_per_boundary","old_sections":old_sections,"candidate_plan":plan,"deterministic_candidates":boundaries,
        "changed_boundary_contexts":contexts,"degraded":bool(degradation),"degradation":degradation,"boundary_review":audit,"label_preservation":label_preservation,
        "timings_ms":{"deterministic_embedding_dp":deterministic_ms,"qwen_review_and_labels":qwen_ms,"total":deterministic_ms+qwen_ms},
    }
    _write_json(output.resolve(),report);return {**{k:v for k,v in report.items() if k not in {"old_sections","candidate_plan","deterministic_candidates","changed_boundary_contexts","boundary_review"}},"output":str(output.resolve()),"old_sections":len(old_sections),"candidate_sections":len(plan),"changed_boundaries":len(changed),"boundary_review_summary":audit["summary"]}


def curate_section_preview(db:Database,base_preview:Path,ends:list[int],output:Path,label_overrides:dict[str,Any]|None=None)->dict[str,Any]:
    """Create an auditable expert-reviewed candidate without mutating the KB."""
    base=json.loads(base_preview.resolve().read_text(encoding="utf-8"));source_id=str(base["source_id"]);data=db.source_inspection(source_id)
    if not data:raise KeyError(source_id)
    if data["source"].get("current_build_id")!=base.get("parent_build_id"):raise RuntimeError("基础候选对应的Build已经过期")
    sentences=_sentences_from_inspection(data);ends=[int(x) for x in ends]
    valid,reason=_valid_reviewed_ends(ends,sentences,720000,60000)
    if not valid:raise ValueError(reason)
    ranges=[];start=1
    for end in ends+[len(sentences)]:ranges.append((start,end));start=end+1
    old_plan=base.get("candidate_plan",[]);overrides=label_overrides or {};plan=[]
    for position,(start,end) in enumerate(ranges,1):
        # Carry the label from the candidate range with the greatest sentence
        # overlap. This remains stable when expert review inserts, removes or
        # shifts boundaries; split topics can be replaced explicitly below.
        best=max(old_plan,key=lambda item:max(0,min(end,int(item["end_id"]))-max(start,int(item["start_id"]))+1),default={})
        label=dict(best)
        label.update(overrides.get(str(position),overrides.get(position,{})))
        title=str(label.get("title") or "").strip();summary=str(label.get("summary") or "").strip();keywords=label.get("keywords") or []
        if not title or not summary or not keywords:raise ValueError(f"专家候选第{position}章缺少完整标题、摘要或关键词")
        plan.append({"start_id":start,"end_id":end,"title":title,"summary":summary,"keywords":keywords,"boundary_meta":{"method":"expert_semantic_review","base_preview":base_preview.name}})
    report={
        "schema_version":1,"source_id":source_id,"source_title":base.get("source_title",data["source"]["title"]),"parent_build_id":base["parent_build_id"],"created_at":now_iso(),
        "method":str(base.get("method","embedding_silence_dp+qwen_per_boundary"))+"+expert_semantic_review","degraded":False,"degradation":[],"candidate_plan":plan,
        "timings_ms":{**base.get("timings_ms",{}),"expert_review_runtime_ms":None},"boundary_review":base.get("boundary_review",{}),
        "expert_review":{"based_on":str(base_preview.resolve()),"internal_boundary_sentence_ids":ends,"label_override_positions":sorted(int(x) for x in overrides)},
    }
    _write_json(output.resolve(),report)
    return {"output":str(output.resolve()),"source_id":source_id,"parent_build_id":base["parent_build_id"],"sections":len(plan),"ends":ends,"degraded":False}


def activate_section_preview(config:dict[str,Any],db:Database,preview_path:Path)->dict[str,Any]:
    preview=json.loads(preview_path.resolve().read_text(encoding="utf-8"));source_id=str(preview["source_id"]);source=db.source(source_id)
    if not source or source.get("current_build_id")!=preview.get("parent_build_id"):raise RuntimeError("当前Build已经变化，拒绝激活过期的边界候选")
    data=db.source_inspection(source_id);sentences=_sentences_from_inspection(data);plan=preview["candidate_plan"]
    old_sections=[{"position":item["position"],"start_id":item["start_sentence_id"],"end_id":item["end_sentence_id"],"title":item["title"],"summary":item["summary"],"keywords":item.get("keywords_json",[])} for item in data["sections"]]
    label_preservation=_preserve_complete_parent_labels(plan,old_sections);build_id=uuid.uuid4().hex
    embedding=get_embedding_engine(project_path(config,config["models"]["embedding"]),int(config["models"]["embedding_max_tokens"]),str(config["models"].get("embedding_device","cpu")))
    glossary=_source_glossary(config,source);sections,chunks=build_sections_chunks(sentences,plan,source_id,build_id,embedding,glossary,config)
    parent=next(x for x in db.list_builds(source_id) if x["id"]==source["current_build_id"])
    manifest_path=project_path(config,config["project"]["models_dir"])/"manifest.json";config_for_hash={k:v for k,v in config.items() if not k.startswith("_")}
    build={"id":build_id,"parent_build_id":source["current_build_id"],"started_at":now_iso(),"config_hash":json_hash(config_for_hash),"model_manifest_hash":sha256_file(manifest_path) if manifest_path.is_file() else "missing","glossary_version":glossary.version(),"boundary_method":preview["method"],"asr_provenance":parent.get("asr_provenance",{}),"asr_preflight":parent.get("asr_preflight",{}),"degraded":bool(preview.get("degradation")),"degradation":preview.get("degradation",[]),"boundary_candidates":preview.get("deterministic_candidates",[]),"boundary_review":preview.get("boundary_review",{}),"expert_review":preview.get("expert_review"),"label_preservation":label_preservation,"section_rebuild_timings_ms":preview.get("timings_ms",{})}
    bundle={"source":{"id":source_id,"video_hash":source["video_hash"],"category":source["category"],"title":source["title"],"video_path":source["video_path"],"media_duration_ms":source["media_duration_ms"]},"build":build,"sentences":sentences,"sections":sections,"chunks":chunks,"correction_candidates":[]}
    parent_dir=project_path(config,config["project"]["data_dir"])/"builds"/source_id/source["current_build_id"]
    for key in ("asr_raw","asr_preflight"):
        path=parent_dir/f"{key}.json"
        if path.is_file():bundle[key]=json.loads(path.read_text(encoding="utf-8"))
    build_dir=project_path(config,config["project"]["data_dir"])/"builds"/source_id/build_id;write_inspection_files(build_dir,bundle);shutil.copy2(preview_path.resolve(),build_dir/"boundary_rebuild.json")
    timings={};started=time.perf_counter();staged=False
    lock_path=project_path(config,config["project"]["data_dir"])/"runtime"/"build.lock"
    with FileLock(lock_path):
        try:
            db.stage_bundle(bundle);staged=True;timings["database_ms"]=round((time.perf_counter()-started)*1000);stage=time.perf_counter()
            documents=db.index_documents_for_activation(source_id,build_id);index_meta=write_vector_index(documents,embedding,config);timings["faiss_ms"]=round((time.perf_counter()-stage)*1000);stage=time.perf_counter()
            db.activate_build(bundle,index_meta);write_index_manifest(config,index_meta);prune_index_versions(config,index_meta);timings["activate_ms"]=round((time.perf_counter()-stage)*1000)
        except Exception as error:
            if staged:db.fail_build(build_id,str(error))
            raise
    return {"source_id":source_id,"parent_build_id":preview["parent_build_id"],"build_id":build_id,"sections":len(sections),"chunks":len(chunks),"degraded":bool(preview.get("degradation")),"degradation":preview.get("degradation",[]),"label_preservation":label_preservation,"review_timings_ms":preview.get("timings_ms",{}),"activation_timings_ms":timings,"index":index_meta,"build_dir":str(build_dir.resolve())}


class Ingestor:
    def __init__(self,config:dict[str,Any],db:Database):self.config=config;self.db=db

    def ingest(self,input_path:Path,category:str="未分类",title:str|None=None,sentences_json:Path|None=None,use_llm:bool=True,progress:Callable[[str,int,str,list[dict[str,str]]],None]|None=None,cancelled:Callable[[],bool]|None=None,confirm_preflight:Callable[[dict[str,Any]],bool]|None=None,preflight_report:dict[str,Any]|None=None)->dict[str,Any]:
        lock_path=project_path(self.config,self.config["project"]["data_dir"])/"runtime"/"build.lock"
        with FileLock(lock_path):
            return self._ingest_locked(input_path,category,title,sentences_json,use_llm,progress,cancelled,confirm_preflight,preflight_report)

    def _release_qa_resources(self)->dict[str,Any]:
        """Free QA models before CPU/RAM-heavy ASR and remember what to restore."""
        from .search import unload_rerankers
        client=LocalLLMClient(self.config);loaded=client.loaded_models();was_loaded=any(x.get("identifier")==client.model for x in loaded)
        state={"llm_was_loaded":was_loaded}
        try:
            unload_embedding_engines();unload_rerankers()
            if was_loaded:
                client.unload_model();deadline=time.time()+60
                while time.time()<deadline and any(x.get("identifier")==client.model for x in client.loaded_models()):time.sleep(.5)
                if any(x.get("identifier")==client.model for x in client.loaded_models()):raise TimeoutError("llama-server问答模型在60秒内未卸载，拒绝与ASR同时占用内存")
            gc.collect();return state
        except Exception:
            self._restore_qa_resources(state)
            raise

    def _restore_qa_resources(self,state:dict[str,Any])->None:
        """Warm every QA model before an ingestion may be reported complete."""
        from .search import get_reranker_engine
        embedding=get_embedding_engine(project_path(self.config,self.config["models"]["embedding"]),int(self.config["models"]["embedding_max_tokens"]),str(self.config["models"].get("embedding_device","cpu")))
        get_reranker_engine(project_path(self.config,self.config["models"]["reranker"]),str(self.config["models"].get("reranker_device","cpu")))
        if state.get("llm_was_loaded"):
            client=LocalLLMClient(self.config)
            if not any(x.get("identifier")==client.model for x in client.loaded_models()):client.load_model()
        _=embedding

    def _ingest_locked(self,input_path:Path,category:str="未分类",title:str|None=None,sentences_json:Path|None=None,use_llm:bool=True,progress:Callable[[str,int,str,list[dict[str,str]]],None]|None=None,cancelled:Callable[[],bool]|None=None,confirm_preflight:Callable[[dict[str,Any]],bool]|None=None,preflight_report:dict[str,Any]|None=None)->dict[str,Any]:
        storage=[];staged=False;bundle=None
        def add(key,label,path,description):storage.append({"key":key,"label":label,"path":str(Path(path).resolve()),"description":description})
        def emit(stage,percent,message):
            if cancelled and cancelled():raise InterruptedError("用户安全取消入库")
            if progress:progress(stage,percent,message,[dict(x) for x in storage])
        emit("validate",2,"检查输入和配置");input_path=input_path.resolve()
        if not input_path.is_file():raise FileNotFoundError(input_path)
        category=(category or "未分类").strip() or "未分类";video_hash=sha256_file(input_path);source_id=video_hash;source_title=(title or input_path.stem).strip();build_id=uuid.uuid4().hex;existing_source=self.db.source(source_id);parent_build_id=existing_source.get("current_build_id") if existing_source else None
        data_dir=project_path(self.config,self.config["project"]["data_dir"]);build_dir=data_dir/"builds"/source_id/build_id;build_dir.mkdir(parents=True,exist_ok=True);sources=project_path(self.config,self.config["project"]["sources_dir"]);source_dir=sources/f"{safe_name(source_title)}-{source_id[:8]}";source_dir.mkdir(parents=True,exist_ok=True);managed=source_dir/input_path.name
        emit("copy_video",7,"保存受管视频")
        if managed!=input_path and not managed.exists():shutil.copy2(input_path,managed)
        add("managed_video","受管视频",managed,"正式播放和离线重建来源")
        duration=media_duration_ms(project_path(self.config,self.config["project"]["ffmpeg"]),managed)
        glossary_root=project_path(self.config,self.config["project"]["glossaries_dir"]);glossary_paths=[glossary_root/"global.json",glossary_root/"categories"/f"{safe_name(category)}.json",glossary_root/"sources"/f"{source_id}.json"]
        for path in glossary_paths:
            if not path.exists():_write_json(path,{})
        glossary=Glossary(glossary_paths)
        for pos,path in enumerate(glossary_paths):add(f"glossary_{pos}","受控术语表",path,"全局/分类/来源术语映射")
        cache=data_dir/"cache"/source_id;cache.mkdir(parents=True,exist_ok=True);audio=cache/"audio.wav"
        asr_result:ASRResult|None=None;preflight:dict[str,Any]|None=dict(preflight_report) if preflight_report else None
        if sentences_json:
            emit("load_sentences",18,"读取已有完整句JSON");sentences=validate_sentences(json.loads(sentences_json.read_text(encoding="utf-8")));add("sentences_input","外部句子JSON",sentences_json,"跳过ASR的输入")
            asr_provenance={"requested_provider":"external_sentences_json","actual_provider":"external_sentences_json","timestamp_method":"provided_and_validated_ms","input":str(sentences_json.resolve()),"parent_build_id":parent_build_id}
        else:
            if managed.suffix.lower()==".wav":audio=managed
            elif not audio.exists():emit("extract_audio",12,"FFmpeg提取16kHz单声道音频");extract_audio(project_path(self.config,self.config["project"]["ffmpeg"]),managed,audio)
            add("audio","ASR音频",audio,"可再生的ASR工作缓存")
            provider=get_asr_provider(self.config);emit("asr_precheck",16,f"检查{provider.name}本地模型和BF16能力")
            provider_health=provider.doctor()
            if not provider_health.get("available"):raise RuntimeError(f"ASR Provider预检失败：{provider_health}")
            qa_state={}
            if bool(self.config.get("asr",{}).get("resources",{}).get("unload_qa_during_asr",True)):
                emit("release_qa_models",18,"暂停问答并卸载BGE、Reranker和llama-server模型")
                qa_state=self._release_qa_resources()
            request=ASRRequest(
                audio=audio,
                title=source_title,
                category=category,
                hotwords=list(glossary.mapping),
                work_dir=data_dir/"cache"/source_id/"asr-work"/build_id,
                checkpoint_path=data_dir/"cache"/source_id/"qwen3-asr-transcript-checkpoint.json",
                ffmpeg=project_path(self.config,self.config["project"]["ffmpeg"]),
                cancelled=cancelled,
            )
            add(
                "asr_checkpoint",
                "Qwen原始转写检查点",
                request.checkpoint_path,
                "完整Qwen转写后、ForcedAligner前原子保存；相同音频/模型/提示重试时复用",
            )
            try:
                if provider.name=="qwen3_asr" and bool(self.config.get("asr",{}).get("preflight",{}).get("enabled",True)):
                    if preflight is not None:
                        _write_json(build_dir/"asr_preflight.json",preflight);add("asr_preflight","ASR预检报告",build_dir/"asr_preflight.json","从同一失败任务复用的已确认预检报告")
                        emit("asr_preflight_reused",24,"复用同一任务已确认的ASR预检与Qwen转写检查点")
                    else:
                        emit("asr_preflight",20,"用开头、中段、结尾各60秒实测Qwen3-ASR与ForcedAligner")
                        preflight=provider.preflight(request);_write_json(build_dir/"asr_preflight.json",preflight);add("asr_preflight","ASR预检报告",build_dir/"asr_preflight.json","三段实测耗时、预计全片耗时与内存峰值")
                        emit("awaiting_confirmation",24,f"预检完成：预计全片约{preflight['estimated_full_seconds']:.0f}秒，等待确认继续")
                        if confirm_preflight is None or not confirm_preflight(preflight):raise InterruptedError("ASR预检后未确认继续，当前知识库未切换")
                emit("asr",28,f"运行本地{provider.name}并生成真实句级毫秒时间戳")
                asr_result=provider.transcribe(request);sentences=validate_sentences(asr_result.sentences);asr_provenance=asr_result.provenance()
            finally:
                work=request.work_dir.resolve();cache_root=(data_dir/"cache"/source_id/"asr-work").resolve()
                try:
                    if work.exists() and cache_root in work.parents:shutil.rmtree(work)
                except OSError:
                    pass
                if qa_state:
                    emit("restore_qa_models",45,"卸载ASR并重新预热BGE、Reranker和问答Qwen")
                    self._restore_qa_resources(qa_state)
        # Approved glossary entries create a corrected view while raw_text remains
        # immutable and inspectable in every Build.
        for sentence in sentences:
            corrected = glossary.correct_text(sentence.get("approved_text") or sentence["text"])
            if corrected != sentence["text"]:
                sentence["approved_text"] = corrected
        emit("embedding_model",48,"加载BGE v1.5并计算句子语义");embedding=get_embedding_engine(project_path(self.config,self.config["models"]["embedding"]),int(self.config["models"]["embedding_max_tokens"]),str(self.config["models"].get("embedding_device","cpu")))
        plan,boundaries=deterministic_section_plan(sentences,embedding,self.config);client=LocalLLMClient(self.config) if use_llm else None
        if client and not client.health()["available"]:client=None
        emit("semantic_sections",62,"Qwen逐个复核确定性候选并生成标题" if client else "使用确定性边界和回退标题");plan,degradation,boundary_review=review_and_label(plan,sentences,client,self.config)
        emit("chunks",73,"按完整句构建Chunk并校验512-token硬上限");sections,chunks=build_sections_chunks(sentences,plan,source_id,build_id,embedding,glossary,self.config);candidates=glossary.candidates(sentences,source_id,build_id)
        manifest_path=project_path(self.config,self.config["project"]["models_dir"])/"manifest.json";model_manifest_hash=sha256_file(manifest_path) if manifest_path.is_file() else "missing";config_for_hash={k:v for k,v in self.config.items() if not k.startswith("_")}
        bundle={"source":{"id":source_id,"video_hash":video_hash,"category":category,"title":source_title,"video_path":portable_asset_path(self.config,managed),"media_duration_ms":duration or sentences[-1]["end"]},"build":{"id":build_id,"parent_build_id":parent_build_id,"started_at":now_iso(),"config_hash":json_hash(config_for_hash),"model_manifest_hash":model_manifest_hash,"glossary_version":glossary.version(),"boundary_method":"embedding_silence_dp+qwen_per_boundary","asr_provenance":asr_provenance,"asr_preflight":{k:v for k,v in (preflight or {}).items() if k!="raw"},"degraded":bool(degradation),"degradation":degradation,"boundary_candidates":boundaries,"boundary_review":boundary_review},"sentences":sentences,"sections":sections,"chunks":chunks,"correction_candidates":candidates,"asr_raw":asr_result.raw if asr_result else None,"asr_preflight":preflight}
        paths=write_inspection_files(build_dir,bundle)
        for key,path in paths.items():add(key,f"{key}.json",path,"当前Build人工检查材料")
        try:
            emit("database",82,"暂存不可见Build");self.db.stage_bundle(bundle);staged=True
            emit("faiss",90,"生成完整新索引，旧Build继续可查询");documents=self.db.index_documents_for_activation(source_id,build_id);index_meta=write_vector_index(documents,embedding,self.config)
            emit("activate",97,"原子切换current_build_id和索引版本");self.db.activate_build(bundle,index_meta)
            # Activation is the commit point. Manifest writing and old-version GC
            # are maintenance only; a failure here must never misreport a ready,
            # queryable Build as an ingestion failure.
            try:
                write_index_manifest(self.config,index_meta);prune_index_versions(self.config,index_meta)
            except Exception as maintenance_error:
                degradation.append(f"激活后索引维护警告：{maintenance_error}")
            add("database","SQLite主数据",self.db.path,"知识库唯一正式主副本");add("faiss","当前FAISS索引",index_dir(self.config)/index_meta["index"],"由数据库重建的向量索引")
        except Exception as error:
            if staged:self.db.fail_build(build_id,str(error),"cancelled" if isinstance(error,InterruptedError) else "failed")
            raise
        emit("completed",100,"入库Build已完整切换，可以提问")
        return {"source_id":source_id,"build_id":build_id,"category":category,"title":source_title,"sentences":len(sentences),"sections":len(sections),"chunks":len(chunks),"asr_provider":asr_provenance.get("actual_provider"),"degraded":degradation,"index":index_meta,"storage":storage}
