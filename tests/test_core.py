from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from local_rag.db import Database, lexical_tokens
from local_rag.corrections import preview_sentence_edits
from local_rag.asr import ASRRequest, Qwen3ASRProvider, alignment_safe_transcript, extract_funasr_segments, normalize_overlong_sentence_boundaries, timestamps_to_sentences
from local_rag.lmstudio import _parse_json_object, _validate_json
from local_rag.pipeline import Glossary, deterministic_section_plan, prune_index_versions, review_and_label, stable_id, validate_sentences, write_inspection_files
from local_rag.search import aggregate_section_score, normalize_retrieval_query


class FakeEmbedding:
    max_tokens=512
    path=Path("fake")
    def encode(self,texts):
        values=[]
        for i,_ in enumerate(texts):
            angle=(i//4)*0.8
            values.append([np.cos(angle),np.sin(angle)])
        return np.asarray(values,dtype="float32")
    def token_count(self,text):return len(text)


def sample_bundle(build_id="b1"):
    sentences=[{"id":i,"text":f"第{i}句。","start":i*60000-60000,"end":i*60000-100} for i in range(1,9)]
    return {
        "source":{"id":"hash","video_hash":"hash","category":"测试","title":"标题","video_path":"x.mp4","media_duration_ms":480000},
        "build":{"id":build_id,"config_hash":"c","model_manifest_hash":"m","glossary_version":"g","boundary_method":"test","started_at":"now"},
        "sentences":sentences,
        "sections":[{"id":f"sec-{build_id}","title":"主题","summary":"","keywords":[],"start_id":1,"end_id":8,"start":0,"end":479900,"boundary_meta":{"method":"test"}}],
        "chunks":[{"id":f"chunk-{build_id}","section_id":f"sec-{build_id}","position":1,"start_id":1,"end_id":8,"start":0,"end":479900,"token_count":30,"search_text":"正文","embedding_text":"主题\n正文"}],
        "correction_candidates":[],
    }


class CoreTests(unittest.TestCase):
    def test_ids_and_json_schema(self):
        self.assertEqual(stable_id("a",1),stable_id("a",1));self.assertNotEqual(stable_id("a",1),stable_id("a",2))
        schema={"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":False}
        value=_parse_json_object("```json\n{\"ok\":true}\n```");_validate_json(value,schema)
        with self.assertRaises(ValueError):_validate_json({"ok":True,"extra":1},schema)
        constrained={"type":"object","properties":{"action":{"type":"string","enum":["accept"]},"shift":{"type":"integer","minimum":-1,"maximum":1}},"required":["action","shift"],"additionalProperties":False}
        _validate_json({"action":"accept","shift":0},constrained)
        with self.assertRaises(ValueError):_validate_json({"action":"reject","shift":0},constrained)

    def test_sentence_validation(self):
        value=validate_sentences([{"id":1,"text":"完整句。","start":0,"end":1000}]);self.assertEqual(value[0]["text"],"完整句。")
        with self.assertRaises(ValueError):validate_sentences([{"id":2,"text":"错ID。","start":0,"end":1000}])

    def test_asr_adapters_never_invent_timestamps(self):
        rows=extract_funasr_segments([{"sentence_info":[{"text":"第一句。","start":10,"end":600}]}])
        self.assertEqual(rows,[{"id":1,"text":"第一句。","start":10,"end":600}])
        with self.assertRaises(ValueError):extract_funasr_segments([{"text":"没有句级时间"}])
        aligned=timestamps_to_sentences("你好。TCP/IP是什么？",[
            {"text":"你","start_time":.1,"end_time":.2},{"text":"好","start_time":.2,"end_time":.4},
            {"text":"TCP/IP","start_time":.6,"end_time":1.0},{"text":"是","start_time":1.0,"end_time":1.1},
            {"text":"什","start_time":1.1,"end_time":1.2},{"text":"么","start_time":1.2,"end_time":1.4},
        ],1000)
        self.assertEqual([x["text"] for x in aligned],["你好。","TCP/IP是什么？"])
        self.assertEqual((aligned[0]["start"],aligned[-1]["end"]),(1100,2400))
        adjustments=[]
        point=timestamps_to_sentences("嗯。",[{"text":"嗯","start_time":2.5,"end_time":2.5}],adjustments=adjustments)
        self.assertEqual((point[0]["start"],point[0]["end"]),(2500,2501))
        self.assertEqual(adjustments[0]["reason"],"zero_duration_after_ms_rounding")
        repeated=timestamps_to_sentences("你好。。下一句？！",[
            {"text":"你","start_time":0,"end_time":.1},{"text":"好","start_time":.1,"end_time":.2},
            {"text":"下","start_time":.3,"end_time":.4},{"text":"一","start_time":.4,"end_time":.5},
            {"text":"句","start_time":.5,"end_time":.6},
        ])
        self.assertEqual([x["text"] for x in repeated],["你好。。","下一句？！"])
        # Official Qwen long-audio mode concatenates chunk texts first.  A
        # sentence may therefore begin in one 180s chunk and end in the next.
        cross_chunk_text="第一句跨块"+"继续到这里。第二句。"
        cross_chunk=timestamps_to_sentences(cross_chunk_text,[
            {"text":"第一句跨块","start_time":179.0,"end_time":180.0},
            {"text":"继续到这里","start_time":180.1,"end_time":181.0},
            {"text":"第二句","start_time":181.2,"end_time":182.0},
        ])
        self.assertEqual([x["text"] for x in cross_chunk],["第一句跨块继续到这里。","第二句。"])
        self.assertEqual((cross_chunk[0]["start"],cross_chunk[0]["end"]),(179000,181000))
        self.assertEqual(alignment_safe_transcript("token。F F N一！"),"token。 F F N一！ ")
        latin_boundary=timestamps_to_sentences("token。F F N一！",[
            {"text":"token","start_time":0,"end_time":.5},
            {"text":"F","start_time":.6,"end_time":.7},{"text":"F","start_time":.7,"end_time":.8},
            {"text":"N","start_time":.8,"end_time":.9},{"text":"一","start_time":.9,"end_time":1.0},
        ])
        self.assertEqual([x["text"] for x in latin_boundary],["token。","F F N一！"])
        with self.assertRaises(ValueError):timestamps_to_sentences("没有句号",[{"text":"没有句号","start_time":0,"end_time":1}])
        with self.assertRaises(ValueError):timestamps_to_sentences("。。",[{"text":"。","start_time":0,"end_time":1}])

    def test_qwen_transcript_checkpoint_is_strictly_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/"model";aligner=root/"aligner";model.mkdir();aligner.mkdir()
            (model/"model.safetensors").write_bytes(b"model")
            audio=root/"audio.wav"
            with wave.open(str(audio),"wb") as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b"\0\0"*16000)
            config={"_root":str(root),"asr":{"providers":{"qwen3_asr":{"model":"model","aligner":"aligner","max_new_tokens":512}}}}
            provider=Qwen3ASRProvider(config);checkpoint=root/"checkpoint.json";clip_file=root/"chunk.wav";clip_file.write_bytes(b"wav")
            request=ASRRequest(audio=audio,title="标题",category="分类",hotwords=[],work_dir=root,ffmpeg=root/"ffmpeg",checkpoint_path=checkpoint)
            clips=[{"path":clip_file,"start_ms":0,"end_ms":1000}]
            segments=[{**clips[0],"language":"Chinese","raw_transcription":"跨块句子。","generated_tokens":8}]
            prompt=provider._prompt(request);provider._save_checkpoint(request,clips,prompt,segments)
            loaded=provider._load_checkpoint(request,clips,prompt)
            self.assertEqual(loaded[0]["raw_transcription"],"跨块句子。")
            self.assertIsNone(provider._load_checkpoint(request,[{**clips[0],"end_ms":999}],prompt))

    def test_overlong_qwen_sentence_uses_only_existing_clause_boundaries(self):
        original="首先介绍功能，"+"这是没有句号的长说明，"*20+"最后完成。"
        normalized,changed=normalize_overlong_sentence_boundaries(original,64)
        self.assertTrue(changed);self.assertGreater(normalized.count("。"),1)
        lexical=lambda value:"".join(x for x in value if x.isalnum() or "\u3400"<=x<="\u9fff")
        self.assertEqual(lexical(normalized),lexical(original))
        self.assertTrue(all(len(lexical(piece))<=96 for piece in normalized.split("。") if piece))

    def test_glossary_longest_and_ascii_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"global.json";path.write_text('{"TCP/IP":["Pcpip"],"MoE":["ME"]}',encoding="utf-8")
            glossary=Glossary([path]);self.assertIn("TCP/IP",glossary.normalize("Pcpip协议"));self.assertNotIn("MoE",glossary.normalize("SOMETHING"))

    def test_deterministic_plan_covers_all_sentences(self):
        sentences=[{"id":i,"text":f"第{i}句。","start":i*60000-60000,"end":i*60000-100} for i in range(1,21)]
        config={"sectioning":{"window_sentences":3,"semantic_weight":.65,"silence_weight":.2,"discourse_weight":.15,"target_min_ms":120000,"ideal_min_ms":180000,"ideal_max_ms":360000,"target_max_ms":480000,"hard_max_ms":720000}}
        plan,_=deterministic_section_plan(sentences,FakeEmbedding(),config);owned=[i for section in plan for i in range(section["start_id"],section["end_id"]+1)];self.assertEqual(owned,list(range(1,21)));self.assertTrue(all(sentences[x["end_id"]-1]["end"]-sentences[x["start_id"]-1]["start"]<=720000 for x in plan))

    def test_boundary_requests_are_isolated_and_partial_failure_falls_back(self):
        sentences=[{"id":i,"text":f"第{i}句。","start":i*60000-60000,"end":i*60000-100} for i in range(1,7)]
        plan=[
            {"start_id":1,"end_id":2,"boundary_meta":{"method":"embedding_silence_dp","qwen_reviewed":False}},
            {"start_id":3,"end_id":4,"boundary_meta":{"method":"embedding_silence_dp","qwen_reviewed":False}},
            {"start_id":5,"end_id":6,"boundary_meta":{"method":"embedding_silence_dp","qwen_reviewed":False}},
        ]
        class Client:
            def structured(self,messages,name,schema,max_tokens,**kwargs):
                if name=="boundary_review":
                    boundary_id=schema["properties"]["boundary_id"]["const"]
                    if boundary_id==4:raise RuntimeError("simulated malformed JSON")
                    return {"boundary_id":boundary_id,"action":"reject","shift":0}
                if name=="boundary_replacement":return {"boundary_id":0}
                positions=schema["properties"]["labels"]["items"]["properties"]["position"]["enum"]
                return {"labels":[{"position":p,"title":f"章节{p}","summary":"摘要","keywords":["词"]} for p in positions]}
        config={"sectioning":{"qwen_review_radius":3,"qwen_boundary_attempts":2,"qwen_boundary_max_tokens":128,"qwen_label_batch_size":2,"qwen_label_attempts":2,"qwen_label_max_tokens":512,"hard_max_ms":720000}}
        reviewed,degraded,audit=review_and_label(plan,sentences,Client(),config)
        self.assertEqual([(x["start_id"],x["end_id"]) for x in reviewed],[(1,4),(5,6)])
        self.assertEqual(audit["summary"]["boundary_success"],1)
        self.assertIn("1/2",degraded[0])

    def test_build_activation_is_a_single_pointer_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            db=Database(Path(directory)/"db.sqlite");bundle=sample_bundle();db.stage_bundle(bundle)
            self.assertEqual(db.list_sources(),[]);self.assertEqual(db.active_chunk_ids(),[])
            meta={"version":"v","index":"i","mapping":"m","count":1,"dimension":2,"chunk_ids_hash":"h"};db.activate_build(bundle,meta)
            self.assertEqual(len(db.list_sources()),1);self.assertEqual(db.active_chunk_ids(),["chunk-b1"]);self.assertEqual(db.meta("active_index")["version"],"v");self.assertTrue(db.integrity()["sqlite_ok"])

    def test_failed_replacement_keeps_current_build(self):
        with tempfile.TemporaryDirectory() as directory:
            db=Database(Path(directory)/"db.sqlite");first=sample_bundle("b1");db.stage_bundle(first);db.activate_build(first,{"version":"v1","index":"i","mapping":"m","count":1,"dimension":2,"chunk_ids_hash":"h"})
            second=sample_bundle("b2");db.stage_bundle(second);db.fail_build("b2","simulated")
            self.assertEqual(db.source("hash")["current_build_id"],"b1");self.assertEqual(db.active_chunk_ids(),["chunk-b1"])

    def test_inspection_files_reference_text_once(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle=sample_bundle();paths=write_inspection_files(Path(directory),bundle)
            sections=json.loads(paths["sections"].read_text(encoding="utf-8"));chunks=json.loads(paths["chunks"].read_text(encoding="utf-8"));sentences=json.loads(paths["sentences"].read_text(encoding="utf-8"))
            self.assertIn("raw_text",sentences[0]);self.assertNotIn("text",sections[0]);self.assertNotIn("search_text",chunks[0])

    def test_lexical_tokens_keep_frequency(self):
        value=lexical_tokens("DNS DNS 域名");self.assertGreaterEqual(value.lower().count("dns"),2)

    def test_colloquial_how_query_is_normalized_without_inventing_answer_terms(self):
        self.assertEqual(normalize_retrieval_query("路由是怎么运作的？"),"路由的工作原理和具体过程是什么？")
        self.assertEqual(normalize_retrieval_query("DNS如何运行"),"DNS的工作原理和具体过程是什么？")
        self.assertEqual(normalize_retrieval_query("路由有哪些类型？"),"路由有哪些类型？")

    def test_section_score_does_not_reward_having_more_chunks(self):
        self.assertEqual(aggregate_section_score([0.87,0.78]),0.87)
        self.assertGreater(aggregate_section_score([0.945]),aggregate_section_score([0.87,0.78]))

    def test_sentence_edit_preview_is_auditable_and_glossary_is_opt_in(self):
        sentences=[{"id":1,"raw_text":"使用软麦克斯选择专家。","approved_text":None,"start":0,"end":1000}]
        preview=preview_sentence_edits(sentences,[{"sentence_id":1,"approved_text":"使用Softmax选择专家。"}])
        self.assertEqual(preview["changes"][0]["before"],"使用软麦克斯选择专家。")
        self.assertEqual(preview["changes"][0]["after"],"使用Softmax选择专家。")
        self.assertEqual(len(preview["glossary_candidates"]),1)
        self.assertEqual(preview["glossary_candidates"][0]["variant"],"软麦克斯")
        with self.assertRaises(ValueError):preview_sentence_edits(sentences,[{"sentence_id":1,"approved_text":"缺少句号"}])

    def test_index_gc_keeps_active_and_one_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);indexes=root/"data"/"indexes";indexes.mkdir(parents=True)
            versions=["1"*32,"2"*32,"3"*32]
            for version in versions:
                (indexes/f"faiss-{version}.index").write_bytes(b"i")
                (indexes/f"chunks-{version}.json").write_text("[]",encoding="utf-8")
            config={"_root":str(root),"project":{"data_dir":"data"}}
            removed=prune_index_versions(config,{"version":versions[-1]},keep=2)
            self.assertEqual(len(removed),2)
            self.assertTrue((indexes/f"faiss-{versions[-1]}.index").is_file())
            self.assertEqual(len(list(indexes.glob("faiss-*.index"))),2)


if __name__=="__main__":unittest.main()
