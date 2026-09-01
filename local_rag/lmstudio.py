from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx


class LocalLLMUnavailable(RuntimeError):
    pass


def _validate_json(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    type_map = {
        "object": dict, "array": list, "string": str, "boolean": bool,
        "integer": int, "number": (int, float),
    }
    if expected in type_map and (not isinstance(value, type_map[expected]) or (expected in {"integer", "number"} and isinstance(value, bool))):
        raise ValueError(f"{path}类型不符合JSON schema：需要{expected}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}必须等于{schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}不在允许值{schema['enum']}中")
    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}小于最小值{schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}大于最大值{schema['maximum']}")
    if expected == "string" and "maxLength" in schema and len(value) > int(schema["maxLength"]):
        raise ValueError(f"{path}字符串过长")
    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}缺少字段{key}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(schema.get("properties", {}))
            if extras:
                raise ValueError(f"{path}包含额外字段：{sorted(extras)}")
        for key, child in schema.get("properties", {}).items():
            if key in value:
                _validate_json(value[key], child, f"{path}.{key}")
    elif expected == "array":
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path}数组过短")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path}数组过长")
        child = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_json(item, child, f"{path}[{index}]")


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型没有返回JSON对象")
    return value


class LocalLLMClient:
    def __init__(self, project_config: dict[str, Any]):
        self.project_config=project_config;self.config=project_config["llm"]
        self.base_url=self.config["base_url"].rstrip("/");self.model=self.config["model"]
        self.timeout=float(self.config.get("timeout_seconds",180));self.max_retries=max(1,int(self.config.get("max_retries",1)))

    def _path(self,value:str|Path)->Path:
        path=Path(os.path.expandvars(str(value))).expanduser()
        return (path if path.is_absolute() else Path(self.project_config["_root"])/path).resolve()

    def _pid_path(self)->Path:
        return self._path(self.project_config["project"]["data_dir"])/"runtime"/"llama-server.pid"

    def _headers(self)->dict[str,str]:
        key=str(self.config.get("api_key","")).strip()
        return {"Authorization":f"Bearer {key}"} if key else {}

    def _model_entries(self)->list[dict[str,Any]]:
        try:
            response=httpx.get(f"{self.base_url}/models",headers=self._headers(),timeout=5);response.raise_for_status()
            return list(response.json().get("data",[]))
        except Exception as error:raise LocalLLMUnavailable(f"独立llama-server不可用：{self.base_url}") from error

    def models(self) -> list[str]:
        return [str(item["id"]) for item in self._model_entries()]

    def health(self) -> dict[str, Any]:
        try:
            entries=self._model_entries();models=[str(x["id"]) for x in entries];active=self._select_model(models)
            match=next(x for x in entries if str(x["id"])==active);actual=match.get("meta",{}).get("n_ctx")
            if actual is not None and int(actual)!=int(self.config["context_window"]):raise LocalLLMUnavailable(f"llama-server实际上下文{actual}与配置{self.config['context_window']}不一致")
            return {
                "available": True, "models": models, "configured_model": self.model,
                "active_model": active, "context_length": actual,"backend":"llama-server",
            }
        except LocalLLMUnavailable as error:
            return {"available": False, "models": [], "configured_model": self.model, "error": str(error)}

    def start_server(self) -> int:
        if self.health().get("available"):
            path=self._pid_path()
            if not path.is_file():raise RuntimeError(f"端口已被非本项目管理的模型服务占用：{self.base_url}")
            return int(path.read_text().strip())
        server=self._path(self.config["server_path"]);model=self._path(self.config["model_path"])
        if not server.is_file():raise FileNotFoundError(f"找不到llama-server：{server}")
        if not model.is_file():raise FileNotFoundError(f"找不到生成模型：{model}")
        port=self.base_url.split(":")[-1].split("/")[0];runtime=self._pid_path().parent;runtime.mkdir(parents=True,exist_ok=True)
        flash_value=self.config.get("flash_attention","on");flash="on" if flash_value is True else ("off" if flash_value is False else str(flash_value).lower())
        command=[str(server),"-m",str(model),"--alias",self.model,"--host","127.0.0.1","--port",port,"-c",str(int(self.config["context_window"])),"-np","1","-ngl",str(self.config.get("gpu_layers","all")),"--flash-attn",flash,"--jinja","--reasoning","off","--reasoning-budget","0","--api-key",str(self.config.get("api_key","local-rag-loopback"))]
        flags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
        with (runtime/"llama-server.stdout.log").open("a",encoding="utf-8") as out,(runtime/"llama-server.stderr.log").open("a",encoding="utf-8") as err:
            process=subprocess.Popen(command,cwd=server.parent,stdout=out,stderr=err,creationflags=flags)
        self._pid_path().write_text(str(process.pid),encoding="ascii")
        deadline=time.time()+float(self.config.get("load_timeout_seconds",180))
        while time.time()<deadline:
            if process.poll() is not None:
                self._pid_path().unlink(missing_ok=True);raise RuntimeError("llama-server启动失败，请查看knowledge_base/data/runtime/llama-server.stderr.log")
            if self.health().get("available"):return process.pid
            time.sleep(.5)
        self.stop_server();raise TimeoutError("llama-server模型加载超时")

    def stop_server(self) -> None:
        path=self._pid_path()
        if not path.is_file():return
        try:pid=int(path.read_text(encoding="ascii").strip())
        except Exception:pid=0
        if pid>0:subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True)
        path.unlink(missing_ok=True)

    def loaded_models(self) -> list[dict[str, Any]]:
        try:
            return [{"identifier":x["id"],"contextLength":x.get("meta",{}).get("n_ctx")} for x in self._model_entries()]
        except LocalLLMUnavailable:return []

    def load_model(self) -> None:
        self.start_server()

    def unload_model(self) -> None:
        self.stop_server()

    def context_budget(self, configured: int, max_output_tokens: int) -> int:
        health = self.health()
        actual = int(health.get("context_length") or configured)
        if actual != int(configured):
            raise RuntimeError(f"llama-server实际上下文{actual}与项目配置{configured}不一致")
        reserve = max(1024, int(max_output_tokens) + 768)
        return max(512, int((actual - reserve) * float(self.config.get("evidence_budget_ratio", 0.5))))

    def _select_model(self, models: list[str]) -> str:
        if self.model in models:
            return self.model
        partial = [item for item in models if self.model.lower() in item.lower()]
        if len(partial) == 1:
            return partial[0]
        raise LocalLLMUnavailable(
            f"配置模型“{self.model}”未由llama-server提供。当前模型：{models or '无'}"
        )

    def _active_model(self) -> str:
        health = self.health()
        if not health.get("available"):
            raise LocalLLMUnavailable(health.get("error","llama-server模型未加载"))
        return str(health["active_model"])

    def structured(
        self,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int = 4096,
        *,
        attempts: int | None = None,
        audit: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system_prompt = "\n".join(item["content"] for item in messages if item["role"] == "system")
        user_input = "\n\n".join(item["content"] for item in messages if item["role"] != "system")
        user_input += (
            "\n\n只返回一个JSON对象，不要Markdown代码块，不要解释。JSON必须严格符合以下schema：\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        payload = {
            "model": self._active_model(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        last_error: Exception | None = None
        request_timeout = max(self.timeout, 180) if schema_name in {"semantic_sections","boundary_review","section_labels"} else self.timeout
        total_attempts = max(1, int(attempts if attempts is not None else self.max_retries))
        for attempt in range(1, total_attempts + 1):
            entry: dict[str, Any] = {"attempt": attempt, "max_tokens": int(max_tokens)}
            started = time.perf_counter()
            try:
                response = httpx.post(f"{self.base_url}/chat/completions",json=payload,headers=self._headers(),timeout=request_timeout)
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices", [])
                content = str(choices[0].get("message", {}).get("content", "")) if choices else ""
                entry.update({
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "usage": body.get("usage"),
                    "content": content,
                })
                result = _parse_json_object(content)
                _validate_json(result, schema)
                entry["valid"] = True
                if audit is not None:
                    audit.append(entry)
                return result
            except Exception as error:
                last_error = error
                entry.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000))
                entry.update({"valid": False, "error": str(error)})
                if audit is not None:
                    audit.append(entry)
        raise RuntimeError(f"llama-server结构化输出失败：{last_error}")

    def count_tokens(self, text: str) -> int:
        """Use llama-server's exact tokenizer; stay conservative on fallback."""
        try:
            response=httpx.post(self.base_url.removesuffix("/v1")+"/tokenize",json={"content":text},headers=self._headers(),timeout=10);response.raise_for_status()
            return len(response.json()["tokens"])
        except Exception:
            # Chinese Qwen tokenization is often near character granularity; this fallback
            # intentionally overestimates ASCII-heavy text instead of risking overflow.
            return max(1, len(text.encode("utf-8")) // 2)

    def expand_query(self, query: str, limit: int) -> list[str]:
        schema = {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": limit}},
            "required": ["queries"], "additionalProperties": False,
        }
        result = self.structured(
            [
                {"role": "system", "content": "你只生成有助于召回本地知识库原文的中文检索表达，不回答问题，不添加事实。"},
                {"role": "user", "content": f"原问题：{query}\n给出最多{limit}条同义或术语化检索表达。"},
            ],
            "query_expansions", schema, 128,
        )
        return [item.strip() for item in result.get("queries", []) if item.strip() and item.strip() != query]


# Kept temporarily for third-party imports; the project itself uses the generic name.
LMStudioClient=LocalLLMClient
