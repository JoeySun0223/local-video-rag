from __future__ import annotations

import json
import ctypes
import msvcrt
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx

from .config import project_path
from .lmstudio import LocalLLMClient


class FileLock:
    def __init__(self, path: Path):
        self.path=path;self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open("a+b")
        if self.path.stat().st_size==0:self.file.write(b"0");self.file.flush()
        self.file.seek(0)
        try:msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.file.close();raise RuntimeError("已有另一个知识库构建/删除操作正在运行")
        return self
    def __exit__(self,*_):
        if self.file:
            self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1);self.file.close()


def _state_path(config:dict[str,Any])->Path:
    return project_path(config,config["project"]["data_dir"])/"runtime"/"state.json"


def _write_state(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)


def process_alive(pid:int)->bool:
    if pid<=0:return False
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:ctypes.windll.kernel32.CloseHandle(handle)


def start_stack(config:dict[str,Any],open_browser:bool=True)->dict[str,Any]:
    state_path=_state_path(config)
    if state_path.is_file():
        try:
            old=json.loads(state_path.read_text(encoding="utf-8"));pid=int(old.get("web_pid",0))
            if pid and process_alive(pid):return {**old,"already_running":True}
        except Exception:pass
    base=f'http://{config["web"]["host"]}:{config["web"]["port"]}'
    try:
        if httpx.get(base+"/api/health",timeout=1).status_code<500:
            raise RuntimeError(f"端口已被未受当前启动器管理的服务占用：{base}")
    except httpx.HTTPError:
        pass
    client=LocalLLMClient(config);web_process=None;llm_pid=0
    log_dir=state_path.parent;out=(log_dir/"web.stdout.log").open("a",encoding="utf-8");err=(log_dir/"web.stderr.log").open("a",encoding="utf-8")
    try:
        llm_pid=client.start_server()
        if config["llm"]["model"] not in [x.get("identifier") for x in client.loaded_models()]:client.load_model()
        command=[sys.executable,str(Path(config["_root"])/"rag.py"),"--config",str(config["_config_path"]),"serve"]
        flags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)
        web_process=subprocess.Popen(command,cwd=config["_root"],stdout=out,stderr=err,creationflags=flags)
        deadline=time.time()+90
        health_payload=None
        while time.time()<deadline:
            if web_process.poll() is not None:raise RuntimeError("Web进程启动失败，请查看data/runtime/web.stderr.log")
            try:
                response=httpx.get(base+"/api/health",timeout=2)
                if response.status_code==200:health_payload=response.json();break
            except Exception:pass
            time.sleep(.5)
        else:raise TimeoutError("Web健康检查超时")
        prewarm=httpx.post(base+"/api/runtime/prewarm",timeout=300);prewarm.raise_for_status()
        state={"web_pid":int(health_payload.get("web_pid",web_process.pid)),"launcher_pid":web_process.pid,"llm_pid":llm_pid,"llm_backend":"llama-server","base_url":base,"model":config["llm"]["model"],"started_at":time.time(),"status":"running"};_write_state(state_path,state)
        if open_browser:webbrowser.open(base)
        return state
    except Exception:
        if web_process and web_process.poll() is None:web_process.terminate()
        client.unload_model();client.stop_server();raise
    finally:out.close();err.close()


def stop_stack(config:dict[str,Any])->dict[str,Any]:
    state_path=_state_path(config);state={}
    if state_path.is_file():
        try:state=json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:state={}
    pid=int(state.get("web_pid",0) or 0)
    if pid and process_alive(pid):
        base=state.get("base_url") or f'http://{config["web"]["host"]}:{config["web"]["port"]}'
        try:
            active=httpx.get(base+"/api/tasks-active",timeout=3).json()
        except Exception:
            active=[]
        if active:
            choice="c"
            if sys.stdin.isatty():
                print(f"当前有 {len(active)} 个入库任务。输入 W 等待安全完成，或 C 请求安全取消后关闭。")
                choice=(input("[W/c]: ").strip().lower() or "w")
            if choice.startswith("w"):
                print("正在等待入库任务完成；按 Ctrl+C 可中止本次关闭操作。")
            else:
                for task in active:
                    try:httpx.post(base+f'/api/tasks/{task["task_id"]}/cancel',timeout=3)
                    except Exception:pass
                print("已请求安全取消，正在等待当前模型调用返回并结束暂存Build。")
            deadline=time.time()+900
            while time.time()<deadline:
                try:
                    active=httpx.get(base+"/api/tasks-active",timeout=3).json()
                    if not active:break
                except Exception:break
                time.sleep(1)
            else:
                raise TimeoutError("入库任务在15分钟内未安全结束；Web和模型保持运行")
        # Active Builds have already completed or reached a safe cancelled state.
        # Force-ending this dedicated worker is the reliable Windows equivalent of
        # unloading its in-process BGE and reranker objects.
        subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True)
        deadline=time.time()+5
        while time.time()<deadline and process_alive(pid):time.sleep(.25)
        if process_alive(pid):subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True)
    client=LocalLLMClient(config);client.stop_server()
    _write_state(state_path,{"status":"stopped","stopped_at":time.time(),"model":config["llm"]["model"],"llm_backend":"llama-server"})
    return {"stopped":True,"web_pid":pid,"models_loaded":client.loaded_models()}
