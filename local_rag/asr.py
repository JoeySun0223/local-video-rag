from __future__ import annotations

import gc
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
import wave
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import project_path


SENTENCE_END = re.compile(r".+?(?:[。！？!?]+|\.(?=\s|$))", re.DOTALL)
ALIGNMENT_SENTENCE_END = re.compile(r"(?:[。！？!?]+|\.(?=\s|$))")
CLAUSE_BOUNDARY = re.compile(r"[，,；;：:]")


@dataclass
class ASRRequest:
    audio: Path
    title: str
    category: str
    hotwords: list[str]
    work_dir: Path
    ffmpeg: Path
    checkpoint_path: Path | None = None
    cancelled: Callable[[], bool] | None = None


@dataclass
class ASRResult:
    sentences: list[dict[str, Any]]
    requested_provider: str
    actual_provider: str
    model_path: str
    model_revision: str | None
    timestamp_method: str
    used_vad: bool
    used_punctuation: bool
    hotwords_applied: list[str]
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("sentences", None)
        value.pop("raw", None)
        return value


def _clean_model_output(value: Any) -> Any:
    """Keep raw model evidence JSON-serializable without hiding its structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean_model_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_model_output(item) for item in value]
    if hasattr(value, "tolist"):
        return _clean_model_output(value.tolist())
    return str(value)


def extract_funasr_segments(raw_results: Any) -> list[dict[str, Any]]:
    """Adapt FunASR sentence_info to the project's canonical sentence contract."""
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    if not isinstance(raw_results, list):
        raise ValueError("FunASR没有返回结果数组")
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        segments = result.get("sentence_info")
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("sentence") or segment.get("text") or "").strip()
            start, end = segment.get("start"), segment.get("end")
            if text and start is not None and end is not None:
                rows.append({"id": len(rows) + 1, "text": text, "start": int(start), "end": int(end)})
    if not rows:
        raise ValueError("FunASR没有返回可用的sentence_info；不会伪造句级时间戳")
    return rows


def _model_revision(path: Path) -> str | None:
    metadata = path / "download_provenance.json"
    if metadata.is_file():
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            return str(value.get("revision") or value.get("commit") or value.get("requested_revision") or "") or None
        except Exception:
            pass
    return None


def _unload(*objects: Any) -> None:
    for item in objects:
        try:
            del item
        except Exception:
            pass
    gc.collect()


class FunASRProvider:
    name = "funasr"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def doctor(self) -> dict[str, Any]:
        models = self.config["models"]
        paths = {key: project_path(self.config, models[key]) for key in ("asr", "vad", "punctuation")}
        try:
            import funasr  # noqa: F401
            runtime = True
        except Exception:
            runtime = False
        return {"provider": self.name, "runtime": runtime, "assets": {k: str(v) for k, v in paths.items()}, "available": runtime and all(v.exists() for v in paths.values())}

    def transcribe(self, request: ASRRequest) -> ASRResult:
        from funasr import AutoModel

        models = self.config["models"]
        model_path = project_path(self.config, models["asr"])
        started = time.perf_counter()
        model = AutoModel(
            model=str(model_path),
            vad_model=str(project_path(self.config, models["vad"])),
            punc_model=str(project_path(self.config, models["punctuation"])),
            disable_update=True,
            disable_pbar=True,
        )
        loaded = time.perf_counter()
        kwargs: dict[str, Any] = {
            "input": str(request.audio), "batch_size_s": 300,
            "return_timestamp": True, "sentence_timestamp": True,
            "return_raw_text": True, "disable_pbar": True,
        }
        if request.hotwords:
            kwargs["hotword"] = " ".join(request.hotwords)
        try:
            raw = model.generate(**kwargs)
            generated = time.perf_counter()
            sentences = extract_funasr_segments(raw)
            return ASRResult(
                sentences=sentences, requested_provider=self.name, actual_provider=self.name,
                model_path=str(model_path), model_revision=_model_revision(model_path),
                timestamp_method="funasr_sentence_info_ms", used_vad=True, used_punctuation=True,
                hotwords_applied=list(request.hotwords),
                timings_ms={
                    "model_init": round((loaded - started) * 1000),
                    "generate": round((generated - loaded) * 1000),
                    "adapt": round((time.perf_counter() - generated) * 1000),
                },
                raw={"funasr": _clean_model_output(raw)},
            )
        finally:
            del model
            gc.collect()


def _wav_info(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Qwen3-ASR输入必须是16-bit单声道WAV")
        return wav.getframerate(), wav.getnframes(), wav.getnchannels()


def _quiet_cut(wav: wave.Wave_read, target: int, radius: int, window: int) -> int:
    total = wav.getnframes()
    best_position, best_energy = target, None
    step = max(window, wav.getframerate() // 10)
    lower = max(window, target - radius)
    upper = min(total - window, target + radius)
    for position in range(lower, upper + 1, step):
        wav.setpos(position)
        samples = array("h", wav.readframes(window))
        if os.sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            continue
        energy = sum(int(x) * int(x) for x in samples) / len(samples)
        if best_energy is None or energy < best_energy:
            best_position, best_energy = position + window // 2, energy
    return best_position


def silence_aware_ranges(audio: Path, target_seconds: int = 180, search_seconds: int = 5) -> list[tuple[int, int]]:
    """Return non-overlapping sample ranges, adjusting each nominal cut to local low energy."""
    with wave.open(str(audio), "rb") as wav:
        rate, total = wav.getframerate(), wav.getnframes()
        target = max(rate, int(target_seconds * rate))
        radius = max(rate, int(search_seconds * rate))
        window = max(1, int(0.1 * rate))
        cuts = [0]
        nominal = target
        while nominal < total:
            cut = _quiet_cut(wav, nominal, radius, window)
            if cut - cuts[-1] < rate:
                cut = min(total, cuts[-1] + target)
            cuts.append(cut)
            nominal = cut + target
        cuts.append(total)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def _extract_range(ffmpeg: Path, audio: Path, output: Path, start_seconds: float, duration_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_seconds:.6f}", "-i", str(audio), "-t", f"{duration_seconds:.6f}",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ], check=True)
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(f"无法生成ASR音频片段：{output}")


def _complete_sentences(text: str) -> list[str]:
    compact = text.strip()
    if not compact:
        raise ValueError("Qwen3-ASR返回空文本")
    pieces = [match.group(0).strip() for match in SENTENCE_END.finditer(compact) if match.group(0).strip()]
    matches: list[str] = []
    leading_symbols = ""
    for piece in pieces:
        # Punctuation models can legitimately emit runs such as ``。。`` or
        # ``？！``.  The regex sees a trailing symbol as another sentence, but
        # it has no acoustic unit of its own.  Preserve every character while
        # attaching such a symbol-only piece to a neighbouring lexical
        # sentence instead of inventing an unalignable sentence.
        if _alignment_key(piece):
            matches.append(leading_symbols + piece)
            leading_symbols = ""
        elif matches:
            matches[-1] += piece
        else:
            leading_symbols += piece
    if leading_symbols:
        if matches:
            matches[-1] += leading_symbols
        else:
            raise ValueError("转写结果只有标点或符号，不包含可对齐文字")
    consumed = "".join(pieces)
    normalize = lambda value: re.sub(r"\s+", "", value)
    if not matches or normalize(consumed) != normalize(compact):
        raise ValueError("转写结果仍含没有句末标点的片段；不会把不完整片段伪装成完整句")
    return matches


def normalize_overlong_sentence_boundaries(text: str, max_lexical_chars: int = 240) -> tuple[str, bool]:
    """Promote existing clause punctuation only when Qwen emits an unusably long sentence.

    No lexical character is added, removed, or rewritten. The original model
    transcript remains in the ASR checkpoint; this normalized form is used for
    sentence alignment, chunking, and inspection.
    """
    if max_lexical_chars < 32:
        raise ValueError("Qwen句子长度上限不能小于32个可对齐字符")
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in ALIGNMENT_SENTENCE_END.finditer(text):
        ranges.append((start, match.end()))
        start = match.end()
    if start < len(text):
        ranges.append((start, len(text)))
    if not ranges and text:
        ranges.append((0, len(text)))

    result: list[str] = []
    changed = False
    for span_start, span_end in ranges:
        remaining = text[span_start:span_end]
        while len(_alignment_key(remaining)) > max_lexical_chars:
            candidates: list[tuple[int, int]] = []
            for match in CLAUSE_BOUNDARY.finditer(remaining):
                lexical = len(_alignment_key(remaining[:match.end()]))
                if lexical >= max_lexical_chars // 2:
                    candidates.append((match.start(), lexical))
            before = [item for item in candidates if item[1] <= max_lexical_chars]
            if before:
                cut = before[-1][0]
            else:
                after = [item for item in candidates if item[1] <= max_lexical_chars * 3 // 2]
                if not after:
                    break
                cut = after[0][0]
            result.append(remaining[:cut] + "。")
            remaining = remaining[cut + 1:]
            changed = True
        result.append(remaining)
    normalized = "".join(result)
    if _alignment_key(normalized) != _alignment_key(text):
        raise AssertionError("句界规范化不得改变转写文字")
    return normalized, changed


def _alignment_key(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or "\u3400" <= char <= "\u9fff")


def alignment_safe_transcript(text: str) -> str:
    """Keep Qwen text intact while forcing the official aligner tokenizer to flush at sentence ends.

    The official tokenizer drops punctuation without flushing its Latin-word
    buffer, so ``token。FFN`` otherwise becomes one timestamp item
    ``tokenFFN`` spanning two sentences.  Added whitespace is not part of the
    canonical transcript and creates no textual or timestamp content.
    """
    return ALIGNMENT_SENTENCE_END.sub(lambda match: match.group(0) + " ", text)


def timestamps_to_sentences(
    transcript: str,
    timestamps: list[dict[str, Any]],
    offset_ms: int = 0,
    adjustments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sentences = _complete_sentences(transcript)
    units = []
    for item in timestamps:
        key = _alignment_key(str(item.get("text", "")))
        if not key:
            continue
        start = item.get("start_time", item.get("start"))
        end = item.get("end_time", item.get("end"))
        if start is None or end is None:
            raise ValueError("ForcedAligner时间戳缺少start/end")
        units.append({"key": key, "start": float(start), "end": float(end), "raw": item})
    if not units:
        raise ValueError("ForcedAligner没有返回有效词/字时间戳")
    rows = []
    unit_index = 0
    previous_end = offset_ms
    for sentence in sentences:
        target = _alignment_key(sentence)
        if not target:
            raise ValueError("完整句不包含可对齐文字")
        collected = ""
        first = unit_index
        while unit_index < len(units) and len(collected) < len(target):
            collected += units[unit_index]["key"]
            unit_index += 1
        if collected != target:
            raise ValueError(f"ForcedAligner文字与转写句子不一致：{sentence[:30]}")
        start = offset_ms + round(units[first]["start"] * 1000)
        end = offset_ms + round(units[unit_index - 1]["end"] * 1000)
        if start < previous_end and previous_end - start <= 20:
            if adjustments is not None:
                adjustments.append({
                    "sentence_id": len(rows) + 1,
                    "reason": "sub_20ms_overlap_clamped",
                    "original_start": start,
                    "adjusted_start": previous_end,
                    "end": end,
                })
            start = previous_end
        # The official aligner can return an instantaneous point for a very
        # short filler (for example a final "嗯").  Millisecond rounding then
        # produces start == end.  Preserve that real point as the start and
        # give it the smallest interval accepted by the canonical schema.
        if end == start and start >= previous_end:
            if adjustments is not None:
                adjustments.append({
                    "sentence_id": len(rows) + 1,
                    "reason": "zero_duration_after_ms_rounding",
                    "start": start,
                    "original_end": end,
                    "adjusted_end": end + 1,
                })
            end += 1
        if start < previous_end or end <= start:
            raise ValueError(
                "ForcedAligner返回重叠或倒序的句级时间戳："
                f"sentence={len(rows)+1} previous_end={previous_end} "
                f"start={start} end={end} text={sentence[:80]!r}"
            )
        rows.append({"id": len(rows) + 1, "text": sentence, "start": start, "end": end})
        previous_end = end
    if unit_index != len(units):
        raise ValueError("ForcedAligner仍有未归属任何完整句的时间戳")
    return rows


class _MemorySampler:
    def __init__(self):
        self._stop = threading.Event()
        self.peak_working_set = 0
        self.peak_private = 0
        self.minimum_available = 0
        self._thread: threading.Thread | None = None

    @staticmethod
    def _snapshot() -> tuple[int, int, int]:
        if platform.system() != "Windows":
            return 0, 0, 0
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        pmc = PMC(); pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(pmc.WorkingSetSize), int(pmc.PrivateUsage), int(status.ullAvailPhys)

    def __enter__(self):
        def run():
            while not self._stop.wait(0.25):
                working, private, available = self._snapshot()
                self.peak_working_set = max(self.peak_working_set, working)
                self.peak_private = max(self.peak_private, private)
                self.minimum_available = available if not self.minimum_available else min(self.minimum_available, available)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def report(self) -> dict[str, float]:
        gib = 1024 ** 3
        return {
            "peak_working_set_gib": round(self.peak_working_set / gib, 3),
            "peak_private_gib": round(self.peak_private / gib, 3),
            "minimum_system_available_gib": round(self.minimum_available / gib, 3),
        }


class Qwen3ASRProvider:
    name = "qwen3_asr"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        settings = config["asr"]["providers"][self.name]
        self.settings = settings
        self.model_path = project_path(config, settings["model"])
        self.aligner_path = project_path(config, settings["aligner"])

    def _runtime(self):
        import torch
        requested = str(self.settings.get("device", "cpu")).lower()
        if requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("配置要求CUDA，但当前PyTorch/CUDA不可用")
        dtype_name = str(self.settings.get("dtype", "bfloat16")).lower()
        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if dtype_name not in dtypes:
            raise ValueError(f"不支持的Qwen ASR dtype：{dtype_name}")
        return torch.device(device), dtypes[dtype_name], dtype_name

    def doctor(self) -> dict[str, Any]:
        try:
            import torch
            import transformers
            runtime = tuple(int(part) for part in transformers.__version__.split(".")[:2]) >= (5, 13)
            device,dtype,dtype_name = self._runtime(); compute = False
            if runtime:
                value = torch.ones((16, 16), dtype=dtype, device=device)
                compute = (value @ value).dtype == dtype
        except Exception as error:
            runtime = False; compute = False; device = str(self.settings.get("device","cpu")); dtype_name = str(self.settings.get("dtype","bfloat16"));runtime_error=str(error)
        assets = {"model": str(self.model_path), "aligner": str(self.aligner_path)}
        return {
            "provider": self.name, "runtime": runtime, "compute_dtype_ok": compute,
            "device": str(device), "dtype": dtype_name, "error": locals().get("runtime_error"),
            "assets": assets, "available": runtime and compute and self.model_path.exists() and self.aligner_path.exists(),
        }

    def _check(self) -> None:
        report = self.doctor()
        if not report["runtime"]:
            raise RuntimeError("Qwen3-ASR原生HF版要求Transformers>=5.13和可用PyTorch")
        if not report["compute_dtype_ok"]:
            raise RuntimeError(f"配置的ASR设备或精度不可用：{report}")
        for label, path in (("ASR", self.model_path), ("ForcedAligner", self.aligner_path)):
            if not path.is_dir() or not (path / "config.json").is_file():
                raise FileNotFoundError(f"{label}官方本地模型不完整：{path}")

    def _prompt(self, request: ASRRequest) -> str | None:
        parts = []
        if request.title:
            parts.append(f"Title: {request.title}")
        if request.category and request.category != "未分类":
            parts.append(f"Category: {request.category}")
        if request.hotwords:
            parts.append("Vocabulary: " + ", ".join(request.hotwords))
        return ". ".join(parts) or None

    def _segment_files(self, request: ASRRequest, ranges: list[tuple[int, int]], prefix: str) -> list[dict[str, Any]]:
        rate, _, _ = _wav_info(request.audio)
        files = []
        for index, (start, end) in enumerate(ranges, 1):
            if request.cancelled and request.cancelled():
                raise InterruptedError("用户安全取消入库")
            path = request.work_dir / f"{prefix}-{index:03d}.wav"
            _extract_range(request.ffmpeg, request.audio, path, start / rate, (end - start) / rate)
            files.append({"path": path, "start_ms": round(start * 1000 / rate), "end_ms": round(end * 1000 / rate)})
        return files

    def _checkpoint_signature(self, request: ASRRequest, clips: list[dict[str, Any]], prompt: str | None) -> dict[str, Any]:
        audio_stat = request.audio.stat()
        model_file = self.model_path / "model.safetensors"
        model_stat = model_file.stat()
        return {
            "schema_version": 1,
            "provider": self.name,
            "audio_size": audio_stat.st_size,
            "audio_mtime_ns": audio_stat.st_mtime_ns,
            "model_size": model_stat.st_size,
            "model_mtime_ns": model_stat.st_mtime_ns,
            "model_revision": _model_revision(self.model_path),
            "prompt": prompt,
            "max_new_tokens": int(self.settings.get("max_new_tokens", 2048)),
            "clips": [{"start_ms": x["start_ms"], "end_ms": x["end_ms"]} for x in clips],
        }

    def _load_checkpoint(self, request: ASRRequest, clips: list[dict[str, Any]], prompt: str | None) -> list[dict[str, Any]] | None:
        path = request.checkpoint_path
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self._checkpoint_signature(request, clips, prompt):
                return None
            saved = payload.get("segments")
            if not isinstance(saved, list) or len(saved) != len(clips):
                return None
            rows = []
            for clip, item in zip(clips, saved, strict=True):
                if not isinstance(item, dict) or not isinstance(item.get("raw_transcription"), str):
                    return None
                rows.append({**item, **clip})
            return rows
        except (OSError, ValueError, TypeError):
            return None

    def _save_checkpoint(self, request: ASRRequest, clips: list[dict[str, Any]], prompt: str | None, segments: list[dict[str, Any]]) -> None:
        path = request.checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": self._checkpoint_signature(request, clips, prompt),
            "segments": _clean_model_output(segments),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _run(
        self,
        request: ASRRequest,
        clips: list[dict[str, Any]],
        allow_checkpoint: bool = False,
        build_sentences: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], bool]:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor

        device,dtype,_ = self._runtime()

        prompt = self._prompt(request)
        timings: dict[str, int] = {}
        raw_segments = self._load_checkpoint(request, clips, prompt) if allow_checkpoint else None
        checkpoint_reused = raw_segments is not None
        if raw_segments is None:
            raw_segments = []
            started = time.perf_counter()
            processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            model = AutoModelForMultimodalLM.from_pretrained(str(self.model_path), dtype=dtype, local_files_only=True).to(device).eval()
            timings["asr_model_init"] = round((time.perf_counter() - started) * 1000)
            generate_started = time.perf_counter()
            try:
                for clip in clips:
                    if request.cancelled and request.cancelled():
                        raise InterruptedError("用户安全取消入库")
                    inputs = processor.apply_transcription_request(audio=str(clip["path"]), prompt=prompt)
                    inputs = inputs.to(model.device, model.dtype)
                    max_new_tokens = int(self.settings.get("max_new_tokens", 2048))
                    with torch.inference_mode():
                        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                    generated = output_ids[:, inputs["input_ids"].shape[1]:]
                    eos = model.generation_config.eos_token_id
                    eos_ids = set(eos if isinstance(eos, list) else [eos])
                    if generated.shape[1] >= max_new_tokens and int(generated[0, -1]) not in eos_ids:
                        raise RuntimeError(f"Qwen3-ASR输出达到{max_new_tokens} token上限；拒绝保存可能被截断的转写")
                    parsed = processor.decode(generated, return_format="parsed")[0]
                    text = str(parsed.get("transcription", "")).strip()
                    language = str(parsed.get("language") or "")
                    raw_segments.append({**clip, "language": language, "raw_transcription": text, "generated_tokens": int(generated.shape[1])})
                    del inputs, output_ids, generated
            finally:
                del model, processor
                gc.collect()
            timings["asr_generate"] = round((time.perf_counter() - generate_started) * 1000)
            if allow_checkpoint:
                self._save_checkpoint(request, clips, prompt, raw_segments)
        else:
            timings["asr_checkpoint_reused"] = 1

        align_started = time.perf_counter()
        align_processor = AutoProcessor.from_pretrained(str(self.aligner_path), local_files_only=True)
        aligner = AutoModelForTokenClassification.from_pretrained(str(self.aligner_path), dtype=dtype, local_files_only=True).to(device).eval()
        timings["aligner_model_init"] = round((time.perf_counter() - align_started) * 1000)
        merged_timestamps: list[dict[str, Any]] = []
        punctuation_used = False
        forward_started = time.perf_counter()
        try:
            for segment in raw_segments:
                if request.cancelled and request.cancelled():
                    raise InterruptedError("用户安全取消入库")
                if not segment["raw_transcription"].strip():
                    segment["timestamps"] = []
                    continue
                canonical_transcription, changed = normalize_overlong_sentence_boundaries(
                    segment["raw_transcription"], int(self.settings.get("max_sentence_chars", 240)),
                )
                punctuation_used = punctuation_used or changed
                if changed:
                    segment["canonical_transcription"] = canonical_transcription
                alignment_transcription = alignment_safe_transcript(canonical_transcription)
                if alignment_transcription != canonical_transcription:
                    segment["alignment_transcription"] = alignment_transcription
                inputs, word_lists = align_processor.prepare_forced_aligner_inputs(
                    audio=str(segment["path"]), transcript=alignment_transcription, language=segment["language"],
                )
                inputs = inputs.to(aligner.device, aligner.dtype)
                with torch.inference_mode():
                    output = aligner(**inputs)
                timestamps = align_processor.decode_forced_alignment(
                    logits=output.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                    timestamp_token_id=aligner.config.timestamp_token_id,
                )[0]
                segment["timestamps"] = _clean_model_output(timestamps)
                offset_seconds = int(segment["start_ms"]) / 1000.0
                for item in timestamps:
                    cleaned = _clean_model_output(item)
                    cleaned["start_time"] = round(float(cleaned.get("start_time", cleaned.get("start"))) + offset_seconds, 3)
                    cleaned["end_time"] = round(float(cleaned.get("end_time", cleaned.get("end"))) + offset_seconds, 3)
                    merged_timestamps.append(cleaned)
                del inputs, output
        finally:
            del aligner, align_processor
            gc.collect()
        timings["aligner_forward"] = round((time.perf_counter() - forward_started) * 1000)
        merged_text = "".join(segment.get("canonical_transcription", segment["raw_transcription"]) for segment in raw_segments)
        timestamp_adjustments: list[dict[str, Any]] = []
        rows = timestamps_to_sentences(merged_text, merged_timestamps, adjustments=timestamp_adjustments) if build_sentences else []
        raw = {
            "provider": self.name, "prompt": prompt, "segments": _clean_model_output(raw_segments),
            "merged_transcription": merged_text,
            "merged_timestamps": _clean_model_output(merged_timestamps),
            "timestamp_adjustments": timestamp_adjustments,
            "audio_segmentation": "official_qwen3_asr_lowest_100ms_energy_within_5s_no_overlap_no_gap",
            "merge_method": "official_chunk_text_concatenation_and_timestamp_offset_then_global_sentence_split",
            "asr_checkpoint_reused": checkpoint_reused,
        }
        return rows, raw, timings, punctuation_used

    def preflight(self, request: ASRRequest) -> dict[str, Any]:
        self._check()
        rate, frames, _ = _wav_info(request.audio)
        duration = frames / rate
        sample = min(float(self.config["asr"]["preflight"].get("sample_seconds", 60)), duration)
        starts = [0.0, max(0.0, duration / 2 - sample / 2), max(0.0, duration - sample)]
        unique = []
        for start in starts:
            pair = (round(start * rate), round(min(duration, start + sample) * rate))
            if pair not in unique:
                unique.append(pair)
        clips = self._segment_files(request, unique, "preflight")
        with _MemorySampler() as memory:
            started = time.perf_counter()
            rows, raw, timings, punctuation_used = self._run(
                request, clips, allow_checkpoint=False, build_sentences=False,
            )
            elapsed = time.perf_counter() - started
        sampled_seconds = sum((item[1] - item[0]) / rate for item in unique)
        fixed_ms = timings.get("asr_model_init", 0) + timings.get("aligner_model_init", 0)
        variable_seconds = max(0.0, elapsed - fixed_ms / 1000)
        estimate = fixed_ms / 1000 + variable_seconds * duration / max(sampled_seconds, 1.0)
        return {
            "provider": self.name, "samples": [{"start_ms": x["start_ms"], "end_ms": x["end_ms"]} for x in clips],
            "sampled_seconds": round(sampled_seconds, 3), "elapsed_seconds": round(elapsed, 3),
            "estimated_full_seconds": round(estimate, 1), "sentences": len(rows),
            "punctuation_fallback_used": punctuation_used, "timings_ms": timings,
            "memory": memory.report(), "raw": raw,
        }

    def transcribe(self, request: ASRRequest) -> ASRResult:
        self._check()
        ranges = silence_aware_ranges(
            request.audio, int(self.settings.get("chunk_seconds", 180)), int(self.settings.get("silence_search_seconds", 5)),
        )
        clips = self._segment_files(request, ranges, "full")
        with _MemorySampler() as memory:
            started = time.perf_counter()
            rows, raw, timings, punctuation_used = self._run(request, clips, allow_checkpoint=True)
            timings["total"] = round((time.perf_counter() - started) * 1000)
        raw["memory"] = memory.report()
        return ASRResult(
            sentences=rows, requested_provider=self.name, actual_provider=self.name,
            model_path=str(self.model_path), model_revision=_model_revision(self.model_path),
            timestamp_method="official_qwen3_chunk_align_offset_merge_then_global_sentence_ms",
            used_vad=False, used_punctuation=punctuation_used,
            hotwords_applied=list(request.hotwords), timings_ms=timings,
            raw=raw,
        )


def get_asr_provider(config: dict[str, Any], name: str | None = None):
    provider = str(name or config.get("asr", {}).get("primary", "funasr")).strip().lower()
    if provider == "funasr":
        return FunASRProvider(config)
    if provider == "qwen3_asr":
        return Qwen3ASRProvider(config)
    raise ValueError(f"未知ASR Provider：{provider}")
