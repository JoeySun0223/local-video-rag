"""FFmpeg operations used by the transcription stage."""

from __future__ import annotations

import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def extract_audio(ffmpeg: Path, video: Path, output_wav: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(output_wav),
        ],
        check=True,
    )


def duration_ms(ffmpeg: Path, video: Path) -> int:
    ffprobe = ffmpeg.with_name("ffprobe.exe")
    if not ffprobe.is_file():
        return 0
    result = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return round(float(result.stdout.strip()) * 1000)


def video_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"视频路径不存在：{path}")
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)


def title_from_filename(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        possible_title, possible_hash = stem.rsplit("__", 1)
        if len(possible_hash) == 16 and all(character in "0123456789abcdef" for character in possible_hash.lower()):
            return possible_title
    return stem
