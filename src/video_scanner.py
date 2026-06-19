from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def scan_videos(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    videos = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos
