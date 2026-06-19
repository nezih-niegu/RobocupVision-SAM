from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

ProgressStatus = Literal["ok", "error"]


class ProgressTracker:
    def __init__(self, progress_path: Path) -> None:
        self.progress_path = progress_path
        self._completed: set[tuple[str, float]] = set()
        self._load()

    def _load(self) -> None:
        if not self.progress_path.exists():
            return
        with self.progress_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") == "ok":
                    self._completed.add((record["video_path"], float(record["timestamp_sec"])))

    def is_done(self, video_path: Path, timestamp_sec: float) -> bool:
        return (str(video_path), round(timestamp_sec, 3)) in self._completed

    def mark(self, video_path: Path, timestamp_sec: float, status: ProgressStatus, detail: str = "") -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "video_path": str(video_path),
            "timestamp_sec": round(timestamp_sec, 3),
            "status": status,
            "detail": detail,
        }
        with self.progress_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if status == "ok":
            self._completed.add((str(video_path), round(timestamp_sec, 3)))
