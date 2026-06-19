from __future__ import annotations


def frame_filename(timestamp_sec: float) -> str:
    """Unique JPEG name from timestamp (ms precision)."""
    return f"frame_{int(round(timestamp_sec * 1000)):08d}.jpg"
