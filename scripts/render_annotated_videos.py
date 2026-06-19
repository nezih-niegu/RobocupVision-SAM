#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coco_loader import load_coco_annotations
from src.config_loader import load_config
from src.mask_renderer import render_frame_overlay
from src.video_scanner import scan_videos


def build_video_index(input_dir: Path) -> dict[str, Path]:
    return {video.stem: video for video in scan_videos(input_dir)}


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open video writer for {output_path}")


def render_video(
    video_path: Path,
    coco_path: Path,
    output_path: Path,
    render_config: dict,
) -> dict:
    coco = load_coco_annotations(coco_path)
    if not coco.keyframes:
        raise RuntimeError(f"No keyframes in COCO file: {coco_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = create_video_writer(output_path, fps, width, height)
    frames_written = 0
    keyframes_used: set[float] = set()
    frame_idx = 0

    with tqdm(total=frame_count or None, desc=video_path.name, leave=False) as pbar:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            time_sec = frame_idx / fps if fps > 0 else 0.0
            keyframe = coco.keyframe_at(time_sec)
            if keyframe is not None:
                keyframes_used.add(keyframe.timestamp_sec)
                frame = render_frame_overlay(frame, keyframe, render_config)

            writer.write(frame)
            frames_written += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    return {
        "video": str(video_path),
        "coco": str(coco_path),
        "output": str(output_path),
        "frames_written": frames_written,
        "keyframes_used": len(keyframes_used),
        "keyframes_total": len(coco.keyframes),
        "errors": "",
    }


def discover_jobs(
    config: dict,
    video_arg: Path | None,
    limit: int,
    skip_existing: bool,
) -> list[tuple[Path, Path, Path]]:
    output_dir = config["annotation_render"]["output_path"]
    video_index = build_video_index(config["input_dir"])
    coco_dir = config["output_dir"] / "coco"

    if video_arg:
        stem = video_arg.stem
        coco_path = coco_dir / f"{stem}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"COCO file not found: {coco_path}")
        output_path = output_dir / f"{stem}.mp4"
        return [(video_arg, coco_path, output_path)]

    coco_files = sorted(
        p
        for p in coco_dir.glob("*.json")
        if not p.name.startswith("_") and not p.name.endswith("_test.json")
    )

    jobs: list[tuple[Path, Path, Path]] = []
    for coco_path in coco_files:
        video_path = video_index.get(coco_path.stem)
        if video_path is None:
            continue
        output_path = output_dir / f"{coco_path.stem}.mp4"
        if skip_existing and output_path.exists():
            continue
        jobs.append((video_path, coco_path, output_path))
        if limit > 0 and len(jobs) >= limit:
            break

    return jobs


def write_summary(rows: list[dict], summary_path: Path) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotated videos with COCO mask overlays")
    parser.add_argument("--video", type=Path, help="Render a single source video")
    parser.add_argument("--limit", type=int, default=0, help="Process only N videos (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with existing MP4")
    parser.add_argument("--output-dir", type=Path, help="Override output directory (e.g. outputs_0.1s)")
    args = parser.parse_args()

    config = load_config(output_dir=args.output_dir)
    render_config = config["annotation_render"]
    jobs = discover_jobs(config, args.video, args.limit, args.skip_existing)

    if not jobs:
        print("No videos to render.")
        return

    print(f"Rendering {len(jobs)} video(s)")
    summary_rows: list[dict] = []

    for video_path, coco_path, output_path in tqdm(jobs, desc="Videos"):
        try:
            stats = render_video(video_path, coco_path, output_path, render_config)
            summary_rows.append(stats)
            print(f"Saved: {output_path}")
        except Exception as exc:
            summary_rows.append(
                {
                    "video": str(video_path),
                    "coco": str(coco_path),
                    "output": str(output_path),
                    "frames_written": 0,
                    "keyframes_used": 0,
                    "keyframes_total": 0,
                    "errors": str(exc),
                }
            )

    summary_path = config["output_dir"] / "render_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
