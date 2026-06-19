#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.frame_extractor import extract_frames_at_interval, save_frame_jpeg
from src.frame_naming import frame_filename
from src.roboflow_client import RoboflowSegmentationClient
from src.response_parser import parse_workflow_response
from src.coco_exporter import CocoExporter
from src.progress_tracker import ProgressTracker
from src.video_scanner import scan_videos


def process_video(
    video_path: Path,
    config: dict,
    client: RoboflowSegmentationClient,
    progress: ProgressTracker,
) -> dict:
    classes = config["classes"]
    output_dir = config["output_dir"]
    interval = config["frame_interval_sec"]
    max_side = config.get("max_image_side", 1280)

    frames = extract_frames_at_interval(video_path, interval, max_side)
    exporter = CocoExporter(classes, video_path.stem)
    frames_dir = output_dir / "frames" / video_path.stem

    stats = {
        "video": str(video_path),
        "frames_total": len(frames),
        "frames_processed": 0,
        "frames_skipped": 0,
        "frames_error": 0,
        "annotations": 0,
        **{f"ann_{c.replace(' ', '_')}": 0 for c in classes},
        "errors": "",
    }

    for frame in tqdm(frames, desc=video_path.name, leave=False):
        if progress.is_done(video_path, frame.timestamp_sec):
            stats["frames_skipped"] += 1
            continue

        frame_name = frame_filename(frame.timestamp_sec)
        h, w = frame.image.shape[:2]

        try:
            raw = client.segment_frame(frame.image)
            detections = parse_workflow_response(
                raw, w, h, classes, config.get("min_annotation_area", 100)
            )
            save_frame_jpeg(frame.image, frames_dir / frame_name)
            exporter.add_frame(
                frame_name, w, h, frame.frame_index, frame.timestamp_sec, detections
            )
            progress.mark(video_path, frame.timestamp_sec, "ok")
            stats["frames_processed"] += 1
            stats["annotations"] += len(detections)
            for det in detections:
                key = f"ann_{det.class_name.replace(' ', '_')}"
                if key in stats:
                    stats[key] += 1
        except Exception as exc:
            progress.mark(video_path, frame.timestamp_sec, "error", str(exc))
            stats["frames_error"] += 1
            stats["errors"] = str(exc)

    coco_path = output_dir / "coco" / f"{video_path.stem}.json"
    if stats["frames_processed"] > 0 or not coco_path.exists():
        exporter.save(coco_path)
    return stats


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
    parser = argparse.ArgumentParser(description="Segment all videos via Roboflow SAM3")
    parser.add_argument("--limit", type=int, default=0, help="Process only N videos (0 = all)")
    parser.add_argument("--video", type=Path, help="Process a single video path")
    parser.add_argument("--output-dir", type=Path, help="Override output directory (e.g. outputs_0.1s)")
    args = parser.parse_args()

    config = load_config(output_dir=args.output_dir)
    output_dir = config["output_dir"]
    progress = ProgressTracker(output_dir / "progress.jsonl")

    client = RoboflowSegmentationClient(
        api_url=config["roboflow"]["api_url"],
        api_key=config["roboflow"]["api_key"],
        workspace=config["roboflow"]["workspace"],
        workflow_id=config["roboflow"]["workflow_id"],
        classes=config["classes"],
        max_retries=config.get("max_retries", 3),
        api_delay_sec=config.get("api_delay_sec", 0.5),
        segmentation_mode=config.get("segmentation_mode", "sam3_api"),
        output_prob_thresh=config.get("output_prob_thresh", 0.25),
    )

    if args.video:
        videos = [args.video]
    else:
        videos = scan_videos(config["input_dir"])
        if args.limit > 0:
            videos = videos[: args.limit]

    print(f"Found {len(videos)} video(s) to process")
    summary_rows: list[dict] = []

    for video_path in tqdm(videos, desc="Videos"):
        if not video_path.exists():
            summary_rows.append({"video": str(video_path), "errors": "file not found"})
            continue
        stats = process_video(video_path, config, client, progress)
        summary_rows.append(stats)

    summary_path = output_dir / "summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
