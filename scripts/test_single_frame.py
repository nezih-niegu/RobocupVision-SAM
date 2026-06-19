#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.frame_extractor import extract_frames_at_interval, save_frame_jpeg
from src.frame_naming import frame_filename
from src.roboflow_client import RoboflowSegmentationClient
from src.response_parser import parse_workflow_response
from src.coco_exporter import CocoExporter


def draw_preview(image_rgb, detections, output_path: Path) -> None:
    import numpy as np

    bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for det in detections:
        pts = [
            (int(det.segmentation[i]), int(det.segmentation[i + 1]))
            for i in range(0, len(det.segmentation), 2)
        ]
        if len(pts) >= 3:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(bgr, [pts_np], True, (0, 255, 0), 2)
        x, y, _, _ = [int(v) for v in det.bbox]
        cv2.putText(
            bgr,
            det.class_name,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAM3 segmentation on a single video frame")
    parser.add_argument("--video", type=Path, help="Path to video file")
    parser.add_argument("--timestamp", type=float, default=0.0, help="Timestamp in seconds")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    config = load_config(output_dir=args.output_dir)
    output_dir = config["output_dir"]

    if args.video:
        video_path = args.video
    else:
        candidates = list(config["input_dir"].rglob("video-988_singular_display.mov"))
        if not candidates:
            candidates = list(config["input_dir"].rglob("*.mov")) + list(config["input_dir"].rglob("*.mp4"))
        if not candidates:
            raise SystemExit("No videos found under AllVideos/")
        video_path = candidates[0]

    print(f"Using video: {video_path}")

    frames = extract_frames_at_interval(
        video_path,
        interval_sec=config["frame_interval_sec"],
        max_image_side=config.get("max_image_side", 1280),
    )
    if not frames:
        raise SystemExit("Could not extract any frames")

    target = min(frames, key=lambda f: abs(f.timestamp_sec - args.timestamp))
    image = target.image
    h, w = image.shape[:2]

    client = RoboflowSegmentationClient(
        api_url=config["roboflow"]["api_url"],
        api_key=config["roboflow"]["api_key"],
        workspace=config["roboflow"]["workspace"],
        workflow_id=config["roboflow"]["workflow_id"],
        classes=config["classes"],
        max_retries=config.get("max_retries", 3),
        api_delay_sec=0,
        segmentation_mode=config.get("segmentation_mode", "sam3_api"),
        output_prob_thresh=config.get("output_prob_thresh", 0.25),
    )

    print("Calling Roboflow API...")
    raw_response = client.segment_frame(image)

    raw_path = output_dir / "test_raw_response.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as f:
        json.dump(raw_response, f, indent=2, default=str)
    print(f"Raw response saved to: {raw_path}")

    detections = parse_workflow_response(
        raw_response, w, h, config["classes"], config.get("min_annotation_area", 100)
    )
    print(f"Parsed {len(detections)} detections:")
    for det in detections:
        print(f"  - {det.class_name} (conf={det.confidence:.3f}, area={det.area:.0f})")

    frame_name = frame_filename(target.timestamp_sec)
    frames_dir = output_dir / "frames" / video_path.stem
    frame_path = frames_dir / frame_name
    save_frame_jpeg(image, frame_path)

    exporter = CocoExporter(config["classes"], video_path.stem)
    exporter.add_frame(frame_name, w, h, target.frame_index, target.timestamp_sec, detections)
    coco_path = output_dir / "coco" / f"{video_path.stem}_test.json"
    exporter.save(coco_path)
    print(f"COCO test saved to: {coco_path}")

    preview_path = output_dir / "annotated" / f"{video_path.stem}_preview.jpg"
    draw_preview(image, detections, preview_path)
    print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()
