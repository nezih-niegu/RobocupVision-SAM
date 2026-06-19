from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CocoAnnotation:
    category_id: int
    category_name: str
    segmentation: list[float]
    bbox: list[float]
    score: float


@dataclass
class CocoKeyframe:
    timestamp_sec: float
    width: int
    height: int
    annotations: list[CocoAnnotation]


@dataclass
class CocoVideoAnnotations:
    video_stem: str
    categories: dict[int, str]
    keyframes: list[CocoKeyframe]
    timestamps: list[float]

    def keyframe_at(self, time_sec: float) -> CocoKeyframe | None:
        if not self.keyframes:
            return None
        idx = bisect_right(self.timestamps, time_sec) - 1
        if idx < 0:
            return self.keyframes[0]
        return self.keyframes[idx]


def load_coco_annotations(coco_path: Path) -> CocoVideoAnnotations:
    with coco_path.open() as f:
        data: dict[str, Any] = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[int, list[CocoAnnotation]] = {}

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = categories.get(category_id, "unknown")
        seg = ann.get("segmentation", [])
        polygon = seg[0] if seg and isinstance(seg[0], list) else seg
        if not polygon:
            continue
        anns_by_image.setdefault(image_id, []).append(
            CocoAnnotation(
                category_id=category_id,
                category_name=category_name,
                segmentation=[float(v) for v in polygon],
                bbox=[float(v) for v in ann.get("bbox", [0, 0, 0, 0])],
                score=float(ann.get("score", 0.0) or 0.0),
            )
        )

    keyframes: list[CocoKeyframe] = []
    for image in sorted(data.get("images", []), key=lambda img: float(img.get("timestamp_sec", 0))):
        image_id = image["id"]
        keyframes.append(
            CocoKeyframe(
                timestamp_sec=float(image.get("timestamp_sec", 0.0)),
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=anns_by_image.get(image_id, []),
            )
        )

    timestamps = [kf.timestamp_sec for kf in keyframes]
    return CocoVideoAnnotations(
        video_stem=coco_path.stem,
        categories=categories,
        keyframes=keyframes,
        timestamps=timestamps,
    )
