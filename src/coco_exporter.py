from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.response_parser import ParsedDetection


class CocoExporter:
    def __init__(self, classes: list[str], video_stem: str) -> None:
        self.video_stem = video_stem
        self.categories = [
            {"id": idx + 1, "name": name, "supercategory": "robocup"}
            for idx, name in enumerate(classes)
        ]
        self.class_to_id = {name: idx + 1 for idx, name in enumerate(classes)}
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._next_image_id = 1
        self._next_ann_id = 1

    def add_frame(
        self,
        file_name: str,
        width: int,
        height: int,
        frame_index: int,
        timestamp_sec: float,
        detections: list[ParsedDetection],
    ) -> int:
        image_id = self._next_image_id
        self._next_image_id += 1

        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
            }
        )

        for det in detections:
            category_id = self.class_to_id.get(det.class_name)
            if category_id is None:
                continue
            self.annotations.append(
                {
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [det.segmentation],
                    "bbox": det.bbox,
                    "area": det.area,
                    "iscrowd": 0,
                    "score": det.confidence,
                }
            )
            self._next_ann_id += 1

        return image_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": {
                "description": f"SAM3 segmentation for {self.video_stem}",
                "version": "1.0",
            },
            "licenses": [],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def merge_coco_files(coco_paths: list[Path], output_path: Path) -> dict[str, Any]:
    if not coco_paths:
        raise ValueError("No COCO files to merge")

    merged: dict[str, Any] | None = None
    next_image_id = 1
    next_ann_id = 1

    for path in sorted(coco_paths):
        with path.open() as f:
            data = json.load(f)
        if merged is None:
            merged = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
                "images": [],
                "annotations": [],
            }

        image_id_map: dict[int, int] = {}
        for image in data.get("images", []):
            old_id = image["id"]
            new_id = next_image_id
            next_image_id += 1
            image_id_map[old_id] = new_id
            new_image = dict(image)
            new_image["id"] = new_id
            merged["images"].append(new_image)

        for ann in data.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = image_id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)

    assert merged is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(merged, f, indent=2)
    return merged
