#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coco_exporter import merge_coco_files
from src.config_loader import load_config


def validate_coco_file(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    issues: list[str] = []
    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not categories:
        issues.append("missing categories")
    if not images:
        issues.append("no images")

    cat_ids = {c["id"] for c in categories}
    image_ids = {img["id"] for img in images}

    for ann in annotations:
        if ann.get("image_id") not in image_ids:
            issues.append(f"annotation {ann.get('id')} has invalid image_id")
        if ann.get("category_id") not in cat_ids:
            issues.append(f"annotation {ann.get('id')} has invalid category_id")
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list):
            issues.append(f"annotation {ann.get('id')} missing segmentation")

    class_counts = {c["name"]: 0 for c in categories}
    for ann in annotations:
        cat = next((c for c in categories if c["id"] == ann["category_id"]), None)
        if cat:
            class_counts[cat["name"]] += 1

    frames_without_ann = sum(
        1 for img in images if not any(a["image_id"] == img["id"] for a in annotations)
    )

    return {
        "file": str(path),
        "valid": len(issues) == 0,
        "issues": issues,
        "images": len(images),
        "annotations": len(annotations),
        "frames_without_annotations": frames_without_ann,
        "class_counts": class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate COCO outputs and optionally merge")
    parser.add_argument("--merge", action="store_true", help="Merge all per-video COCO files")
    parser.add_argument("--output-dir", type=Path, help="Override output directory (e.g. outputs_0.1s)")
    args = parser.parse_args()

    config = load_config(output_dir=args.output_dir)
    coco_dir = config["output_dir"] / "coco"
    if not coco_dir.exists():
        raise SystemExit(f"No COCO directory found at {coco_dir}")

    coco_files = sorted(
        p
        for p in coco_dir.glob("*.json")
        if not p.name.startswith("_") and not p.name.endswith("_test.json")
    )
    if not coco_files:
        raise SystemExit("No COCO JSON files found")

    report_path = config["output_dir"] / "validation_report.json"
    reports = [validate_coco_file(p) for p in coco_files]
    with report_path.open("w") as f:
        json.dump(reports, f, indent=2)

    valid = sum(1 for r in reports if r["valid"])
    print(f"Validated {len(reports)} files ({valid} valid, {len(reports) - valid} with issues)")
    print(f"Report: {report_path}")

    if args.merge:
        merge_path = coco_dir / "_merged_dataset.json"
        merge_coco_files(coco_files, merge_path)
        print(f"Merged dataset: {merge_path}")


if __name__ == "__main__":
    main()
