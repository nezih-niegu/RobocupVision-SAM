from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.coco_loader import CocoAnnotation, CocoKeyframe

# Draw order: field first, then robots, ball on top
DRAW_ORDER = ["playing field", "allied robots", "rival robots", "ball"]


def _scale_polygon(polygon: list[float], scale_x: float, scale_y: float) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    return pts.astype(np.int32)


def _alpha_for_class(class_name: str, render_config: dict[str, Any]) -> float:
    by_class = render_config.get("mask_alpha_by_class", {})
    if class_name in by_class:
        return float(by_class[class_name])
    return float(render_config.get("mask_alpha_default", 0.45))


def _color_for_class(class_name: str, render_config: dict[str, Any]) -> tuple[int, int, int]:
    colors = render_config.get("class_colors_bgr", {})
    color = colors.get(class_name, [200, 200, 200])
    return int(color[0]), int(color[1]), int(color[2])


def _sorted_annotations(annotations: list[CocoAnnotation]) -> list[CocoAnnotation]:
    order_map = {name: idx for idx, name in enumerate(DRAW_ORDER)}

    def sort_key(ann: CocoAnnotation) -> tuple[int, float]:
        return (order_map.get(ann.category_name, 99), -ann.score)

    return sorted(annotations, key=sort_key)


def render_frame_overlay(
    frame_bgr: np.ndarray,
    keyframe: CocoKeyframe,
    render_config: dict[str, Any],
) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    scale_x = frame_w / keyframe.width if keyframe.width else 1.0
    scale_y = frame_h / keyframe.height if keyframe.height else 1.0

    output = frame_bgr.copy()
    for ann in _sorted_annotations(keyframe.annotations):
        pts = _scale_polygon(ann.segmentation, scale_x, scale_y)
        if len(pts) < 3:
            continue

        color = _color_for_class(ann.category_name, render_config)
        alpha = _alpha_for_class(ann.category_name, render_config)

        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], color)
        output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

        x, y, w, h = [int(v) for v in ann.bbox]
        x = int(x * scale_x)
        y = int(y * scale_y)
        label = ann.category_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(text_h + 4, y)
        cv2.rectangle(
            output,
            (x, label_y - text_h - 4),
            (x + text_w + 4, label_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            label,
            (x + 2, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output
