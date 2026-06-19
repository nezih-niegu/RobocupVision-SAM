from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ParsedDetection:
    class_name: str
    confidence: float
    segmentation: list[float]
    bbox: list[float]
    area: float


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().rstrip(".")


def _bbox_from_polygon(polygon: list[float]) -> list[float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _polygon_area(polygon: list[float]) -> float:
    if len(polygon) < 6:
        return 0.0
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    return float(cv2.contourArea(pts))


def _mask_to_polygon(mask: np.ndarray) -> list[float] | None:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1.0:
        return None
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    flat = approx.reshape(-1, 2).astype(float).flatten().tolist()
    return flat if len(flat) >= 6 else None


def _decode_rle_counts(rle: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    counts = rle.get("counts")
    size = rle.get("size", [height, width])
    if not counts:
        return None
    h, w = int(size[0]), int(size[1])
    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    if isinstance(counts, list):
        for count in counts:
            mask[idx : idx + int(count)] = val
            idx += int(count)
            val = 1 - val
        return mask.reshape((h, w), order="F")
    return None


def _prediction_to_polygon(pred: dict[str, Any], image_h: int, image_w: int) -> list[float] | None:
    if "points" in pred and pred["points"]:
        points = pred["points"]
        if isinstance(points, list) and points and isinstance(points[0], dict):
            flat = []
            for pt in points:
                flat.extend([float(pt["x"]), float(pt["y"])])
            return flat if len(flat) >= 6 else None
        if isinstance(points, list) and points and isinstance(points[0], (list, tuple)):
            flat = [float(v) for pair in points for v in pair]
            return flat if len(flat) >= 6 else None

    if "segmentation" in pred:
        seg = pred["segmentation"]
        if isinstance(seg, list) and seg:
            if all(isinstance(v, (int, float)) for v in seg):
                return [float(v) for v in seg]
            if isinstance(seg[0], list):
                flat = [float(v) for sub in seg for v in sub]
                return flat if len(flat) >= 6 else None

    if "mask" in pred and isinstance(pred["mask"], list):
        mask_arr = np.array(pred["mask"], dtype=np.uint8)
        return _mask_to_polygon(mask_arr)

    if "rle" in pred and isinstance(pred["rle"], dict):
        mask_arr = _decode_rle_counts(pred["rle"], image_h, image_w)
        if mask_arr is not None:
            return _mask_to_polygon(mask_arr)

    x = pred.get("x")
    y = pred.get("y")
    w = pred.get("width")
    h = pred.get("height")
    if None not in (x, y, w, h):
        cx, cy = float(x), float(y)
        hw, hh = float(w) / 2.0, float(h) / 2.0
        return [
            cx - hw, cy - hh,
            cx + hw, cy - hh,
            cx + hw, cy + hh,
            cx - hw, cy + hh,
        ]

    bbox = pred.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        bx, by, bw, bh = [float(v) for v in bbox]
        return [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]

    return None


def _iter_predictions(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if any(k in node for k in ("class", "class_name", "label", "category")) and any(
                k in node for k in ("x", "width", "points", "segmentation", "mask", "bbox")
            ):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _class_from_prediction(pred: dict[str, Any]) -> str:
    for key in ("class", "class_name", "label", "category"):
        if key in pred and pred[key]:
            return str(pred[key])
    return "unknown"


def _polygon_from_point_pairs(points: list[Any]) -> list[float] | None:
    if not points:
        return None
    if isinstance(points[0], (list, tuple)) and len(points[0]) == 2:
        flat = [float(v) for pair in points for v in pair]
        return flat if len(flat) >= 6 else None
    return None


def _parse_sam3_concept_response(
    response: dict[str, Any],
    known_classes: list[str] | None = None,
    min_area: float = 100.0,
) -> list[ParsedDetection]:
    detections: list[ParsedDetection] = []
    known_normalized = {_normalize_class_name(c): c for c in (known_classes or [])}

    for prompt_result in response.get("prompt_results", []):
        echo = prompt_result.get("echo", {})
        raw_class = str(echo.get("text", "unknown"))
        norm = _normalize_class_name(raw_class)
        class_name = known_normalized.get(norm, raw_class)

        for pred in prompt_result.get("predictions", []):
            confidence = float(pred.get("confidence", 0.0) or 0.0)
            masks = pred.get("masks", [])
            if not isinstance(masks, list):
                continue

            for mask in masks:
                polygon = _polygon_from_point_pairs(mask)
                if not polygon:
                    continue
                bbox = _bbox_from_polygon(polygon)
                area = _polygon_area(polygon)
                if area < min_area:
                    continue
                detections.append(
                    ParsedDetection(
                        class_name=class_name,
                        confidence=confidence,
                        segmentation=polygon,
                        bbox=bbox,
                        area=area,
                    )
                )

    return detections


def parse_workflow_response(
    response: Any,
    image_width: int,
    image_height: int,
    known_classes: list[str] | None = None,
    min_area: float = 100.0,
) -> list[ParsedDetection]:
    if isinstance(response, dict) and "prompt_results" in response:
        return _parse_sam3_concept_response(response, known_classes, min_area)

    predictions = _iter_predictions(response)
    detections: list[ParsedDetection] = []
    known_normalized = {_normalize_class_name(c): c for c in (known_classes or [])}

    for pred in predictions:
        polygon = _prediction_to_polygon(pred, image_height, image_width)
        if not polygon:
            continue

        raw_class = _class_from_prediction(pred)
        norm = _normalize_class_name(raw_class)
        class_name = known_normalized.get(norm, raw_class)

        confidence = float(pred.get("confidence", pred.get("score", 0.0)) or 0.0)
        bbox = _bbox_from_polygon(polygon)
        area = _polygon_area(polygon)
        if area < min_area:
            continue
        detections.append(
            ParsedDetection(
                class_name=class_name,
                confidence=confidence,
                segmentation=polygon,
                bbox=bbox,
                area=area,
            )
        )

    return detections
