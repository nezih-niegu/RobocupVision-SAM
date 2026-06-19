from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(
    config_path: Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open() as f:
        config = yaml.safe_load(f)

    output_override = output_dir or os.getenv("OUTPUT_DIR")
    if output_override:
        config["output_dir"] = str(output_override)

    rf = config.setdefault("roboflow", {})
    rf["api_url"] = os.getenv("ROBOFLOW_API_URL", rf.get("api_url", "https://serverless.roboflow.com"))
    rf["workspace"] = os.getenv("ROBOFLOW_WORKSPACE", rf.get("workspace", "hackatonfemsa"))
    rf["workflow_id"] = os.getenv("ROBOFLOW_WORKFLOW_ID", rf.get("workflow_id", "general-segmentation-api"))
    rf["api_key"] = os.getenv("ROBOFLOW_API_KEY", "")

    classes_env = os.getenv("ROBOFLOW_CLASSES")
    if classes_env:
        config["classes"] = [c.strip() for c in classes_env.split(",") if c.strip()]

    config["input_dir"] = PROJECT_ROOT / config.get("input_dir", "AllVideos")
    config["output_dir"] = PROJECT_ROOT / config.get("output_dir", "outputs_0.1s")
    config.setdefault("segmentation_mode", os.getenv("SEGMENTATION_MODE", "sam3_api"))
    config.setdefault("output_prob_thresh", 0.25)

    render_cfg = config.setdefault("annotation_render", {})
    render_cfg.setdefault("output_dir", "annotated")
    render_cfg.setdefault("mask_alpha_default", 0.45)
    render_cfg.setdefault("mask_alpha_by_class", {"playing field": 0.20})
    render_cfg.setdefault(
        "class_colors_bgr",
        {
            "playing field": [0, 180, 0],
            "allied robots": [255, 120, 0],
            "rival robots": [0, 0, 255],
            "ball": [0, 255, 255],
        },
    )
    render_cfg["output_path"] = config["output_dir"] / render_cfg["output_dir"]

    return config
