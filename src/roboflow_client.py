from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image


class RoboflowSegmentationClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        workspace: str,
        workflow_id: str,
        classes: list[str],
        max_retries: int = 3,
        api_delay_sec: float = 0.5,
        segmentation_mode: str = "sam3_api",
        output_prob_thresh: float = 0.25,
    ) -> None:
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is required. Copy .env.example to .env and set your key.")
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.classes = classes
        self.max_retries = max_retries
        self.api_delay_sec = api_delay_sec
        self.segmentation_mode = segmentation_mode
        self.output_prob_thresh = output_prob_thresh
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    def _image_to_base64(self, image_rgb: np.ndarray) -> str:
        pil = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _segment_via_sam3_api(self, image_b64: str) -> dict[str, Any]:
        prompts = [{"type": "text", "text": class_name} for class_name in self.classes]
        return self.client.sam3_concept_segment(
            inference_input=image_b64,
            prompts=prompts,
            format="polygon",
            output_prob_thresh=self.output_prob_thresh,
        )

    def _segment_via_workflow(self, image_b64: str) -> Any:
        classes_str = ", ".join(self.classes)
        return self.client.run_workflow(
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            images={"image": image_b64},
            parameters={"classes": classes_str},
            use_cache=True,
        )

    def segment_frame(self, image_rgb: np.ndarray) -> Any:
        image_b64 = self._image_to_base64(image_rgb)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if self.segmentation_mode == "workflow":
                    result = self._segment_via_workflow(image_b64)
                else:
                    result = self._segment_via_sam3_api(image_b64)

                if self.api_delay_sec > 0:
                    time.sleep(self.api_delay_sec)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        raise RuntimeError(f"Roboflow API failed after {self.max_retries} attempts: {last_error}")
