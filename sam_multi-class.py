import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from sam3.model_builder import build_sam3_video_predictor

# =========================================================
# Confirmed against sam3_base_predictor.py:
# - add_prompt accepts obj_id (used to register independent
#   prompts/objects in the same session).
# - Each frame's `outputs` dict has parallel arrays:
#     out_obj_ids:     (N,)        int64  -> object id per slot
#     out_probs:       (N,)        float32-> confidence per slot
#     out_boxes_xywh:  (N, 4)      float32
#     out_binary_masks:(N, H, W)   bool
#   Masks are matched to a class via position in out_obj_ids,
#   not a per-record dict.
# =========================================================
MIN_PROB = 0.0  # raise this (e.g. 0.3) to filter out low-confidence detections

# ---- Config ----
video_path = "video-297_test.mp4"
frame_idx = 0
fps_override = None
alpha = 0.45
output_base_dir = "outputs"
os.makedirs(output_base_dir, exist_ok=True)

# Class definitions: label -> (obj_id, text prompt, BGR color for overlay)
CLASSES = {
    #"field":  (1, "playing field",                                  (60, 60, 60)),   # gray
    "ally":   (2, "robot with a green flat cover on top",           (0, 200, 0)),    # green
    "rival":  (3, "robot with two vertically stacked components",   (0, 0, 220)),    # red
    #"ball":   (4, "ball",                                           (0, 220, 220)),  # yellow
}


def load_video_frames(path):
    """Load RGB frames for visualization (separate from what the model consumes)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def extract_obj_mask(outputs, obj_id, min_prob=MIN_PROB):
    """
    Pull the mask for a given obj_id out of a frame's `outputs` dict,
    which has the form:
        {
          "out_obj_ids": int64[N],
          "out_probs": float32[N],
          "out_boxes_xywh": float32[N, 4],
          "out_binary_masks": bool[N, H, W],
          "frame_stats": {...},
        }
    Returns a 2D boolean numpy array, or None if obj_id isn't present
    this frame (or its confidence is below min_prob).
    """
    if not outputs:
        return None

    out_obj_ids = outputs.get("out_obj_ids")
    out_masks = outputs.get("out_binary_masks")
    out_probs = outputs.get("out_probs")

    if out_obj_ids is None or out_masks is None or len(out_obj_ids) == 0:
        return None

    matches = np.where(np.asarray(out_obj_ids) == obj_id)[0]
    if len(matches) == 0:
        return None

    idx = matches[0]
    if out_probs is not None and float(out_probs[idx]) < min_prob:
        return None

    return np.asarray(out_masks[idx]).astype(bool)


def add_all_prompts(predictor, session_id, frame_idx, classes):
    for label, (obj_id, text, _color) in classes.items():
        print(f"  Adding prompt for '{label}' (obj_id={obj_id}): \"{text}\"")
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                obj_id=obj_id,
                text=text,
            )
        )


def render_overlay_frame(frame_rgb, outputs, classes, alpha):
    """Composite all class masks onto a single frame with distinct colors,
    plus a small legend in the corner."""
    overlay = frame_rgb.copy()
    h, w = frame_rgb.shape[:2]

    for label, (obj_id, _text, color_bgr) in classes.items():
        mask = extract_obj_mask(outputs, obj_id)
        if mask is None or mask.sum() == 0:
            continue
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        color_layer = np.zeros_like(frame_rgb)
        color_layer[mask] = color_rgb
        overlay = np.where(
            mask[..., None],
            (overlay * (1 - alpha) + color_layer * alpha).astype(np.uint8),
            overlay,
        )

    # Legend
    legend_x, legend_y = 10, 10
    for i, (label, (_obj_id, _text, color_bgr)) in enumerate(classes.items()):
        y = legend_y + i * 22
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        cv2.rectangle(overlay, (legend_x, y), (legend_x + 16, y + 16), color_rgb, -1)
        cv2.putText(
            overlay, label, (legend_x + 22, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return overlay


def save_overlay_video(frames, outputs_per_frame, classes, out_path, fps, alpha):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for idx, frame_rgb in enumerate(frames):
        outputs = outputs_per_frame.get(idx)
        composed_rgb = render_overlay_frame(frame_rgb, outputs, classes, alpha)
        writer.write(cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    print("Loading SAM3 video predictor...")
    video_predictor = build_sam3_video_predictor()

    print(f"Loading frames from: {video_path}")
    video_frames = load_video_frames(video_path)
    print(f"Loaded {len(video_frames)} frames")

    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = fps_override or source_fps

    # ---- Start session ----
    response = video_predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    # ---- Add one prompt per class ----
    print("Adding class prompts...")
    add_all_prompts(video_predictor, session_id, frame_idx, CLASSES)

    # ---- Propagate across the full video ----
    print(f"Propagating {len(CLASSES)} classes across video (fps={fps})...")
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    print(f"Got outputs for {len(outputs_per_frame)} frames")

    # ---- Clean up session ----
    video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

    # ---- Save overlay video ----
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{video_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.mp4"

    save_overlay_video(video_frames, outputs_per_frame, CLASSES, output_path, fps, alpha)
    print(f"Saved multi-class masked video to: {output_path}")


if __name__ == "__main__":
    main()