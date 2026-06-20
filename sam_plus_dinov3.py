import os
import cv2
from pathlib import Path
from datetime import datetime
import json

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import save_masklet_video
from cluster_robotsv2 import main as cluster_main, save_team_ball_and_field_video, save_team_colored_video

# ---- Config ----
video_path = "clip_4robots.mp4"#"video-297_test.mp4"  # confirm this path # video-357_singular_display.mov <- Salen los 4.
prompt = "robot" # TEAM 2"robot with purple handle" # TEAM 1: "robot with purple handle"
frame_idx = 0
fps_override = None  # leave None to use the source video's fps
alpha = 0.5  # mask overlay transparency
output_base_dir = "outputs"

os.makedirs(output_base_dir, exist_ok=True)


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


# ---- Load model ----
print("Loading SAM3 video predictor...")
video_predictor = build_sam3_video_predictor()

# ---- Load frames + fps for visualization ----
print(f"Loading frames from: {video_path}")
video_frames = load_video_frames(video_path)
print(f"Loaded {len(video_frames)} frames")

cap = cv2.VideoCapture(video_path)
source_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
fps = fps_override or source_fps

# ---- Session 1: robot detection (full lifecycle) ----
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = response["session_id"]

video_predictor.handle_request(
    request=dict(type="add_prompt", session_id=session_id,
                 frame_index=frame_idx, text=prompt)
)

print(f"Propagating '{prompt}' across video (fps={fps})...")
outputs_per_frame = propagate_in_video(video_predictor, session_id)
print(f"Got masks for {len(outputs_per_frame)} frames")

video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

# ---- Session 2: ball detection (full lifecycle, starts only after session 1 is closed) ----
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
ball_session_id = response["session_id"]

video_predictor.handle_request(
    request=dict(type="add_prompt", session_id=ball_session_id,
                 frame_index=frame_idx, text="ball")
)

print("Propagating 'ball' across video...")
ball_outputs_per_frame = propagate_in_video(video_predictor, ball_session_id)
print(f"Got ball masks for {len(ball_outputs_per_frame)} frames")

video_predictor.handle_request(request=dict(type="close_session", session_id=ball_session_id))

# ---- Session 3: field detection (full lifecycle, after both prior sessions are closed) ----
response = video_predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
field_session_id = response["session_id"]

video_predictor.handle_request(
    request=dict(type="add_prompt", session_id=field_session_id,
                 frame_index=frame_idx, text="green area")
)

print("Propagating 'field' across video...")
field_outputs_per_frame = propagate_in_video(video_predictor, field_session_id)
print(f"Got field masks for {len(field_outputs_per_frame)} frames")

video_predictor.handle_request(request=dict(type="close_session", session_id=field_session_id))


# ---- Set up output dir (moved up so both overlay videos can use it) ----
video_name = Path(video_path).stem
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(output_base_dir) / f"{video_name}_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Cluster detected robots into 2 teams ----
team_assignments, sim_matrix, valid_ids = cluster_main(video_frames, outputs_per_frame, n_teams=2)

# ---- Save team + ball overlay ----
team_ball_field_path = output_dir / "teams_ball_field.mp4"
save_team_ball_and_field_video(video_frames, outputs_per_frame, ball_outputs_per_frame,
                                field_outputs_per_frame, team_assignments,
                                out_path=team_ball_field_path, fps=fps, alpha=0.5)

# ---- Save team-colored overlay (robots only, no ball) ----
team_video_path = output_dir / "teams_colored.mp4"
save_team_colored_video(video_frames, outputs_per_frame, team_assignments,
                         out_path=team_video_path, fps=fps, alpha=0.5)

# ---- Save original SAM3 overlay video (robots only, default coloring) ----
output_path = output_dir / "result2.mp4"
save_masklet_video(
    video_frames,
    outputs_per_frame,
    out_path=str(output_path),
    alpha=alpha,
    fps=fps,
)
print(f"Saved masked video to: {output_path}")