"""
Takes SAM3 per-frame outputs (out_obj_ids, out_binary_masks) + the raw video frames,
extracts masked crops per tracked object ID across all frames it appears in,
embeds them with DINOv2, averages embeddings per ID, and clusters into 2 teams.

Assumes you already have:
    video_frames      -> list of RGB frames (np.uint8, HxWx3), from load_video_frames()
    outputs_per_frame  -> dict[frame_idx] = {'out_obj_ids':..., 'out_binary_masks':..., ...}

Run this AFTER your existing sam_demo_script.py pipeline, reusing those two variables,
or adapt the bottom __main__ block to reload them from disk if you saved them.
"""

import numpy as np
import cv2
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans
from collections import defaultdict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "facebook/dinov2-base"

MAX_CROPS_PER_ID = 8          # cap how many frames we sample per robot, for speed
MIN_MASK_AREA_PX = 400        # skip tiny/noisy masks (sliver detections)


def load_dino():
    print(f"Loading {DINO_MODEL_NAME} on {DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE).eval()
    return processor, model


def mask_to_crop(frame_rgb, mask_bool):
    """
    Given a full-frame RGB image and a full-frame boolean mask,
    return a tight crop with background blanked to white (DINOv2 normalizes
    so white vs black background doesn't matter much, but pick one consistently).
    Returns None if mask is empty/too small.
    """
    ys, xs = np.where(mask_bool)
    if len(ys) < MIN_MASK_AREA_PX:
        return None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = frame_rgb[y0:y1, x0:x1].copy()
    mask_crop = mask_bool[y0:y1, x0:x1]

    # blank out everything outside the mask within the crop
    crop[~mask_crop] = 255

    return crop


@torch.no_grad()
def embed_crops(crops, processor, model, batch_size=16):
    """Embed a list of RGB numpy crops with DINOv2, return (N, D) numpy array."""
    if len(crops) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    all_embeds = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        # CLS token pooled output, shape (B, D)
        pooled = outputs.last_hidden_state[:, 0, :]
        all_embeds.append(pooled.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)


def collect_crops_per_id(video_frames, outputs_per_frame):
    """
    Walk every frame's outputs, bucket masked crops by obj_id.
    Returns dict[obj_id] -> list of RGB crop arrays.
    """
    crops_by_id = defaultdict(list)

    for frame_idx, out in outputs_per_frame.items():
        obj_ids = out["out_obj_ids"]
        masks = out["out_binary_masks"]

        if len(obj_ids) == 0:
            continue

        frame_rgb = video_frames[frame_idx]

        for obj_id, mask in zip(obj_ids, masks):
            if len(crops_by_id[obj_id]) >= MAX_CROPS_PER_ID:
                continue
            crop = mask_to_crop(frame_rgb, mask)
            if crop is not None:
                crops_by_id[obj_id].append(crop)

    return crops_by_id


def main(video_frames, outputs_per_frame, n_teams=2):
    crops_by_id = collect_crops_per_id(video_frames, outputs_per_frame)

    unique_ids = sorted(crops_by_id.keys())
    print(f"\nFound {len(unique_ids)} unique tracked object IDs: {unique_ids}")
    for oid in unique_ids:
        print(f"  id={oid}: {len(crops_by_id[oid])} usable crops collected")

    if len(unique_ids) != 4:
        print(
            f"\n⚠️  Expected 4 robots but found {len(unique_ids)} unique IDs.\n"
            "   If this is MORE than 4: SAM3 is likely re-assigning new IDs when a robot\n"
            "   leaves/re-enters frame (track breaks), and these will need to be merged\n"
            "   by similarity before team clustering, not after.\n"
            "   If LESS than 4: some robots may never have triggered a confident detection\n"
            "   (check if the missing ones are partially occluded or off-prompt)."
        )

    processor, model = load_dino()

    avg_embeddings = []
    valid_ids = []
    for oid in unique_ids:
        crops = crops_by_id[oid]
        if len(crops) == 0:
            continue
        embeds = embed_crops(crops, processor, model)
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)  # L2 normalize
        avg_embed = embeds.mean(axis=0)
        avg_embed = avg_embed / np.linalg.norm(avg_embed)  # re-normalize after averaging
        avg_embeddings.append(avg_embed)
        valid_ids.append(oid)

    avg_embeddings = np.stack(avg_embeddings, axis=0)

    # cosine similarity matrix
    sim_matrix = avg_embeddings @ avg_embeddings.T
    print("\nCosine similarity matrix (rows/cols = obj_ids in order below):")
    print(valid_ids)
    np.set_printoptions(precision=3, suppress=True)
    print(sim_matrix)

    # cluster into n_teams groups
    kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=0)
    labels = kmeans.fit_predict(avg_embeddings)

    print("\nTeam assignment:")
    for oid, label in zip(valid_ids, labels):
        print(f"  obj_id={oid}  ->  team {label}")

    return dict(zip(valid_ids, labels)), sim_matrix, valid_ids


# BGR colors (since this writes via cv2/ffmpeg pipelines that expect BGR for output)
TEAM_COLORS_BGR = {
    0: (255, 80, 80),    # team 0 -> blue-ish
    1: (80, 80, 255),    # team 1 -> red-ish
}
UNASSIGNED_COLOR_BGR = (200, 200, 200)  # gray, for any obj_id not in team_assignments
OVERLAP_COLOR_BGR = (0, 255, 255)       # yellow, flags two masks overlapping in this frame


def save_team_colored_video(video_frames, outputs_per_frame, team_assignments,
                             out_path, fps=30.0, alpha=0.5):
    """
    Writes an overlay video where each detected robot's mask is tinted by its
    team color (from team_assignments: dict[obj_id] -> team_label), rather than
    SAM3's default per-object-id coloring. Lets you visually sanity-check the
    clustering result on every frame instead of just trusting the printed labels.

    Pixels claimed by more than one mask in the same frame (robots overlapping/
    occluding each other) are painted a distinct overlap color instead of being
    silently overwritten by whichever object happened to be processed last --
    that overwrite-then-alpha-blend used to produce ambiguous purple-ish colors
    in overlap regions, since the 50% blend exposed the *other* robot's true
    pixel color mixing optically with the new tint.
    """
    if len(video_frames) == 0:
        raise ValueError("video_frames is empty")

    h, w = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        for frame_idx, frame_rgb in enumerate(video_frames):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()

            out = outputs_per_frame.get(frame_idx)
            if out is not None and len(out["out_obj_ids"]) > 0:
                overlay = frame_bgr.copy()
                claimed = np.zeros((h, w), dtype=bool)   # pixels already colored this frame
                overlap = np.zeros((h, w), dtype=bool)    # pixels claimed by >1 mask

                for obj_id, mask in zip(out["out_obj_ids"], out["out_binary_masks"]):
                    team = team_assignments.get(obj_id)
                    color = TEAM_COLORS_BGR.get(team, UNASSIGNED_COLOR_BGR)

                    new_overlap = mask & claimed
                    overlap |= new_overlap

                    fresh = mask & ~claimed
                    overlay[fresh] = color
                    claimed |= mask

                overlay[overlap] = OVERLAP_COLOR_BGR
                frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"Saved team-colored overlay video to: {out_path}")


BALL_COLOR_BGR = (0, 165, 255)  # orange, fixed -- no clustering needed for a single ball


def save_team_and_ball_video(video_frames, robot_outputs_per_frame, ball_outputs_per_frame,
                              team_assignments, out_path, fps=30.0, alpha=0.5):
    """
    Like save_team_colored_video, but also overlays a ball mask from a SEPARATE
    SAM3 session (ball_outputs_per_frame), painted a fixed color since there's
    only one ball and no clustering involved.

    robot_outputs_per_frame and ball_outputs_per_frame come from two independent
    propagate_in_video() sessions (one prompted "robot", one prompted "ball").
    Their obj_ids are NOT compared against each other -- each dict is scoped to
    its own session, so there's no risk of a ball obj_id colliding with a robot
    obj_id and getting accidentally team-colored.

    Overlap detection covers robot-robot AND robot-ball overlaps (e.g. ball
    occluded by a robot's foot) -- any pixel claimed by more than one mask in
    the frame, regardless of source, gets flagged with OVERLAP_COLOR_BGR.
    """
    if len(video_frames) == 0:
        raise ValueError("video_frames is empty")

    h, w = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        for frame_idx, frame_rgb in enumerate(video_frames):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
            overlay = frame_bgr.copy()
            claimed = np.zeros((h, w), dtype=bool)
            overlap = np.zeros((h, w), dtype=bool)

            robot_out = robot_outputs_per_frame.get(frame_idx)
            if robot_out is not None and len(robot_out["out_obj_ids"]) > 0:
                for obj_id, mask in zip(robot_out["out_obj_ids"], robot_out["out_binary_masks"]):
                    team = team_assignments.get(obj_id)
                    color = TEAM_COLORS_BGR.get(team, UNASSIGNED_COLOR_BGR)

                    new_overlap = mask & claimed
                    overlap |= new_overlap
                    fresh = mask & ~claimed
                    overlay[fresh] = color
                    claimed |= mask

            ball_out = ball_outputs_per_frame.get(frame_idx)
            if ball_out is not None and len(ball_out["out_obj_ids"]) > 0:
                # if multiple ball-prompt detections fire on one frame, just take
                # the highest-confidence one -- there should only be one real ball
                probs = ball_out["out_probs"]
                best_idx = int(np.argmax(probs)) if len(probs) > 0 else None
                if best_idx is not None:
                    mask = ball_out["out_binary_masks"][best_idx]

                    new_overlap = mask & claimed
                    overlap |= new_overlap
                    fresh = mask & ~claimed
                    overlay[fresh] = BALL_COLOR_BGR
                    claimed |= mask

            overlay[overlap] = OVERLAP_COLOR_BGR
            frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"Saved team+ball overlay video to: {out_path}")


FIELD_OUTLINE_COLOR_BGR = (255, 255, 255)  # white outline
FIELD_OUTLINE_THICKNESS = 2


def save_team_ball_and_field_video(video_frames, robot_outputs_per_frame, ball_outputs_per_frame,
                                    field_outputs_per_frame, team_assignments, out_path,
                                    fps=30.0, alpha=0.5):
    """
    Same as save_team_and_ball_video, plus a field boundary traced as an outline
    (not a filled region -- filling the whole field green-on-green would be
    invisible, and any other fill color would bury the robot/ball overlay under
    one giant block of color). field_outputs_per_frame comes from a THIRD
    independent SAM3 session (prompted e.g. "field" or "green playing field"),
    fully separate from the robot and ball sessions -- same reasoning as before:
    no shared obj_id space, no risk of cross-contamination between detections.

    If multiple field-prompt detections fire on one frame (shouldn't happen for
    a single field, but just in case), only the highest-confidence one is drawn,
    same pattern as the ball selection logic.
    """
    if len(video_frames) == 0:
        raise ValueError("video_frames is empty")

    h, w = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    try:
        for frame_idx, frame_rgb in enumerate(video_frames):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
            overlay = frame_bgr.copy()
            claimed = np.zeros((h, w), dtype=bool)
            overlap = np.zeros((h, w), dtype=bool)

            robot_out = robot_outputs_per_frame.get(frame_idx)
            if robot_out is not None and len(robot_out["out_obj_ids"]) > 0:
                for obj_id, mask in zip(robot_out["out_obj_ids"], robot_out["out_binary_masks"]):
                    team = team_assignments.get(obj_id)
                    color = TEAM_COLORS_BGR.get(team, UNASSIGNED_COLOR_BGR)

                    new_overlap = mask & claimed
                    overlap |= new_overlap
                    fresh = mask & ~claimed
                    overlay[fresh] = color
                    claimed |= mask

            ball_out = ball_outputs_per_frame.get(frame_idx)
            if ball_out is not None and len(ball_out["out_obj_ids"]) > 0:
                probs = ball_out["out_probs"]
                best_idx = int(np.argmax(probs)) if len(probs) > 0 else None
                if best_idx is not None:
                    mask = ball_out["out_binary_masks"][best_idx]

                    new_overlap = mask & claimed
                    overlap |= new_overlap
                    fresh = mask & ~claimed
                    overlay[fresh] = BALL_COLOR_BGR
                    claimed |= mask

            overlay[overlap] = OVERLAP_COLOR_BGR
            frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            # field outline drawn AFTER the alpha blend, directly onto the final
            # frame at full opacity -- it's a thin contour line, not a filled
            # region, so it doesn't need to participate in the overlap logic above
            field_out = field_outputs_per_frame.get(frame_idx)
            if field_out is not None and len(field_out["out_obj_ids"]) > 0:
                probs = field_out["out_probs"]
                best_idx = int(np.argmax(probs)) if len(probs) > 0 else None
                if best_idx is not None:
                    field_mask = field_out["out_binary_masks"][best_idx]
                    contours, _ = cv2.findContours(
                        field_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(frame_bgr, contours, -1, FIELD_OUTLINE_COLOR_BGR,
                                      FIELD_OUTLINE_THICKNESS)

            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"Saved team+ball+field overlay video to: {out_path}")


if __name__ == "__main__":
    # Adapt this block: reuse video_frames / outputs_per_frame from your
    # existing sam_demo_script.py run, or load them from disk if you pickled them.
    raise SystemExit(
        "Import main(video_frames, outputs_per_frame) into your existing script "
        "after propagate_in_video() instead of running this file standalone."
    )