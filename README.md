# SamRoboSoccer — Team Classification & Field/Ball Detection Pipeline

This document summarizes the pipeline built so far for detecting robots, clustering them
into teams, and visualizing robots/ball/field on robot-soccer footage using SAM3 (video
segmentation) and DINOv2 (visual embeddings).

## Goal

Given a video of a robot soccer match (2 robots per team, 4 robots total, plus a ball),
automatically:

1. Detect and track each robot across the video using SAM3.
2. Determine which 2 robots belong to the same team, without relying on hardcoded team
   colors — using visual similarity instead.
3. Detect the ball.
4. Detect the playing field boundary.
5. Produce an overlay video that visually confirms all of the above (team-colored robot
   masks, a fixed-color ball mask, and a field outline).

## Why this approach instead of a simpler one

Two alternatives were considered and rejected early on:

- **VQA-style team classification with a vision-language model** (e.g. asking a model
  "is this robot on the blue or red team?"). Rejected because it's heavier per-crop than
  needed for a binary classification task, and slower for a real-time pipeline.
- **CLIP/SigLIP-style text-prompted classification** (e.g. embedding "a blue team robot"
  as a text anchor). Rejected in favor of **DINOv2**, because DINOv2 is trained purely on
  visual self-similarity (no text alignment), which makes it better suited to distinguishing
  two structurally-identical robot kits that differ mainly by paint/marker color — a
  fine-grained visual identity task, not a semantic category task.

## Pipeline Overview

```
Video file
   │
   ├─► SAM3 Session 1 — prompt: "robot"   → outputs_per_frame (per-frame masks + obj_ids)
   ├─► SAM3 Session 2 — prompt: "ball"    → ball_outputs_per_frame
   ├─► SAM3 Session 3 — prompt: "field"   → field_outputs_per_frame
   │
   ├─► cluster_main(video_frames, outputs_per_frame)
   │      1. Collect masked crops per robot obj_id across all frames it appears in
   │      2. Embed each crop with DINOv2 (facebook/dinov2-base)
   │      3. Average + L2-normalize embeddings per robot
   │      4. Cosine similarity matrix between all robots
   │      5. K-means (k=2) clustering → team_assignments: dict[obj_id] -> team_label
   │
   └─► save_team_ball_and_field_video(...)
          Renders one overlay video:
            - robot masks tinted by team_assignments (blue / red)
            - ball mask tinted a fixed color (orange)
            - field mask drawn as a white outline (not filled)
            - overlapping robot/ball masks flagged yellow (instead of blending
              into an ambiguous color)
```

## Why 3 separate SAM3 sessions, not 1

SAM3's text-prompted video grounding was only ever validated with a single prompt per
session. Stacking multiple prompts (`"robot"`, `"ball"`, `"field"`) in one session would
risk an unlabeled, flattened `out_obj_ids` list with no way to tell which detection came
from which prompt. Running 3 independent sessions guarantees:

- No risk of the ball or field accidentally being clustered or colored as a robot team.
- Each session's `obj_id`s are scoped to that session only — no cross-session ID collisions.

**Tradeoff:** ~3x total inference time (3 full propagation passes instead of 1).

### Important: sessions must run fully sequentially, not interleaved

Each `start_session` call loads the **entire video** onto the GPU as one tensor. Starting
a second session before fully finishing (propagating + closing) the first one means both
full-video tensors sit in GPU memory simultaneously — this caused a real `CUDA out of memory`
error during development. The fix: always run `start_session → add_prompt → propagate →
close_session` to completion before starting the next session.

## Key Implementation Details & Decisions

### Masked crops, not bounding-box crops
When extracting a robot's image for DINOv2 embedding, the SAM3 *mask* is used to blank out
all non-robot pixels (background field, other robots) before embedding — not just a
bounding-box crop. A plain bbox crop would let background field bleed into the embedding,
which is especially bad here since the field looks identical behind every robot and would
dilute the team-identity signal.

### Averaging embeddings per robot across multiple frames
Each robot gets multiple masked crops sampled across the frames it's tracked in (capped at
8 per robot, `MAX_CROPS_PER_ID`), embedded individually, then averaged and re-normalized.
This smooths out pose/lighting noise from any single frame.

### Detection lag is expected and handled
SAM3's text-prompted tracking takes roughly ~1 second (~30 frames) to lock onto a stable,
confident detection after a prompt is added. Frame 0 of a clip will often have zero
detections (`out_obj_ids` empty) — this is normal, not a bug. The clustering code scans
across all frames and buckets crops by whichever `obj_id`s eventually appear, rather than
assuming frame 0 has everything.

### ID-count sanity check doubles as a track-stability check
After collecting crops, the pipeline prints how many *unique* `obj_id`s were found. This
is a free diagnostic:
- Fewer than expected → some robot(s) never triggered a stable detection (occlusion, bad
  angle, or simply not present in that clip).
- More than expected → SAM3 is likely re-assigning new IDs when a robot leaves and
  re-enters frame (a track break), which would require ID-merging before clustering, not
  after.

### Overlap handling in the overlay video
Originally, when two robot masks overlapped, the renderer simply overwrote pixel-by-pixel
in loop order — whichever robot was processed last "won" the overlapping region. After
alpha-blending with the original frame, this produced a misleading purple-ish blend in
overlap zones (the new color tint mixing optically with the *other* robot's true pixel
color showing through). 

**Fix:** overlap regions are now explicitly detected (pixels claimed by more than one mask
in the same frame) and painted a distinct flag color (yellow) instead of silently
resolving to an ambiguous blended color. This applies to robot-robot overlaps and
robot-ball overlaps.

### Ball and field rendering
- **Ball:** single fixed color (orange), no clustering needed since there's only one ball.
  If multiple detections fire on one frame (shouldn't normally happen), only the
  highest-confidence one (by `out_probs`) is kept.
- **Field:** rendered as a **white outline** (via `cv2.findContours` + `cv2.drawContours`),
  not a filled region — filling green-on-green would be invisible, and any other fill color
  would visually bury the robot/ball overlay underneath it. Only the outer boundary
  (`cv2.RETR_EXTERNAL`) is drawn, so internal gaps (e.g. a robot standing on the field
  splitting the mask) don't produce stray extra outlines.

## Validation So Far

| Test | Robots detected | Result |
|---|---|---|
| 3-robot clip (1 robot never appeared in this clip) | 3/3 | Correct team split |
| 4-robot clip (`clip_4robots.mp4`, trimmed around the 2s mark where all 4 appear) | 4/4 | Correct team split (0,3 vs 1,2) |

**Caveat:** in the 4-robot test, the similarity margin between the correct same-team pairs
(0.639, 0.664) and the highest cross-team pair (0.721) was uncomfortably thin — the single
highest similarity value in the whole matrix wasn't even within the correct cluster.
K-means still chose the right split because it optimizes overall variance, not just the
single highest pair, but this margin is not comfortably wide. Two correct results is not
yet strong enough evidence that the whole-body DINOv2 embedding approach is robust across
different lighting/angles — more clips need to be tested.

## Open Questions / Next Steps

- **Run on more clips** to check whether the same-team similarity margin holds up
  consistently, or whether it was a lucky margin on the tested clips.
- **Consider tighter cropping** (e.g. cropping just the handle/marker region instead of
  the whole robot body) if the margin proves inconsistent — team identity is likely
  concentrated in a small visual feature (the colored handle), and averaging over the
  whole robot body may be diluting that signal with pose/shape noise that's identical
  across both teams.
- **GPU memory headroom**: full-length clips (e.g. ~2800 frames) reliably OOM on a 16GB
  GPU regardless of source resolution, since SAM3 resizes frames internally to a fixed
  size before allocating the GPU tensor (memory cost scales with frame count, not source
  resolution). Clips need to be trimmed to short windows (a few hundred frames) before
  running the pipeline.
- **Prompt wording matters**: SAM3's text grounding is sensitive to phrasing (e.g.
  `"robot with purple handle"` failed to detect anything, while plain `"robot"` worked).
  If `"field"`-style prompts don't grab a clean mask, alternate phrasings
  (`"green carpet"`, `"playing surface"`) are worth trying.

## File Reference

- `sam_plus_dino.py` (your script) — orchestrates the 3 SAM3 sessions, calls clustering
  and rendering.
- `cluster_robotsv2.py` — contains:
  - `main(video_frames, outputs_per_frame, n_teams=2)` — the DINOv2 embedding +
    clustering pipeline for robots.
  - `save_team_colored_video(...)` — robots only, team-colored, with overlap flagging.
  - `save_team_and_ball_video(...)` — robots + ball.
  - `save_team_ball_and_field_video(...)` — robots + ball + field outline (current
    full version).
