"""
This a manual annotation. press key on your keyboard to annotate the following class:
  F     = annotate FOREHAND  
  B     = annotate BACKHAND
  S     = annotate SERVE
  D     = Undo
  SPACE = pause / resume
  [ / ] = back/forward 10 frame
  Q     = exit and save
it will take 30 frames (including the annotated frame) before the annotated frame.
  
how to use:
  python extract.py --video forehand.mp4

Tips:
  - annotate at the exact moment racket touch the ball.
  - Minimum of 200+ annotation per class for better result.
  - can annotate different class in 1 video. 
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import argparse
import os
from collections import deque

MODEL_PATH  = "pose_landmarker.task"
OUTPUT_CSV  = "tennis_dataset.csv"
SEQ_LENGTH  = 30    

LABEL_KEYS = {
    ord('f'): "forehand",
    ord('b'): "backhand",
    ord('s'): "serve",
}

COLORS_BGR = {
    "forehand": (29, 158, 117),
    "backhand": (83,  74, 183),
    "serve":    (30, 139, 195),
}

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
]

def extract_keypoints(lm):
    return np.array([[l.x, l.y, l.z] for l in lm]).flatten()

def draw_skeleton(frame, lm):
    h, w = frame.shape[:2]
    pts = {i: (int(l.x*w), int(l.y*h)) for i, l in enumerate(lm)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (245, 66, 230), 2)
    for p in pts.values():
        cv2.circle(frame, p, 4, (245, 117, 66), -1)

def draw_ui(frame, frame_idx, total, annotations, last_flash, paused, buffer_ready):
    h, w = frame.shape[:2]

    progress = frame_idx / max(total, 1)
    cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (int(w * progress), 28), (60, 100, 180), -1)
    cv2.putText(frame, f"Frame {frame_idx}/{total}  {'|| PAUSED' if paused else ''}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    ctrl_x = w - 250
    cv2.rectangle(frame, (ctrl_x, 0), (w, 110), (20, 20, 20), -1)
    hints = [
        "F = forehand",
        "B = backhand",
        "S = serve",
        "D = undo",
        "SPACE = pause  [ ] = seek",
        "Q = exit & save",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (ctrl_x + 8, 18 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    counts = {}
    for _, lbl, _ in annotations:
        counts[lbl] = counts.get(lbl, 0) + 1

    panel_h = 20 + 22 * (len(LABEL_KEYS) + 1)
    cv2.rectangle(frame, (0, h - panel_h), (220, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Saved Annotations: {len(annotations)}",
                (8, h - panel_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    for i, lbl in enumerate(["forehand", "backhand", "serve"]):
        c = COLORS_BGR[lbl]
        cv2.putText(frame, f"  {lbl:<12} {counts.get(lbl, 0):>3}x",
                    (8, h - panel_h + 36 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    buf_color = (80, 200, 80) if buffer_ready else (80, 80, 80)
    cv2.putText(frame, "BUF READY" if buffer_ready else "BUFFERING...",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, buf_color, 1)

    if last_flash:
        lbl, remaining = last_flash
        c = COLORS_BGR.get(lbl, (200, 200, 200))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), c, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, f"+ {lbl.upper()} ANNOTATED",
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the file video")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: file not found: {args.video}")
        return

    print("loading video to the  memory...")
    cap_tmp = cv2.VideoCapture(args.video)
    fps   = cap_tmp.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    while True:
        ret, f = cap_tmp.read()
        if not ret:
            break
        all_frames.append(f)
    cap_tmp.release()
    total = len(all_frames)
    print(f"Video loaded: {total} frame @ {fps:.0f}fps")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmark_cache = {}

    annotations = []
    flash_counter = 0  
    frame_idx    = 0
    paused       = False
    monotonic_ts = 0   

    skeleton_cache = {}  

    print("\nVideo opened. Press F/B/S in the impact moment.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if not paused and frame_idx < total - 1:
                frame_idx += 1

            frame = all_frames[frame_idx].copy()

            if frame_idx not in landmark_cache:
                monotonic_ts += 33   
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, monotonic_ts)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    landmark_cache[frame_idx] = extract_keypoints(lm)
                    skeleton_cache[frame_idx] = lm
                else:
                    landmark_cache[frame_idx] = None
                    skeleton_cache[frame_idx] = None

            if skeleton_cache.get(frame_idx) is not None:
                draw_skeleton(frame, skeleton_cache[frame_idx])

            seq_start = frame_idx - SEQ_LENGTH + 1
            buffer_ready = (
                seq_start >= 0 and
                all(landmark_cache.get(i) is not None for i in range(seq_start, frame_idx + 1))
            )

            last_flash = None
            if flash_counter > 0 and annotations:
                last_flash = (annotations[-1][1], flash_counter)
                flash_counter -= 1

            draw_ui(frame, frame_idx, total, annotations, last_flash, paused, buffer_ready)

            cv2.imshow("Tennis Annotator", frame)
            key = cv2.waitKey(20 if not paused else 50) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                paused = not paused

            elif key == ord('['):
                frame_idx = max(0, frame_idx - 10)
                paused = True

            elif key == ord(']'):
                frame_idx = min(total - 1, frame_idx + 10)
                paused = True

            elif key == ord('d'):
                if annotations:
                    removed = annotations.pop()
                    print(f"  Undo: {removed[1]} in frame {removed[0]}")
                else:
                    print("No annotation can be undone.")

            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]

                seq_start = frame_idx - SEQ_LENGTH + 1
                if seq_start < 0:
                    print(f"  [skip] not enough previous frame (need {SEQ_LENGTH}, available {frame_idx+1})")
                    continue

                seq_kps = [landmark_cache.get(i) for i in range(seq_start, frame_idx + 1)]
                missing = sum(1 for kp in seq_kps if kp is None)

                if missing > SEQ_LENGTH * 0.3:  
                    print(f"  [skip] to many frames without pose ({missing}/{SEQ_LENGTH}). try in another moment")
                    continue

                filled = []
                last_valid = None
                for kp in seq_kps:
                    if kp is not None:
                        last_valid = kp
                    filled.append(last_valid if last_valid is not None else np.zeros(99))

                annotations.append((frame_idx, label, filled))
                flash_counter = 8
                print(f"  [{frame_idx:>5}] {label.upper()} — total: {len(annotations)}")
                paused = True  

    cv2.destroyAllWindows()

    if not annotations:
        print("No annotations. exit to save.")
        return

    print(f"\nTotal annotations: {len(annotations)}")
    print("saving to CSV...")

    cols = []
    for i in range(33):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols.append("label")

    rows = []
    for _, label, seq_kps in annotations:
        for kp in seq_kps:
            rows.append(list(kp) + [label])

    df_new = pd.DataFrame(rows, columns=cols)

    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDataset saved: {OUTPUT_CSV}")
    print(f"Total frame in CSV: {len(df_combined)}")
    print("Label Distribution:")
    print(df_combined["label"].value_counts().to_string())

    counts = {}
    for _, lbl, _ in annotations:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"\nAnnotation summary: ")
    for lbl, cnt in counts.items():
        print(f"  {lbl:<12} : {cnt} annotation ({cnt * SEQ_LENGTH} frame)")

if __name__ == "__main__":
    main()
