import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import tensorflow as tf
import json
import argparse
import os
from collections import deque

POSE_MODEL    = "pose_landmarker.task"
LSTM_MODEL    = "tennis_model.h5"
LABELS_FILE   = "tennis_labels.json"
SEQ_LENGTH    = 30
CONF_THRESH   = 0.65
PREDICT_EVERY = 5   

COLORS_BGR = {
    "forehand": (29, 158, 117),
    "backhand": (83,  74, 183),
    "serve":    (30, 139, 195),
}
DEFAULT_COLOR = (120, 120, 120)

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
]

def extract_keypoints(lm):
    return np.array([[l.x, l.y, l.z] for l in lm]).flatten()

def draw_skeleton(frame, lm, color=(245, 66, 230)):
    h, w = frame.shape[:2]
    pts = {i: (int(l.x*w), int(l.y*h)) for i, l in enumerate(lm)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts.values():
        cv2.circle(frame, p, 5, (245, 117, 66), -1)

def smooth_predictions(pred_history):
    if not pred_history:
        return None, 0.0
    avg = np.mean(pred_history, axis=0)
    idx = np.argmax(avg)
    return idx, float(avg[idx])

def draw_overlay(frame, label, conf, counters, frame_idx, total_frames, buffer_len):
    h, w = frame.shape[:2]
    is_active = label is not None and conf >= CONF_THRESH
    color = COLORS_BGR.get(label, DEFAULT_COLOR) if is_active else DEFAULT_COLOR

    panel_h = 34 + 28 * len(counters)
    cv2.rectangle(frame, (0, 0), (230, panel_h), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 0), (230, panel_h), (60, 60, 60), 1)
    cv2.putText(frame, "SHOT COUNTER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    for i, (shot, count) in enumerate(counters.items()):
        y = 46 + i * 28
        c = COLORS_BGR.get(shot, DEFAULT_COLOR)
        if shot == label and is_active:
            cv2.rectangle(frame, (4, y-18), (226, y+8), c, -1)
            txt_col = (255, 255, 255)
        else:
            txt_col = c
        cv2.putText(frame, f"{shot.upper():<12} {count:>3}x",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 1)

    progress = frame_idx / max(total_frames, 1)
    bar_w = 200
    bar_x = w - bar_w - 10
    cv2.rectangle(frame, (bar_x, 8), (bar_x + bar_w, 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, 8), (bar_x + int(bar_w * progress), 20), (100, 100, 200), -1)
    cv2.putText(frame, f"{frame_idx}/{total_frames}",
                (bar_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    buf_fill = int((buffer_len / SEQ_LENGTH) * 230)
    cv2.rectangle(frame, (0, h - 5), (230, h), (40, 40, 40), -1)
    ready_color = (80, 200, 80) if buffer_len == SEQ_LENGTH else (80, 80, 80)
    cv2.rectangle(frame, (0, h - 5), (buf_fill, h), ready_color, -1)

    cv2.rectangle(frame, (0, h - 60), (w, h - 6), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, h - 60), (w, h - 6), color, 1)

    if is_active:
        display  = label.upper()
        conf_str = f"{conf * 100:.0f}%"
    elif buffer_len < SEQ_LENGTH:
        display  = "-- BUFFERING --"
        conf_str = ""
    else:
        display  = "-- IDLE --"
        conf_str = ""

    cv2.putText(frame, display,  (12, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, conf_str, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if is_active:
        bw = int(conf * (w - 210))
        cv2.rectangle(frame, (205, h - 26), (w - 8, h - 14), (50, 50, 50), -1)
        cv2.rectangle(frame, (205, h - 26), (205 + bw, h - 14), color, -1)

    cv2.putText(frame, "SPACE=pause  Q=quit", (w - 185, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       required=True,      help="Path to the file video input")
    parser.add_argument("--save",        action="store_true", help="save output video")
    parser.add_argument("--no-preview",  action="store_true", help="no preview")
    parser.add_argument("--predict-every", type=int, default=PREDICT_EVERY,
                        help=f"run LSTM every N frame (default: {PREDICT_EVERY})")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: file not found: {args.video}")
        return

    predict_every = args.predict_every

    print("Loading LSTM model...")
    model = tf.keras.models.load_model(LSTM_MODEL)
    model(np.zeros((1, SEQ_LENGTH, 99)))  

    with open(LABELS_FILE) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    print(f"class: {label_map}")
    print(f"predict every {predict_every} frame")

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap   = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video}  ({W}x{H}, {fps:.0f}fps, {total} frames)")

    writer = None
    if args.save:
        base, _ = os.path.splitext(args.video)
        out_path = base + "_predicted.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        print(f"Output: {out_path}")

    sequence      = deque(maxlen=SEQ_LENGTH)
    pred_hist     = deque(maxlen=5)
    counters      = {v: 0 for v in label_map.values()}
    prev_label    = None
    hold_count    = 0
    HOLD_THRESH   = 8

    current_label = None
    current_conf  = 0.0
    frame_idx     = 0
    paused        = False
    timeline      = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

            ts_ms = int(frame_idx * 1000 / fps)

            if not paused:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, ts_ms)

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    skel_color = COLORS_BGR.get(current_label, (245, 66, 230)) if current_label else (245, 66, 230)
                    draw_skeleton(frame, lm, skel_color)

                    kp = extract_keypoints(lm)
                    sequence.append(kp)

                    if len(sequence) == SEQ_LENGTH and frame_idx % predict_every == 0:
                        X     = np.expand_dims(np.array(sequence), axis=0)
                        probs = model.predict(X, verbose=0)[0]
                        pred_hist.append(probs)

                        idx, conf = smooth_predictions(list(pred_hist))
                        if idx is not None:
                            current_label = label_map[idx]
                            current_conf  = conf
                            timeline.append((frame_idx, current_label, round(conf, 3)))

                            if current_label == prev_label and conf >= CONF_THRESH:
                                hold_count += 1
                                if hold_count == HOLD_THRESH:
                                    counters[current_label] += 1
                                    print(f"  [{frame_idx:>5}/{total}] {current_label.upper()} "
                                          f"#{counters[current_label]}  ({conf*100:.0f}%)")
                            else:
                                hold_count = 0
                            prev_label = current_label

                if frame_idx % 100 == 0:
                    print(f"Progress: {frame_idx}/{total} ({frame_idx/total*100:.0f}%)", end="\r")

            draw_overlay(frame, current_label, current_conf,
                         counters, frame_idx, total, len(sequence))

            if writer:
                writer.write(frame)

            if not args.no_preview:
                cv2.imshow("Tennis Prediction", frame)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n\n" + "=" * 60)
    print("Video Prediction Summary")
    print("=" * 60)
    print(f"Total frame processed : {frame_idx}")
    print("Total shot detected:")
    for shot, count in counters.items():
        print(f"  {shot:<12} : {count}x")

    if timeline:
        import pandas as pd
        df = pd.DataFrame(timeline, columns=["frame", "label", "confidence"])
        print("\nprediction distribution:")
        print(df["label"].value_counts().to_string())
        print("\nmean confidence per class:")
        print(df.groupby("label")["confidence"].mean().round(3).to_string())

        stats_path = os.path.splitext(args.video)[0] + "_stats.csv"
        df.to_csv(stats_path, index=False, encoding="utf-8")
        print(f"\nTimeline saved: {stats_path}")

    if args.save:
        print(f"Video output      : {out_path}")

if __name__ == "__main__":
    main()
