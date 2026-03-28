"""
predict_video.py — Prediksi Pukulan Tenis (Real-Time Probability Bars)
=======================================================================
Mengganti shot counter dengan probabilitas berjalan (live bar chart)
yang menampilkan confidence setiap kelas secara real-time, mirip
tampilan di siaran TV tenis profesional.

Layout overlay:
  ┌─────────────────────────────────────────────────┐
  │  [progress bar atas]              [frame count] │
  │                                                 │
  │                  VIDEO FRAME                    │
  │                                                 │
  │  ┌──────────────────────────────────────────┐  │
  │  │ FOREHAND ████████████████████░░░░░  94%  │  │
  │  │ BACKHAND ████░░░░░░░░░░░░░░░░░░░░   4%  │  │
  │  │ SERVE    ░░░░░░░░░░░░░░░░░░░░░░░░   2%  │  │
  │  └──────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────┘

Cara pakai:
  python predict_video.py --video input.mp4
  python predict_video.py --video input.mp4 --save
  python predict_video.py --video input.mp4 --save --no-preview

Kontrol:
  SPACE = pause / resume
  Q     = keluar
"""

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

# ── Konfigurasi ──────────────────────────────────────────────────
POSE_MODEL    = "pose_landmarker.task"
LSTM_MODEL    = "tennis_model.h5"
LABELS_FILE   = "tennis_labels.json"
SEQ_LENGTH    = 30
CONF_THRESH   = 0.65          # threshold label aktif (warna menyala)
PREDICT_EVERY = 5             # jalankan LSTM setiap N frame
SMOOTH_WINDOW = 7             # rata-rata N prediksi terakhir untuk smoothing

# ── Warna BGR per kelas ───────────────────────────────────────────
COLORS_BGR = {
    "forehand": (29,  158, 117),   # hijau teal
    "backhand": (183,  74,  83),   # ungu kemerahan
    "serve":    (195, 139,  30),   # kuning emas
}
DEFAULT_COLOR  = (120, 120, 120)
BAR_BG_COLOR   = (40, 40, 40)
OVERLAY_BG     = (15, 15, 15)
TEXT_MUTED     = (140, 140, 140)
TEXT_BRIGHT    = (230, 230, 230)

# ── Koneksi skeleton ──────────────────────────────────────────────
CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
]


# ═══════════════════════════════════════════════════════════════════
# Pose utilities
# ═══════════════════════════════════════════════════════════════════

def extract_keypoints(lm):
    return np.array([[l.x, l.y, l.z] for l in lm]).flatten()


def draw_skeleton(frame, lm, color=(245, 66, 230)):
    h, w = frame.shape[:2]
    pts = {i: (int(l.x * w), int(l.y * h)) for i, l in enumerate(lm)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts.values():
        cv2.circle(frame, p, 5, (245, 117, 66), -1)


# ═══════════════════════════════════════════════════════════════════
# Prediction smoothing
# ═══════════════════════════════════════════════════════════════════

def smooth_probs(pred_history):
    """
    Rata-rata probabilitas dari riwayat prediksi terakhir.
    Returns: (avg_probs array, best_label_idx, best_conf)
    """
    if not pred_history:
        return None, None, 0.0
    avg  = np.mean(pred_history, axis=0)
    idx  = int(np.argmax(avg))
    conf = float(avg[idx])
    return avg, idx, conf


# ═══════════════════════════════════════════════════════════════════
# Overlay drawing
# ═══════════════════════════════════════════════════════════════════

def draw_probability_bars(frame, avg_probs, label_map, active_label, active_conf, buffer_len):
    """
    Gambar panel probabilitas berjalan di bagian bawah frame.
    Setiap kelas punya bar sendiri yang terus diperbarui.
    """
    h, w = frame.shape[:2]
    n_classes = len(label_map)

    # ── Dimensi panel ─────────────────────────────────────────────
    BAR_H       = 28          # tinggi setiap bar row
    BAR_PAD_V   = 10          # jarak vertikal antar baris
    LABEL_W     = 130         # lebar kolom label teks
    PCT_W       = 60          # lebar kolom persentase
    BAR_MARGIN  = 12          # margin kiri/kanan panel
    PANEL_PAD   = 10          # padding dalam panel

    panel_h = PANEL_PAD * 2 + n_classes * BAR_H + (n_classes - 1) * BAR_PAD_V + 24
    panel_y = h - panel_h - 8
    panel_x = 0
    panel_w = w

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, h - 4),
                  OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Garis atas panel
    top_lbl = active_label.upper() if (active_label and active_conf >= CONF_THRESH) else ("BUFFERING..." if buffer_len < SEQ_LENGTH else "-- IDLE --")
    top_color = COLORS_BGR.get(active_label, DEFAULT_COLOR) if (active_label and active_conf >= CONF_THRESH) else DEFAULT_COLOR
    cv2.line(frame, (0, panel_y), (w, panel_y), top_color, 2)

    # Label aktif (besar, di pojok kiri panel)
    cv2.putText(frame, top_lbl,
                (BAR_MARGIN, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, top_color, 2)

    # Hints kanan atas panel
    cv2.putText(frame, "SPACE=pause  Q=quit",
                (w - 190, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1)

    if avg_probs is None:
        # Belum ada prediksi → bar kosong
        for ci in range(n_classes):
            cls = label_map.get(ci, str(ci))
            row_y = panel_y + PANEL_PAD + 22 + ci * (BAR_H + BAR_PAD_V)
            cv2.putText(frame, cls.upper(),
                        (BAR_MARGIN, row_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1)
        return

    # ── Bar per kelas ─────────────────────────────────────────────
    bar_x_start = BAR_MARGIN + LABEL_W
    bar_x_end   = w - BAR_MARGIN - PCT_W - 10
    bar_max_w   = bar_x_end - bar_x_start

    for ci in range(n_classes):
        cls      = label_map.get(ci, str(ci))
        prob     = float(avg_probs[ci]) if avg_probs is not None else 0.0
        is_top   = (cls == active_label and active_conf >= CONF_THRESH)
        color    = COLORS_BGR.get(cls, DEFAULT_COLOR)
        row_y    = panel_y + PANEL_PAD + 22 + ci * (BAR_H + BAR_PAD_V)

        # Label kelas
        lbl_color = color if is_top else TEXT_MUTED
        cv2.putText(frame, cls.upper(),
                    (BAR_MARGIN, row_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lbl_color,
                    2 if is_top else 1)

        # Background bar
        cv2.rectangle(frame,
                      (bar_x_start, row_y + 2),
                      (bar_x_end,   row_y + BAR_H - 4),
                      BAR_BG_COLOR, -1)

        # Isi bar — lebar proporsional terhadap probabilitas
        fill_w = int(bar_max_w * prob)
        if fill_w > 0:
            bar_color = color if is_top else tuple(int(c * 0.55) for c in color)
            cv2.rectangle(frame,
                          (bar_x_start,              row_y + 2),
                          (bar_x_start + fill_w,     row_y + BAR_H - 4),
                          bar_color, -1)

            # Efek glow: garis terang tipis di tepi kanan bar (hanya kelas aktif)
            if is_top and fill_w > 4:
                cv2.rectangle(frame,
                              (bar_x_start + fill_w - 3, row_y + 2),
                              (bar_x_start + fill_w,     row_y + BAR_H - 4),
                              tuple(min(255, int(c * 1.4)) for c in color), -1)

        # Teks persentase
        pct_str = f"{prob * 100:.0f}%"
        pct_color = color if is_top else TEXT_MUTED
        cv2.putText(frame, pct_str,
                    (bar_x_end + 8, row_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, pct_color,
                    2 if is_top else 1)

        # Tick marks pada bar (25%, 50%, 75%)
        for tick_pct in [0.25, 0.50, 0.75]:
            tx = bar_x_start + int(bar_max_w * tick_pct)
            cv2.line(frame, (tx, row_y + BAR_H - 6), (tx, row_y + BAR_H - 4),
                     (70, 70, 70), 1)


def draw_top_bar(frame, frame_idx, total_frames, buffer_len, active_label, active_conf):
    """Progress bar dan info di bagian atas frame."""
    h, w = frame.shape[:2]

    # Progress bar
    progress = frame_idx / max(total_frames, 1)
    cv2.rectangle(frame, (0, 0), (w, 5), (30, 30, 30), -1)
    color = COLORS_BGR.get(active_label, (80, 80, 200)) if (active_label and active_conf >= CONF_THRESH) else (80, 80, 180)
    cv2.rectangle(frame, (0, 0), (int(w * progress), 5), color, -1)

    # Frame counter
    cv2.putText(frame, f"{frame_idx} / {total_frames}",
                (w - 140, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1)

    # Buffer indicator
    if buffer_len < SEQ_LENGTH:
        pct = int(buffer_len / SEQ_LENGTH * 100)
        cv2.putText(frame, f"BUFFERING {pct}%",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 160, 100), 1)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tennis Shot Prediction — Real-Time Probability Bars"
    )
    parser.add_argument("--video",         required=True,       help="Path ke file video input")
    parser.add_argument("--save",          action="store_true", help="Simpan output video")
    parser.add_argument("--no-preview",    action="store_true", help="Tidak tampilkan preview")
    parser.add_argument("--predict-every", type=int, default=PREDICT_EVERY,
                        help=f"Jalankan LSTM setiap N frame (default: {PREDICT_EVERY})")
    parser.add_argument("--smooth",        type=int, default=SMOOTH_WINDOW,
                        help=f"Window rata-rata probabilitas (default: {SMOOTH_WINDOW})")
    parser.add_argument("--conf",          type=float, default=CONF_THRESH,
                        help=f"Threshold confidence untuk label aktif (default: {CONF_THRESH})")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: file tidak ditemukan: {args.video}")
        return

    predict_every   = args.predict_every
    smooth_window   = args.smooth
    conf_threshold  = args.conf

    # ── Load model ────────────────────────────────────────────────
    print("Memuat model LSTM...")
    model = tf.keras.models.load_model(LSTM_MODEL)
    model(np.zeros((1, SEQ_LENGTH, 99)))   # warm-up
    print("Model siap.")

    with open(LABELS_FILE) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # Urutkan kelas secara konsisten (untuk urutan bar)
    sorted_labels = {i: label_map[i] for i in sorted(label_map.keys())}
    print(f"Kelas: {sorted_labels}")
    print(f"Prediksi setiap {predict_every} frame, smooth={smooth_window}")

    # ── Setup MediaPipe ───────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── Buka video ────────────────────────────────────────────────
    cap   = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video}  ({W}×{H}, {fps:.0f}fps, {total} frame)")

    # ── Output writer ─────────────────────────────────────────────
    writer = None
    if args.save:
        base, _ = os.path.splitext(args.video)
        out_path = base + "_predicted.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        print(f"Output: {out_path}")

    # ── State ─────────────────────────────────────────────────────
    sequence      = deque(maxlen=SEQ_LENGTH)
    pred_hist     = deque(maxlen=smooth_window)

    current_label = None
    current_conf  = 0.0
    current_probs = None          # array probabilitas semua kelas (sudah di-smooth)
    frame_idx     = 0
    paused        = False
    timeline      = []            # untuk ringkasan akhir & stats CSV

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

                    # Warna skeleton mengikuti kelas aktif
                    skel_color = COLORS_BGR.get(current_label, (245, 66, 230)) \
                                 if (current_label and current_conf >= conf_threshold) \
                                 else (245, 66, 230)
                    draw_skeleton(frame, lm, skel_color)

                    kp = extract_keypoints(lm)
                    sequence.append(kp)

                    # ── Jalankan LSTM setiap predict_every frame ───────────
                    if len(sequence) == SEQ_LENGTH and frame_idx % predict_every == 0:
                        X     = np.expand_dims(np.array(sequence), axis=0)
                        probs = model.predict(X, verbose=0)[0]   # shape (n_classes,)
                        pred_hist.append(probs)

                    # ── Smooth probabilitas ────────────────────────────────
                    if pred_hist:
                        avg_probs, best_idx, best_conf = smooth_probs(list(pred_hist))
                        current_probs = avg_probs
                        current_label = sorted_labels.get(best_idx)
                        current_conf  = best_conf

                        # Log untuk statistik akhir
                        if frame_idx % predict_every == 0 and current_label:
                            timeline.append((frame_idx, current_label, round(best_conf, 3)))

                if frame_idx % 150 == 0:
                    print(f"Progress: {frame_idx}/{total} "
                          f"({frame_idx/total*100:.0f}%)"
                          + (f"  [{current_label} {current_conf*100:.0f}%]"
                             if current_label else ""), end="\r")

            # ── Overlay ───────────────────────────────────────────
            draw_top_bar(frame, frame_idx, total, len(sequence),
                         current_label, current_conf)
            draw_probability_bars(frame, current_probs, sorted_labels,
                                  current_label, current_conf, len(sequence))

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

    # ── Ringkasan ──────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("RINGKASAN PREDIKSI VIDEO")
    print("=" * 60)
    print(f"Total frame diproses : {frame_idx}")

    if timeline:
        import pandas as pd
        df = pd.DataFrame(timeline, columns=["frame", "label", "confidence"])

        print("\nDistribusi prediksi per kelas:")
        print(df["label"].value_counts().to_string())

        print("\nRata-rata confidence per kelas:")
        print(df.groupby("label")["confidence"].mean().round(3).to_string())

        # Simpan stats CSV
        stats_path = os.path.splitext(args.video)[0] + "_stats.csv"
        df.to_csv(stats_path, index=False, encoding="utf-8")
        print(f"\nTimeline disimpan : {stats_path}")

    if args.save:
        print(f"Video output      : {out_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
