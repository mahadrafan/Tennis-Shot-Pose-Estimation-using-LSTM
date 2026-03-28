"""
make_gif.py — Buat GIF animasi dari koordinat pose per kelas
=============================================================
Membaca tennis_dataset.csv dan menghasilkan 3 file GIF:
  forehand.gif / backhand.gif / serve.gif

Setiap GIF menampilkan satu contoh sekuens (30 frame) dari
kelas tersebut, dirender sebagai stick-figure animasi dengan
warna berbeda per kelas.

Cara pakai:
  python make_gif.py
  python make_gif.py --csv tennis_dataset.csv --fps 15 --scale 400
  python make_gif.py --sample 3   (ambil contoh ke-3 dari tiap kelas)

Output:
  forehand.gif
  backhand.gif
  serve.gif
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import argparse
import os

# ── Konfigurasi ──────────────────────────────────────────────────
SEQ_LENGTH = 30

# Koneksi skeleton (MediaPipe index)
CONNECTIONS = [
    # Tubuh atas
    (11, 12),  # bahu kiri - bahu kanan
    (11, 13),  # bahu kiri - siku kiri
    (13, 15),  # siku kiri - pergelangan kiri
    (12, 14),  # bahu kanan - siku kanan
    (14, 16),  # siku kanan - pergelangan kanan
    # Torso
    (11, 23),  # bahu kiri - pinggul kiri
    (12, 24),  # bahu kanan - pinggul kanan
    (23, 24),  # pinggul kiri - pinggul kanan
    # Kaki
    (23, 25),  # pinggul kiri - lutut kiri
    (24, 26),  # pinggul kanan - lutut kanan
    (25, 27),  # lutut kiri - pergelangan kaki kiri
    (26, 28),  # lutut kanan - pergelangan kaki kanan
]

# Landmark yang divisualisasikan
VISIBLE_JOINTS = set()
for a, b in CONNECTIONS:
    VISIBLE_JOINTS.add(a)
    VISIBLE_JOINTS.add(b)

# Warna per kelas
CLASS_CONFIG = {
    "forehand": {
        "color":      "#1D9E75",   # hijau teal
        "joint_color":"#0F6E56",
        "bg":         "#0a0a0a",
        "title_color":"#1D9E75",
    },
    "backhand": {
        "color":      "#534AB7",   # ungu
        "joint_color":"#3C3489",
        "bg":         "#0a0a0a",
        "title_color":"#7B75DD",
    },
    "serve": {
        "color":      "#1E8BBF",   # biru
        "joint_color":"#0C5F8A",
        "bg":         "#0a0a0a",
        "title_color":"#4BAFD9",
    },
}

# Lebar garis per koneksi (lengan lebih tebal)
ARM_CONNECTIONS = {(11,13),(13,15),(12,14),(14,16)}
SHOULDER_CONN   = {(11,12)}

def get_line_width(conn):
    if conn in ARM_CONNECTIONS:
        return 3.5
    if conn in SHOULDER_CONN:
        return 2.5
    return 2.0


def build_sequences(df, seq_length=30):
    """Ambil satu sekuens per kelas (non-overlapping blocks)."""
    labels   = df["label"].values
    features = df.drop("label", axis=1).values.astype(np.float32)
    seqs_by_class = {}

    i = 0
    while i + seq_length <= len(df):
        window_labels = labels[i : i + seq_length]
        if len(set(window_labels)) == 1:
            cls = window_labels[0]
            if cls not in seqs_by_class:
                seqs_by_class[cls] = []
            seqs_by_class[cls].append(features[i : i + seq_length])
            i += seq_length
        else:
            i += 1

    return seqs_by_class


def render_frame(ax, keypoints_flat, cfg, frame_num, total_frames, alpha=1.0):
    """Render satu frame pose ke axis matplotlib."""
    ax.clear()
    ax.set_facecolor(cfg["bg"])

    # Parse koordinat: keypoints_flat shape (99,) → x0,y0,z0,x1,y1,z1,...
    kp = keypoints_flat.reshape(33, 3)
    xs = kp[:, 0]
    ys = kp[:, 1]

    # Flip Y agar kaki di bawah (koordinat mediapipe: y=0 atas, y=1 bawah)
    ys_flipped = 1.0 - ys

    # Gambar koneksi skeleton
    for (a, b) in CONNECTIONS:
        lw = get_line_width((a, b))
        ax.plot(
            [xs[a], xs[b]],
            [ys_flipped[a], ys_flipped[b]],
            color=cfg["color"],
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # Gambar joint
    for j in VISIBLE_JOINTS:
        size = 60 if j in {15, 16} else 35  # pergelangan lebih besar
        ax.scatter(
            xs[j], ys_flipped[j],
            c=cfg["joint_color"],
            s=size,
            zorder=5,
            alpha=alpha,
            edgecolors=cfg["color"],
            linewidths=0.8,
        )

    # Progress bar bawah
    bar_y   = 0.03
    bar_h   = 0.018
    bar_x0  = 0.05
    bar_w   = 0.90
    prog    = (frame_num + 1) / total_frames

    ax.barh(bar_y, bar_w, height=bar_h, left=bar_x0,
            color="#222222", transform=ax.transAxes, zorder=10)
    ax.barh(bar_y, bar_w * prog, height=bar_h, left=bar_x0,
            color=cfg["color"], transform=ax.transAxes, zorder=11, alpha=0.85)

    # Frame counter
    ax.text(0.5, 0.005, f"Frame {frame_num+1}/{total_frames}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            color="#888888", fontsize=7, fontfamily="monospace")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def make_gif(class_name, sequences, sample_idx, fps, output_path, canvas_size):
    """Buat GIF animasi untuk satu kelas."""
    if not sequences:
        print(f"  [skip] tidak ada data untuk kelas: {class_name}")
        return

    idx    = min(sample_idx, len(sequences) - 1)
    seq    = sequences[idx]          # shape (30, 99)
    cfg    = CLASS_CONFIG[class_name]
    n_frames = len(seq)

    # Buat trailing ghost frames (efek motion trail)
    trail_len = 5

    fig, ax = plt.subplots(
        figsize=(canvas_size / 100, canvas_size / 100),
        facecolor=cfg["bg"]
    )
    fig.patch.set_facecolor(cfg["bg"])

    # Judul kelas
    label_text = ax.text(
        0.5, 0.96,
        class_name.upper(),
        transform=ax.transAxes,
        ha="center", va="top",
        color=cfg["title_color"],
        fontsize=14, fontweight="bold", fontfamily="monospace",
    )

    frames_data = []

    def animate(frame_num):
        render_frame(ax, seq[frame_num], cfg, frame_num, n_frames)

        # Ghost trail — gambar ulang frame sebelumnya dengan opacity turun
        for t in range(1, trail_len + 1):
            ghost_idx = frame_num - t
            if ghost_idx < 0:
                break
            alpha_ghost = 0.25 * (1 - t / (trail_len + 1))
            kp = seq[ghost_idx].reshape(33, 3)
            xs = kp[:, 0]
            ys = 1.0 - kp[:, 1]
            for (a, b) in CONNECTIONS:
                lw = get_line_width((a, b)) * 0.7
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                        color=cfg["color"], linewidth=lw,
                        alpha=alpha_ghost, solid_capstyle="round")

        # Pastikan label tetap ada setelah ax.clear()
        ax.text(
            0.5, 0.96, class_name.upper(),
            transform=ax.transAxes, ha="center", va="top",
            color=cfg["title_color"], fontsize=14,
            fontweight="bold", fontfamily="monospace",
        )
        return []

    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(
        fig, animate,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    ani.save(
        output_path,
        writer="pillow",
        fps=fps,
        dpi=100,
        savefig_kwargs={"facecolor": cfg["bg"]},
    )
    plt.close(fig)
    print(f"  Saved: {output_path}  ({n_frames} frames @ {fps}fps, sample #{idx})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pose animation GIF per class from tennis_dataset.csv"
    )
    parser.add_argument("--csv",    default="tennis_dataset.csv",
                        help="Path ke CSV dataset (default: tennis_dataset.csv)")
    parser.add_argument("--fps",    type=int,   default=12,
                        help="Frame per second GIF (default: 12)")
    parser.add_argument("--scale",  type=int,   default=380,
                        help="Ukuran canvas dalam pixel (default: 380)")
    parser.add_argument("--sample", type=int,   default=0,
                        help="Index contoh yang diambil per kelas (default: 0 = pertama)")
    parser.add_argument("--outdir", default=".",
                        help="Direktori output GIF (default: direktori saat ini)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: file tidak ditemukan: {args.csv}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Membaca dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Total frame: {len(df)}")
    print("Distribusi:", df["label"].value_counts().to_dict())

    print("\nMembangun sekuens per kelas...")
    seqs_by_class = build_sequences(df, SEQ_LENGTH)

    for cls, seqs in seqs_by_class.items():
        print(f"  {cls}: {len(seqs)} sekuens tersedia")

    print(f"\nMembuat GIF (fps={args.fps}, canvas={args.scale}px, sample=#{args.sample})...")
    for cls in ["forehand", "backhand", "serve"]:
        out = os.path.join(args.outdir, f"{cls}.gif")
        make_gif(
            class_name  = cls,
            sequences   = seqs_by_class.get(cls, []),
            sample_idx  = args.sample,
            fps         = args.fps,
            output_path = out,
            canvas_size = args.scale,
        )

    print("\nSelesai! File GIF yang dihasilkan:")
    for cls in ["forehand", "backhand", "serve"]:
        out = os.path.join(args.outdir, f"{cls}.gif")
        if os.path.exists(out):
            size_kb = os.path.getsize(out) // 1024
            print(f"  {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
