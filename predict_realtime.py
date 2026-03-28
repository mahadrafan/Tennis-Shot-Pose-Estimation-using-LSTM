import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import tensorflow as tf
import json
import threading
import time
from collections import deque

POSE_MODEL   = "pose_landmarker.task"
LSTM_MODEL   = "tennis_model.h5"
LABELS_FILE  = "tennis_labels.json"
SEQ_LENGTH   = 30
CONF_THRESH  = 0.65
PREDICT_EVERY = 3    

COLORS = {
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

class WebcamReader:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

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

def smooth_predictions(pred_history):
    if not pred_history:
        return None, 0.0
    avg = np.mean(pred_history, axis=0)
    idx = np.argmax(avg)
    return idx, float(avg[idx])

def draw_ui(frame, label, conf, counters, buffer_len, fps):
    h, w = frame.shape[:2]
    color = COLORS.get(label, DEFAULT_COLOR) if label and conf >= CONF_THRESH else DEFAULT_COLOR

    panel_h = 34 + 28 * len(counters)
    cv2.rectangle(frame, (0, 0), (225, panel_h), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 0), (225, panel_h), (60, 60, 60), 1)
    cv2.putText(frame, "SHOT COUNTER", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    for i, (shot, count) in enumerate(counters.items()):
        y = 46 + i * 28
        c = COLORS.get(shot, DEFAULT_COLOR)
        active = (shot == label and conf >= CONF_THRESH)
        if active:
            cv2.rectangle(frame, (4, y-18), (221, y+8), c, -1)
            txt_col = (255, 255, 255)
        else:
            txt_col = c
        cv2.putText(frame, f"{shot.upper():<12} {count:>3}x",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w-90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

    bar_fill = int((buffer_len / SEQ_LENGTH) * 225)
    cv2.rectangle(frame, (0, h-5), (225, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, h-5), (bar_fill, h),
                  (80, 180, 80) if buffer_len == SEQ_LENGTH else (80, 80, 80), -1)

    cv2.rectangle(frame, (0, h-58), (w, h-6), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, h-58), (w, h-6), color, 1)

    display  = (label.upper() if label and conf >= CONF_THRESH else "-- IDLE --")
    conf_str = (f"{conf*100:.0f}%" if label and conf >= CONF_THRESH else "")

    cv2.putText(frame, display,  (12, h-32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, conf_str, (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    if label and conf >= CONF_THRESH:
        bw = int(conf * (w - 210))
        cv2.rectangle(frame, (205, h-24), (w-8, h-12), (50,50,50), -1)
        cv2.rectangle(frame, (205, h-24), (205+bw, h-12), color, -1)

    cv2.putText(frame, "Q=quit  R=reset", (w-150, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1)

def main():
    print("Loading LSTM model...")
    model = tf.keras.models.load_model(LSTM_MODEL)
    model(np.zeros((1, SEQ_LENGTH, 99)))

    with open(LABELS_FILE) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    print(f"class: {label_map}")

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    sequence   = deque(maxlen=SEQ_LENGTH)
    pred_hist  = deque(maxlen=5)
    counters   = {v: 0 for v in label_map.values()}
    prev_label = None
    hold_count = 0
    HOLD_THRESH = 10

    current_label = None
    current_conf  = 0.0
    frame_count   = 0

    fps_times = deque(maxlen=30)

    print("opening webcam... press Q to exit, R to reset.")
    cam = WebcamReader(0)
    time.sleep(0.5)  

    with PoseLandmarker.create_from_options(options) as landmarker:
        ts_ms = 0
        while True:
            t0 = time.time()
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1
            ts_ms += 33  

            small = cv2.resize(frame, (640, 480))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]

                draw_skeleton(frame, lm)

                kp = extract_keypoints(lm)
                sequence.append(kp)

                if len(sequence) == SEQ_LENGTH and frame_count % PREDICT_EVERY == 0:
                    X = np.expand_dims(np.array(sequence), axis=0)
                    probs = model.predict(X, verbose=0)[0]
                    pred_hist.append(probs)

                    idx, conf = smooth_predictions(list(pred_hist))
                    if idx is not None:
                        current_label = label_map[idx]
                        current_conf  = conf

                        if current_label == prev_label and conf >= CONF_THRESH:
                            hold_count += 1
                            if hold_count == HOLD_THRESH:
                                counters[current_label] += 1
                                print(f"{current_label.upper()} {counters[current_label]}  ({conf*100:.0f}%)")
                        else:
                            hold_count = 0
                        prev_label = current_label

            fps_times.append(time.time() - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

            draw_ui(frame, current_label, current_conf, counters, len(sequence), fps)
            cv2.imshow("Tennis Shot Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counters   = {v: 0 for v in label_map.values()}
                sequence.clear()
                pred_hist.clear()
                current_label = None
                current_conf  = 0.0
                print("Reset.")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
