"""
webcam_predict.py — Live hand gesture recognition using your webcam.

Usage:
    python webcam_predict.py

Controls:
    Q  — quit
    S  — save a screenshot
    P  — pause / unpause

Requirements:
    pip install opencv-python tensorflow numpy
"""

import cv2
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

# ─── Settings ─────────────────────────────────────────────────────────────────

MODEL_PATH      = "saved_model/gesture_model.h5"
CLASSES_PATH    = "class_names.json"
IMAGE_SIZE      = (64, 64)
CONFIDENCE_MIN  = 0.4        # only show prediction if confidence > 40%
CAMERA_INDEX    = 0          # 0 = default webcam, try 1 if not working
FRAME_SKIP      = 2          # predict every N frames (higher = faster display)

# ─── Colors (BGR format for OpenCV) ───────────────────────────────────────────
COLOR_GREEN  = (0, 220, 100)
COLOR_RED    = (0, 60, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_BG     = (20, 20, 20)


# ─── Load model and class names ───────────────────────────────────────────────

def load_resources():
    print("[Loading] Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[Loading] Class names...")
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    print(f"[Ready] {len(class_names)} gesture classes loaded!")
    print("[Info] Press Q to quit, S to screenshot, P to pause\n")
    return model, class_names


# ─── Preprocess one frame for the model ───────────────────────────────────────

def preprocess_frame(frame):
    """
    Resize and normalize a webcam frame for model input.
    Crops the center square first to avoid stretching.
    """
    h, w = frame.shape[:2]

    # Crop center square
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    frame = frame[y1:y1+size, x1:x1+size]

    # Resize to model input size
    frame = cv2.resize(frame, IMAGE_SIZE)

    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(frame, axis=0)


# ─── Draw overlay on frame ────────────────────────────────────────────────────

def draw_overlay(frame, gesture, confidence, fps, paused, all_probs, class_names):
    """
    Draw prediction results, confidence bar, and controls on the frame.
    """
    h, w = frame.shape[:2]

    # ── Top banner ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), COLOR_BG, -1)

    if paused:
        cv2.putText(frame, "PAUSED", (w//2 - 60, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 2)
    else:
        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)

        # Confidence %
        conf_text = f"{confidence*100:.1f}%"
        cv2.putText(frame, conf_text, (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_YELLOW, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # ── Confidence bar ────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, h - 30, w - 20, 15
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * confidence)
    bar_color = COLOR_GREEN if confidence > 0.7 else COLOR_YELLOW if confidence > 0.4 else COLOR_RED
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_color, -1)
    cv2.putText(frame, "Confidence", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    # ── Top 3 predictions sidebar ─────────────────────────────────────────────
    if all_probs is not None:
        top3_idx = np.argsort(all_probs)[::-1][:3]
        sidebar_x = w - 180
        cv2.rectangle(frame, (sidebar_x - 5, 75), (w - 5, 200), COLOR_BG, -1)
        cv2.putText(frame, "Top 3:", (sidebar_x, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        for i, idx in enumerate(top3_idx):
            name = class_names[idx]
            prob = all_probs[idx]
            color = COLOR_GREEN if i == 0 else COLOR_WHITE
            cv2.putText(frame, f"{i+1}. {name}: {prob*100:.0f}%",
                        (sidebar_x, 120 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── Controls hint ─────────────────────────────────────────────────────────
    cv2.putText(frame, "Q=Quit  S=Screenshot  P=Pause", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ── Green detection box ───────────────────────────────────────────────────
    margin = 20
    cv2.rectangle(frame, (margin, 75), (w - margin - 185, h - 55),
                  COLOR_GREEN if confidence > CONFIDENCE_MIN else (80, 80, 80), 2)
    cv2.putText(frame, "Place hand here", (margin + 5, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    # Load model
    model, class_names = load_resources()

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {CAMERA_INDEX}")
        print("  Try changing CAMERA_INDEX to 1 at the top of this file")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[Camera] Webcam opened successfully!")
    print("[Camera] Show your hand gesture to the camera...\n")

    # State
    paused       = False
    frame_count  = 0
    gesture      = "..."
    confidence   = 0.0
    all_probs    = None
    fps          = 0.0
    prev_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read from camera")
            break

        # Flip horizontally (mirror effect — more natural)
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # ── FPS calculation ───────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # ── Predict every N frames ────────────────────────────────────────────
        if not paused and frame_count % FRAME_SKIP == 0:
            processed   = preprocess_frame(frame)
            probs       = model.predict(processed, verbose=0)[0]
            top_idx     = np.argmax(probs)
            confidence  = float(probs[top_idx])
            all_probs   = probs

            if confidence >= CONFIDENCE_MIN:
                gesture = class_names[top_idx]
            else:
                gesture = "?"

        # ── Draw overlay ──────────────────────────────────────────────────────
        frame = draw_overlay(frame, gesture, confidence, fps,
                             paused, all_probs, class_names)

        # ── Show window ───────────────────────────────────────────────────────
        cv2.imshow("Hand Gesture Recognition", frame)

        # ── Key controls ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("[Quit] Closing...")
            break

        elif key == ord('s') or key == ord('S'):
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f"[Screenshot] Saved: {filename}")

        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"[{'Paused' if paused else 'Resumed'}]")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Webcam closed.")


if __name__ == "__main__":
    main()
