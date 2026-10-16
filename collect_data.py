"""
collect_data.py — Collect hand gesture images from your webcam.

Features:
- Crops ONLY the hand region (no face, no background)
- Saves images with the same format as your original dataset
- Guides you through all 32 gestures one by one
- Press SPACE to start collecting, SPACE again to move to next gesture
- Press Q to quit anytime

Usage:
    python collect_data.py

Images saved to: webcam_dataset/train/<gesture>/ and webcam_dataset/val/<gesture>/
"""

import cv2
import os
import time
import numpy as np

# ─── Settings ─────────────────────────────────────────────────────────────────

SAVE_DIR        = "webcam_dataset"   # where to save collected images
IMAGES_PER_CLASS = 400               # how many images to collect per gesture
VAL_SPLIT        = 0.2               # 20% goes to val, 80% to train
CAMERA_INDEX     = 0
IMAGE_SIZE       = (64, 64)

# ─── All 32 gesture classes ───────────────────────────────────────────────────

GESTURES = [
    # Letters A-Z
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # Words
    "bye", "hello", "no", "perfect", "thankyou", "yes"
]

# ─── Gesture instructions ─────────────────────────────────────────────────────

INSTRUCTIONS = {
    "A": "Closed fist, thumb to the side",
    "B": "Four fingers up, thumb folded",
    "C": "Curved hand like letter C",
    "D": "Index up, others curved to thumb",
    "E": "All fingers curled down",
    "F": "Index and thumb touching, others up",
    "G": "Index pointing sideways, thumb out",
    "H": "Index and middle pointing sideways",
    "I": "Pinky finger up, others closed",
    "J": "Pinky up, draw J motion",
    "K": "Index and middle up, thumb between them",
    "L": "Index up, thumb out (L shape)",
    "M": "Three fingers folded over thumb",
    "N": "Two fingers folded over thumb",
    "O": "All fingers and thumb form O shape",
    "P": "Like K but pointing down",
    "Q": "Like G but pointing down",
    "R": "Index and middle crossed",
    "S": "Closed fist, thumb over fingers",
    "T": "Thumb between index and middle",
    "U": "Index and middle up together",
    "V": "Index and middle up, spread apart (V sign)",
    "W": "Index, middle, ring up spread",
    "X": "Index finger hooked/bent",
    "Y": "Thumb and pinky out (hang loose)",
    "Z": "Index finger draws Z in air",
    "bye":      "Open hand waving gesture",
    "hello":    "Open flat hand, palm facing out",
    "no":       "Index and middle finger crossed or shaking",
    "perfect":  "Index and thumb forming OK circle",
    "thankyou": "Flat hand from chin moving forward",
    "yes":      "Closed fist nodding up and down",
}

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN  = (0, 220, 100)
RED    = (0, 60, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 215, 255)
CYAN   = (255, 220, 0)


# ─── Create folder structure ──────────────────────────────────────────────────

def create_dirs():
    for gesture in GESTURES:
        os.makedirs(os.path.join(SAVE_DIR, "train", gesture), exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, "val",   gesture), exist_ok=True)
    print(f"[Setup] Dataset folders created in: {SAVE_DIR}/")


# ─── Count existing images ────────────────────────────────────────────────────

def count_existing(gesture):
    train_dir = os.path.join(SAVE_DIR, "train", gesture)
    val_dir   = os.path.join(SAVE_DIR, "val",   gesture)
    count = 0
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            count += len([f for f in os.listdir(d) if f.endswith(".jpg")])
    return count


# ─── Extract hand region ──────────────────────────────────────────────────────

def extract_hand_region(frame):
    """
    Extract just the hand region from the frame.
    Uses skin color detection to find the hand.
    Returns the cropped and resized hand image.
    """
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    mask = mask1 + mask2

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def get_hand_bbox(frame, roi):
    """
    Get bounding box of hand inside the ROI region.
    """
    mask = extract_hand_region(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 3000:
            x, y, w, h = cv2.boundingRect(largest)
            return x, y, w, h
    return None


# ─── Draw UI ──────────────────────────────────────────────────────────────────

def draw_ui(frame, gesture, instruction, count, total, state, progress):
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 80), BLACK, -1)

    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2)

    # Count
    cv2.putText(frame, f"{count}/{total}", (w-120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)

    # Instruction
    cv2.putText(frame, instruction, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # State message
    if state == "waiting":
        msg = "Place hand in GREEN BOX, then press SPACE to start"
        color = YELLOW
    elif state == "collecting":
        msg = "COLLECTING... Hold gesture steady!"
        color = GREEN
    elif state == "done":
        msg = "Done! Press SPACE for next gesture or Q to quit"
        color = CYAN
    else:
        msg = ""
        color = WHITE

    cv2.putText(frame, msg, (10, h-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.putText(frame, "Q=Quit  SPACE=Start/Next", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    # Progress bar
    bar_w = int((progress / total) * (w - 20))
    cv2.rectangle(frame, (10, h-12), (w-10, h-5), (60,60,60), -1)
    cv2.rectangle(frame, (10, h-12), (10+bar_w, h-5), GREEN, -1)

    return frame


# ─── Main collection loop ─────────────────────────────────────────────────────

def collect_gesture(cap, gesture, instruction, gesture_idx, total_gestures):
    """
    Collect images for one gesture.
    Returns True to continue, False to quit.
    """
    train_dir = os.path.join(SAVE_DIR, "train", gesture)
    val_dir   = os.path.join(SAVE_DIR, "val",   gesture)

    # Count already collected
    existing = count_existing(gesture)
    if existing >= IMAGES_PER_CLASS:
        print(f"[Skip] {gesture} already has {existing} images")
        return True

    target    = IMAGES_PER_CLASS - existing
    collected = 0
    state     = "waiting"

    # ROI box (where hand should be placed)
    roi_x, roi_y = 100, 100
    roi_size = 300

    print(f"\n[Collecting] {gesture} ({gesture_idx+1}/{total_gestures}): {instruction}")
    print(f"  Place your hand in the green box and press SPACE")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Adjust ROI if frame is small
        roi_x    = min(roi_x, w - roi_size - 10)
        roi_y    = min(roi_y, h - roi_size - 10)
        roi_size = min(roi_size, min(w, h) - 120)

        # Draw ROI box
        box_color = GREEN if state == "collecting" else YELLOW
        cv2.rectangle(frame,
                      (roi_x, roi_y),
                      (roi_x + roi_size, roi_y + roi_size),
                      box_color, 3)
        cv2.putText(frame, "Place hand here", (roi_x+5, roi_y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Extract ROI
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

        # Collect image if in collecting state
        if state == "collecting":
            # Save the ROI (hand region only)
            save_img = cv2.resize(roi, IMAGE_SIZE)

            # Decide train or val
            total_so_far = existing + collected
            if total_so_far < int(IMAGES_PER_CLASS * (1 - VAL_SPLIT)):
                save_path = os.path.join(train_dir, f"{gesture}_{total_so_far:04d}.jpg")
            else:
                save_path = os.path.join(val_dir, f"{gesture}_{total_so_far:04d}.jpg")

            cv2.imwrite(save_path, save_img)
            collected += 1

            # Small delay between captures
            time.sleep(0.05)

            if collected >= target:
                state = "done"
                print(f"  ✔ Collected {collected} images for '{gesture}'")

        # Draw UI
        frame = draw_ui(frame, gesture, instruction,
                        existing + collected, IMAGES_PER_CLASS,
                        state, existing + collected)

        # Gesture progress indicator
        cv2.putText(frame, f"Gesture {gesture_idx+1}/{total_gestures}",
                    (w-200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.imshow("Data Collector — Hand Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            return False

        elif key == ord(' '):
            if state == "waiting":
                state = "collecting"
                print(f"  Recording...")
            elif state == "done":
                return True

    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def preview_mode(cap):
    """
    Preview mode — shows exactly what will be saved before collecting.
    Press Q to quit preview, SPACE to start collecting.
    """
    print("\n[Preview] Showing what will be saved...")
    print("  Move your hand into the green box")
    print("  Press SPACE to start collecting, Q to quit\n")

    roi_x, roi_y, roi_size = 100, 100, 300

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        roi_size = min(roi_size, min(w, h) - 120)

        # Draw ROI box on main frame
        cv2.rectangle(frame, (roi_x, roi_y),
                      (roi_x+roi_size, roi_y+roi_size), GREEN, 3)

        # Extract ROI and show preview
        roi        = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size].copy()
        preview    = cv2.resize(roi, (200, 200))

        # Show preview in top right corner
        ph, pw = preview.shape[:2]
        frame[10:10+ph, w-pw-10:w-10] = preview
        cv2.rectangle(frame, (w-pw-10, 10), (w-10, 10+ph), YELLOW, 2)
        cv2.putText(frame, "PREVIEW (what gets saved)",
                    (w-pw-10, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

        cv2.putText(frame, "PREVIEW MODE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
        cv2.putText(frame, "This is what will be saved (top right)",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, "SPACE = Start Collecting   Q = Quit",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        cv2.imshow("Data Collector — Hand Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            return False
        elif key == ord(' '):
            return True

    return False


def main():
    print("=" * 60)
    print("  Hand Gesture Data Collector")
    print("=" * 60)
    print(f"  Gestures to collect : {len(GESTURES)}")
    print(f"  Images per gesture  : {IMAGES_PER_CLASS}")
    print(f"  Save location       : {SAVE_DIR}/")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE = Start collecting / Move to next gesture")
    print("  Q     = Quit anytime (progress is saved)")
    print("\nTips for best results:")
    print("  - Use good lighting")
    print("  - Keep ONLY your hand in the green box")
    print("  - Move your hand slightly while collecting (adds variety)")
    print("  - Plain/simple background works best")
    print("\nPress Enter to start...")
    input()

    create_dirs()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Preview mode first ───────────────────────────────────────────────────
    print("\nStarting PREVIEW MODE first...")
    print("You will see exactly what gets saved before collecting anything.")
    print("Press SPACE when happy with the view, Q to quit.\n")
    if not preview_mode(cap):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Check which gestures already have enough images
    remaining = []
    for g in GESTURES:
        existing = count_existing(g)
        if existing < IMAGES_PER_CLASS:
            remaining.append(g)
        else:
            print(f"[Skip] {g} already complete ({existing} images)")

    print(f"\n{len(remaining)} gestures to collect: {remaining}\n")

    for idx, gesture in enumerate(remaining):
        instruction = INSTRUCTIONS.get(gesture, "Perform the gesture")
        result = collect_gesture(cap, gesture, instruction, idx, len(remaining))
        if not result:
            print("\n[Quit] Progress saved. Run again to continue.")
            break
    else:
        print("\n" + "="*60)
        print("  ALL GESTURES COLLECTED!")
        print("="*60)
        print(f"\nDataset saved to: {SAVE_DIR}/")
        print("\nNow retrain with your new webcam images:")
        print(f'  python train.py --data "{os.path.abspath(SAVE_DIR)}" --epochs 20 --batch 64')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
