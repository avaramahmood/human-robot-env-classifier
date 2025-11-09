# sequential_inference.py
import os
import cv2
import torch
from collections import deque
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "runs/detect/train/weights/best.pt"   # path to your trained model
VIDEO_PATH = r"D:\yolo-human-robot-classifier\test_video.mp4"  # absolute path recommended
SEQUENCE_LENGTH = 5        # how many frames to consider for voting
CONF_THRESHOLD = 0.35      # detection confidence threshold
CLASS_NAMES = ['stairs', 'obstacle', 'door']  # must match data.yaml order

# --- Basic checks ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if isinstance(VIDEO_PATH, str) and VIDEO_PATH != "0" and not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# --- Convert VIDEO_PATH to webcam int if needed ---
video_source = 0 if VIDEO_PATH == 0 or VIDEO_PATH == "0" else VIDEO_PATH

# --- Load model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
model = YOLO(MODEL_PATH)
# ultralytics model auto-selects device in inference call; ok to keep model as is.

# --- Open video capture ---
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_source}. Check path or device.")

# --- Sequence memory ---
pred_queue = deque(maxlen=SEQUENCE_LENGTH)

print("Starting sequential inference. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or can't read frame.")
        break

    # Run YOLO inference (single-frame)
    results = model(frame)[0]  # returns Results object; index 0 for first batch item

    # Build per-frame detection summary (class names detected above threshold)
    current_preds = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf >= CONF_THRESHOLD:
            if 0 <= cls_id < len(CLASS_NAMES):
                current_preds.append(CLASS_NAMES[cls_id])
    pred_queue.append(current_preds)

    # Majority vote across queue
    all_preds = [p for sub in pred_queue for p in sub]
    if all_preds:
        final_pred = max(set(all_preds), key=all_preds.count)
    else:
        final_pred = "none"

    # Annotate frame using ultralytics plotting and add the sequence label
    annotated = results.plot()  # draws boxes + labels
    cv2.putText(annotated, f"SEQ: {final_pred.upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Sequential YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished.")
