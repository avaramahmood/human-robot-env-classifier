import os
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from lstm_model import TemporalClassifier

# ---------------- CONFIG ----------------
YOLO_PATH = "runs/detect/train/weights/best.pt"
LSTM_PATH = "lstm_training/lstm_model.pth"
VIDEO_PATH = r"D:\\yolo-human-robot-classifier\\test_video.mp4"  # or "0" for webcam

CLASS_NAMES = ['stairs', 'obstacle', 'door', 'floor']
CONF_THR = 0.20         # LOWER threshold to catch more
SEQ_LEN = 10
STABLE_COUNT = 6
HIDDEN_DIM = 64
TOP_N_SHOW = 5          # Show up to N detections even if conf is low
# ----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# Load YOLO
yolo = YOLO(YOLO_PATH)
print("YOLO model loaded.")

# Load LSTM
if os.path.exists(LSTM_PATH):
    lstm = TemporalClassifier(input_dim=len(CLASS_NAMES),
                              hidden_dim=HIDDEN_DIM,
                              num_layers=1,
                              num_classes=len(CLASS_NAMES))
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
    lstm.to(DEVICE).eval()
    use_lstm = True
    print("LSTM loaded successfully.")
else:
    lstm = None
    use_lstm = False
    print("No LSTM weights found, fallback to voting only.")

# Video setup
cap = cv2.VideoCapture(0 if VIDEO_PATH in ["0", 0] else VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video source")

seq_buffer = deque(maxlen=SEQ_LEN)
recent_preds = deque(maxlen=STABLE_COUNT)

def normalize(v):
    s = v.sum()
    if s == 0:
        return v
    return v / s

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO inference ---
    results = yolo.predict(frame, imgsz=640, conf=0.2, iou=0.6, verbose=False, device=DEVICE)[0]
    conf_vec = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    # Weighted per-class confidences (accumulate all detections)
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if 0 <= cls < len(CLASS_NAMES):
            conf_vec[cls] = max(conf_vec[cls], conf)

    conf_vec = normalize(conf_vec)
    seq_buffer.append(conf_vec)

    # --- Temporal inference ---
    if use_lstm and len(seq_buffer) == SEQ_LEN:
        seq_np = np.stack(seq_buffer, axis=0)
        seq_t = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = lstm(seq_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs))
            conf_score = float(probs[0, pred_idx])
        final_label = CLASS_NAMES[pred_idx]
    else:
        votes = [CLASS_NAMES[np.argmax(f)] for f in seq_buffer if np.sum(f) > 0]
        final_label = max(set(votes), key=votes.count) if votes else "none"
        conf_score = max([max(f) for f in seq_buffer]) if seq_buffer else 0.0

    recent_preds.append(final_label)
    stable_pred = max(set(recent_preds), key=recent_preds.count)

    # --- Annotate frame ---
    annotated = results.plot()

    # Draw top-N detections (sorted by confidence)
    detections = sorted(results.boxes, key=lambda b: float(b.conf), reverse=True)[:TOP_N_SHOW]
    for box in detections:
        cls = int(box.cls)
        conf = float(box.conf)
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if label == stable_pred else (100, 100, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display stable LSTM label
    cv2.putText(annotated, f"LSTM: {stable_pred.upper()} ({conf_score:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("YOLO + LSTM (sensitive)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished.")
