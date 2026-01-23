import cv2
import numpy as np
import time
from collections import deque

# ---------------- AOI SETTINGS ----------------
AOI_X = 430
AOI_Y = 175
AOI_W = 40
AOI_H = 115    # corresponds to 25 cm
CM_PER_PIXEL = 25 / AOI_H

# ---------------- HSV VALUES ----------------
lower = np.array([0, 0, 0])
upper = np.array([80, 255, 255])

# ---------------- SMOOTHING ----------------
alpha = 0.3
smooth_top = None

# ---------------- HISTORY BUFFER ----------------
history = deque()
WINDOW_SECONDS = 3.0

# ---------------- PD CONTROLLER (not used for motor here) ----------------
SETPOINT = 12.75     # cm
TOLERANCE = 0.5
Kp = 15
Kd = 5

last_error = 0
last_time = time.time()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Running NO-SERIAL test. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("WARNING: Invalid frame received.")
        continue

    # ---------------- AOI DRAW ----------------
    frame_disp = frame.copy()
    cv2.rectangle(frame_disp,
                  (AOI_X, AOI_Y),
                  (AOI_X + AOI_W, AOI_Y + AOI_H),
                  (0, 0, 255), 1)

    # ---------------- ROI EXTRACTION ----------------
    roi = frame[AOI_Y:AOI_Y + AOI_H, AOI_X:AOI_X + AOI_W]
    if roi.size == 0:
        print("WARNING: ROI empty — skipping frame.")
        continue

    # ---------------- HSV CONVERSION ----------------
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except:
        print("HSV conversion error")
        continue

    # ---------------- THRESHOLD ----------------
    mask = cv2.inRange(hsv, lower, upper)

    # ---------------- MORPH CLEANUP ----------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ---------------- IGNORE TOP 10% ----------------
    valid_mask = mask.copy()
    valid_mask[:int(AOI_H * 0.10), :] = 0

    ys, xs = np.where(valid_mask == 255)
    roi_marked = roi.copy()

    if len(ys) > 0:
        # Top of blob
        top_y = int(np.min(ys))

        # Smooth
        if smooth_top is None:
            smooth_top = top_y
        else:
            smooth_top = int(alpha * top_y + (1 - alpha) * smooth_top)

        # Height
        height_cm = (AOI_H - smooth_top) * CM_PER_PIXEL

        cv2.line(roi_marked, (0, smooth_top), (AOI_W, smooth_top), (0, 255, 0), 2)
        cv2.putText(roi_marked, f"{height_cm:.1f} cm",
                    (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # History buffer
        now = time.time()
        history.append((now, height_cm))
        while history and (now - history[0][0]) > WINDOW_SECONDS:
            history.popleft()

        heights = [h for (_, h) in history]
        if len(heights) > 1:
            avg_height = np.mean(heights)
            height_range = np.max(heights) - np.min(heights)

            cv2.putText(roi_marked, f"avg {avg_height:.1f}",
                        (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(roi_marked, f"rng {height_range:.1f}",
                        (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ---------------- DISPLAY ----------------
    try:
        frame_disp[AOI_Y:AOI_Y + AOI_H, AOI_X:AOI_X + AOI_W] = roi_marked
        cv2.imshow("Blob Height", frame_disp)
        cv2.imshow("ROI", roi_marked)
        cv2.imshow("Mask", mask)
    except:
        print("Display error — skipping frame.")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
