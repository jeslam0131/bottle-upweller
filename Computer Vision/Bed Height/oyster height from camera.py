import cv2
import numpy as np
import time
from collections import deque

cap = cv2.VideoCapture(1)

# --- AOI coordinates ---
AOI_X = 430
AOI_Y = 175
AOI_W = 40
AOI_H = 115      # <-- corresponds to 25 cm in real life
# ------------------------------------------------------

# Conversion scale
CM_PER_PIXEL = 25 / AOI_H   # physical height (25 cm) divided by pixel height

# --- HARD-CODED HSV VALUES ---
lower = np.array([0, 0, 0])
upper = np.array([80, 255, 255])
# -----------------------------

# Smoothing factor
alpha = 0.3
smooth_top = None

# --- Rolling buffer for last 3 seconds ---
history = deque()   # stores (timestamp, height_cm)

WINDOW_SECONDS = 3.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Draw AOI rectangle
    cv2.rectangle(display, 
                  (AOI_X, AOI_Y),
                  (AOI_X + AOI_W, AOI_Y + AOI_H),
                  (0, 0, 255), 
                  1)

    # Extract AOI
    roi = frame[AOI_Y:AOI_Y+AOI_H, AOI_X:AOI_X+AOI_W]

    # HSV segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Mask cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Ignore top 10% (noise)
    valid_mask = mask.copy()
    valid_mask[0:int(AOI_H * 0.10), :] = 0

    ys, xs = np.where(valid_mask == 255)
    roi_marked = roi.copy()

    height_from_bottom_cm = None

    if len(ys) > 0:

        # Top of blob
        top_y = int(np.min(ys))
        bottom_y = int(np.max(ys))

        # Smooth top
        if smooth_top is None:
            smooth_top = top_y
        else:
            smooth_top = int(alpha * top_y + (1 - alpha) * smooth_top)

        # Compute height (bottom of AOI -> blob top)
        height_from_bottom_px = AOI_H - smooth_top
        height_from_bottom_cm = height_from_bottom_px * CM_PER_PIXEL

        # Draw green line
        cv2.line(roi_marked, 
                 (0, smooth_top), 
                 (AOI_W, smooth_top), 
                 (0, 255, 0), 
                 2)

        # Display height text
        cv2.putText(roi_marked, 
                    f"{height_from_bottom_cm:.1f} cm",
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 
                    (0, 255, 0), 
                    2)

        # -----------------------------
        # Add value to rolling history
        # -----------------------------
        now = time.time()
        history.append((now, height_from_bottom_cm))

        # Remove old entries (> 3 sec)
        while len(history) > 0 and (now - history[0][0]) > WINDOW_SECONDS:
            history.popleft()

        # Compute stats if enough data
        if len(history) > 1:
            heights = [h for (_, h) in history]
            avg_height = np.mean(heights)
            height_range = np.max(heights) - np.min(heights)

            print(f"Last 3s avg = {avg_height:.2f} cm   range = {height_range:.2f} cm")

            # Draw stats on ROI
            cv2.putText(roi_marked, 
                        f"avg: {avg_height:.1f} cm",
                        (0, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 255), 
                        1)

            cv2.putText(roi_marked, 
                        f"rng: {height_range:.1f} cm",
                        (0, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 255), 
                        1)

    # Put marked ROI back into display
    display[AOI_Y:AOI_Y+AOI_H, AOI_X:AOI_X+AOI_W] = roi_marked

    cv2.imshow("Blob Height", display)
    cv2.imshow("ROI", roi_marked)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
