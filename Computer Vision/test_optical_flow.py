import numpy as np
import cv2 as cv
from collections import deque

# ===================== USER PARAMETERS =====================

VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\Flow_sweep_crop.mp4"

FRAME_LAG = 3            # frames between comparisons (reveals slow drift)
MIN_SPEED = 0.1        # px/frame noise floor
MAX_SPEED = 2          # px/frame → full green
MIN_AREA = 500           # pixels; removes spatially isolated noise
ALPHA_OVERLAY = 0.6

# ==========================================================


# ------------------ Video Setup ---------------------------

cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv.CAP_PROP_FPS)
print("Video FPS:", fps)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

# Buffer for multi-frame lag
scale=0.5
frame = cv.resize(frame, None, fx=scale, fy=scale)
gray_buffer = deque()

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (7, 7), 0)  # helps coherence
gray_buffer.append(gray)

WINDOW_NAME = "Optical Flow (Only Pixels Above Speed Threshold)"
cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
cv.resizeWindow(WINDOW_NAME, 900, 600)


# ------------------ Main Loop -----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break
    scale=0.5
    frame = cv.resize(frame, None, fx=scale, fy=scale)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray_buffer.append(gray)

    if len(gray_buffer) <= FRAME_LAG:
        cv.imshow(WINDOW_NAME, frame)
        cv.waitKey(1)
        continue

    gray_old = gray_buffer[0]
    gray_new = gray_buffer[FRAME_LAG]
    gray_buffer.popleft()

    # --------- Dense Optical Flow ---------
    flow = cv.calcOpticalFlowFarneback(
        gray_old, gray_new, None,
        pyr_scale=0.5,
        levels=3,
        winsize=20,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # --------- Absolute Speed (px/frame) ---------
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mag = mag / FRAME_LAG

    # --------- Threshold: what is motion at all? ---------
    motion_mask = mag > MIN_SPEED

    # --------- Spatial Coherence Filtering ---------
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        motion_mask.astype(np.uint8),
        connectivity=8
    )

    clean_mask = np.zeros_like(motion_mask, dtype=np.uint8)

    for i in range(1, num_labels):  # skip background
        if stats[i, cv.CC_STAT_AREA] >= MIN_AREA:
            clean_mask[labels == i] = 1

    # Keep only coherent motion
    mag = mag * clean_mask

    # --------- Map speed to color (absolute, no normalization) ---------
    s = np.clip((mag - MIN_SPEED) / (MAX_SPEED - MIN_SPEED), 0, 1)

    speed_color = np.zeros_like(frame)
    speed_color[..., 1] = (s * 255).astype(np.uint8)        # Green
    speed_color[..., 2] = ((1 - s) * 255).astype(np.uint8)  # Yellow → green

    # --------- Blend full images once ---------
    blended = cv.addWeighted(
        frame, 1 - ALPHA_OVERLAY,
        speed_color, ALPHA_OVERLAY,
        0
    )

    # --------- Apply color ONLY where speed > threshold ---------
    overlay = frame.copy()
    mask = clean_mask.astype(bool)   # MUST be boolean

    overlay[mask] = blended[mask]

    cv.imshow(WINDOW_NAME, overlay)

    if cv.waitKey(1) == 27:  # ESC
        break


# ------------------ Cleanup -------------------------------

cap.release()
cv.destroyAllWindows()
