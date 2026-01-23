import numpy as np
import cv2 as cv
from collections import deque

# ===================== USER PARAMETERS =====================

VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\good_flow_1.mp4"

FRAME_LAG = 5              # frames between flow measurements
MIN_SPEED = 0.01           # px/frame (numerical noise floor)
MAX_SPEED = 1            # px/frame → full green
MIN_AREA = 400             # minimum connected area (pixels)
CONFIRM_FRAMES = 3         # temporal persistence requirement
ALPHA_OVERLAY = 0.3        # overlay opacity
SCALE = 0.5                # downscale for speed (0.5 = 4× faster)

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

# Resize immediately (CRITICAL for performance)
frame = cv.resize(frame, None, fx=SCALE, fy=SCALE)

# Buffer for frame-lag
gray_buffer = deque()

# Temporal persistence counter
motion_count = None

WINDOW_NAME = "Optical Flow (Temporal + Spatial Filtering)"
cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
cv.resizeWindow(WINDOW_NAME, 900, 600)


# ------------------ Main Loop -----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, None, fx=SCALE, fy=SCALE)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)  # safe input blur
    gray_buffer.append(gray)

    # Wait until buffer fills
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
        levels=4,
        winsize=31,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # --------- Absolute Speed (px/frame) ---------
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    mag = mag / FRAME_LAG

    # --------- Initialize temporal counter ---------
    if motion_count is None:
        motion_count = np.zeros_like(mag, dtype=np.uint8)

    # --------- Instantaneous motion mask ---------
    motion_mask = mag > MIN_SPEED

    # --------- TEMPORAL CONSISTENCY FILTER ---------
    motion_count[motion_mask] += 1
    motion_count[~motion_mask] = 0

    confirmed_motion = motion_count >= CONFIRM_FRAMES

    # --------- SPATIAL COHERENCE FILTER ---------
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        confirmed_motion.astype(np.uint8),
        connectivity=8
    )

    clean_mask = np.zeros_like(confirmed_motion, dtype=np.uint8)

    for i in range(1, num_labels):  # skip background
        if stats[i, cv.CC_STAT_AREA] >= MIN_AREA:
            clean_mask[labels == i] = 1

    # Keep only trusted motion
    mag = mag * clean_mask

    # --------- Map speed to color (absolute) ---------
    s = np.clip((mag - MIN_SPEED) / (MAX_SPEED - MIN_SPEED), 0, 1)

    speed_color = np.zeros_like(frame)
    speed_color[..., 1] = (s * 255).astype(np.uint8)        # Green
    speed_color[..., 2] = ((1 - s) * 255).astype(np.uint8)  # Yellow → green

    # --------- Blend full image once ---------
    blended = cv.addWeighted(
        frame, 1 - ALPHA_OVERLAY,
        speed_color, ALPHA_OVERLAY,
        0
    )

    # --------- Apply color ONLY where motion is trusted ---------
    overlay = frame.copy()
    mask = clean_mask.astype(bool)

    overlay[mask] = blended[mask]

    # --------- Overlay video time ---------
    frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    video_time = frame_idx / fps

    cv.putText(
        overlay,
        f"Video Time: {video_time:6.2f} s",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    cv.imshow(WINDOW_NAME, overlay)

    if cv.waitKey(1) == 27:  # ESC
        break


# ------------------ Cleanup -------------------------------

cap.release()
cv.destroyAllWindows()
