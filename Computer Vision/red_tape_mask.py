import cv2
import numpy as np

# =============================
# Video input
# =============================
VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\good_flow_1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000 / fps) if fps > 0 else 1

# =============================
# Red HSV threshold (hard-coded)
# =============================
LOW_S = 60
LOW_V = 70

LOW_RED_1  = np.array([0,   LOW_S, LOW_V])  
HIGH_RED_1 = np.array([20,  255,   255])
LOW_RED_2  = np.array([160, LOW_S, LOW_V])
HIGH_RED_2 = np.array([179, 255,   255])

# =============================
# Morphology kernels
# =============================
KERNEL_CLOSE_BIG = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 35))
KERNEL_CLOSE_MED = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
KERNEL_FINAL     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# =============================
# Temporal smoothing
# =============================
ALPHA = 0.85  # higher = smoother, less responsive
smoothed_mask = None

cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)

# =============================
# Main loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Red threshold ---
    mask1 = cv2.inRange(hsv, LOW_RED_1, HIGH_RED_1)
    mask2 = cv2.inRange(hsv, LOW_RED_2, HIGH_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # =============================
    # PRE-FILL STABILIZATION
    # =============================

    # Blur to remove speckle + soften gaps
    red_mask = cv2.GaussianBlur(red_mask, (11, 11), 0)

    # Aggressively close gaps in tape
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE_BIG)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE_MED)

    # Binarize again
    _, red_mask = cv2.threshold(red_mask, 127, 255, cv2.THRESH_BINARY)

    # =============================
    # FLOOD FILL (outside → inside)
    # =============================
    h, w = red_mask.shape
    flood = red_mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from guaranteed outside
    cv2.floodFill(flood, ff_mask, (0, 0), 255)

    # Invert flood → interior
    flood_inv = cv2.bitwise_not(flood)
    filled_region = cv2.bitwise_or(red_mask, flood_inv)

    # Final cleanup
    filled_region = cv2.morphologyEx(filled_region, cv2.MORPH_CLOSE, KERNEL_FINAL)

    # =============================
    # TEMPORAL SMOOTHING (KEY PART)
    # =============================
    if smoothed_mask is None:
        smoothed_mask = filled_region.astype(np.float32)
    else:
        smoothed_mask = (
            ALPHA * smoothed_mask +
            (1 - ALPHA) * filled_region.astype(np.float32)
        )

    # Convert back to binary
    _, smoothed_mask_bin = cv2.threshold(
        smoothed_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
    )

    # =============================
    # Apply mask
    # =============================
    segmented = cv2.bitwise_and(frame, frame, mask=smoothed_mask_bin)

    # =============================
    # Visualization
    # =============================
    vis_red    = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    vis_fill   = cv2.cvtColor(smoothed_mask_bin, cv2.COLOR_GRAY2BGR)

    combined = np.vstack((
        frame,
        vis_red,
        vis_fill,
        segmented
    ))

    combined = cv2.resize(combined, None, fx=0.6, fy=0.6)
    cv2.imshow("Debug View", combined)

    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
