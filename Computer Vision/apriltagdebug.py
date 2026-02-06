import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque

# ================= CONFIG =================
CAMERA_INDEX = 1
MASK_PATH = r"C:\Users\jessi\Documents\research SM\Computer Vision\mask_temp.png"
TAG_FAMILY = "tag25h9"

# MM Real-World Values
TAG_SIZE_MM = 77.8        # Your physical AprilTag size
MASK_HEIGHT_MM = 250.0    # Desired real-world height of the green mask
ANGLE_OFFSET_DEG = -8    # Adjust this if the mask is tilted relative to the tube

# MM offsets (Moves the mask relative to the Tag center)
# X: Positive = Right, Y: Negative = Up (toward the top of the tube)
MASK_OFFSET_X_MM = 195
MASK_OFFSET_Y_MM = 415

SMOOTHING_FRAMES = 6
# =========================================

pos_history = deque(maxlen=SMOOTHING_FRAMES)
angle_history = deque(maxlen=SMOOTHING_FRAMES)
scale_history = deque(maxlen=SMOOTHING_FRAMES)

at_detector = Detector(families=TAG_FAMILY, nthreads=1,
    quad_decimate=0.5,
    quad_sigma=0.8,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0)
mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise FileNotFoundError("Could not find mask_temp.png")
mh, mw = mask_img.shape[:2]

cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)
    stencil = np.zeros(frame.shape[:2], dtype=np.uint8)

    if results:
        tag = results[0]
        corners = tag.corners.astype(np.float32)
        
        # 1. STABLE SCALE (From ChatGPT logic)
        edge_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        avg_px_mm = np.mean(edge_lengths) / TAG_SIZE_MM

        # 2. STABLE ANGLE (Aligned to Tag X-axis)
        v = corners[1] - corners[0]
        # We use negative here because image Y is inverted
        angle_rad = -np.arctan2(v[1], v[0])
        avg_angle_deg = np.degrees(angle_rad) + ANGLE_OFFSET_DEG

        # 3. SMOOTHING
        pos_history.append(tag.center)
        angle_history.append(avg_angle_deg)
        scale_history.append(avg_px_mm)

        smoothed_center = np.mean(pos_history, axis=0)
        smoothed_angle = np.mean(angle_history)
        smoothed_px_mm = np.mean(scale_history)

        # 4. UNIFIED POSITIONING (The Fix)
        # We calculate the target center by rotating the MM offset 
        # based on the tag's current tilt.
        rad = np.radians(-smoothed_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # This math "glues" the offset to the tag's local axes
        off_x_px = (MASK_OFFSET_X_MM * cos_a - MASK_OFFSET_Y_MM * sin_a) * smoothed_px_mm
        off_y_px = (MASK_OFFSET_X_MM * sin_a + MASK_OFFSET_Y_MM * cos_a) * smoothed_px_mm
        target_px = (smoothed_center[0] + off_x_px, smoothed_center[1] + off_y_px)

        # 5. UNIFIED TRANSFORMATION MATRIX
        # Scale the PNG to match MASK_HEIGHT_MM
        mask_scale = (MASK_HEIGHT_MM * smoothed_px_mm) / mh
        
        # Create one matrix that handles Rotation and Scaling simultaneously
        M = cv2.getRotationMatrix2D(target_px, -smoothed_angle, mask_scale)
        
        # Adjust so the mask's center is at target_px
        M[0, 2] -= (mw / 2) * mask_scale
        M[1, 2] -= (mh / 2) * mask_scale

        # 6. DRAWING & AXES
        cv2.warpAffine(mask_img, M, (frame.shape[1], frame.shape[0]), 
                       dst=stencil, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Red line showing the Tag's X-axis for alignment verification
        cv2.line(frame, tuple(smoothed_center.astype(int)), 
                 (int(smoothed_center[0] + cos_a * 60), int(smoothed_center[1] - sin_a * 60)), (0, 0, 255), 3)

        # Green Overlay
        overlay = frame.copy()
        overlay[stencil > 200] = (0, 255, 0)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    cv2.imshow("Unified Alignment (Red Line = Tag X-Axis)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()