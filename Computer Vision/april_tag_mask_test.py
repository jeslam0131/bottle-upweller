import cv2
import numpy as np
from pupil_apriltags import Detector

# ================= CONFIG =================
CAMERA_INDEX = 1
MASK_PATH = r"C:\Users\jessi\Documents\research SM\Computer Vision\mask_temp.png"
TAG_FAMILY = "tag25h9"

TAG_SIZE_MM = 50

# MM offsets (tag center -> mask top-left)
MASK_OFFSET_X_MM = -10
MASK_OFFSET_Y_MM = 1  # try 10mm

# Manual scale (mask size still hardcoded)
MASK_SCALE = 0.30
SCALE_STEP = 0.02
AXIS_LEN_PX = 60
# =========================================

mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise FileNotFoundError("Mask image not found")
AOI_MASK = np.where(mask_img > 200, 255, 0).astype(np.uint8)

detector = Detector(
    families=TAG_FAMILY,
    nthreads=1,
    quad_decimate=0.5,
    quad_sigma=0.8,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    if detections:
        tag = detections[0]

        # KEEP FLOATS FOR MATH
        corners_f = tag.corners.astype(np.float32)   # (4,2)
        cx_f, cy_f = tag.center.astype(np.float32)

        # ints only for drawing
        corners_i = corners_f.astype(int)
        cx_i, cy_i = int(cx_f), int(cy_f)

        # Draw tag outline
        for i in range(4):
            cv2.line(frame, tuple(corners_i[i]), tuple(corners_i[(i + 1) % 4]), (0, 255, 0), 2)

        # Compute rotation (use float corners)
        v = corners_f[1] - corners_f[0]
        angle_rad = -np.arctan2(v[1], v[0])
        angle_deg = np.degrees(angle_rad)

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        # Compute px/mm (use float corners)
        edge_lengths = [np.linalg.norm(corners_f[i] - corners_f[(i + 1) % 4]) for i in range(4)]
        tag_size_px = float(np.mean(edge_lengths))
        px_per_mm = tag_size_px / TAG_SIZE_MM

        # Draw axes (unchanged)
        x_axis = R @ np.array([AXIS_LEN_PX, 0], dtype=np.float32)
        y_axis = R @ np.array([0, AXIS_LEN_PX], dtype=np.float32)

        cv2.arrowedLine(frame, (cx_i, cy_i), (int(cx_f + x_axis[0]), int(cy_f + x_axis[1])), (0, 0, 255), 2)
        cv2.arrowedLine(frame, (cx_i, cy_i), (int(cx_f + y_axis[0]), int(cy_f + y_axis[1])), (255, 0, 0), 2)

        # Scale mask (still hardcoded)
        scaled_mask = cv2.resize(AOI_MASK, None, fx=MASK_SCALE, fy=MASK_SCALE, interpolation=cv2.INTER_NEAREST)
        mh, mw = scaled_mask.shape

        # Rotate mask
        M = cv2.getRotationMatrix2D((mw // 2, mh // 2), angle_deg, 1.0)
        rotated_mask = cv2.warpAffine(scaled_mask, M, (mw, mh), flags=cv2.INTER_NEAREST)

        # MM offset -> PX offset (this is correct)
        offset_vec_px = np.array([MASK_OFFSET_X_MM * px_per_mm,
                                  MASK_OFFSET_Y_MM * px_per_mm], dtype=np.float32)
        offset_px = R @ offset_vec_px

        x0 = int(cx_f + offset_px[0])
        y0 = int(cy_f + offset_px[1])
        x1 = x0 + mw
        y1 = y0 + mh

        # Clip
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(frame.shape[1], x1), min(frame.shape[0], y1)

        mx0, my0 = fx0 - x0, fy0 - y0
        mx1, my1 = mx0 + (fx1 - fx0), my0 + (fy1 - fy0)

        if fx1 > fx0 and fy1 > fy0:
            mask_crop = rotated_mask[my0:my1, mx0:mx1]
            overlay = frame.copy()
            overlay[fy0:fy1, fx0:fx1][mask_crop == 255] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Debug: show computed scale
        cv2.putText(frame, f"tag_px={tag_size_px:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"px_per_mm={px_per_mm:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AprilTag + Mask (MM offsets, manual scale)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('='):
        MASK_SCALE += SCALE_STEP
    elif key == ord('-'):
        MASK_SCALE = max(0.05, MASK_SCALE - SCALE_STEP)

cap.release()
cv2.destroyAllWindows()
