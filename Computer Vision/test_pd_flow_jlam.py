import cv2
import numpy as np
import time
from collections import deque

# ============================================================
# VIDEO INPUT (CHANGE THIS)
# ============================================================
VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\flow_sweep.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps
video_time = 0.0

cv2.namedWindow("Control View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control View", 900, 700)

# ============================================================
# FRAME OF INTEREST (FOI) — HARD CODED (will likely switch to a png overlay with april tags)
# ============================================================
FRAME_X = 800
FRAME_Y = 350
FRAME_W = 220
FRAME_H = 700   # ~25 cm physical height

# ============================================================
# BED COLOR SEGMENTATION
# ============================================================
lower = np.array([0, 0, 0])
upper = np.array([80, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# ============================================================
# BED HEIGHT SMOOTHING
# ============================================================
alpha = 0.3
smoothed_bed_height_y = None

# ============================================================
# ROI UPDATE TIMING (update the area of interest- bottom half of the fluidized bed  once every 60 seconds)
# ============================================================
ROI_UPDATE_PERIOD = 60.0
last_roi_update = 0.0
roi_top_y = None

# ============================================================
# MOTION + CONTROL PARAMETERS 
# ============================================================
V_MOTION = 0.5                 # px/frame that triggers go (will switch this to probably mm/sec or something in the future)
MOTION_WINDOW_SEC = 1.0        # period of time (sec) data is collected to determine if a pixel has moved above the threshold
CONTROL_PERIOD = 1.0           # time between commands sent to arduino in second
COVERAGE_TARGET = 0.85         # ideal % of area of interst 

# hystersis band
COVERAGE_BAND = 0.02 # hystersis band
COVERAGE_MIN = COVERAGE_TARGET-COVERAGE_BAND

#pick between Min coverage value or Setpoint coverage value
MIN_HYSTERSIS_METHOD = True # set false to use setpoint

#state
correcting = False # out of hystersis band
prev_error = 0.0 
last_control_time = 0.0 

#command controls
Kp = 300.0
Kd = 50.0

DELTA_POS_MAX= 5000 #max command per (10,000 limit for valve)

# ============================================================
# STATE
# ============================================================
prev_gray = None
motion_buffer = deque()   # stores (time, moved_mask)
last_control_time = 0.0
prev_error = 0.0
control_command = 0.0

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, camera_frame = cap.read()
    if not ret:
        break

    video_time += dt
    display = camera_frame.copy()

    # --------------------------------------------------------
    # DRAW FRAME OF INTEREST
    # --------------------------------------------------------
    cv2.rectangle(
        display,
        (FRAME_X, FRAME_Y),
        (FRAME_X + FRAME_W, FRAME_Y + FRAME_H),
        (0, 0, 255),
        3
    )

    # --------------------------------------------------------
    # EXTRACT FRAME OF INTEREST
    # --------------------------------------------------------
    frame = camera_frame[
        FRAME_Y:FRAME_Y + FRAME_H,
        FRAME_X:FRAME_X + FRAME_W
    ]

    # --------------------------------------------------------
    # BED HEIGHT DETECTION
    # --------------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    valid_mask = mask.copy()
    valid_mask[0:int(FRAME_H * 0.10), :] = 0

    ys, _ = np.where(valid_mask == 255)
    if len(ys) == 0:
        cv2.imshow("Control View", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    measured_top_y = int(np.min(ys))

    if smoothed_bed_height_y is None:
        smoothed_bed_height_y = measured_top_y
    else:
        smoothed_bed_height_y = int(
            alpha * measured_top_y +
            (1 - alpha) * smoothed_bed_height_y
        )

    bed_height_y = smoothed_bed_height_y
    bed_thickness = FRAME_H - bed_height_y

    # --------------------------------------------------------
    # UPDATE ROI (ONCE PER MINUTE)
    # --------------------------------------------------------
    if roi_top_y is None or (video_time - last_roi_update) > ROI_UPDATE_PERIOD:
        roi_top_y = int(bed_height_y + 0.50 * bed_thickness)
        last_roi_update = video_time

    # --------------------------------------------------------
    # DENSE OPTICAL FLOW
    # --------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Motion detection inside ROI
        moved_now = (mag > V_MOTION)
        roi_mask = np.zeros_like(moved_now)
        roi_mask[roi_top_y:FRAME_H, :] = True
        moved_now &= roi_mask

        # Store in motion buffer
        motion_buffer.append((video_time, moved_now))

        # Remove old entries (> 1 sec)
        while motion_buffer and (video_time - motion_buffer[0][0]) > MOTION_WINDOW_SEC:
            motion_buffer.popleft()

        # OR motion over last 1 second
        moved_recent = np.zeros_like(moved_now)
        for _, m in motion_buffer:
            moved_recent |= m

        # Coverage
        roi_area = roi_mask.sum()
        moved_area = moved_recent.sum()
        coverage = moved_area / roi_area if roi_area > 0 else 0.0

        # ----------------------------------------------------
        # PD CONTROLLER (ONCE PER SECOND)
        # ----------------------------------------------------
        if video_time - last_control_time >= CONTROL_PERIOD:
           
           #Hystersis Controller 
            if MIN_HYSTERSIS_METHOD:
                if correcting:
                    if coverage >= COVERAGE_TARGET:
                        correcting = False
                else:
                    if coverage < COVERAGE_MIN:
                        correcting = True
            else:
                LOW  = COVERAGE_TARGET - COVERAGE_BAND
                HIGH = COVERAGE_TARGET + COVERAGE_BAND

                if correcting:
                    if LOW <= coverage <= HIGH:
                        correcting = False
                else:
                    if coverage < LOW or coverage > HIGH:
                        correcting = True            
            if correcting:        
                error = COVERAGE_TARGET - coverage
                delt_T_err = CONTROL_PERIOD
                d_error = (error - prev_error)/delt_T_err

                delta_pos = Kp*error + Kd*d_error
                delta_pos = delta_pos = max(-DELTA_POS_MAX, min(delta_pos, DELTA_POS_MAX)) #limits pos to be in max change
                
                print(
                f"t={video_time:6.1f}s  "
                f"coverage={coverage:5.2f}  "
                f"error={error:6.3f}  "
                f"command={delta_pos:8.1f}"
                )
                
                
            else:
                delta_pos = 0
                print(
                f"t={video_time:6.1f}s  "
                f"coverage={coverage:5.2f}  "
                f"error={0}  "
                f"command={delta_pos:8.1f}"
                )
                
            #send_to_arduino (delta_pos)
            

            prev_error = error
            last_control_time = video_time
        # ----------------------------------------------------
        # BED COLOR SEGMENTATION OVERLAY (RED) — DEBUG
        # ----------------------------------------------------
        bed_overlay = frame.copy()
        bed_overlay[mask == 255] = (0, 0, 255)   # RED = detected spat
        frame = cv2.addWeighted(bed_overlay, 0.25, frame, 0.75, 0)
        # ----------------------------------------------------
        # VISUAL OVERLAYS
        # ----------------------------------------------------
        overlay = frame.copy()
        overlay[moved_recent] = (0, 255, 0)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Bed height (blue)
        cv2.line(frame, (0, bed_height_y), (FRAME_W, bed_height_y), (255, 0, 0), 2)

        # ROI top (red)
        cv2.line(frame, (0, roi_top_y), (FRAME_W, roi_top_y), (0, 255, 0), 2)

        # Coverage text
        cv2.putText(
            frame,
            f"Coverage: {coverage*100:.1f}%",
            (2, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2
        )

    prev_gray = gray

    # --------------------------------------------------------
    # PUT FRAME BACK INTO CAMERA VIEW
    # --------------------------------------------------------
    display[
        FRAME_Y:FRAME_Y + FRAME_H,
        FRAME_X:FRAME_X + FRAME_W
    ] = frame

    cv2.imshow("Control View", display)
    delay_ms = int(1000 / fps)
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
