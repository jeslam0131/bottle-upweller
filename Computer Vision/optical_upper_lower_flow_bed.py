import cv2
import numpy as np
import time
from collections import deque
import serial

# ============================================================
# Serial Communication
# ============================================================
ser = serial.Serial('COM7', 9600, timeout=0.1)  # change COM port
time.sleep(2)
SEND_PERIOD = 1          # send to Arduino every 5 seconds
last_send_time = 0.0

# ============================================================
# VIDEO INPUT (CHANGE THIS)
# ============================================================
VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\flow_sweep.mp4"
#cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print("Camera failed to open")
    exit()


#fps = cap.get(cv2.CAP_PROP_FPS)
fps=30
dt = 1.0 / fps
video_time = 0.0

cv2.namedWindow("Control View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control View", 900, 700)

# ============================================================
# FRAME OF INTEREST (FOI) — HARD CODED (will switch to a png overlay with april tags)
# ============================================================
FRAME_X = 600
FRAME_Y = 520
FRAME_W = 120
FRAME_H = 300   # ~25 cm physical height

# ============================================================
# BED COLOR SEGMENTATION
# ============================================================
lower = np.array([0, 0, 0])
upper = np.array([170, 30, 150])
kernel = np.ones((5, 5), np.uint8)

# ============================================================
# BED HEIGHT SMOOTHING
# ============================================================
alpha = 0.3 # Exponential Moving Average 
smoothed_bed_height_y = None
height_ignore = 0.5 #top percent of bed to ignore


# ============================================================
# ROI UPDATE TIMING (update the area of interest- bottom half of the fluidized bed  once every 60 seconds)
# ============================================================
STARTUP_DELAY = 3.0  # seconds
ROI_UPDATE_PERIOD_LB = 60.0
ROI_UPDATE_PERIOD_UB = 10.0
last_roi_update_lb=0.0
last_roi_update_ub = 0.0
roi_top_y= None 
upper_y_low = None
upper_y_high = None


# ============================================================
# MOTION + CONTROL PARAMETERS 
# ============================================================

# Lower Bed control parameter ============================================================
V_MOTION = 1                # px/frame that triggers go (will switch this to probably mm/sec or something in the future)
MOTION_WINDOW_SEC = 3.5        # period of time (sec) data is collected to determine if a pixel has moved above the threshold
COVERAGE_TARGET_LB = 0.85         # ideal % of area of interst 
LOWER_ROI_UPPER_FRAC = 0.5

# hystersis band
COVERAGE_BAND = 0.02 # hystersis band
COVERAGE_MIN_lb = COVERAGE_TARGET_LB-COVERAGE_BAND
MIN_HYSTERSIS_METHOD = True 

#lower bed control state
correcting_lb = False # false = in hystersis band | true = out of hystersis band
prev_error_lb = 0.0 
last_control_time = 0.0 
delta_pos_lb = 0.0


#command controls lower bed
Kp_lb = 3000
Kd_lb = 1000
DELTA_POS_LB_MAX= 5000

# Bed Height Control  ============================================================
BED_HEIGHT_TARGET = 400 # will switch to cm currently pixel

#lower bed control state
prev_bed_height = None

#command controls lower bed
Kp_bh = 5.0
Kd_bh = 1.0
DELTA_POS_BH_MAX = 800

# Upper Bed Control  ============================================================
V_FAST_MAX  = 4 # max allowed % fast pixels 
# same motion window time as lower bed
COVERAGE_TARGET_UB = 0.1
UPPER_ROI_LOW_FRAC = 0.65
UPPER_ROI_HIGH_FRAC = 0.85

# hystersis band
COVERAGE_BAND_UB = 0.02 # hystersis band
COVERAGE_MAX_UB = COVERAGE_TARGET_UB+COVERAGE_BAND_UB


#upper bed control state
correcting_ub = False # false = in hystersis band | true = out of hystersis band
prev_error_ub = 0.0 
delta_pos_ub = 0.0
#same control time as lower bed

#command controls upper bed
Kp_ub = 1000
Kd_ub = 100
DELTA_POS_UB_MAX = 500





CONTROL_PERIOD = 1.0           # time between commands sent to arduino in second
DELTA_POS_TOT_MAX= 5000 #max command per (10,000 limit for valve)

# ============================================================
# STATE
# ============================================================
prev_gray = None
motion_buffer = deque()   # stores (time, moved_mask) for lower bed 
upper_motion_buffer=deque() # store upper bed
control_command = 0.0

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    if ser.in_waiting > 0:
        try:
            print("ARDUINO:", ser.readline().decode().strip())
        except:
            pass
    
    ret, camera_frame = cap.read()
    if not ret:
        break

    video_time += dt
    display = camera_frame.copy()

    # --------------------------------------------------------
    # DRAW FRAME OF INTEREST // will be swaped out with april tag
    # --------------------------------------------------------
    cv2.rectangle(
        display,
        (FRAME_X, FRAME_Y),
        (FRAME_X + FRAME_W, FRAME_Y + FRAME_H),
        (0, 0, 255),
        3
    )

    # --------------------------------------------------------
    # EXTRACT FRAME OF INTEREST / maksed by png
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

    measured_top_y = int(np.percentile(ys, height_ignore))

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
    # UPDATE ROI (ONCE PER MINUTE LOWER ONCE PER 10 SEC UPPER)
    # --------------------------------------------------------
    if video_time > STARTUP_DELAY:
        if roi_top_y is None or (video_time - last_roi_update_lb) > ROI_UPDATE_PERIOD_LB:
            roi_top_y = int(bed_height_y + LOWER_ROI_UPPER_FRAC * bed_thickness)
            last_roi_update_lb = video_time
            
        if upper_y_low is None or (video_time - last_roi_update_ub) > ROI_UPDATE_PERIOD_UB:
            upper_y_low  = int(bed_height_y + (1-UPPER_ROI_HIGH_FRAC) * bed_thickness) #swapped because lower on the bottle has a higher corresponding pixel valueq
            upper_y_high = int(bed_height_y + (1-UPPER_ROI_LOW_FRAC) * bed_thickness)
            
            last_roi_update_ub = video_time
            

    # --------------------------------------------------------
    # DENSE OPTICAL FLOW lower bed
    # --------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        ## Full Bed Dense Optical Flow
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Motion detection inside Bottom ROI
        moved_now = (mag > V_MOTION)
        roi_mask = np.zeros_like(moved_now)
        roi_mask[roi_top_y:FRAME_H, :] = True
        moved_now &= roi_mask
        
        #Motion dection in Upper ROI
        upper_roi_mask = np.zeros_like(mag, dtype=bool)
        upper_roi_mask[upper_y_low:upper_y_high, :] = True
        moved_upper_now = (mag > V_FAST_MAX) & upper_roi_mask

        # Store in motion buffer 
        motion_buffer.append((video_time, moved_now)) # lower bed
        upper_motion_buffer.append((video_time, moved_upper_now))
        

        # Remove old entries (> 1 sec)
        while motion_buffer and (video_time - motion_buffer[0][0]) > MOTION_WINDOW_SEC:
            motion_buffer.popleft()
        
        while upper_motion_buffer and (video_time - upper_motion_buffer[0][0]) > MOTION_WINDOW_SEC:
            upper_motion_buffer.popleft()

        # OR motion over last 1 second
        moved_recent = np.zeros_like(moved_now)
        for _, m in motion_buffer:
            moved_recent |= m
        
        moved_upper_recent = np.zeros_like(moved_upper_now)
        for _, m in upper_motion_buffer:
            moved_upper_recent |= m

        # Coverage
        roi_area = roi_mask.sum()
        moved_area = moved_recent.sum()
        coverage_lb = moved_area / roi_area if roi_area > 0 else 0.0
        
        upper_area = upper_roi_mask.sum()
        upper_moved_area = moved_upper_recent.sum()
        coverage_ub = upper_moved_area / upper_area if upper_area > 0 else 0.0
        
        

        # ----------------------------------------------------
        # PD CONTROLLER (ONCE PER SECOND)
        # ----------------------------------------------------
        if video_time - last_control_time >= CONTROL_PERIOD:
            dt_control = CONTROL_PERIOD

            # Hysteresis band bottom 50% of bed
            if correcting_lb:
                if coverage_lb >= COVERAGE_TARGET_LB:
                    correcting_lb = False
            else:
                if coverage_lb < COVERAGE_MIN_lb:
                    correcting_lb = True

            # PD controller bottom 50% of bed
            error_lb = COVERAGE_TARGET_LB - coverage_lb
            d_error_lb = (error_lb - prev_error_lb) / dt_control  
                              
            if correcting_lb:
                delta_pos_lb = Kp_lb * error_lb + Kd_lb * d_error_lb
            else:
                delta_pos_lb = 0.0

            delta_pos_lb = np.clip(delta_pos_lb, -DELTA_POS_LB_MAX, DELTA_POS_LB_MAX)
            prev_error_lb = error_lb
            
            if correcting_ub:
                if coverage_ub<=COVERAGE_TARGET_UB: 
                    correcting_ub = False
            else:
                if coverage_ub > COVERAGE_MAX_UB:
                    correcting_ub=True
            
            error_ub = coverage_ub - COVERAGE_TARGET_UB
            d_error_ub = (error_ub - prev_error_ub) / dt_control
            
            if correcting_ub:
                delta_pos_ub = - (Kp_ub * error_ub + Kd_ub * d_error_ub)
            else:
                delta_pos_ub = 0.0

            delta_pos_ub = np.clip(delta_pos_ub, -DELTA_POS_UB_MAX, 0.0) # clamps values between 0 and max contribution from upper
            prev_error_ub = error_ub
                
            # =====================================================
            # COMBINE COMMANDS
            # =====================================================
            delta_pos_total = delta_pos_lb + delta_pos_ub
            delta_pos_total = np.clip(delta_pos_total, -DELTA_POS_TOT_MAX, DELTA_POS_TOT_MAX)
            command_to_send = int(delta_pos_total)

            

            print(
                f"t={video_time:6.1f}s  "
                f"LB={coverage_lb:5.2f}  "
                f"UB={coverage_ub:5.2f}  "
                f"cmd_lb={delta_pos_lb:7.1f}  "
                f"cmd_ub={delta_pos_ub:7.1f}  "
                f"cmd_tot={delta_pos_total:8.1f}"
            )
            last_control_time = video_time

            # send_to_arduino(delta_pos_total)
            
            # Deadband: send 0 if small
            if abs(command_to_send) < 50:
                command_to_send = 0

            
            if video_time - last_send_time >= SEND_PERIOD:
                ser.write(f"{int(command_to_send)}\n".encode())

                print(f"Sent to Arduino: {command_to_send}")

                last_send_time = video_time
    
   
        
        # ----------------------------------------------------
        # BED COLOR SEGMENTATION OVERLAY (RED) — DEBUG
        # ----------------------------------------------------
        bed_overlay = frame.copy()
        bed_overlay[mask == 255] = (0, 0, 255)   # RED = detected spat
        frame = cv2.addWeighted(bed_overlay, 0.25, frame, 0.75, 0)
        # ----------------------------------------------------
        # VISUAL OVERLAYS
        # ----------------------------------------------------
        # overlay bottom bed above threshold (green)
        overlay = frame.copy()
        overlay[moved_recent] = (0, 255, 0)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        #overlay upper bed above threshold (purple)
        upper_overlay=frame.copy()
        upper_overlay[moved_upper_recent] = (255, 0, 255)  # PINK (BGR)
        frame = cv2.addWeighted(upper_overlay, 0.35, frame, 0.65, 0)

        # Bed height (blue)
        cv2.line(frame, (0, bed_height_y), (FRAME_W, bed_height_y), (255, 0, 0), 2)

        # ROI top lower bed (red)
        cv2.line(frame, (0, roi_top_y), (FRAME_W, roi_top_y), (0, 255, 0), 2)
        
        #ROI Upper bed (pink)
        cv2.line(frame, (0, upper_y_low), (FRAME_W, upper_y_low), (255, 0, 255), 2)
        cv2.line(frame, (0, upper_y_high), (FRAME_W, upper_y_high), (255, 0, 255), 2)   
        # Coverage text
        cv2.putText(
            frame,
            f"Coverage: {coverage_lb*100:.1f}%",
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
