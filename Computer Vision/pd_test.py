import cv2
import numpy as np
import time
import serial
from collections import deque

# --------------------------------------------------------
# SERIAL CONNECTION (9600 baud, stable)
# --------------------------------------------------------
try:
    ser = serial.Serial('COM3', 9600, timeout=0.05)
    time.sleep(2)
except Exception as e:
    print("ERROR opening serial:", e)
    ser = None

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------
AOI_X = 430
AOI_Y = 175
AOI_W = 40
AOI_H = 115
CM_PER_PIXEL = 25 / AOI_H

lower = np.array([0, 0, 0])
upper = np.array([80, 255, 255])

alpha = 0.3
smooth_top = None

history = deque()
WINDOW_SECONDS = 3.0

SETPOINT = 12.75
TOLERANCE = 0.5
Kp = 20
Kd = 8

last_error = 0
last_time = time.time()

# --------------------------------------------------------
# SERIAL RATE LIMIT
# --------------------------------------------------------
SERIAL_INTERVAL = 0.05  # send max 20 commands/sec
last_serial_send = 0

# --------------------------------------------------------
# CAMERA
# --------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting PD loop...")

# ========================================================
# ======================= MAIN LOOP ======================
# ========================================================
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            continue

        frame_disp = frame.copy()

        # Draw AOI
        cv2.rectangle(frame_disp,
                      (AOI_X, AOI_Y),
                      (AOI_X + AOI_W, AOI_Y + AOI_H),
                      (0, 0, 255), 1)

        roi = frame[AOI_Y:AOI_Y+AOI_H, AOI_X:AOI_X+AOI_W]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Morph cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Ignore top 10%
        valid_mask = mask.copy()
        valid_mask[:int(AOI_H*0.10), :] = 0

        ys, xs = np.where(valid_mask == 255)
        roi_marked = roi.copy()

        if len(ys) > 0:
            top_y = int(np.min(ys))

            # Smooth
            if smooth_top is None:
                smooth_top = top_y
            else:
                smooth_top = int(alpha * top_y + (1 - alpha) * smooth_top)

            height_cm = (AOI_H - smooth_top) * CM_PER_PIXEL

            cv2.line(roi_marked, (0, smooth_top), (AOI_W, smooth_top), (0, 255, 0), 2)
            cv2.putText(roi_marked, f"{height_cm:.1f} cm",
                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # HISTORY BUFFER
            now = time.time()
            history.append((now, height_cm))
            while history and now - history[0][0] > WINDOW_SECONDS:
                history.popleft()

            heights = [h for _, h in history]

            if len(heights) > 1:
                avg_height = np.mean(heights)
                height_range = np.max(heights) - np.min(heights)

                cv2.putText(roi_marked, f"avg {avg_height:.1f}",
                            (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(roi_marked, f"rng {height_range:.1f}",
                            (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                # --------------------------------------------------------
                # PD CONTROLLER
                # --------------------------------------------------------
                dt = now - last_time
                error = SETPOINT - height_cm
                d_error = (error - last_error) / dt if dt > 0 else 0

                command = Kp * error + Kd * d_error
                last_error = error
                last_time = now

                if abs(error) < TOLERANCE:
                    command = 0

                command = max(min(command, 100), -100)

                # --------------------------------------------------------
                # SERIAL DRAIN + RATE LIMIT + SAFE SEND
                # --------------------------------------------------------
                if ser:
                    try:
                        # DRAIN INPUT BUFFER
                        if ser.in_waiting > 0:
                            ser.read(ser.in_waiting)

                        # RATE LIMITED WRITE
                        if (now - last_serial_send) >= SERIAL_INTERVAL:
                            msg = f"CMD {int(command)}\n"
                            ser.write(msg.encode())
                            last_serial_send = now

                    except Exception as e:
                        print("Serial exception:", e)

                print(f"height={height_cm:.2f} | cmd={int(command)}")

        # DISPLAY
        frame_disp[AOI_Y:AOI_Y+AOI_H, AOI_X:AOI_X+AOI_W] = roi_marked
        cv2.imshow("Height", frame_disp)
        cv2.imshow("ROI", roi_marked)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as main_error:
        print("MAIN LOOP ERROR:", main_error)

# CLEANUP
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
