import cv2
import numpy as np

# =================CONFIGURATION=================
VIDEO_SOURCE =  r"c:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\low_flow.mp4"

# --- TIGHT ROI: CROP OUT THE BOTTLE EDGES ---
# Based on your video, focus ONLY on the center-mass of the sand
ROI_TOP = 0.65    
ROI_BOT = 0.95    
ROI_LEFT = 0.42   # Increase this to hide the left red tape
ROI_RIGHT = 0.55  # Decrease this to hide the right red tape

# --- Sensitivity Tuning ---
MIN_SPEED = 0.2   # Threshold to kill noise
MAX_SPEED = 12.0  # Range for Green color
# ===============================================

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # CLAHE is the "magic bullet" for low-contrast sand grains
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    ret, first_frame = cap.read()
    if not ret: return

    h, w = first_frame.shape[:2]
    y1, y2 = int(h * ROI_TOP), int(h * ROI_BOT)
    x1, x2 = int(w * ROI_LEFT), int(w * ROI_RIGHT)
    
    # Process first ROI
    prev_roi = first_frame[y1:y2, x1:x2]
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    prev_gray = clahe.apply(prev_gray) 

    # Buffer to average movement over 5 frames (Kills flicker)
    mag_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray) # BOOST CONTRAST

        # Calculate Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # --- THE FIX: ONLY LOOK AT VERTICAL (dy) ---
        # Pump vibration is mostly horizontal/shaking. Upweller flow is vertical.
        dy = flow[..., 1]
        mag = np.abs(dy) 

        # Temporal Smoothing: Average frames to stabilize slow movement
        mag_buffer.append(mag)
        if len(mag_buffer) > 5: mag_buffer.pop(0)
        avg_mag = np.mean(mag_buffer, axis=0)

        # Create Visualization
        mask = avg_mag > MIN_SPEED
        hsv = np.zeros_like(roi)
        hsv[..., 1] = 255
        
        # Speed mapping
        norm_speed = np.clip((avg_mag - MIN_SPEED) / (MAX_SPEED - MIN_SPEED), 0, 1)
        hsv[..., 0] = (norm_speed * 60).astype(np.uint8) 
        hsv[..., 2] = np.where(mask, 255, 0).astype(np.uint8)

        bgr_overlay = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined_roi = cv2.addWeighted(roi, 0.7, bgr_overlay, 0.3, 0)
        
        # Replace only the ROI in the original frame
        frame[y1:y2, x1:x2] = combined_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Show result scaled for your screen
        cv2.imshow('Upweller Flow', cv2.resize(frame, (0,0), fx=0.6, fy=0.6))
        
        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()