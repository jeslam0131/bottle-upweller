import numpy as np
import cv2 as cv
import csv
import os


video_path = r"C:\Users\angzh\OneDrive\Desktop\UROP IAP 2026\WIN_20260106_10_22_59_Pro.mp4"
##cap = cv.VideoCapture(0) --> live
cap = cv.VideoCapture(video_path)

###for ease? video_path = input("Enter video path: ")
##cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video:", video_path)
    raise SystemExit

fps = cap.get(cv.CAP_PROP_FPS) #frames per second: pixels moved per frame --> pixels per second
print("FPS:", fps)

points_per_second = 15.0 #target sampling rate
samples_per_frame = max(int(round(fps/points_per_second)), 1) #sampling every x frames (samples/second)
print("Sampling every", samples_per_frame, "frames")

ret, frame1 = cap.read()
if not ret or frame1 is None:
    print("Cannot read first frame")
    cap.release()
    exit()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

cv.namedWindow("Original", cv.WINDOW_NORMAL)
cv.namedWindow("Optical Flow", cv.WINDOW_NORMAL)

##choice 1 --> full frame
frame = prvs.shape
height = frame[0] ##height of frame
width = frame[1] ##weight of frame

#choice 2 --> crop out tube


n_slices = 20
slice_height = height // n_slices

frame_counter = 0
times = [] #time (secs) for each saved sample
slice_vals = [] #list of lists: each row = [val_slice0, ..., val_slice19]
percentile = 10 #10th percentile

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(1):
    ret, frame2 = cap.read()
    if not ret or frame2 is None:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    frame_counter += 1

    #Collect data every samples_per_frame frames
    if frame_counter % samples_per_frame == 0:
        t = frame_counter / fps
        row = []

        for i in range(n_slices):
            y1 = i*slice_height
            if i == n_slices - 1:
                y2 = height        # last slice takes all remaining pixels
            else:
                y2 = (i + 1) * slice_height

            slice_mag = mag[y1:y2, :] #pixels in horiz band

            if percentile == 0: #if i set percentile to 0 (min)
                val = float(np.min(slice_mag))
            else:
                val = float(np.percentile(slice_mag, percentile))

            row.append(val)

        times.append(t)
        slice_vals.append(row)

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow("Original", frame2)
    cv.imshow("Optical Flow", bgr)
    #cv.imshow('frame2', bgr)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)

    prvs = next
cap.release()

out_csv = os.path.splitext(video_path)[0] + f"_slices_p{percentile}.csv"

cv.destroyAllWindows()
