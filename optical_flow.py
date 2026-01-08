import numpy as np
import cv2 as cv

video_path = r"C:\Users\angzh\OneDrive\Desktop\UROP IAP 2026\WIN_20260106_10_22_59_Pro.mp4"
##cap = cv.VideoCapture("WIN_20260106_10_22_59_Pro")
##cap = cv.VideoCapture(0)
cap = cv.VideoCapture(video_path)

###for ease? video_path = input("Enter video path: ")
##cap = cv.VideoCapture(video_path)

fps = cap.get(cv.CAP_PROP_FPS) #frames per second: pixels moved per frame --> pixels per second
print("FPS:", fps)

points_per_second = 12 #target sampling rate
samples_per_second = max(int(round(fps/points_per_second)), 1) #sampling every x frames (samples/second)
print("Sampling every", samples_per_second, "frames")

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

frame = prvs.shape
height = frame[0] ##height of frame
width = frame[1] ##weight of frame

n_slices = 20
slice_height = height / n_slices


hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()
