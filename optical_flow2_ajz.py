import numpy as np
import cv2 as cv
import csv
import os

##video_path = r"C:\Users\angzh\OneDrive\Desktop\UROP IAP 2026\IMG_8574.mov" ## test video

video_path = r"C:\Users\angzh\OneDrive\Desktop\UROP IAP 2026\WIN_20260106_10_24_28_Pro.mp4" ##actual trial video

##cap = cv.VideoCapture(0) ##live
cap = cv.VideoCapture(video_path)

points_per_second = 10.0 #data points per second to record
n_slices = 20  #num ofhorizontal slivers
percentile = 10
save_debug_overlay = False #set True if you want to save a sample overlay image

tube_diameter_mm = 100 #need to measure

#files to save/reuse clicking work
mask_file = os.path.splitext(video_path)[0] + "_tube_mask.npy"
calib_file = os.path.splitext(video_path)[0] + "_tube_width_px.txt"

#polygon mask from clicks
def collect_polygon_mask(frame_bgr, window_name="Click tube outline"):
    """
    Click 4+ points around the tube interior boundary.
    ENTER = finish, BACKSPACE = undo, ESC = cancel.
    Returns: mask (uint8 0/255)
    """
    points = []
    disp = frame_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print("Point:", (x, y))

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, on_mouse)

    while True:
        disp = frame_bgr.copy()

        #draw points
        for p in points:
            cv.circle(disp, p, 4, (0, 255, 0), -1)

        #draw polyline preview
        if len(points) > 1:
            cv.polylines(disp, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)

        cv.putText(disp, "Click tube outline (inside). ENTER=done | BACKSPACE=undo | ESC=cancel",
                   (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow(window_name, disp)
        k = cv.waitKey(20) & 0xFF

        if k == 13:  #ENTER
            break
        elif k == 8:  #BACKSPACE
            if points:
                points.pop()
        elif k == 27:  #ESC
            cv.destroyWindow(window_name)
            return None

    cv.destroyWindow(window_name)

    if len(points) < 3:
        print("Need at least 3 points to form a polygon.")
        return None

    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


#pixel distance between 2 clicks
def measure_two_points_distance(frame_bgr, window_name="Click 2 points"):
    """
    Click two points (e.g., left inner wall and right inner wall).
    Returns: distance in pixels (float)
    """
    pts = []
    disp = frame_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            print("Clicked:", (x, y))

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, on_mouse)

    while True:
        disp = frame_bgr.copy()

        for p in pts:
            cv.circle(disp, p, 6, (0, 0, 255), -1)

        if len(pts) == 2:
            cv.line(disp, pts[0], pts[1], (0, 0, 255), 2)
            dist = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
            cv.putText(disp, f"Distance: {dist:.1f}px (ENTER to accept)",
                       (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv.putText(disp, "Click 2 points across tube width. ENTER=done | ESC=cancel",
                   (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv.imshow(window_name, disp)
        k = cv.waitKey(20) & 0xFF

        if k == 13 and len(pts) == 2:  #ENTER
            break
        elif k == 27:  #ESC
            cv.destroyWindow(window_name)
            return None

    cv.destroyWindow(window_name)
    dist = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    return dist


#open video and read first frame
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video:", video_path)
    raise SystemExit

fps = cap.get(cv.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0  # fallback
print("FPS:", fps)

sample_stride = max(int(round(fps / points_per_second)), 1)
print("Sampling every", sample_stride, "frames (~", fps / sample_stride, "samples/sec )")

ret, frame1 = cap.read()
if not ret or frame1 is None:
    print("Cannot read first frame")
    cap.release()
    raise SystemExit

#load/create mask
if os.path.exists(mask_file):
    mask = np.load(mask_file)
    print("Loaded mask:", mask_file)
else:
    mask = collect_polygon_mask(frame1, "Mask: click tube outline")
    if mask is None:
        cap.release()
        raise SystemExit("Mask not created.")
    np.save(mask_file, mask)
    print("Saved mask:", mask_file)

#load/measure tube width in pixels (optional)
tube_width_px = None
if os.path.exists(calib_file):
    with open(calib_file, "r") as f:
        tube_width_px = float(f.read().strip())
    print("Loaded tube width (px):", tube_width_px)
else:
    #only prompt for measurement if you want to start calibration now
    #(You can skip by pressing ESC)
    tube_width_px = measure_two_points_distance(frame1, "Calibration: click tube walls (optional)")
    if tube_width_px is not None:
        with open(calib_file, "w") as f:
            f.write(str(tube_width_px))
        print("Saved tube width (px):", tube_width_px)


#decide conversion px/sec or mm/sec
mm_per_pixel = None
if tube_diameter_mm is not None and tube_width_px is not None and tube_width_px > 0:
    mm_per_pixel = tube_diameter_mm / tube_width_px
    print("mm_per_pixel:", mm_per_pixel)

units = "px_per_sec" if mm_per_pixel is None else "mm_per_sec"
print("Output units:", units)

#prep for optical flow
prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
H, W = prev_gray.shape

slice_height = H // n_slices

#prep HSV for visualization
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

#windows
#####cv.namedWindow("Original", cv.WINDOW_NORMAL) ##comment out if don't need original
cv.namedWindow("Optical Flow", cv.WINDOW_NORMAL)
cv.namedWindow("Debug (mask+slivers)", cv.WINDOW_NORMAL)

#storage
frame_counter = 0
times = []
slice_vals = []


#main loop
while True:
    ret, frame2 = cap.read()
    if not ret or frame2 is None:
        print("No frames grabbed!")
        break

    frame_counter += 1
    next_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    #apply mask: ignore outside tube
    mag_masked = mag.astype(np.float32).copy()
    mag_masked[mask == 0] = np.nan

    #collects data every sample_stride frames
    if frame_counter % sample_stride == 0:
        t = frame_counter / fps
        row = []

        for i in range(n_slices):
            y1 = i * slice_height
            y2 = H if i == n_slices - 1 else (i + 1) * slice_height

            slice_mag = mag_masked[y1:y2, :]

            #uses nanpercentile so outside-tube pixels don't count
            valid = np.isfinite(slice_mag)

            if not np.any(valid):
                val_px_per_frame = np.nan #if no tube pixels in this slice

                # valid = np.isfinite(slice_mag)
                # if not np.any(valid):
                #    val_px_per_frame = np.nan #if no tube pixels in this slice
                # else:
                #     val_px_per_frame = float(
                #     np.nanpercentile(slice_mag, percentile)
            else:
                val_px_per_frame = float(
            np.nanpercentile(slice_mag, percentile)
    )

            # Convert to px/sec
            val_px_per_sec = val_px_per_frame * fps

            # Convert to mm/sec if calibration exists
            if mm_per_pixel is not None:
                val_out = val_px_per_sec * mm_per_pixel
            else:
                val_out = val_px_per_sec

            row.append(val_out)

        times.append(t)
        slice_vals.append(row)

    #visualization: optical flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(np.nan_to_num(mag_masked, nan=0.0), None, 0, 255, cv.NORM_MINMAX)
    flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    #debug overlay: show mask + sliver lines on original
    debug = frame2.copy()
    mask_col = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    overlay = debug.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.2).astype(np.uint8)  # darken outside tube

    for i in range(1, n_slices):
        y = i * slice_height
        cv.line(overlay, (0, y), (W - 1, y), (255, 255, 255), 1)

    cv.imshow("Original", frame2)
    cv.imshow("Optical Flow", flow_bgr)
    cv.imshow("Debug (mask+slivers)", overlay)

    k = cv.waitKey(30) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('s'):
        cv.imwrite("debug_overlay.png", overlay)
        cv.imwrite("optical_flow_vis.png", flow_bgr)
        print("Saved debug_overlay.png and optical_flow_vis.png")

    prev_gray = next_gray

cap.release()
cv.destroyAllWindows()

#write CSV
out_csv = os.path.splitext(video_path)[0] + f"_slice_p{percentile}_{units}.csv"

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_sec"] + [f"slice_{i}_{units}" for i in range(n_slices)])
    for t, row in zip(times, slice_vals):
        writer.writerow([t] + row)

print("Saved CSV:", out_csv)
print("Collected samples:", len(times))
