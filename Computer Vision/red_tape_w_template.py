import cv2
import numpy as np
import time


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
VIDEO_PATH = r"C:\Users\jessi\MIT Dropbox\Jessica Lam\BUPSY stuff\Test Videos for Optical Flow\low_flow.mp4"
TEMPLATE_PATH = r"C:\Users\jessi\Documents\research SM\Computer Vision\mask_temp.png"

UPDATE_INTERVAL = 20



# =========================================================
# 1. Robust frame grab (prevents early EOF)
# =========================================================
def safe_read(cap):
    ret, frame = cap.read()
    if not ret:
        # Try a second time
        cap.grab()
        ret, frame = cap.retrieve()

    return ret, frame



# =========================================================
# 2. RANSAC line fitting
# =========================================================
def ransac_line(points, iterations=300, threshold=2.0):
    if len(points) < 20:
        return None

    xs = points[:, 0]
    ys = points[:, 1]
    N = len(points)

    best_inliers = 0
    best_model = None

    for _ in range(iterations):
        i1, i2 = np.random.choice(N, 2, replace=False)
        (x1, y1), (x2, y2) = points[i1], points[i2]

        if x1 == x2 and y1 == y2:
            continue

        a = (y1 - y2)
        b = (x2 - x1)
        c = x1 * y2 - x2 * y1

        norm = np.sqrt(a*a + b*b)
        if norm < 1e-6:
            continue

        a /= norm
        b /= norm
        c /= norm

        dist = np.abs(a*xs + b*ys + c)
        inliers = np.sum(dist < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_model = (a, b, c)

    return best_model



# =========================================================
# 3. Intersection
# =========================================================
def intersect(L1, L2):
    if L1 is None or L2 is None:
        return None
    a1,b1,c1 = L1
    a2,b2,c2 = L2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    x = (b1*c2 - b2*c1)/det
    y = (c1*a2 - c2*a1)/det
    return np.array([x, y], dtype=np.float32)



# =========================================================
# 4. Debug draw line
# =========================================================
def draw_line(img, line, color):
    if line is None: return
    a,b,c = line
    H,W = img.shape[:2]
    pts=[]
    for y in [0, H]:
        if abs(a) > 1e-6:
            x = -(b*y + c)/a
            pts.append((int(x), int(y)))
    if len(pts)==2:
        cv2.line(img, pts[0], pts[1], color, 2)



# =========================================================
# 5. Load mask template (white = inside)
# =========================================================
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
if template.shape[2] == 4:
    alpha = template[:,:,3]
    _, roi_mask_template = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
else:
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, roi_mask_template = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

h_temp, w_temp = roi_mask_template.shape

template_corners = np.array([
    [0,0],
    [w_temp,0],
    [w_temp,h_temp],
    [0,h_temp],
], dtype=np.float32)



# =========================================================
# 6. Create resizable windows
# =========================================================
for name in ["Red Mask", "Red Overlay", "Filtered Tape Contours", "ROI"]:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 400)



# =========================================================
# 7. Main loop
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)

roi_mask_global = None
H_global = None
last_update = 0

while True:

    ret, frame = safe_read(cap)
    if not ret:
        print("End of file reached cleanly.")
        break

    Hf, Wf = frame.shape[:2]
    now = time.time()

    # -----------------------------------------------------
    # Step A: HSV threshold
    # -----------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    LOW_S = 60
    LOW_V = 70

    LOW_RED_1  = np.array([0,   LOW_S, LOW_V])
    HIGH_RED_1 = np.array([20,  255,   255])
    LOW_RED_2  = np.array([160, LOW_S, LOW_V])
    HIGH_RED_2 = np.array([179, 255,   255])

    mask1 = cv2.inRange(hsv, LOW_RED_1, HIGH_RED_1)
    mask2 = cv2.inRange(hsv, LOW_RED_2, HIGH_RED_2)
    red_mask = mask1 | mask2

    red_mask = cv2.medianBlur(red_mask, 5)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    cv2.imshow("Red Mask", red_mask)

    overlay = frame.copy()
    overlay[red_mask>0] = (0,0,255)
    cv2.imshow("Red Overlay", overlay)


    # -----------------------------------------------------
    # Step B: Contour filtering
    # -----------------------------------------------------
    contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    verticals=[]
    horizontals=[]
    bottoms=[]

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 200:
            continue

        aspect = h/(w+1e-6)

        if aspect > 2.5 and h > 40:
            verticals.append(c)
        elif aspect < 0.4 and w > 40:
            horizontals.append(c)
        elif area > 800:
            bottoms.append(c)

    contour_debug = frame.copy()
    for c in verticals: cv2.drawContours(contour_debug,[c],-1,(255,0,0),2)
    for c in horizontals: cv2.drawContours(contour_debug,[c],-1,(0,255,0),2)
    for c in bottoms: cv2.drawContours(contour_debug,[c],-1,(0,0,255),2)
    cv2.imshow("Filtered Tape Contours", contour_debug)


    # Only update ROI occasionally
    if (H_global is None) or (now - last_update > UPDATE_INTERVAL):

        if not verticals or not horizontals or not bottoms:
            cv2.imshow("ROI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # Top tape = highest y
        horizontals_sorted = sorted(horizontals, key=lambda c: cv2.boundingRect(c)[1])
        top_tape = horizontals_sorted[0]
        tx,ty,tw,th = cv2.boundingRect(top_tape)

        # Left & right
        v_sorted = sorted(verticals, key=lambda c: cv2.boundingRect(c)[0])
        left_tape  = v_sorted[0]
        right_tape = v_sorted[-1]
        lx,ly,lw,lh = cv2.boundingRect(left_tape)
        rx,ry,rw,rh = cv2.boundingRect(right_tape)

        # Bottom = largest y
        bottoms_sorted = sorted(bottoms, key=lambda c: cv2.boundingRect(c)[1], reverse=True)
        bottom_tape = bottoms_sorted[0]
        bx,by,bw,bh = cv2.boundingRect(bottom_tape)

        TL = np.array([lx + lw/2, ty + th/2], np.float32)
        TR = np.array([rx + rw/2, ty + th/2], np.float32)
        BL = np.array([lx + lw/2, by + bh/2], np.float32)
        BR = np.array([rx + rw/2, by + bh/2], np.float32)

        detected_corners = np.array([TL,TR,BR,BL], np.float32)

        H_global = cv2.getPerspectiveTransform(template_corners, detected_corners)
        warped = cv2.warpPerspective(roi_mask_template, H_global, (Wf,Hf))
        roi_mask_global = warped > 0

        last_update = now


    # -----------------------------------------------------
    # Step C: Apply ROI
    # -----------------------------------------------------
    result = frame.copy()
    if roi_mask_global is not None:
        result[~roi_mask_global] = 0

    cv2.imshow("ROI", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
