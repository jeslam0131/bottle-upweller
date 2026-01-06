import cv2
import numpy as np

# Tape constants
TOP_TAPE_THICKNESS_IN = 0.75
TOP_TAPE_THICKNESS_CM = TOP_TAPE_THICKNESS_IN * 2.54

# Expected sizes (inner ROI)
EXPECTED_H_CM = 25
EXPECTED_H_TOL = 25
EXPECTED_W_CM = 10
EXPECTED_W_TOL = 10

def classify_tape_pieces(contours):
    """Classify tape contours into top, bottom, left, right."""
    top = bottom = left = right = None

    print("\n--- NEW FRAME ---")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue  # ignore noise

        aspect = w / float(h)

        print(f"Contour: x={x}, y={y}, w={w}, h={h}, aspect={aspect:.2f}")

        # Horizontal tape = wide + short
        if aspect > 3:
            print(" → horizontal candidate")
            if top is None or y < top[1]:
                top = (x, y, w, h)
            elif bottom is None or y > bottom[1]:
                bottom = (x, y, w, h)

        # Vertical tape = tall + thin
        elif aspect < 0.33:
            print(" → vertical candidate")
            if left is None or x < left[0]:
                left = (x, y, w, h)
            elif right is None or x > right[0]:
                right = (x, y, w, h)

    print("CLASSIFIED:")
    print(" top   :", top)
    print(" bottom:", bottom)
    print(" left  :", left)
    print(" right :", right)
    return top, bottom, left, right


# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # DEBUG: show loose red thresholds
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([165, 50, 50])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"FOUND {len(contours)} red contours")

    # Step 1: Classify pieces
    top, bottom, left, right = classify_tape_pieces(contours)

    # If any are missing → skip ROI processing
    if not (top and bottom and left and right):
        print("❌ Missing one or more tape segments → cannot compute ROI")
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Unpack
    tx, ty, tw, th = top
    bx, by, bw, bh = bottom
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right

    # Scaling from top tape thickness
    if th > 0:
        pixels_per_cm = th / TOP_TAPE_THICKNESS_CM
    else:
        pixels_per_cm = 1

    print(f"pixels_per_cm = {pixels_per_cm:.2f}")

    # ROI boundaries (inner edges)
    roi_left   = lx + lw
    roi_right  = rx
    roi_top    = ty + th
    roi_bottom = by

    # Interior ROI dims in pixels
    roi_w_px = roi_right - roi_left
    roi_h_px = roi_bottom - roi_top

    print(f"ROI px: width={roi_w_px}, height={roi_h_px}")

    if roi_w_px <= 0 or roi_h_px <= 0:
        print("❌ ROI invalid (negative or zero size)")
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Convert to cm
    roi_w_cm = roi_w_px / pixels_per_cm
    roi_h_cm = roi_h_px / pixels_per_cm

    print(f"ROI cm: width={roi_w_cm:.2f}, height={roi_h_cm:.2f}")

    # Validate against ranges
    height_valid = (EXPECTED_H_CM - EXPECTED_H_TOL <= roi_h_cm <= EXPECTED_H_CM + EXPECTED_H_TOL)
    width_valid  = (EXPECTED_W_CM - EXPECTED_W_TOL <= roi_w_cm <= EXPECTED_W_CM + EXPECTED_W_TOL)

    print(f"height_valid = {height_valid}, width_valid = {width_valid}")

    # Draw classified tape pieces
    cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (0,255,255), 2)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,255,255), 2)
    cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (255,255,0), 2)
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255,255,0), 2)

    # Draw ROI
    color = (0,255,0) if (height_valid and width_valid) else (0,0,255)
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), color, 3)

    # Print dimensions on frame
    cv2.putText(frame, f"W: {roi_w_cm:.1f} cm  H: {roi_h_cm:.1f} cm",
                (roi_left, roi_top-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
