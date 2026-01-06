import cv2
import numpy as np

# Initialize the webcam feed (0 is default camera)
cap = cv2.VideoCapture(0)

# Create a window for the sliders
cv2.namedWindow('Control Panel')

# Define the initial values for the trackbars (default values)
initial_values = [0, 120, 70, 10, 255, 255]

# Create trackbars for adjusting the HSV range
cv2.createTrackbar('Lower H', 'Control Panel', initial_values[0], 179, lambda x: None)
cv2.createTrackbar('Lower S', 'Control Panel', initial_values[1], 255, lambda x: None)
cv2.createTrackbar('Lower V', 'Control Panel', initial_values[2], 255, lambda x: None)
cv2.createTrackbar('Upper H', 'Control Panel', initial_values[3], 179, lambda x: None)
cv2.createTrackbar('Upper S', 'Control Panel', initial_values[4], 255, lambda x: None)
cv2.createTrackbar('Upper V', 'Control Panel', initial_values[5], 255, lambda x: None)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Get current values of the trackbars
    lower_h = cv2.getTrackbarPos('Lower H', 'Control Panel')
    lower_s = cv2.getTrackbarPos('Lower S', 'Control Panel')
    lower_v = cv2.getTrackbarPos('Lower V', 'Control Panel')

    upper_h = cv2.getTrackbarPos('Upper H', 'Control Panel')
    upper_s = cv2.getTrackbarPos('Upper S', 'Control Panel')
    upper_v = cv2.getTrackbarPos('Upper V', 'Control Panel')

    # Define the lower and upper bounds for color segmentation
    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])

    # Convert the captured frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the specified color range
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Apply the mask to the original frame to get the segmented result
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original frame and the segmented frame
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Segmented Frame', segmented_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
q
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
