import cv2
import numpy as np
import os

## Cropping and rotations settings worked on my laptop but don't work on the pi (unclear why but not relevant to our usecase)
# Open the video
video_path = "/home/upweller/Desktop/Bottle_upweller/bottle-upweller/videos/Video from ARC.MP4" #(couldn't get relative path on pi work edit to local storage path) 
#video_path = os.path.join(os.path.dirname(__file__), "videos", "Video from ARC.mp4") #(uncomment on Jess Laptop)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

# Define fixed HSV range values
h_min = 0
s_min = 12
v_min = 0
h_max = 90
s_max = 255
v_max = 255
lower_hsv = np.array([h_min, s_min, v_min])  # Example values, replace with your own
upper_hsv = np.array([h_max, s_max, v_max])

# Initialize variables for tracking oyster_height and frames
oyster_heights = []  # To store the oyster_height values of the last 10 frames
frame_count = 0

# Set cropping region (x, y, width, height)

crop_x, crop_y, crop_w, crop_h = 100, 0, 750, 500  # Example values, adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Fix orientation if the video is sideways (rotate 90 degrees clockwise)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # You can also use ROTATE_90_COUNTERCLOCKWISE

    # Crop the frame
    cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # Convert the cropped frame to HSV
    hsv_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the largest bounding box
    largest_area = 0
    oyster_height = None
    bounding_image = cropped_frame.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        min_width = 200
        min_height = 200 
        x, y, w, h = cv2.boundingRect(contour)
        if area > largest_area and w >= min_width and h >= min_height:  # Check if this contour is the largest so far
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            oyster_height = y  # Update the y value of the largest bounding box

            # Draw the largest bounding box
            cv2.rectangle(bounding_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Print bounding box information
            print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")

    # Update the oyster_height list with the current frame's oyster_height
    if oyster_height is not None:
        oyster_heights.append(oyster_height)
        if len(oyster_heights) > 10:
            oyster_heights.pop(0)  # Keep only the last 10 heights

        # Calculate the average oyster_height over the last 10 frames
        average_oyster_height = np.mean(oyster_heights)

        # Draw a line at the average oyster_height
        cv2.line(bounding_image, (0, int(average_oyster_height)), (bounding_image.shape[1], int(average_oyster_height)), (0, 0, 255), 2)
        print(f"Average oyster height (y-coordinate) over the last 10 frames: {average_oyster_height}")

    # Display the frames
    #cv2.imshow("Original Frame", frame)
    #cv2.imshow("Cropped Frame", cropped_frame)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Bounding Box with Line", bounding_image)

    # Wait for a key press and close if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
