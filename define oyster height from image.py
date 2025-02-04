import cv2 
import matplotlib.pyplot as plt
import numpy as np


def nothing(x):
    "callback function for trackbar"
    pass

image_path= r"C:\Users\jessi\Documents\research SM\videos\picARC1.png"
full_image= cv2.imread(image_path)
y=100
x=0
h=550
w=300
image =full_image[x:w,y:h]
#img.show()
hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#cv2.namedWindow("Threshold Adjuster")
# Create trackbars for HSV ranges
""" cv2.createTrackbar("H Min", "Threshold Adjuster", 0, 179, nothing)
cv2.createTrackbar("H Max", "Threshold Adjuster", 179, 179, nothing)
cv2.createTrackbar("S Min", "Threshold Adjuster", 0, 255, nothing)
cv2.createTrackbar("S Max", "Threshold Adjuster", 255, 255, nothing)
cv2.createTrackbar("V Min", "Threshold Adjuster", 0, 255, nothing)
cv2.createTrackbar("V Max", "Threshold Adjuster", 255, 255, nothing)

while True:
    # Get current positions of trackbars
    h_min = cv2.getTrackbarPos("H Min", "Threshold Adjuster")
    h_max = cv2.getTrackbarPos("H Max", "Threshold Adjuster")
    s_min = cv2.getTrackbarPos("S Min", "Threshold Adjuster")
    s_max = cv2.getTrackbarPos("S Max", "Threshold Adjuster")
    v_min = cv2.getTrackbarPos("V Min", "Threshold Adjuster")
    v_max = cv2.getTrackbarPos("V Max", "Threshold Adjuster")

    # Define the HSV range
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Threshold the HSV image
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the original, mask, and result images
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Image", result)

    # Print current HSV values
    print(f"Lower HSV: {lower_hsv}, Upper HSV: {upper_hsv}", end="\r")

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break """


 # Define the HSV range
h_min = 0
s_min = 12
v_min = 0
h_max = 90
s_max = 255
v_max = 255
lower_hsv = np.array([h_min, s_min, v_min])  # Example values, replace with your own
upper_hsv = np.array([h_max, s_max, v_max])

# Threshold the HSV image
mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Finding largest bounding box
largest_area=0
y_box= None

bounding_image = image.copy()
for contour in contours:
    area = cv2.contourArea(contour)
    if  area > largest_area:  #Check for largest area
        largest_area=area
        x, y, w, h = cv2.boundingRect(contour)
        oyster_height = y
        bounding_image = image.copy()
        cv2.rectangle(bounding_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

if oyster_height is not None:
    print(f"The y-coordinate of the largest bounding box is: {y_box}")
    cv2.line(bounding_image, (0, oyster_height), (bounding_image.shape[1], oyster_height), (0, 0, 255), 2)

else:
    print("No bounding box found.")
    
# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Filtered Image", result)
cv2.imshow("Bounding Box", bounding_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
