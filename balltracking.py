import cv2
import numpy as np

# Open the video file
video_path = "/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/20240808182614 DL iPhone.mov"
cap = cv2.VideoCapture(video_path)

# Create a window to display the video
cv2.namedWindow('Golf Ball Tracking')

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit if the video has ended

    # Convert frame to HSV color space for better color detection
   
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the golf ball (tuned for a white ball)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Create a binary mask where the white regions are isolated
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Find contours of the ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours around the detected golf ball
    for contour in contours:
        # Filter out small contours to ignore noise
        if cv2.contourArea(contour) > 50:  # You may need to adjust this threshold
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw the circle and the center
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)  # Center point

            # Print the coordinates of the ball (optional)
            print(f"Ball center: {center}")

    # Display the frame with the tracked ball
    cv2.imshow('Golf Ball Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
