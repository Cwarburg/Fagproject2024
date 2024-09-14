import cv2
import numpy as np

# Open the video file
video_path = "/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/20240808182614 DL iPhone.mov"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)

kalman.statePre = np.zeros((4, 1), dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

measurement = np.zeros((2, 1), dtype=np.float32)

# Create a window to display the video
cv2.namedWindow('Golf Ball Tracking')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the ROI
    roi = frame[int(height/2):height, 0:width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Color thresholds
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 800:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y + height / 2))
            radius = int(radius)

            # Update measurement
            measurement[0] = np.float32(center[0])
            measurement[1] = np.float32(center[1])

            # Correct the Kalman filter
            kalman.correct(measurement)

            # Draw detection
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            detected = True
            break  # Assuming only one ball

    if not detected:
        # Predict the next position
        prediction = kalman.predict()
        predicted_center = (int(prediction[0]), int(prediction[1]))

        # Draw the predicted position
        cv2.circle(frame, predicted_center, 5, (0, 0, 255), -1)  # Red dot for prediction
        cv2.putText(frame, 'Predicted', (predicted_center[0] + 10, predicted_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        # Even when detected, predict to update the internal state
        kalman.predict()

    # Display the frame
    cv2.imshow('Golf Ball Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    print(f'Center {center}')

# Release resources
cap.release()
cv2.destroyAllWindows()
