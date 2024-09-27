import cv2
import numpy as np
import matplotlib.pyplot as plt
# Parameters for Lucas-Kanade optical flow
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Set up the blob detector with default parameters
blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByArea = True
blob_params.minArea = 200  # Adjust this as per your requirements
blob_params.filterByCircularity = False
blob_params.filterByConvexity = False
blob_params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(blob_params)

# Load the GIF (replace with actual file path)
cap = cv2.VideoCapture('/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/ballspinning.gif')

# Get the first frame
ret, old_frame = cap.read()

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect blobs in the first frame
keypoints = detector.detect(old_gray)

if keypoints:
    p0 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when there are no more frames

        # Convert the current frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the previous and current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Convert to integers for OpenCV drawing functions
            a, b = int(a), int(b)
            c, d = int(c), int(d)
            cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
            cv2.line(frame, (a, b), (c, d), (255, 0, 0), 2)

        # Display the frame with the tracking
        cv2.imshow('Optical Flow', frame)

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Break on ESC key press
        if cv2.waitKey(1000) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("No blobs detected in the first frame!")

# Plot all frames in the GIF
    

