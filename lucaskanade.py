import cv2
import numpy as np


# Parameters for ShiTomasi corner detection (goodFeaturesToTrack)
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,  # Use pyramidal LK
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Capture video from file or camera
cap = cv2.VideoCapture('/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/20240808182614 DL iPhone.mov')

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Find initial points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing (same size as frame)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (pyramidal Lucas-Kanade)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw tracks
        for i, (new_pt, old_pt) in enumerate(zip(good_new, good_old)):
            a, b = new_pt.ravel()
            c, d = old_pt.ravel()
            # Convert coordinates to integers
            a, b, c, d = int(a), int(b), int(c), int(d)
            # Draw the motion vector
            mask = cv2.line(mask, (a, b), (c, d), color[i % len(color)].tolist(), 2)
            # Draw current point
            frame = cv2.circle(frame, (a, b), 5, color[i % len(color)].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('Optical Flow - Lucas-Kanade', img)
    else:
        cv2.imshow('Optical Flow - Lucas-Kanade', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
