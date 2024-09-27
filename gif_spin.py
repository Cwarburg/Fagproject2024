import cv2
import numpy as np
from PIL import Image

# Load GIF and extract frames
gif_path = "/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/ballspinning.gif"
gif = Image.open(gif_path)
frames = []
try:
    while True:
        frame = gif.copy()
        frames.append(frame)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

# Convert frames to OpenCV format
opencv_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in frames]
#Initiliserer blob detector
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 500
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs and track displacements
keypoints_frames = []
displacements = []

for i, frame in enumerate(opencv_frames):
    keypoints = detector.detect(frame)
    keypoints_frames.append(keypoints)

    # Draw keypoints on the frame
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Blobs", im_with_keypoints)
    cv2.waitKey(100)
    
    if i > 0:
        prev_kp = keypoints_frames[i-1]
        curr_kp = keypoints_frames[i]
        for kp_prev in prev_kp:
            closest_blob = min(curr_kp, key=lambda kp_curr: np.linalg.norm(np.array(kp_curr.pt) - np.array(kp_prev.pt)))
            displacement = np.linalg.norm(np.array(closest_blob.pt) - np.array(kp_prev.pt))
            displacements.append(displacement)

# Close all OpenCV windows
cv2.destroyAllWindows()

# Calculate angular velocity from displacements (assuming ball radius and frame rate)
ball_radius_pixels = frame.shape[0] / 2 # By looking at the ball in the gif, I can see that the ball edges touches each side of the framegif_sp
frame_rate = 25  # Example frame rate, adjust based on your GIF
angular_displacements = [d / ball_radius_pixels for d in displacements]
time_between_frames = 1 / frame_rate
angular_velocity = [theta / time_between_frames for theta in angular_displacements]

# Convert to RPS and RPM
rps = [omega / (2 * np.pi) for omega in angular_velocity]
rpm = [r * 60 for r in rps]
average_rpm = np.mean(rpm)

print(f"Estimated RPM: {rpm}\n Average RPM: {average_rpm}")
