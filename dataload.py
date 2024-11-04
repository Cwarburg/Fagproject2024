import os
import json
import cv2
from tqdm import tqdm

def process_annotations(json_folder, video_folder, output_image_folder, output_label_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc='Processing annotation files'):
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the video filename from the JSON data
        video_file_path = data['Blob']['video_file_path']
        
        # Replace backslashes with forward slashes to handle Windows paths on Unix
        video_file_path_unix = video_file_path.replace('\\', '/')
        
        # Extract the filename from the path
        video_filename = os.path.basename(video_file_path_unix)
        
        # Construct the full path to the video file
        video_path = os.path.join(video_folder, video_filename)

        # Verify that the video file exists
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist. Skipping.")
            continue

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video {video_path}. Skipping.")
            continue

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        annotations = data.get('Annotations', {})
        annotated_frames = sorted([int(k) for k in annotations.keys() if k.isdigit()])
        annotated_frames_set = set(annotated_frames)

        current_frame = 0
        pbar = tqdm(total=total_frames, desc=f'Processing {video_filename}', leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in annotated_frames_set:
                try:
                    image_filename = f"{os.path.splitext(video_filename)[0]}_frame_{current_frame}.jpg"
                    image_path = os.path.join(output_image_folder, image_filename)
                    cv2.imwrite(image_path, frame)

                    boxes = annotations[str(current_frame)]
                    yolo_annotations = []

                    for box in boxes:
                        category = box.get('Category')
                        if category != 'Ball':
                            continue

                        class_id = 0
                        center_u = box.get('CenterU', 0)
                        center_v = box.get('CenterV', 0)
                        bbox_width = box.get('Width', 0)
                        bbox_height = box.get('Height', 0)

                        x_center = center_u / width
                        y_center = center_v / height
                        w = bbox_width / width
                        h = bbox_height / height

                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {w} {h}")

                    label_filename = f"{os.path.splitext(video_filename)[0]}_frame_{current_frame}.txt"
                    label_path = os.path.join(output_label_folder, label_filename)
                    with open(label_path, 'w') as label_file:
                        label_file.write('\n'.join(yolo_annotations))
                except Exception as e:
                    print(f"Error processing frame {current_frame} in video {video_filename}: {e}")

            current_frame += 1
            pbar.update(1)

            # Optional: Stop after last annotated frame
            if current_frame > max(annotated_frames):
                break

        cap.release()
        pbar.close()

if __name__ == '__main__':
    import os

    # Set your directories here
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    json_folder = '/Users/christianwarburg/Desktop/Fagproject/Annotations/AnnotationFiles'   # Update with your annotations folder name
    video_folder = '/Users/christianwarburg/Desktop/Fagproject/videos'    # Update with your videos folder name
    output_image_folder = '/Users/christianwarburg/Desktop/Fagproject/dataset/images/train'
    output_label_folder = '/Users/christianwarburg/Desktop/Fagproject/dataset/labels/train'

    # Debugging: Print paths to verify
    print(f"JSON folder path: {json_folder}")
    print(f"Video folder path: {video_folder}")
    print(f"Output image folder: {output_image_folder}")
    print(f"Output label folder: {output_label_folder}")

    # Proceed if paths exist
    if os.path.exists(json_folder) and os.path.exists(video_folder):
        process_annotations(json_folder, video_folder, output_image_folder, output_label_folder)
    else:
        print("One or more paths do not exist. Please check the paths and try again.")
