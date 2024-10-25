import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import albumentations as A
import cv2
import json
import os


video_path = r"C:\Projects\Annotations\Datasets\PuttingVids\AnnotationFiles"
annotation_files = [os.path.join(video_path, annos) for annos in os.listdir(video_path) if annos.endswith(".json")]


file = annotation_files[0]
def load_annotations(annotation_file):
        
        """
        Takes a json file and returns the data in the file
        """
        with open (annotation_file) as f:
            data = json.load(f)
        return data


def video_to_frame(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def draw_boxes(video,annotation):
    """
    Draws boxes on the frame for the viz
    
    """
    #all video names
    video_path = r'C:\Projects\PuttingVideos'
    annotation_files = [os.path.join(video_path,video) for video in os.listdir('C:\Projects\PuttingVideos')]

    #Get index of frame from the video (list of frames)
    frames = video_to_frame(video)

    annotation = load_annotations(annotation)
    centerU = annotation



    
    return frame

video_name = r'C:\\Projects\\PuttingVideos\\IMG_4943.mov'
annotation = load_annotations()
