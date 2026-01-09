import cv2
from numpy.typing import NDArray
from cv2.typing import MatLike
import numpy as np

def read_videos(video_path: str) -> list[MatLike]:
    cap = cv2.VideoCapture(video_path)
    frames: list[MatLike] = []
    while True:
        ret, frame = cap.read()
        if not ret: # The video is ended?
            break
        frames.append(frame)

    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') # We define an output format
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release