import os
import cv2
import numpy as np
from zipfile import ZipFile

def extract_fraction(zip_path, output_path, fraction=10):
    """Extract a fraction of files from a zip archive."""
    with ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        portion_size = max(1, total_files // fraction)

        # Randomly shuffle and select a subset of files
        random.shuffle(file_list)
        selected_files = file_list[:portion_size]

        # Extract only the selected files
        for file in selected_files:
            zip_ref.extract(file, output_path)

        print(f'Extracted {portion_size} out of {total_files} files.')

def video_to_npy(video_path, output_path, frame_size=(224, 224), num_frames=16):
    """Convert a video to a numpy array with specified number of frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)

    cap.release()
    return np.array(frames) if frames else None
