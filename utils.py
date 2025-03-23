import cv2
import numpy as np

def get_optical_flow(video_path, num_frames=16, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    flow_frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_frames.append(flow)

        prev_gray = gray

    cap.release()
    return np.array(flow_frames) if flow_frames else None
