from src.vision.face_mesh import FaceMeshDetector
import numpy as np
from pathlib import Path
import cv2

def preprocess(frames_file: Path, ppg_array: np.ndarray, frames_path: Path, ppg_path: Path):
    stream = cv2.VideoCapture(str(frames_file))
    face_mesh_detector = FaceMeshDetector()

    frame_idx = 0
    faces = []
    img_h, img_w = None, None
    checked_frames = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret: 
                break

            if img_h is None:
                img_h, img_w = frame.shape[:2]

            if frame_idx % 15 == 0:
                checked_frames += 1
                face_box = face_mesh_detector.get_face_box(frame=frame, timestamp_ms=int(frame_idx * 1000 / 30))
                if face_box is not None:
                     faces.append(face_box)
            frame_idx += 1

    finally:
        stream.release()

    if len(faces) == 0:
        return None, None, None
    
    faces = np.array(faces)
    xmin, ymin = faces[:, :2].min(axis=0)
    xmax, ymax = faces[:, 2:].max(axis=0)

    cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2

    max_span = int(max(xmax - xmin, ymax - ymin) * 1.1)
    side = min(img_h, img_w, max_span)

    x0 = int(cx - side / 2)
    x1 = x0 + side
    y0 = int(cy - side / 2)
    y1 = y0 + side

    if x0 < 0:
        x1 = side
        x0 = 0
    if x1 > img_w:
        x1 = img_w
        x0 = img_w - side

    if y0 < 0:
        y1 = side
        y0 = 0
    if y1 > img_h:
        y1 = img_h
        y0 = img_h - side

    stream = cv2.VideoCapture(str(frames_file))
    cropped_frames = []

    try:
        while True:
            ret, frame = stream.read()
            if not ret: 
                break

            face = frame[y0:y1, x0:x1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (72, 72), cv2.INTER_AREA)
            cropped_frames.append(face)
    
        num_frames = min(len(cropped_frames), len(ppg_array))
        cropped_frames = np.array(cropped_frames)[:num_frames]
        ppg_array = ppg_array[:num_frames]

        np.save(frames_path, cropped_frames)
        np.save(ppg_path, ppg_array)

    finally:
        stream.release()

    return num_frames, checked_frames, len(faces)