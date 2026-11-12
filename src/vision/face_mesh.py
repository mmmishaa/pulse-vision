from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import numpy as np

from src.utils.paths import FACE_MESH_MODEL_WEIGHTS_PATH

class FaceMeshDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=str(FACE_MESH_MODEL_WEIGHTS_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

    def frame_preprocessing(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return mp_image

    def process(self, frame, timestamp_ms):
        rgb_frame = self.frame_preprocessing(frame)

        results = self.face_mesh.detect_for_video(rgb_frame, timestamp_ms)

        if results.face_landmarks:
            return results.face_landmarks[0]
        else:
            return None

    def landmarks_to_pixels(self, roi_landmarks: list, frame_dim: tuple) -> tuple[int, int, int, int]:
        height, width = frame_dim
        lm_x0, lm_y0, lm_x1, lm_y1 = roi_landmarks
        x0 = int(lm_x0 * width)
        y0 = int(lm_y0 * height)
        x1 = int(lm_x1 * width)
        y1 = int(lm_y1 * height)
        
        return x0, y0, x1, y1


    def get_roi_box(self, frame: np.ndarray, timestamp_ms, roi_indices: list = (10, 67, 109, 151, 297, 338)) -> tuple[int, int, int, int]:
        landmarks = self.process(frame, timestamp_ms)
        if landmarks is None: 
            return None

        roi_landmarks = [landmarks[i] for i in roi_indices]
        x = [p.x for p in roi_landmarks]
        y = [p.y for p in roi_landmarks]

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        x0, y0, x1, y1 = self.landmarks_to_pixels(
            roi_landmarks=(x_min, y_min, x_max, y_max),
            frame_dim=frame.shape[:2]
        )

        return x0, y0, x1, y1

    def get_face_box(self, frame: np.ndarray, timestamp_ms) -> tuple[int, int, int, int]:
        landmarks = self.process(frame, timestamp_ms)
        if landmarks is None: 
            return None

        x = [p.x for p in landmarks]
        y = [p.y for p in landmarks]

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        x0, y0, x1, y1 = self.landmarks_to_pixels(
            roi_landmarks=(x_min, y_min, x_max, y_max),
            frame_dim=frame.shape[:2]
        )

        return x0, y0, x1, y1
        