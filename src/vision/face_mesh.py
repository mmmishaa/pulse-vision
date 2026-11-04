import mediapipe as mp
import cv2
import numpy as np

class FaceMeshDetector:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def frame_preprocessing(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def process(self, frame):
        rgb_frame = self.frame_preprocessing(frame)

        results = self.face_mesh(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
        else:
            raise RuntimeError("На кадре не обнаружено лицо!")

        return face_landmarks.landmark

    def landmarks_to_pixels(self, roi_landmarks: list, frame_dim: tuple):
        height, width = frame_dim
        lm_x0, lm_y0, lm_x1, lm_y1 = roi_landmarks
        x0 = int(lm_x0 * width)
        y0 = int(lm_y0 * height)
        x1 = int(lm_x1 * width)
        y1 = int(lm_y1 * height)
        
        return x0, y0, x1, y1


    def get_roi_box(self, frame: np.ndarray, roi_indices: list = (10, 67, 109, 151, 297, 338)) -> tuple[float, float, float, float]:
        landmarks = self.process(frame)
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

        