import cv2
from typing import Generator

def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(camera_id)

    if not cam.isOpened():
        cam.release()
        raise RuntimeError("Не удалось открыть камеру!")

    return cam

def close_camera(cam: cv2.VideoCapture):
    cam.release()

def get_frames(cam: cv2.VideoCapture) -> Generator:
    while True:
        ret, frame = cam.read()

        if not ret:
            cam.release()
            raise RuntimeError("Не удалось получить кадр!")

        yield frame