import cv2
import time
import numpy as np
import torch
from collections import deque

from src.capture.camera import open_camera, close_camera, get_frames
from src.vision.face_mesh import FaceMeshDetector
from scripts.download_weights import download_all_weights
from src.dsp.dsp_processor import estimate_bpm_from_diff_wave
from src.data.preprocess import prepare_general_face_box
from src.models.model import EfficientPhys_C
from src.utils.paths import EFFICIENTPHYS_WEIGHTS_PATH

cam = open_camera()
download_all_weights()

face_mesh_detector = FaceMeshDetector()

start_time = time.time()
last_ts_ms = 0

t_start = time.perf_counter()

is_calibrated = False
cal_face_boxes = []

device = "cpu"

model = EfficientPhys_C(n_frames=100).to(device)
model.load_state_dict(state_dict=torch.load(EFFICIENTPHYS_WEIGHTS_PATH, weights_only=True, map_location=device))
model.eval()

frames_buffer = deque(maxlen=100)

for frame in get_frames(cam):
    curr_ts_ms = int((time.time() - start_time) * 1000)
    if curr_ts_ms <= last_ts_ms:
        curr_ts_ms = last_ts_ms + 1
    last_ts_ms = curr_ts_ms

    if not is_calibrated:
        face_box = face_mesh_detector.get_face_box(frame, curr_ts_ms)
        if face_box is not None:
            cal_face_boxes.append(face_box)

        if len(cal_face_boxes) == 300:
            is_calibrated = True
            general_face_box = prepare_general_face_box(np.array(cal_face_boxes), *frame.shape[:2])
            x0, y0, x1, y1 = general_face_box

            t_end = time.perf_counter()
            real_fps = 300 / (t_end - t_start)

        text = f"Face box calibration ({len(cal_face_boxes)}/{300})..."
        color = (0, 255, 255)

    else:
        frame_face = frame[y0:y1, x0:x1]

        cropped_frame_face = cv2.resize(frame_face, (72, 72), cv2.INTER_AREA)
        cropped_frame_face = cv2.cvtColor(cropped_frame_face, cv2.COLOR_BGR2RGB)

        frames_buffer.append(cropped_frame_face)

        text = f"Frame buffer assembly ({len(frames_buffer)}/{100})..."
        color = (122, 251, 255)

        if len(frames_buffer) == 100:
            batch = np.array(frames_buffer) # (100, 72, 72, 3)
            frames_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0 # (1, 3, 100, 72, 72)

            with torch.no_grad():
                diff_wave = model(frames_tensor)

            diff_wave = diff_wave.squeeze().cpu().numpy()
            bpm = estimate_bpm_from_diff_wave(diff_wave=diff_wave, fps=real_fps)

            text = f"Pulse: {int(bpm)} BPM | FPS: {real_fps:.1f}"
            color = (0, 255, 0)

    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    if is_calibrated:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=2)

    cv2.imshow("PulseVision", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("PulseVision", cv2.WND_PROP_VISIBLE) < 1:
        break

close_camera(cam)
cv2.destroyAllWindows()