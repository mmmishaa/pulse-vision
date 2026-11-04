import urllib.request
import os

from src.utils.paths import FACE_MESH_MODEL_WEIGHTS_PATH

def download_face_mesh_model():
    URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if os.path.exists(FACE_MESH_MODEL_WEIGHTS_PATH):
        print("Веса face_mesh_model уже скачаны\n")
        return
    
    print("Загрузка весов face_mesh_model")
    FACE_MESH_MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url=URL, filename=FACE_MESH_MODEL_WEIGHTS_PATH)
        print("Веса успешно скачаны!\n")
    except (urllib.error.URLError, Exception) as e:
        raise RuntimeError(f"Ошибка во время скачивания весов face_mesh_model. Ошибка: {e}\n")