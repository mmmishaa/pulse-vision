import urllib.request
import os

from src.utils.paths import FACE_MESH_MODEL_WEIGHTS_PATH, EFFICIENTPHYS_WEIGHTS_PATH 

def download_face_mesh_model():
    URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if os.path.exists(FACE_MESH_MODEL_WEIGHTS_PATH):
        print("The face_mesh_model weights have already been downloaded\n")
        return
    
    print("Loading face_mesh_model weights")
    FACE_MESH_MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url=URL, filename=FACE_MESH_MODEL_WEIGHTS_PATH)
        print("Weights successfully downloaded!\n")
    except (urllib.error.URLError, Exception) as e:
        raise RuntimeError(f"An error occurred while downloading the face_mesh_model weights. Error: {e}\n")

def download_efficientphys_model():
    URL = "https://github.com/mmmishaa/pulse-vision/releases/download/v0.1.0/efficientphys_c.pt"
    if os.path.exists(EFFICIENTPHYS_WEIGHTS_PATH):
        print("The efficientphys weights have already been downloaded\n")
        return
    
    print("Loading efficientphys weights")
    EFFICIENTPHYS_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url=URL, filename=EFFICIENTPHYS_WEIGHTS_PATH)
        print("Weights successfully downloaded!\n")
    except (urllib.error.URLError, Exception) as e:
        raise RuntimeError(f"An error occurred while downloading the efficientphys weights. Error: {e}\n")

def download_all_weights():
    download_face_mesh_model()
    download_efficientphys_model()

if __name__ == "__main__":
    download_all_weights()