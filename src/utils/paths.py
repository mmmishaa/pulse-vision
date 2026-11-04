from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = BASE_DIR / "weights"

FACE_MESH_MODEL_WEIGHTS_PATH = WEIGHTS_PATH / "face_landmarker.task"