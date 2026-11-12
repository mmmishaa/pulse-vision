import json
import subprocess
from dotenv import load_dotenv
from src.utils.paths import DATA_DIR

load_dotenv(".env")

def create_dataset_metadata():
    if not (DATA_DIR / "mcd" / "dataset-metadata.json").exists():
        metadata = {
            "title": "MCD rPPG Dataset",
            "id": "mishasavinov/mcd-rppg-dataset",
            "licenses": [{"name": "CC-BY-4.0"}]
        }

        with open (DATA_DIR / "mcd" / "dataset-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

def upload_mcd():
    subprocess.run(["kaggle", "datasets", "create", "-p", str(DATA_DIR / "mcd"), "-r", "tar"], check=True)

def main():
    create_dataset_metadata()
    upload_mcd()

if __name__ == "__main__":
    main()