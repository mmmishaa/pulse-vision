from pathlib import Path
import shutil
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

load_dotenv(".env")

from huggingface_hub import hf_hub_download

from src.utils.paths import DATA_DIR 
from src.data.preprocess import preprocess 

TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = TEMP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MCD_DIR = DATA_DIR / "mcd"
MCD_DIR.mkdir(parents=True, exist_ok=True)

(MCD_DIR / "frames").mkdir(parents=True, exist_ok=True)
(MCD_DIR/ "ppg").mkdir(parents=True, exist_ok=True)

def clear_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_mcd_table():
    db_path = hf_hub_download(
        repo_type="dataset",
        repo_id="kyegorov/mcd_rppg", 
        filename="db.csv",
        cache_dir=CACHE_DIR
    )

    shutil.copy2(db_path, MCD_DIR / "db.csv")
    clear_cache()

def download_mcd_file(filename):
    ...

def main():
    METADATA_PATH = MCD_DIR / "metadata.csv"

    download_mcd_table()
    mcd_df = pd.read_csv(MCD_DIR / "db.csv")
    mcd_df = mcd_df[mcd_df["camera"] == "FullHDwebcam"]

    for row in tqdm(mcd_df.itertuples(), total=len(mcd_df), desc="Video processing in MCD-RPPG"):
        try:
            video_path = row.video
            wave_path = f"ppg_sync_npy/mcd__{row.patient_id}__{row.step}_{row.view}.npy"

            row_frames_file = download_mcd_file(video_path)
            row_ppg_file = download_mcd_file(wave_path)

            ppg_array = np.load(row_ppg_file)

            num_frames = preprocess(
                row_frames_file, 
                ppg_array,
                MCD_DIR / "frames" / f"{row.patient_id}_{row.step}_frames.npy",
                MCD_DIR / "ppg" / f"{row.patient_id}_{row.step}_ppg.npy"
            )

            row_dict = {
                "patient_id": row.patient_id,
                "step": row.step,
                "frames_file": f"frames/{row.patient_id}_{row.step}_frames.npy",
                "ppg_file": f"ppg/{row.patient_id}_{row.step}_ppg.npy",
                "num_frames": num_frames,
                "pulse": row.pulse
            }

            pd.DataFrame([row_dict]).to_csv(METADATA_PATH, mode="a", header=not METADATA_PATH.exists(), index=False)
            break
        finally:
            clear_cache()

if __name__ == "__main__":
    main()