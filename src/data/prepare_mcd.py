from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil
from dotenv import load_dotenv

load_dotenv(".env")

from src.utils.paths import DATA_DIR  

TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = TEMP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MCD_DIR = DATA_DIR / "mcd"
MCD_DIR.mkdir(parents=True, exist_ok=True)

def download_mcd_table():
    db_path = hf_hub_download(
        repo_type="dataset",
        repo_id="kyegorov/mcd_rppg", 
        filename="db.csv",
        cache_dir=CACHE_DIR
    )

    shutil.copy2(db_path, MCD_DIR / "db.csv")

def download_mcd_file(filename):
    ...

def main():
    download_mcd_table()

if __name__ == "__main__":
    main()
    #shutil.rmtree(CACHE_DIR, ignore_errors=True)