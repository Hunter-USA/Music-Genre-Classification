"""
Script to download data for the Music Genre Classification project.


Usage:
    python data_download.py                   # metadata + small (default)
    python data_download.py metadata medium
    python data_download.py metadata large
    python data_download.py metadata full
"""


import argparse
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
from tqdm import tqdm
import requests
import zipfile

BASE_URL = "https://os.unil.cloud.switch.ch/fma/"
DATA_DIR = Path(__file__).resolve().parent / "data"

ARCHIVES = {
    "metadata": ("fma_metadata.zip", "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"),
    "small": ("fma_small.zip", "ade154f733639d52e35e32f5593efe5be76c6d70"),
    "medium": ("fma_medium.zip", "c67b69ea232021025fca9231fc1c7c1a063ab50b"),
    "large": ("fma_large.zip", "497109f4dd721066b5ce5e5f250ec604dc78939e"),
    "full": ("fma_full.zip", "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab"),
}

def sha1sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    size = path.stat().st_size
    with open(path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=f"Verifying {path.name}") as bar:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if start and response.status_code != 206:
            start = 0
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0)) + start
        mode = "ab" if start else "wb"
        with open(part, mode) as f, tqdm(total=total_size, initial=start, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        part.rename(dest)

def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path} to {out_dir}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    except (NotImplementedError, zipfile.BadZipFile) as e:
        sevenzip = shutil.which("7z") or shutil.which("7za")
        if not sevenzip:
            sys.exit(f"Python could not extract {zip_path.name} ({e}). Please install 7-Zip and run: 7z x {str(zip_path)}")
        subprocess.run([sevenzip, "x", "-y", f"-o{out_dir}", str(zip_path)], check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subsets", nargs="*", default=["metadata", "small"],
                        choices=list(ARCHIVES), help="which archives to fetch")
    args = parser.parse_args()

    for name in args.subsets:
        filename, expected_hash = ARCHIVES[name]
        zip_path = DATA_DIR / filename
        out_marker = DATA_DIR / filename.removesuffix(".zip")

        if out_marker.exists():
            print(f"{out_marker.name} already exists, skipping download and extraction.")
            continue

        if not zip_path.exists():
            download(BASE_URL + filename, zip_path)

        actual_hash = sha1sum(zip_path)
        if actual_hash != expected_hash:
            zip_path.unlink()
            sys.exit(f"Checksum verification failed for {zip_path.name}")

        extract(zip_path, DATA_DIR)
        zip_path.unlink()
        print(f"Successfully extracted {zip_path.name} to {DATA_DIR}")

if __name__ == "__main__":
    main()