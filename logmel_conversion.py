"""
Script to convert MP3 audio files to log-mel spectrograms.

Usage:
    python logmel_conversion.py --audio-dir data/fma_small --metadata-dir data/fma_metadata --subset small --out-dir features/fma_small --jobs -1
    python logmel_conversion.py --audio-dir data/fma_medium --metadata-dir data/fma_metadata --subset medium --out-dir features/fma_medium --jobs -1
    python logmel_conversion.py --audio-dir data/fma_large --metadata-dir data/fma_metadata --subset large --out-dir features/fma_large --jobs -1  
"""

from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import librosa
import argparse
from tqdm import tqdm

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30.0
N_FRAMES = 1 + int(SR * DURATION) // HOP_LENGTH

SUBSETS = {
    "small": ["small"],
    "medium": ["small", "medium"],
    "large": ["small", "medium", "large"],
}

def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"

def load_tracks(metadata_dir: Path, subset: str) -> pd.DataFrame:
    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1])
    tracks = tracks[tracks[("set", "subset")].isin(SUBSETS[subset])]
    df = pd.DataFrame({
        "genre_top": tracks[("track", "genre_top")],
        "split": tracks[("set", "split")],
    })
    df.index.name = "track_id"
    return df

def mp3_to_logmel(audio_path: Path) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < N_FRAMES:
        pad = N_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), constant_values=log_mel.min())
    return log_mel[:, :N_FRAMES].astype(np.float32)

def process(track_id: int, audio_dir: Path, output_dir: Path) -> tuple[int, bool]:
    src = get_audio_path(audio_dir, track_id)
    dst = output_dir / src.parent.name / f"{src.stem}.npy"
    if dst.exists():
        return track_id, True
    try:
        log_mel = mp3_to_logmel(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dst.with_suffix(".npy.tmp")
        with open(temp_path, "wb") as f:    
            np.save(f, log_mel)
        temp_path.replace(dst)
        return track_id, True
    except Exception as e:
        print(f"Error processing track {track_id}: {e}")
        return track_id, False

def compute_statistics(manifest: pd.DataFrame, out_path: Path) -> None:
    train = manifest[manifest["ok"] & (manifest["split"] == "training")]
    sum = np.zeros(N_MELS, dtype=np.float64)
    squared_sum = np.zeros(N_MELS, dtype=np.float64)
    n = 0

    for path in tqdm(train["feature_path"], desc="mel statistics"):
        x = np.load(path).astype(np.float64)
        sum += x.sum(axis=1)
        squared_sum += (x ** 2).sum(axis=1)
        n += x.shape[1]

    mean = sum / n
    std = np.sqrt(squared_sum / n - mean ** 2)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             n_tracks=len(train), n_frames=n)
    print(f"Saved per-bin stats from {len(train)} training tracks: {out_path}")

    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("data/fma_small"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/fma_metadata"))
    parser.add_argument("--subset", choices=SUBSETS, default="small")
    parser.add_argument("--out-dir", type=Path, default=Path("features/fma_small"))
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    tracks = load_tracks(args.metadata_dir, args.subset)
    print(f"Loaded {len(tracks)} tracks for subset '{args.subset}'")

    results = Parallel(n_jobs=args.jobs)(
        delayed(process)(tid, args.audio_dir, args.out_dir)
        for tid in tqdm(tracks.index)
    )

    ok = dict(results)  # pyright: ignore
    tracks["ok"] = tracks.index.map(ok)
    tracks["feature_path"] = [
        str(args.out_dir / f"{t:06d}"[:3] / f"{t:06d}.npy") for t in tracks.index
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(args.out_dir / "manifest.csv")

    print(f"Done. {tracks['ok'].sum()} ok, {(~tracks['ok']).sum()} failed. "
          f"Manifest: {args.out_dir / 'manifest.csv'}")

    compute_statistics(tracks, args.out_dir / "mel_stats.npz")

if __name__ == "__main__":
    main()