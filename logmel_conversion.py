
from pathlib import Path
import pandas as pd
import numpy as np
import librosa

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

def load_tracks(matadata_dir: Path, subset: str) -> pd.DataFrame:
    tracks = pd.read_csv(matadata_dir / "tracks.csv", index_col=0, header=[0, 1])
    tracks = tracks[tracks[("set", "subset")].isin(SUBSETS[subset])]
    df = pd.DataFrame({
        "genre_top": tracks[("track", "genre_top")],
        "split": tracks[("set", "split")],
    })
    df.index.name = "track_id"
    return df

def mp3_to_logmel(audio_path: Path) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION, power=2.0)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
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
