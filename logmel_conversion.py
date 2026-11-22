
from pathlib import Path
import pandas as pd

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