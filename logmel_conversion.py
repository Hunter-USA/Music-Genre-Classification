
from pathlib import Path

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