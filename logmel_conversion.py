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