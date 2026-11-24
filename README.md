# Music-Genre-Classification
CNN classification of music genre based on clips of music.

# Installing and mel conversion

This project uses the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset. Audio is downloaded, converted to log-mel spectrograms once, and the saved spectrograms are what the model trains on.

## 1. Set up the environment

```bash
pip install -r requirements.txt
```

PyTorch is not in `requirements.txt` because the right build depends on your hardware. Install it separately using the command for your system from [pytorch.org](https://pytorch.org/get-started/locally/).

## 2. Download the dataset

[data_download.py](data_download.py) downloads the FMA archives into `data/`, checks each one's SHA-1 checksum, extracts it, and deletes the zip. Interrupted downloads resume where they stopped, and any subset that has already been extracted is skipped.

```bash
python data_download.py                    # metadata + small (default)
python data_download.py metadata medium
python data_download.py metadata large
python data_download.py metadata full
```

| Subset   | Tracks  | Genres | Download size |
|----------|---------|--------|---------------|
| small    | 8,000   | 8      | 7.2 GB       |
| medium   | 25,000  | 16     | 22 GB        |
| large    | 106,574 | 161    | 93 GB        |
| full     | 106,574 | 161    | 879 GB (untrimmed audio) |

You always need `metadata`, since it holds `tracks.csv` with the genre labels and train/validation/test splits.

After downloading, `data/` looks like:

```
data/
├── fma_metadata/
│   └── tracks.csv ...
└── fma_small/
    ├── 000/
    │   ├── 000002.mp3
    │   └── ...
    └── ...
```

## 3. Convert audio to log-mel spectrograms

[logmel_conversion.py](logmel_conversion.py) turns each MP3 into a log-mel spectrogram and saves it as a `.npy` file:

```bash
python logmel_conversion.py --audio-dir data/fma_small --metadata-dir data/fma_metadata --subset small --out-dir features/fma_small
```

Running `python logmel_conversion.py` with no arguments processes the small subset with the default paths.

| Argument         | Default              | Description |
|------------------|----------------------|-------------|
| `--audio-dir`    | `data/fma_small`     | Folder of extracted MP3s |
| `--metadata-dir` | `data/fma_metadata`  | Folder containing `tracks.csv` |
| `--subset`       | `small`              | `small`, `medium`, or `large` (each includes the smaller subsets' tracks) |
| `--out-dir`      | `features/fma_small` | Where spectrograms and outputs are written |
| `--jobs`         | `-1`                 | Number of parallel workers (`-1` = all CPU cores) |

A few FMA MP3s are corrupt or empty; those are logged and marked as failed rather than stopping the run.

The output directory will contain:

```
features/fma_small/
├── 000/
│   ├── 000002.npy
│   └── ...
├── ...
├── manifest.csv     # track_id, genre_top, split, ok, feature_path
└── mel_stats.npz    # per-mel-bin mean and std from the training split
```
