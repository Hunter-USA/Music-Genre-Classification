from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import torch

SR = 22050
HOP_LENGTH = 512 
SEGMENT_LENGTH = 3.0  # in seconds
SEGMENT_FRAMES = int(SR * SEGMENT_LENGTH) // HOP_LENGTH

class FMADataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str, stats_path: str | Path, genre_to_index: dict, segment_frames: int = SEGMENT_FRAMES) -> None:
        df = manifest[manifest["ok"] & (manifest["split"] == split)]
        df = df[df["genre_top"].notna()]

        # stats for normalization
        stats = np.load(stats_path)
        self.mean = stats["mean"].astype(np.float32)[:, None]
        self.std = stats["std"].astype(np.float32)[:, None]
        self.seg = segment_frames

        first = np.load(df["feature_path"].iloc[0], mmap_mode="r")
        n_segments = first.shape[1] // self.seg
        self.paths = df["feature_path"].tolist()
        self.labels = [genre_to_index[g] for g in df["genre_top"]]
        self.tracks = df.index.tolist()
        self.items = [(t, s) for t in range(len(self.paths)) for s in range(n_segments)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        t, s = self.items[idx]
        spec = np.load(self.paths[t], mmap_mode="r")
        x = np.array(spec[:, s * self.seg:(s + 1) * self.seg], dtype=np.float32)
        x = (x - self.mean) / self.std
        return (torch.from_numpy(x).unsqueeze(0),
                torch.tensor(self.labels[t], dtype=torch.long),
                self.tracks[t]
        )

def make_dataloader(features_dir: str | Path, batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create data loaders for the FMA dataset. Only thing that should be called in this file is this function.

    Args:
        features_dir (str | Path): Directory containing the FMA dataset features and manifest.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 64.
        num_workers (int, optional): Number of worker processes for the data loaders. Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, list[str]]: Training, validation, and test data loaders, and the list of genres.
    """
    features_dir = Path(features_dir)
    manifest = pd.read_csv(features_dir / "manifest.csv", index_col=0)
    stats_path = features_dir / "mel_stats.npz"
    train_genres = manifest.loc[manifest["ok"] & (manifest["split"] == "training"), "genre_top"].dropna()
    genres = sorted(train_genres.unique())
    genre_to_index = {g: i for i, g in enumerate(genres)}

    def get_loader(split: str, shuffle: bool) -> DataLoader:
        ds = FMADataset(manifest, split, stats_path, genre_to_index)
        print(f"{split}: {len(ds)} segments")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

    return (get_loader("training", shuffle=True),
                get_loader("validation", shuffle=False),
                get_loader("test", shuffle=False), genres)

if __name__ == "__main__":
    train_loader, val_loader, test_loader, genres = make_dataloader("features/fma_small")
    print("Genres:", genres)
    #test a single batch from the training loader
    x, y, tid = next(iter(train_loader))
    print("Batch:", tuple(x.shape), y[:16].tolist())
    print(f"Batch mean {x.mean():.3f}, std {x.std():.3f})")

