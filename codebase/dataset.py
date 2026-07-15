import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from extractSceneflow import buildLoader, loadAnnotation
from paths import SPLITS_FILE


class SceneFlowDataset(Dataset):
    def __init__(self, datasetDir, dataset, split):
        self.loader = buildLoader(datasetDir, dataset, split)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        return loadAnnotation(self.loader, idx)


class DiskCachedDataset(Dataset):
    """Reads samples written by populate.py.

    Indexed by the *loader* index the cache was keyed on, so it is meant to be
    wrapped in a Subset over the frozen split's indices. It deliberately does not
    build a SceneFlowDataloader: that used to run in every DataLoader worker just
    to read len(), which is a large per-worker allocation for no benefit.
    """

    def __init__(self, split, cacheDir):
        self.cacheDir = Path(cacheDir) / split
        self._length = None

    def __len__(self):
        # Only ever bounds a Subset over frozen indices, so scan lazily and once.
        if self._length is None:
            stems = [int(p.stem) for p in self.cacheDir.glob("*.pt")]
            self._length = max(stems) + 1 if stems else 0
        return self._length

    def __getitem__(self, idx):
        path = self.cacheDir / f"{idx}.pt"
        if not path.exists():
            raise FileNotFoundError(f"sample {idx} not in cache at {path}; run populate.py first")
        return torch.load(path, weights_only=False)


def loadSplitIndices(split, splitsFile=SPLITS_FILE):
    """Loader indices for a frozen split ('train' | 'val' | 'pseudoTest')."""
    key = {"train": "trainIndices", "val": "valIndices", "pseudoTest": "pseudoTestIndices"}[split]
    return json.loads(Path(splitsFile).read_text())[key]


def identityCollate(batch):
    return batch[0]
