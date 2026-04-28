from pathlib import Path

import torch
from torch.utils.data import Dataset

from extractSceneflow import buildLoader, loadAnnotation


class SceneFlowDataset(Dataset):
    def __init__(self, datasetDir, dataset, split):
        self.loader = buildLoader(datasetDir, dataset, split)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        return loadAnnotation(self.loader, idx)


class DiskCachedDataset(Dataset):
    def __init__(self, datasetDir, dataset, split, cacheDir):
        self.cacheDir = Path(cacheDir) / split
        loader = buildLoader(datasetDir, dataset, split)
        self.length = len(loader)
        del loader

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.cacheDir / f"{idx}.pt"
        if not path.exists():
            raise FileNotFoundError(f"sample {idx} not in cache at {path}; run populate.py first")
        return torch.load(path, weights_only=False)


def identityCollate(batch):
    return batch[0]
