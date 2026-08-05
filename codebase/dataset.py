import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

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
        sample = torch.load(path, weights_only=False)
        # populate.py wrote pc1 and flow with requires_grad=True (a cache-gen
        # artifact). Left set, the GT flow joins the autograd graph and backward
        # spends itself computing gradients *into the targets*: measured 10.7 ms
        # vs 0.9 ms per sample in phase 2. detach() returns a view, so this is free.
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in sample.items()}


def loadSplitIndices(split, splitsFile=SPLITS_FILE):
    """Loader indices for a frozen split ('train' | 'val' | 'pseudoTest')."""
    key = {"train": "trainIndices", "val": "valIndices", "pseudoTest": "pseudoTestIndices"}[split]
    return json.loads(Path(splitsFile).read_text())[key]


def loadIndexMap(name, splitsFile=SPLITS_FILE):
    """{sourceSplit: [loaderIdx]} for any frozen set, including the derived ones.

    'uncFit' / 'uncHoldout' are the log-level carve of val + pseudoTest the
    uncertainty head is calibrated on, so they span two source splits at once.
    """
    if name in ("uncFit", "uncHoldout"):
        return json.loads(Path(splitsFile).read_text())[f"{name}Indices"]
    # uncFold<k>Fit / uncFold<k>Eval: cross-fitting over all 220 val+pseudoTest logs so
    # every log carries an out-of-fold sigma. Only the eval maps are stored; the fit
    # side is the union of the other folds.
    if name.startswith("uncFold"):
        rest = name[len("uncFold"):]
        if rest.endswith("Eval"):
            k, role = int(rest[:-4]), "Eval"
        elif rest.endswith("Fit"):
            k, role = int(rest[:-3]), "Fit"
        else:
            raise KeyError(f"expected uncFold<k>Fit or uncFold<k>Eval, got {name!r}")
        folds = json.loads(Path(splitsFile).read_text())["uncFoldEvalIndices"]
        if role == "Eval":
            return folds[k]
        merged = {}
        for j, m in enumerate(folds):
            if j == k:
                continue
            for split, idxs in m.items():
                merged.setdefault(split, []).extend(idxs)
        return {split: sorted(v) for split, v in merged.items()}
    return {name: loadSplitIndices(name, splitsFile)}


def buildCachedSubset(indexMap, cacheDir, cap=-1):
    """Concat one cached Subset per source split.

    Concat rather than merge because pseudoTest indices are *train*-loader indices
    and collide numerically with val's, so each split keeps its own cache dir.
    A cap strides evenly so a capped probe still covers every source split.
    """
    ds = ConcatDataset([Subset(DiskCachedDataset(split, cacheDir), indices)
                        for split, indices in sorted(indexMap.items()) if indices])
    if 0 < cap < len(ds):
        stride = len(ds) / cap
        ds = Subset(ds, [int(i * stride) for i in range(cap)])
    return ds


def identityCollate(batch):
    return batch[0]
