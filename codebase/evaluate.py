import faulthandler
faulthandler.enable()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from model import SparseFlowNet, runForward


def _identityCollate(batch):
    return batch[0]


class ValCacheDataset(Dataset):
    def __init__(self, cacheDir, indices):
        self.dir = Path(cacheDir) / "val"
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return torch.load(self.dir / f"{self.indices[i]}.pt", weights_only=False)


def loadModel(checkpointPath, device):
    model = SparseFlowNet(inC=10).to(device)
    ckpt = torch.load(checkpointPath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voxelSize", type=float, default=0.1)
    parser.add_argument("--cacheDir", type=Path,
                        default=Path.home() / "persistent" / "djrhee" / "lidarflow_cache")
    parser.add_argument("--outFile", type=Path, required=True)
    parser.add_argument("--valSamples", type=int, default=-1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]

    args.outFile.parent.mkdir(parents=True, exist_ok=True)

    model = loadModel(args.checkpoint, device)
    print(f"loaded {args.checkpoint}", flush=True)

    valIdx = json.loads((args.cacheDir / "val_indices.json").read_text())
    if args.valSamples > 0:
        valIdx = valIdx[:args.valSamples]

    ds = ValCacheDataset(args.cacheDir, valIdx)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    persistent_workers=True, pin_memory=True, prefetch_factor=4,
                    collate_fn=_identityCollate)
    print(f"running on {len(valIdx)} val sweeps", flush=True)

    cols = {k: [] for k in (
        "predFlowX", "predFlowY", "predFlowZ", "predSigma",
        "gtFlowX", "gtFlowY", "gtFlowZ",
        "isDynamic", "classIdx", "rangeMeters", "density",
    )}

    t0 = time.time()
    for i, sample in enumerate(dl):
        pc0, pc1, flow, _ = sample

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                pred, predLogVar, mask0 = runForward(model, pc0, pc1, args.voxelSize, pointRange, device)

        predSigma = torch.exp(0.5 * predLogVar.float()).cpu().numpy()
        predNp = pred.float().cpu().numpy()  # [M, 3]
        mask0T = mask0.cpu()                 # bool tensor on CPU
        mask0Np = mask0T.numpy()             # numpy version for scipy

        xyzAll = pc0[:, :3].numpy()
        tree = cKDTree(xyzAll)
        xyzMasked = xyzAll[mask0Np]
        try:
            density = tree.query_ball_point(xyzMasked, r=0.5, return_length=True)
        except TypeError:
            density = np.array([len(nb) for nb in tree.query_ball_point(xyzMasked, r=0.5)], dtype=np.intp)

        rangeM = np.linalg.norm(xyzMasked, axis=1).astype(np.float32)

        # index flow fields with CPU bool tensor (consistent with PyTorch)
        gtFlow = flow.flow[mask0T]
        isValid = flow.is_valid[mask0T].numpy().astype(bool)
        isDynamic = flow.is_dynamic[mask0T].numpy().astype(bool)
        catIdx = flow.category_indices[mask0T].numpy()

        if not isValid.any():
            continue

        cols["predFlowX"].append(predNp[isValid, 0])
        cols["predFlowY"].append(predNp[isValid, 1])
        cols["predFlowZ"].append(predNp[isValid, 2])
        cols["predSigma"].append(predSigma[isValid])
        cols["gtFlowX"].append(gtFlow.numpy().astype(np.float32)[isValid, 0])
        cols["gtFlowY"].append(gtFlow.numpy().astype(np.float32)[isValid, 1])
        cols["gtFlowZ"].append(gtFlow.numpy().astype(np.float32)[isValid, 2])
        cols["isDynamic"].append(isDynamic[isValid])
        cols["classIdx"].append(catIdx.astype(np.uint8)[isValid])
        cols["rangeMeters"].append(rangeM[isValid])
        cols["density"].append(density.astype(np.uint16)[isValid])

        if (i + 1) % 50 == 0 or i == len(dl) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dl) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i+1}/{len(dl)}  rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df = df.astype({
        "predFlowX": "float32", "predFlowY": "float32", "predFlowZ": "float32",
        "predSigma": "float32",
        "gtFlowX": "float32", "gtFlowY": "float32", "gtFlowZ": "float32",
        "isDynamic": bool, "classIdx": "uint8",
        "rangeMeters": "float32", "density": "uint16",
    })

    tmpPath = args.outFile.parent / (args.outFile.name + ".tmp")
    df.reset_index(drop=True).to_feather(tmpPath)
    tmpPath.rename(args.outFile)
    print(f"wrote {len(df):,} valid points → {args.outFile}", flush=True)


if __name__ == "__main__":
    print("Starting evaluate.py...")
    main()
