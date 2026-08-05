import faulthandler
faulthandler.enable()

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from dataset import buildCachedSubset, loadIndexMap
from model import SparseFlowNet, loadModelWeights, runForward
from paths import DEFAULT_CACHE_DIR

_SCHEMA = pa.schema([
    ("predFlowX", pa.float32()), ("predFlowY", pa.float32()), ("predFlowZ", pa.float32()),
    ("predSigma", pa.float32()),
    ("gtFlowX", pa.float32()), ("gtFlowY", pa.float32()), ("gtFlowZ", pa.float32()),
    ("isDynamic", pa.bool_()), ("classIdx", pa.uint8()),
    ("rangeMeters", pa.float32()), ("density", pa.uint16()),
    # Log identity, so downstream tooling can group by log. Points within a sweep are
    # heavily dependent, so the log is the exchangeable unit: every honest confidence
    # interval, bootstrap and conformal split has to be taken at this level, and none
    # of that was possible while the dump was a flat bag of points.
    ("logIdx", pa.uint16()),
])
_FLUSH_EVERY = 500


def _identityCollate(batch):
    return batch[0]


def loadModel(checkpointPath, device):
    model = SparseFlowNet(inC=8).to(device)
    ckpt = torch.load(checkpointPath, map_location=device, weights_only=False)
    dropped, missing, unexpected = loadModelWeights(model, ckpt["model"])
    if dropped or missing or unexpected:
        # A pre-uncertainty checkpoint: sigma stays at its constant init, which makes
        # it the constant-sigma control for a sigma-only A/B against a phase-2 run.
        print(f"  dropped={dropped} missing={missing} unexpected={unexpected}", flush=True)
    model.eval()
    return model, ckpt.get("voxelSize")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voxelSize", type=float, default=None,
                        help="defaults to the value stored in the checkpoint")
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--outFile", type=Path, required=True)
    parser.add_argument("--valSamples", type=int, default=-1)
    parser.add_argument("--split", default="val",
                        help="val | pseudoTest | uncFit | uncHoldout | uncFold<k>Eval")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ampDtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--removeGround", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # See train.py: 18-way intra-op thread contention starves the CUDA launch
    # thread, which is the critical path for batch_size=1 sparse convs.
    torch.set_num_threads(1)
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]

    args.outFile.parent.mkdir(parents=True, exist_ok=True)

    model, ckptVoxelSize = loadModel(args.checkpoint, device)
    if args.voxelSize is None:
        if ckptVoxelSize is None:
            raise SystemExit("checkpoint has no voxelSize; pass --voxelSize explicitly")
        args.voxelSize = ckptVoxelSize
    elif ckptVoxelSize is not None and ckptVoxelSize != args.voxelSize:
        raise SystemExit(
            f"voxelSize mismatch: checkpoint {ckptVoxelSize} vs --voxelSize {args.voxelSize}")
    ampDtype = torch.bfloat16 if args.ampDtype == "bf16" else torch.float16
    print(f"loaded {args.checkpoint} (voxelSize={args.voxelSize})", flush=True)

    ds = buildCachedSubset(loadIndexMap(args.split), args.cacheDir, args.valSamples)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    persistent_workers=True, pin_memory=True, prefetch_factor=2,
                    collate_fn=_identityCollate)
    print(f"running on {len(ds)} {args.split} sweeps", flush=True)

    cols = {k: [] for k in (
        "predFlowX", "predFlowY", "predFlowZ", "predSigma",
        "gtFlowX", "gtFlowY", "gtFlowZ",
        "isDynamic", "classIdx", "rangeMeters", "density", "logIdx",
    )}

    tmpPath = args.outFile.parent / (args.outFile.name + ".tmp")
    totalPoints = 0
    writer = ipc.new_file(str(tmpPath), _SCHEMA)
    # Compact log index, assigned in first-seen order. The cache already stores
    # uuid = (logId, timestamp), so no dataset access is needed.
    logIds = {}

    t0 = time.time()
    for i, sample in enumerate(dl):
        pc0, pc1 = sample["pc0"], sample["pc1"]
        gtAll, validAll = sample["flow"], sample["isValid"]
        dynAll, catAll = sample["isDynamic"], sample["categoryIndices"]
        logIdx = logIds.setdefault(sample["uuid"][0], len(logIds))

        # Must mirror training: filter ground before voxelization, and subset the
        # per-point GT identically so mask0 lines up.
        if args.removeGround:
            keep0 = ~sample["isGround0"]
            pc0, pc1 = pc0[keep0], pc1[~sample["isGround1"]]
            gtAll, validAll = gtAll[keep0], validAll[keep0]
            dynAll, catAll = dynAll[keep0], catAll[keep0]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=ampDtype, enabled=args.amp):
                pred, predLogVar, mask0 = runForward(model, pc0, pc1, args.voxelSize, pointRange, device)

        predSigma = torch.exp(0.5 * predLogVar.float()).cpu().numpy()
        predNp = pred.float().cpu().numpy()  # [M, 3]
        mask0T = mask0.cpu()                 # bool tensor on CPU
        mask0Np = mask0T.numpy()             # numpy version for scipy

        xyzAll = pc0.float().numpy()
        tree = cKDTree(xyzAll, balanced_tree=False, compact_nodes=False)
        xyzMasked = xyzAll[mask0Np]
        try:
            density = tree.query_ball_point(xyzMasked, r=0.5, return_length=True, workers=-1)
        except TypeError:
            density = np.array([len(nb) for nb in tree.query_ball_point(xyzMasked, r=0.5)], dtype=np.intp)

        rangeM = np.linalg.norm(xyzMasked, axis=1).astype(np.float32)

        # index flow fields with CPU bool tensor (consistent with PyTorch)
        # detach: the cached flow tensors carry requires_grad=True (a cache-gen
        # artifact), which is harmless in training but breaks .numpy() below.
        gtFlow = gtAll[mask0T].detach().float()
        isValid = validAll[mask0T].numpy().astype(bool)
        isDynamic = dynAll[mask0T].numpy().astype(bool)
        catIdx = catAll[mask0T].numpy()

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
        cols["logIdx"].append(np.full(int(isValid.sum()), logIdx, dtype=np.uint16))

        if (i + 1) % _FLUSH_EVERY == 0 or i == len(dl) - 1:
            if any(len(v) > 0 for v in cols.values()):
                batch = pa.record_batch({
                    "predFlowX": np.concatenate(cols["predFlowX"]).astype(np.float32),
                    "predFlowY": np.concatenate(cols["predFlowY"]).astype(np.float32),
                    "predFlowZ": np.concatenate(cols["predFlowZ"]).astype(np.float32),
                    "predSigma": np.concatenate(cols["predSigma"]).astype(np.float32),
                    "gtFlowX": np.concatenate(cols["gtFlowX"]).astype(np.float32),
                    "gtFlowY": np.concatenate(cols["gtFlowY"]).astype(np.float32),
                    "gtFlowZ": np.concatenate(cols["gtFlowZ"]).astype(np.float32),
                    "isDynamic": np.concatenate(cols["isDynamic"]),
                    "classIdx": np.concatenate(cols["classIdx"]).astype(np.uint8),
                    "rangeMeters": np.concatenate(cols["rangeMeters"]).astype(np.float32),
                    "density": np.concatenate(cols["density"]).astype(np.uint16),
                    "logIdx": np.concatenate(cols["logIdx"]).astype(np.uint16),
                }, schema=_SCHEMA)
                totalPoints += batch.num_rows
                writer.write_batch(batch)
                for k in cols:
                    cols[k] = []

        if (i + 1) % 50 == 0 or i == len(dl) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dl) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i+1}/{len(dl)}  rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    writer.close()
    tmpPath.rename(args.outFile)
    print(f"wrote {totalPoints:,} valid points over {len(logIds)} logs → {args.outFile}", flush=True)


if __name__ == "__main__":
    print("Starting evaluate.py...")
    main()
