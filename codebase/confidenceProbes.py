"""Correspondence-quality confidence signals, scored as standalone error rankers.

The stereo/optical-flow confidence literature settled long ago on the finding that
the informative signals are correspondence-quality measures -- left-right
consistency, matching-cost margins, distinctiveness -- rather than the prediction
itself. The LiDAR analogues need no cost volume:

  fbConsistency  ||f_fwd(p) + f_bwd(p + f_fwd(p))||   forward-backward agreement
  nnResidual     dist(p + f_fwd(p), nearest point in pc1)   the Chamfer term that
                 NSFP / FastNSF / SeFlow optimise as a self-supervised loss, so it is
                 known a priori to track flow error

Both are computed here without retraining anything, and ranked against the learned
sigma and against ||predFlow||. If either beats sigma, the head's input set is the
problem and no amount of loss tuning will fix it.

Both clouds live in the ego0 frame (populate.py ego-compensates pc1), so the reverse
pass is simply runForward(pc1, pc0) and f_fwd + f_bwd should cancel.
"""

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

from dataset import buildCachedSubset, identityCollate, loadIndexMap
from evaluate import loadModel
from model import runFailureHead, runForward, runForwardFeatures, runForwardHeads
from paths import DEFAULT_CACHE_DIR

_SCHEMA = pa.schema([
    ("errMag", pa.float32()), ("predSigma", pa.float32()), ("predNorm", pa.float32()),
    ("fbConsistency", pa.float32()), ("nnResidual", pa.float32()), ("piLogit", pa.float32()),
    ("gtNorm", pa.float32()),
    ("isDynamic", pa.bool_()), ("classIdx", pa.uint8()), ("logIdx", pa.uint16()),
])
_FLUSH_EVERY = 250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="uncHoldout")
    parser.add_argument("--valSamples", type=int, default=-1)
    parser.add_argument("--outFile", type=Path, required=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--removeGround", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)   # see train.py: launch-thread contention costs 4x
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    args.outFile.parent.mkdir(parents=True, exist_ok=True)

    model, voxelSize = loadModel(args.checkpoint, device)
    ds = buildCachedSubset(loadIndexMap(args.split), args.cacheDir, args.valSamples)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True,
                    pin_memory=False, prefetch_factor=4, collate_fn=identityCollate)
    print(f"probing {len(ds)} {args.split} sweeps at voxelSize={voxelSize}", flush=True)

    cols = {k: [] for k in _SCHEMA.names}
    tmp = args.outFile.parent / (args.outFile.name + ".tmp")
    writer = ipc.new_file(str(tmp), _SCHEMA)
    logIds, total, t0 = {}, 0, time.time()

    for i, sample in enumerate(dl):
        logIdx = logIds.setdefault(sample["uuid"][0], len(logIds))
        pc0, pc1 = sample["pc0"], sample["pc1"]
        gtAll, validAll = sample["flow"], sample["isValid"]
        dynAll, catAll = sample["isDynamic"], sample["categoryIndices"]
        if args.removeGround:
            keep0 = ~sample["isGround0"]
            pc0, pc1 = pc0[keep0], pc1[~sample["isGround1"]]
            gtAll, validAll = gtAll[keep0], validAll[keep0]
            dynAll, catAll = dynAll[keep0], catAll[keep0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            d0, p2u, inv, mask0, rel, xyz = runForwardFeatures(
                model, pc0, pc1, voxelSize, pointRange, device)
            fwd, logVar = runForwardHeads(model, d0, p2u, inv, rel, xyz)
            # pi shares the backbone pass, so it is scored on exactly the points the
            # probes are -- the only way this table is a like-for-like ranking.
            piLogit = runFailureHead(model, d0, p2u, inv, rel, xyz)
            bwd, _, mask1 = runForward(model, pc1, pc0, voxelSize, pointRange, device)

        m0 = mask0.cpu().numpy()
        xyz0 = pc0.float().numpy()[m0]
        xyz1 = pc1.float().numpy()[mask1.cpu().numpy()]
        fwdNp = fwd.float().cpu().numpy()
        bwdNp = bwd.float().cpu().numpy()
        if len(xyz1) == 0:
            continue

        warped = xyz0 + fwdNp
        tree = cKDTree(xyz1, balanced_tree=False, compact_nodes=False)
        nnDist, nnIdx = tree.query(warped, workers=-1)
        # f_fwd(p) + f_bwd(q) should cancel for a correct correspondence q.
        fb = np.linalg.norm(fwdNp + bwdNp[nnIdx], axis=1)

        m0T = mask0.cpu()
        gt = gtAll[m0T].detach().float().numpy()
        isValid = validAll[m0T].numpy().astype(bool)
        if not isValid.any():
            continue
        err = np.linalg.norm(fwdNp - gt, axis=1)

        for name, arr in (("errMag", err), ("predSigma", np.exp(0.5 * logVar.float().cpu().numpy())),
                          ("predNorm", np.linalg.norm(fwdNp, axis=1)),
                          ("fbConsistency", fb), ("nnResidual", nnDist),
                          ("piLogit", piLogit.float().cpu().numpy()),
                          # ||gt|| decides whether pi is an error detector or a motion
                          # estimator: on FG_DYNAMIC ||e|| is largely proportional to
                          # ||gt||, so a speed estimator scores high AUC by construction.
                          ("gtNorm", np.linalg.norm(gt, axis=1)),
                          ("isDynamic", dynAll[m0T].numpy().astype(bool)),
                          ("classIdx", catAll[m0T].numpy().astype(np.uint8))):
            cols[name].append(arr[isValid])
        cols["logIdx"].append(np.full(int(isValid.sum()), logIdx, dtype=np.uint16))

        if (i + 1) % _FLUSH_EVERY == 0 or i == len(dl) - 1:
            if cols["errMag"]:
                batch = pa.record_batch(
                    {f.name: np.concatenate(cols[f.name]).astype(f.type.to_pandas_dtype())
                     for f in _SCHEMA}, schema=_SCHEMA)
                total += batch.num_rows
                writer.write_batch(batch)
                for k in cols:
                    cols[k] = []
        if (i + 1) % 50 == 0 or i == len(dl) - 1:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(dl)}  rate={rate:.2f} samp/s  "
                  f"eta={(len(dl)-i-1)/max(rate,1e-9)/60:.1f}min", flush=True)

    writer.close()
    tmp.rename(args.outFile)
    print(f"wrote {total:,} points over {len(logIds)} logs → {args.outFile}", flush=True)


if __name__ == "__main__":
    print("Starting confidenceProbes.py...")
    main()
