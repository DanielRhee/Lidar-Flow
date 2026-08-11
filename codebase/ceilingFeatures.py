"""Dump the uncertainty head's exact 41-d input alongside the realised error.

This exists for one question: **what can ANY function of these frozen features
achieve?** Three independent measurements say the trained head is useless-to-harmful on
FG_DYNAMIC, but that does not distinguish "the objective was wrong" from "the features
carry no signal". A nonparametric model fitted on the same inputs bounds the second.

Foreground only (classIdx != 0). Background's target is identically zero, so |err| =
||pred|| there and the question is vacuous.

Features come from model.uncertaintyInput, the same call runForwardHeads makes, so the
bound is on the head's actual input rather than on a reconstruction of it.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import DataLoader

from classes import stratumOf
from dataset import buildCachedSubset, identityCollate, loadIndexMap
from evaluate import loadModel
from model import UNCERTAINTY_IN, runForwardFeatures, runForwardHeads, uncertaintyInput
from paths import DEFAULT_CACHE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="uncHoldout")
    parser.add_argument("--valSamples", type=int, default=700)
    parser.add_argument("--outFile", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    args.outFile.parent.mkdir(parents=True, exist_ok=True)

    model, voxelSize = loadModel(args.checkpoint, device)
    ds = buildCachedSubset(loadIndexMap(args.split), args.cacheDir, args.valSamples)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True,
                    pin_memory=False, prefetch_factor=4, collate_fn=identityCollate)

    schema = pa.schema([(f"f{i}", pa.float32()) for i in range(UNCERTAINTY_IN)]
                       + [("errMag", pa.float32()), ("predSigma", pa.float32()),
                          ("stratum", pa.int8()), ("logIdx", pa.uint16()),
                          ("gtNorm", pa.float32())])
    tmp = args.outFile.parent / (args.outFile.name + ".tmp")
    writer = ipc.new_file(str(tmp), schema)
    logIds, total, t0 = {}, 0, time.time()
    print(f"dumping {UNCERTAINTY_IN}-d features over {len(ds)} {args.split} sweeps", flush=True)

    for i, sample in enumerate(dl):
        logIdx = logIds.setdefault(sample["uuid"][0], len(logIds))
        keep0 = ~sample["isGround0"]
        pc0, pc1 = sample["pc0"][keep0], sample["pc1"][~sample["isGround1"]]
        gtAll, validAll = sample["flow"][keep0], sample["isValid"][keep0]
        dynAll, catAll = sample["isDynamic"][keep0], sample["categoryIndices"][keep0]

        with torch.no_grad():
            d0, p2u, inv, mask0, rel, xyz = runForwardFeatures(
                model, pc0, pc1, voxelSize, pointRange, device)
            pred, logVar = runForwardHeads(model, d0, p2u, inv, rel, xyz)
            feats = uncertaintyInput(d0.features[p2u[inv]], rel, xyz, pred.detach())

        m0 = mask0.cpu()
        gt = gtAll[m0].detach().float().numpy()
        err = np.linalg.norm(pred.float().cpu().numpy() - gt, axis=1)
        valid = validAll[m0].numpy().astype(bool)
        cat = catAll[m0].numpy()
        strat = stratumOf(cat, dynAll[m0].numpy())
        sel = valid & (cat != 0)                      # foreground, valid
        if not sel.any():
            continue

        F = feats.float().cpu().numpy()[sel]
        cols = {f"f{j}": F[:, j] for j in range(UNCERTAINTY_IN)}
        cols["errMag"] = err[sel].astype(np.float32)
        # sigma travels WITH the features so the head and the ceiling are scored on
        # exactly the same points -- otherwise a population difference masquerades as
        # a capability gap.
        cols["predSigma"] = np.exp(0.5 * logVar.float().cpu().numpy())[sel].astype(np.float32)
        cols["stratum"] = strat[sel].astype(np.int8)
        cols["logIdx"] = np.full(int(sel.sum()), logIdx, dtype=np.uint16)
        # ||gt|| enables the motion-detector / routing experiment: predicting whether a
        # point is TRULY moving is both the gate the deployable system needs and the test
        # of whether the frozen feature holds motion the flow head discards.
        cols["gtNorm"] = np.linalg.norm(gt, axis=1)[sel].astype(np.float32)
        batch = pa.record_batch(cols, schema=schema)
        writer.write_batch(batch)
        total += batch.num_rows

        if (i + 1) % 50 == 0 or i == len(dl) - 1:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(dl)}  {total:,} pts  rate={rate:.2f} samp/s", flush=True)

    writer.close()
    tmp.rename(args.outFile)
    print(f"wrote {total:,} foreground points over {len(logIds)} logs → {args.outFile}", flush=True)


if __name__ == "__main__":
    main()
