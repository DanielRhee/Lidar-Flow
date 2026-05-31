import faulthandler
faulthandler.enable()

import argparse
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from model import SparseFlowNet, runForward


def loadModel(checkpointPath, device):
    model = SparseFlowNet(inC=10).to(device)
    ckpt = torch.load(checkpointPath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def predictSample(model, pc0, pc1, voxelSize, pointRange, device, amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        pred, _, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)
    n = pc0.shape[0]
    fullFlow = torch.zeros((n, 3), dtype=torch.float32, device=device)
    fullFlow[mask0] = pred.float()
    return fullFlow


def loadFramePair(h5File, ts0, ts1):
    grp0 = h5File[str(ts0)]
    grp1 = h5File[str(ts1)]

    pc0Raw = np.array(grp0["lidar"], dtype=np.float32)  # (N, 4): x,y,z,intensity
    pc1Raw = np.array(grp1["lidar"], dtype=np.float32)
    pose0 = np.array(grp0["pose"], dtype=np.float64)    # (4, 4) world_SE3_ego
    pose1 = np.array(grp1["pose"], dtype=np.float64)

    # ego0_SE3_ego1 rotation and translation
    R0, t0_vec = pose0[:3, :3], pose0[:3, 3]
    R1, t1_vec = pose1[:3, :3], pose1[:3, 3]
    R_rel = (R0.T @ R1).astype(np.float32)
    t_rel = (R0.T @ (t1_vec - t0_vec)).astype(np.float32)

    # Zero-pad to 10 features (model trained with inC=10)
    def padTo10(pc):
        padded = np.zeros((pc.shape[0], 10), dtype=np.float32)
        padded[:, :pc.shape[1]] = pc
        return torch.from_numpy(padded)

    pc0 = padTo10(pc0Raw)
    pc1_t = padTo10(pc1Raw)
    pc1XYZ = pc1_t[:, :3] @ torch.from_numpy(R_rel).T + torch.from_numpy(t_rel)
    pc1 = torch.cat([pc1XYZ, pc1_t[:, 3:]], dim=1)
    return pc0, pc1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=Path, required=True,
                        help="directory containing index_eval.pkl and {scene_id}.h5 files")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outDir", type=Path, default=Path("submissions/challenge"))
    parser.add_argument("--voxelSize", type=float, default=0.2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=-1,
                        help="cap samples for debugging; -1 runs all")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    args.outDir.mkdir(parents=True, exist_ok=True)

    import spconv
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"sm={torch.cuda.get_device_capability()} spconv={spconv.__version__}",
        flush=True,
    )

    model = loadModel(args.checkpoint, device)
    print(f"loaded checkpoint from {args.checkpoint}", flush=True)

    with open(args.dataDir / "index_eval.pkl", "rb") as f:
        index = pickle.load(f)

    if args.limit >= 0:
        index = index[:args.limit]

    nSamples = len(index)
    print(f"running inference on {nSamples} eval frames", flush=True)
    print(f"writing outputs to {args.outDir}", flush=True)

    openFiles = {}
    sceneTimestamps = {}  # sceneId -> sorted list of int timestamps

    t0 = time.time()
    for i, (sceneId, timestamp) in enumerate(index):
        h5Path = args.dataDir / f"{sceneId}.h5"
        if sceneId not in openFiles:
            openFiles[sceneId] = h5py.File(h5Path, "r")
            sceneTimestamps[sceneId] = sorted(int(k) for k in openFiles[sceneId].keys())

        h5f = openFiles[sceneId]
        allTs = sceneTimestamps[sceneId]
        tsIdx = allTs.index(timestamp)

        if tsIdx + 1 >= len(allTs):
            # Last frame in scene — no next frame, write zero flow
            n = h5f[str(timestamp)]["lidar"].shape[0]
            fullFlow = np.zeros((n, 3), dtype=np.float32)
        else:
            ts1 = allTs[tsIdx + 1]
            pc0, pc1 = loadFramePair(h5f, timestamp, ts1)
            fullFlow = predictSample(model, pc0, pc1, args.voxelSize, pointRange, device, args.amp).cpu().numpy()

        outPath = args.outDir / sceneId
        outPath.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "flow_tx_m": fullFlow[:, 0],
            "flow_ty_m": fullFlow[:, 1],
            "flow_tz_m": fullFlow[:, 2],
        }).to_feather(outPath / f"{timestamp}.feather")

        if (i + 1) % 50 == 0 or i == nSamples - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (nSamples - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i + 1}/{nSamples}  rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    for f in openFiles.values():
        f.close()

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s. outputs at {args.outDir}", flush=True)
    print(f"zip with: cd {args.outDir.parent} && zip -r {args.outDir.name}.zip {args.outDir.name}/", flush=True)


if __name__ == "__main__":
    print("Starting challenge inference...")
    main()
