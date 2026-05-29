import faulthandler
faulthandler.enable()

import argparse
import time
from pathlib import Path

import torch
from av2.evaluation.scene_flow.utils import get_eval_point_mask, write_output_file

from extractSceneflow import RawSweepLoader
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=Path.home() / "persistent")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--split", default="test",
                        help="dataset split to run inference on; 'test' for leaderboard submission")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to trained checkpoint (e.g. runs/mvp/best.pt)")
    parser.add_argument("--outDir", type=Path, default=Path("submissions/mvp"),
                        help="directory to write per-sweep feather files into; zip this for submission")
    parser.add_argument("--voxelSize", type=float, default=0.2)
    parser.add_argument("--dynamicThreshold", type=float, default=0.05,
                        help="flow magnitude (m) above which a point is marked is_dynamic; "
                             "0.05m/sweep == 0.5m/s at 10Hz")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=-1,
                        help="cap samples (debug); -1 runs the full split")
    parser.add_argument("--maskFile", type=Path, required=True,
                        help="official eval mask archive (from make_mask_files)")
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

    print(f"building loader for {args.split} split...", flush=True)
    loader = RawSweepLoader(args.datasetDir, args.dataset, args.split)
    evalInds = list(range(len(loader)))[::5]
    if args.limit >= 0:
        evalInds = evalInds[:args.limit]
    nSamples = len(evalInds)
    print(f"running inference on {nSamples} samples (split={args.split})", flush=True)
    print(f"writing outputs to {args.outDir}", flush=True)

    t0 = time.time()
    for i, idx in enumerate(evalInds):
        pc0, pc1, _, sweepUuid = loader[idx]

        fullFlow = predictSample(model, pc0, pc1, args.voxelSize, pointRange, device, args.amp)
        mag = torch.linalg.vector_norm(fullFlow, dim=1)
        isDynamic = mag > args.dynamicThreshold

        mask = get_eval_point_mask(sweepUuid, args.maskFile)
        write_output_file(
            fullFlow[mask].cpu().numpy(),
            isDynamic[mask].cpu().numpy(),
            sweepUuid,
            args.outDir,
        )

        if (i + 1) % 50 == 0 or i == nSamples - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (nSamples - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i + 1}/{nSamples}  rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    print(f"done in {time.time() - t0:.1f}s. outputs at {args.outDir}", flush=True)
    print("zip the output dir for leaderboard submission, e.g.:", flush=True)
    print(f"  cd {args.outDir.parent} && zip -r {args.outDir.name}.zip {args.outDir.name}", flush=True)


if __name__ == "__main__":
    print("Starting inference...")
    main()
