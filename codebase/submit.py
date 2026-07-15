import faulthandler
faulthandler.enable()

import argparse
import time
import zipfile
from pathlib import Path

import torch
from av2.evaluation.scene_flow.utils import get_eval_point_mask, write_output_file

from extractSceneflow import RawSweepLoader
from model import SparseFlowNet, runForward
from paths import DEFAULT_DATASET, DEFAULT_DATASET_DIR


def loadModel(checkpointPath, device):
    model = SparseFlowNet(inC=8).to(device)
    ckpt = torch.load(checkpointPath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt.get("voxelSize")


@torch.no_grad()
def predictSample(model, pc0, pc1, voxelSize, pointRange, device, ampDtype, amp):
    """Per-point flow in the ego0 (ego-motion-compensated) frame the model trains in."""
    with torch.autocast("cuda", dtype=ampDtype, enabled=amp):
        pred, _, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)

    n = pc0.shape[0]
    fullFlow = torch.zeros((n, 3), dtype=torch.float32, device=device)
    fullFlow[mask0] = pred.float()
    return fullFlow


def toBenchmarkFrame(pc0, flowEgo0, ego1SE3ego0):
    """Invert the ego0 training convention back to the benchmark's.

    The model predicts motion in the ego0 frame (static == 0); AV2 scores flow
    that maps pc0 into the ego1 frame, so a static point's flow is the ego motion.
    """
    M = ego1SE3ego0.to(pc0.device)
    return (pc0 + flowEgo0) @ M[:3, :3].T + M[:3, 3] - pc0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
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
    parser.add_argument("--ampDtype", choices=["bf16", "fp16"], default="bf16")
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

    model, ckptVoxelSize = loadModel(args.checkpoint, device)
    if ckptVoxelSize is not None and ckptVoxelSize != args.voxelSize:
        raise SystemExit(
            f"voxelSize mismatch: checkpoint was trained at {ckptVoxelSize} but "
            f"--voxelSize is {args.voxelSize}. Re-run with --voxelSize {ckptVoxelSize}."
        )
    ampDtype = torch.bfloat16 if args.ampDtype == "bf16" else torch.float16
    print(f"loaded checkpoint from {args.checkpoint}", flush=True)

    print(f"building loader for {args.split} split...", flush=True)
    loader = RawSweepLoader(args.datasetDir, args.dataset, args.split)
    with zipfile.ZipFile(args.maskFile) as zf:
        maskUuids = {(n.split('/')[0], int(n.split('/')[1].replace('.feather', '')))
                     for n in zf.namelist() if n.endswith('.feather')}
    evalInds = [i for i, (logDir, ts0, ts1) in enumerate(loader._pairs)
                if (logDir.name, ts0) in maskUuids]
    if args.limit >= 0:
        evalInds = evalInds[:args.limit]
    nSamples = len(evalInds)
    print(f"running inference on {nSamples} samples (split={args.split})", flush=True)
    print(f"writing outputs to {args.outDir}", flush=True)

    t0 = time.time()
    for i, idx in enumerate(evalInds):
        sample = loader[idx]
        pc0, sweepUuid = sample["pc0"], sample["uuid"]

        flowEgo0 = predictSample(model, pc0, sample["pc1"], args.voxelSize,
                                 pointRange, device, ampDtype, args.amp)
        # is_dynamic means actually moving, so it is measured on the compensated
        # flow. Measuring it on the benchmark-frame flow would mark every point
        # dynamic whenever the ego is moving.
        isDynamic = torch.linalg.vector_norm(flowEgo0, dim=1) > args.dynamicThreshold
        submitFlow = toBenchmarkFrame(pc0.to(device), flowEgo0, sample["ego1SE3ego0"])

        mask = get_eval_point_mask(sweepUuid, args.maskFile)
        write_output_file(
            submitFlow[mask].cpu().numpy(),
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
