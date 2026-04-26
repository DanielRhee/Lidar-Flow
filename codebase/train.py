import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import SceneFlowDataset, identityCollate
from model import SparseFlowNet, runForward


def epeLoss(pred, gt, valid):
    err = (pred - gt).pow(2).sum(dim=1).clamp(min=1e-8).sqrt()
    validErr = err[valid]
    if validErr.numel() == 0:
        return err.sum() * 0.0
    return validErr.mean()


def runStep(model, sample, device, voxelSize, pointRange):
    pc0, pc1, flow, _ = sample
    pred, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)
    gt = flow.flow.to(device)[mask0]
    valid = flow.is_valid.to(device)[mask0]
    return epeLoss(pred, gt, valid)


def saveCheckpoint(path, model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "globalStep": globalStep,
            "bestVal": bestVal,
            "valEpe": valEpe,
            "args": vars(args),
            "rngState": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        path,
    )


def resolveResumePath(resumeArg, outDir):
    if resumeArg is None:
        return None
    if resumeArg != "auto":
        p = Path(resumeArg)
        return p if p.exists() else None
    for candidate in [outDir / "last.pt", outDir / "step_latest.pt"]:
        if candidate.exists():
            return candidate
    return None


def loadCheckpoint(path, model, opt, sched, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rngState", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])
    startEpoch = ckpt["epoch"] + 1
    globalStep = ckpt.get("globalStep", 0)
    bestVal = ckpt.get("bestVal", float("inf"))
    print(f"resumed from {path} at epoch {startEpoch}, bestVal {bestVal:.4f}")
    return startEpoch, globalStep, bestVal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=Path.home() / "persistent")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--trainSamples", type=int, default=1000)
    parser.add_argument("--valSamples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--voxelSize", type=float, default=0.2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outDir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--resume", type=str, default=None, metavar="PATH|auto")
    parser.add_argument("--checkpointEveryEpochs", type=int, default=5)
    parser.add_argument("--checkpointEverySteps", type=int, default=500)
    args = parser.parse_args()

    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    device = torch.device("cuda")
    args.outDir.mkdir(parents=True, exist_ok=True)

    trainBase = SceneFlowDataset(args.datasetDir, args.dataset, "train")
    valBase = SceneFlowDataset(args.datasetDir, args.dataset, "val")
    trainDs = Subset(trainBase, list(range(min(args.trainSamples, len(trainBase)))))
    valDs = Subset(valBase, list(range(min(args.valSamples, len(valBase)))))

    trainDl = DataLoader(trainDs, batch_size=1, shuffle=True, num_workers=0, collate_fn=identityCollate)
    valDl = DataLoader(valDs, batch_size=1, shuffle=False, num_workers=0, collate_fn=identityCollate)

    model = SparseFlowNet(inC=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weightDecay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(trainDl), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    startEpoch = 0
    globalStep = 0
    bestVal = float("inf")

    resumePath = resolveResumePath(args.resume, args.outDir)
    if resumePath is not None:
        startEpoch, globalStep, bestVal = loadCheckpoint(resumePath, model, opt, sched, scaler, device)

    for epoch in range(startEpoch, args.epochs):
        model.train()
        trainSum, trainN = 0.0, 0
        t0 = time.time()
        for sample in trainDl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = runStep(model, sample, device, args.voxelSize, pointRange)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            trainSum += loss.item()
            trainN += 1
            globalStep += 1

            if args.checkpointEverySteps > 0 and globalStep % args.checkpointEverySteps == 0:
                saveCheckpoint(
                    args.outDir / "step_latest.pt",
                    model, opt, sched, scaler, epoch, globalStep, bestVal, float("nan"), args,
                )

        model.eval()
        valSum, valN = 0.0, 0
        with torch.no_grad():
            for sample in valDl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    loss = runStep(model, sample, device, args.voxelSize, pointRange)
                valSum += loss.item()
                valN += 1

        trainEpe = trainSum / max(trainN, 1)
        valEpe = valSum / max(valN, 1)
        dt = time.time() - t0
        print(f"epoch {epoch}: trainEPE={trainEpe:.4f}  valEPE={valEpe:.4f}  dt={dt:.1f}s")

        saveCheckpoint(args.outDir / "last.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if valEpe < bestVal:
            bestVal = valEpe
            saveCheckpoint(args.outDir / "best.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if args.checkpointEveryEpochs > 0 and (epoch + 1) % args.checkpointEveryEpochs == 0:
            saveCheckpoint(
                args.outDir / f"epoch_{epoch}.pt",
                model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args,
            )

    print(f"best valEPE: {bestVal:.4f}")


if __name__ == "__main__":
    main()
