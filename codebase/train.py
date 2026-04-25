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

    bestVal = float("inf")
    for epoch in range(args.epochs):
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

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "valEpe": valEpe, "args": vars(args)},
            args.outDir / "last.pt",
        )
        if valEpe < bestVal:
            bestVal = valEpe
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "valEpe": valEpe, "args": vars(args)},
                args.outDir / "best.pt",
            )

    print(f"best valEPE: {bestVal:.4f}")


if __name__ == "__main__":
    main()
