import faulthandler
faulthandler.enable()

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import DiskCachedDataset, identityCollate
from model import SparseFlowNet, runForward, runForwardFeatures, runForwardHeads


def epeLoss(pred, gt, valid):
    err = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    validErr = err[valid]
    if validErr.numel() == 0:
        return err.sum() * 0.0
    return validErr.mean()


def betaNllLoss(pred, predLogVar, gt, valid, beta=0.5):
    pred = pred.float()
    predLogVar = predLogVar.float()
    gt = gt.float()

    sqErr = ((pred - gt) ** 2).sum(dim=1)
    var = torch.exp(predLogVar)
    nll = 0.5 * sqErr / var + 1.5 * predLogVar

    weight = var.detach() ** beta
    weighted = nll * weight

    validWeighted = weighted[valid]
    if validWeighted.numel() == 0:
        return weighted.sum() * 0.0
    return validWeighted.sum() / valid.sum().clamp(min=1).float()


def setTrainMode(model, phase):
    model.train()
    if phase not in (2, 3):
        return
    for name, m in model.named_modules():
        if not name:
            continue
        if name.split(".")[0] not in ("head", "uncertaintyHead"):
            m.eval()


def runStep(model, sample, device, voxelSize, pointRange, phase=1, beta=0.5, returnDynamic=False):
    pc0, pc1, flow, _ = sample

    # phase 3 training uses partial no_grad: backbone under no_grad, heads with autograd
    if phase == 3 and not returnDynamic:
        with torch.no_grad():
            d0, pc0ToUnion, inv0Point, mask0 = runForwardFeatures(model, pc0, pc1, voxelSize, pointRange, device)
        pred, predLogVar = runForwardHeads(model, d0, pc0ToUnion, inv0Point)
    else:
        pred, predLogVar, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)

    gt = flow.flow.to(device, non_blocking=True)[mask0]
    valid = flow.is_valid.to(device, non_blocking=True)[mask0]

    if returnDynamic:
        epe = epeLoss(pred, gt, valid)
        dyn = flow.is_dynamic.to(device, non_blocking=True)[mask0]
        dynValid = valid & dyn
        dynEpe = epeLoss(pred, gt, dynValid) if dynValid.any() else torch.tensor(float("nan"), device=device)
        nll = betaNllLoss(pred, predLogVar, gt, valid, beta=beta) if phase in (2, 3) else torch.tensor(float("nan"), device=device)
        return epe, dynEpe, nll

    if phase in (2, 3):
        return betaNllLoss(pred, predLogVar, gt, valid, beta=beta)
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
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rngState", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    if "cuda" in rng:
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    startEpoch = ckpt["epoch"] + 1
    globalStep = ckpt.get("globalStep", 0)
    bestVal = ckpt.get("bestVal", float("inf"))
    print(f"resumed from {path} at epoch {startEpoch}, bestVal {bestVal:.4f}")
    return startEpoch, globalStep, bestVal


def main():
    # Vibecoded the parameters. will fix late rlol. but it looks within reason.
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=Path.home() / "persistent")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--cacheDir", type=Path,
                        default=Path.home() / "persistent" / "djrhee" / "lidarflow_cache")
    parser.add_argument("--trainSamples", type=int, default=-1,
                        help="cap number of training samples; -1 uses the full cached set")
    parser.add_argument("--valSamples", type=int, default=-1,
                        help="cap number of validation samples; -1 uses the full cached set")
    parser.add_argument("--overfit", action="store_true",
                        help="correctness probe: evaluate on the training subset itself")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--voxelSize", type=float, default=0.2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outDir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--resume", type=str, default=None, metavar="PATH|auto")
    parser.add_argument("--checkpointEveryEpochs", type=int, default=5)
    parser.add_argument("--checkpointEverySteps", type=int, default=500)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                        help="1 = EPE training; 2 = beta-NLL uncertainty-only training; "
                             "3 = joint flow+uncertainty training (Option B)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="beta for beta-NLL loss (phases 2 and 3)")
    parser.add_argument("--phase1Ckpt", type=Path, default=None,
                        help="checkpoint to load weights from for phases 2 and 3; "
                             "strict=False so uncertaintyHead keeps its init")
    parser.add_argument("--flowLr", type=float, default=5e-5,
                        help="learning rate for flow head in phase 3 (default: 5e-5)")
    parser.add_argument("--epeBase", type=float, default=None,
                        help="baseline val EPE for phase 3 guardrail; "
                             "defaults to valEpe stored in --phase1Ckpt")
    args = parser.parse_args()

    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    import spconv
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"sm={torch.cuda.get_device_capability()} spconv={spconv.__version__}",
        flush=True,
    )
    args.outDir.mkdir(parents=True, exist_ok=True)

    cacheDir = args.cacheDir
    trainBase = DiskCachedDataset(args.datasetDir, args.dataset, "train", cacheDir)
    valBase = DiskCachedDataset(args.datasetDir, args.dataset, "val", cacheDir)
    trainIdx = json.loads((cacheDir / "train_indices.json").read_text())
    valIdx = json.loads((cacheDir / "val_indices.json").read_text())
    if args.trainSamples > 0:
        trainIdx = trainIdx[:args.trainSamples]
    if args.valSamples > 0:
        valIdx = valIdx[:args.valSamples]
    print(f"using {len(trainIdx)} train / {len(valIdx)} val samples from cache")
    trainDs = Subset(trainBase, trainIdx)
    valDs = Subset(trainBase, trainIdx) if args.overfit else Subset(valBase, valIdx)

    trainDl = DataLoader(trainDs, batch_size=1, shuffle=True, num_workers=6,
                         persistent_workers=True, pin_memory=True, prefetch_factor=4,
                         collate_fn=identityCollate)
    valDl = DataLoader(valDs, batch_size=1, shuffle=False, num_workers=2,
                       persistent_workers=True, pin_memory=True, prefetch_factor=4,
                       collate_fn=identityCollate)

    model = SparseFlowNet(inC=10).to(device)

    epeBase = None
    consecutiveBadEpochs = 0

    if args.phase in (2, 3):
        if args.phase1Ckpt is None:
            raise SystemExit(f"--phase {args.phase} requires --phase1Ckpt PATH")
        ckpt = torch.load(args.phase1Ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"loaded phase-1 weights from {args.phase1Ckpt}")
        print(f"  missing keys (expected uncertaintyHead.*): {list(missing)}")
        print(f"  unexpected keys (should be empty): {list(unexpected)}")
        for name, p in model.named_parameters():
            if not (name.startswith("head.") or name.startswith("uncertaintyHead.")):
                p.requires_grad = False
        nTrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        nTotal = sum(p.numel() for p in model.parameters())
        print(f"phase {args.phase}: trainable = {nTrainable} / {nTotal} ({100.0 * nTrainable / nTotal:.2f}%)")

        if args.phase == 3:
            epeBase = args.epeBase if args.epeBase is not None else ckpt.get("valEpe")
            if epeBase is None or math.isnan(epeBase):
                raise SystemExit("--phase 3 requires --epeBase or a phase1Ckpt with valEpe stored")
            print(f"phase 3: EPE base = {epeBase:.4f}  guardrail = {1.05 * epeBase:.4f}  hard-stop = {1.10 * epeBase:.4f}")

    if args.phase == 3:
        headParams = [p for name, p in model.named_parameters() if name.startswith("head.") and p.requires_grad]
        unchParams = [p for name, p in model.named_parameters() if name.startswith("uncertaintyHead.") and p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": headParams, "lr": args.flowLr},
            {"params": unchParams, "lr": args.lr},
        ], weight_decay=args.weightDecay)
    else:
        trainableParams = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainableParams, lr=args.lr, weight_decay=args.weightDecay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(trainDl), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    startEpoch = 0
    globalStep = 0
    bestVal = float("inf")
    bestNll = float("inf")

    resumePath = resolveResumePath(args.resume, args.outDir)
    if resumePath is not None:
        startEpoch, globalStep, bestVal = loadCheckpoint(resumePath, model, opt, sched, scaler, device)

    for epoch in range(startEpoch, args.epochs):
        setTrainMode(model, args.phase)
        trainSumDev = torch.zeros((), device=device)
        trainN = 0
        t0 = time.time()
        for sample in trainDl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = runStep(model, sample, device, args.voxelSize, pointRange,
                               phase=args.phase, beta=args.beta)
            if not torch.isfinite(loss):
                if args.phase == 1:
                    for m in model.modules():
                        if isinstance(m, torch.nn.BatchNorm1d):
                            m.reset_running_stats()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            trainSumDev += loss.detach()
            trainN += 1
            globalStep += 1

            if args.checkpointEverySteps > 0 and globalStep % args.checkpointEverySteps == 0:
                avgLoss = trainSumDev.item() / max(trainN, 1)
                print(f"  step {globalStep} (epoch {epoch}, {trainN}/{len(trainDl)}): loss={avgLoss:.4f}", flush=True)
                saveCheckpoint(
                    args.outDir / "step_latest.pt",
                    model, opt, sched, scaler, epoch, globalStep, bestVal, float("nan"), args,
                )

        model.eval()
        valEpeSum, valDynSum, valNllSum = 0.0, 0.0, 0.0
        valEpeN, valDynN, valNllN = 0, 0, 0
        with torch.no_grad():
            for sample in valDl:
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    epe, dynEpe, nll = runStep(model, sample, device, args.voxelSize, pointRange,
                                               phase=args.phase, beta=args.beta, returnDynamic=True)
                if torch.isfinite(epe):
                    valEpeSum += epe.item(); valEpeN += 1
                if torch.isfinite(dynEpe):
                    valDynSum += dynEpe.item(); valDynN += 1
                if torch.isfinite(nll):
                    valNllSum += nll.item(); valNllN += 1

        trainLoss = trainSumDev.item() / max(trainN, 1)
        valEpe = valEpeSum / max(valEpeN, 1)
        valDynEpe = valDynSum / max(valDynN, 1)
        valNll = valNllSum / max(valNllN, 1) if valNllN > 0 else float("nan")
        dt = time.time() - t0

        if args.phase in (2, 3):
            print(f"epoch {epoch}: trainNLL={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  valNLL={valNll:.4f}  dt={dt:.1f}s")
        else:
            print(f"epoch {epoch}: trainEPE={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  dt={dt:.1f}s")

        saveCheckpoint(args.outDir / "last.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if args.phase == 3:
            if valEpe < bestVal:
                bestVal = valEpe
            if not math.isnan(valNll) and valNll < bestNll and valEpe <= 1.05 * epeBase:
                bestNll = valNll
                saveCheckpoint(args.outDir / "best.pt", model, opt, sched, scaler, epoch, globalStep, bestNll, valEpe, args)
                print(f"  saved best.pt (valNLL={bestNll:.4f}, valEPE={valEpe:.4f})")
            if valEpe > 1.10 * epeBase:
                consecutiveBadEpochs += 1
                print(f"  WARNING: valEPE {valEpe:.4f} > hard-stop threshold {1.10 * epeBase:.4f} "
                      f"({consecutiveBadEpochs}/2 consecutive epochs)")
                if consecutiveBadEpochs >= 2:
                    print("  Hard stop: EPE degraded for 2 consecutive epochs. Aborting.")
                    break
            else:
                consecutiveBadEpochs = 0
        elif args.phase == 2:
            if valEpe < bestVal:
                bestVal = valEpe
                saveCheckpoint(args.outDir / "best_epe.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)
            if not math.isnan(valNll) and valNll < bestNll:
                bestNll = valNll
                saveCheckpoint(args.outDir / "best_nll.pt", model, opt, sched, scaler, epoch, globalStep, bestNll, valEpe, args)
        else:
            if valEpe < bestVal:
                bestVal = valEpe
                saveCheckpoint(args.outDir / "best.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if args.checkpointEveryEpochs > 0 and (epoch + 1) % args.checkpointEveryEpochs == 0:
            saveCheckpoint(
                args.outDir / f"epoch_{epoch}.pt",
                model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args,
            )

    if args.phase in (2, 3):
        print(f"best valEPE: {bestVal:.4f}  best valNLL: {bestNll:.4f}")
    else:
        print(f"best valEPE: {bestVal:.4f}")


if __name__ == "__main__":
    print("Starting...")
    main()
