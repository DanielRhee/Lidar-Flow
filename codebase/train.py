import faulthandler
faulthandler.enable()

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import DiskCachedDataset, identityCollate, loadSplitIndices
from model import SparseFlowNet, runForward, runForwardFeatures, runForwardHeads
from paths import DEFAULT_CACHE_DIR, DEFAULT_DATASET, DEFAULT_DATASET_DIR


SWEEP_DT = 0.1  # seconds between AV2 sweeps; converts flow magnitude to m/s


def epeLoss(pred, gt, valid):
    err = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    validErr = err[valid]
    if validErr.numel() == 0:
        return err.sum() * 0.0
    return validErr.mean()


def deflowLoss(pred, gt, valid):
    """DeFlow's speed-binned loss: sum of per-bin mean EPE over non-empty bins.

    Points are overwhelmingly static, so a plain mean lets the static majority
    swamp the gradient. Binning by GT speed and averaging within each bin gives
    the fast-moving minority comparable weight.
    """
    err = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    speed = torch.linalg.vector_norm(gt.float(), dim=1) / SWEEP_DT

    bins = (
        speed < 0.4,
        (speed >= 0.4) & (speed <= 1.0),
        speed > 1.0,
    )
    total = err.sum() * 0.0  # keeps dtype/device and stays connected to the graph
    for b in bins:
        m = b & valid
        if m.any():
            total = total + err[m].mean()
    return total


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
        if name.split(".")[0] not in ("head", "refineHead", "uncertaintyHead"):
            m.eval()


def runStep(model, sample, device, voxelSize, pointRange, phase=1, beta=0.5,
            returnDynamic=False, removeGround=True, loss="deflow"):
    pc0, pc1 = sample["pc0"], sample["pc1"]
    gtAll = sample["flow"]
    validAll = sample["isValid"]
    dynamicAll = sample["isDynamic"]

    # Ground removal happens before voxelization, so the per-point GT must be
    # subset the same way: mask0 (the in-range mask) indexes the kept points.
    if removeGround:
        keep0 = ~sample["isGround0"]
        keep1 = ~sample["isGround1"]
        pc0, pc1 = pc0[keep0], pc1[keep1]
        gtAll, validAll, dynamicAll = gtAll[keep0], validAll[keep0], dynamicAll[keep0]

    # phase 3 training uses partial no_grad: backbone under no_grad, heads with autograd
    if phase == 3 and not returnDynamic:
        with torch.no_grad():
            d0, pc0ToUnion, inv0Point, mask0, rel0Point = runForwardFeatures(
                model, pc0, pc1, voxelSize, pointRange, device)
        pred, predLogVar = runForwardHeads(model, d0, pc0ToUnion, inv0Point, rel0Point)
    else:
        pred, predLogVar, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)

    gt = gtAll.to(device, non_blocking=True)[mask0]
    valid = validAll.to(device, non_blocking=True)[mask0]

    if returnDynamic:
        epe = epeLoss(pred, gt, valid)
        dyn = dynamicAll.to(device, non_blocking=True)[mask0]
        dynValid = valid & dyn
        staticValid = valid & ~dyn
        dynEpe = epeLoss(pred, gt, dynValid) if dynValid.any() else torch.tensor(float("nan"), device=device)
        staticEpe = epeLoss(pred, gt, staticValid) if staticValid.any() else torch.tensor(float("nan"), device=device)
        nll = betaNllLoss(pred, predLogVar, gt, valid, beta=beta) if phase in (2, 3) else torch.tensor(float("nan"), device=device)
        return epe, dynEpe, staticEpe, nll

    if phase in (2, 3):
        return betaNllLoss(pred, predLogVar, gt, valid, beta=beta)
    return deflowLoss(pred, gt, valid) if loss == "deflow" else epeLoss(pred, gt, valid)


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
            # Top-level so it can be restored and validated on resume; inside
            # args it was never read back.
            "voxelSize": args.voxelSize,
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


def loadCheckpoint(path, model, opt, sched, scaler, device, voxelSize=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckptVoxelSize = ckpt.get("voxelSize")
    if voxelSize is not None and ckptVoxelSize is not None and ckptVoxelSize != voxelSize:
        raise SystemExit(
            f"voxelSize mismatch: checkpoint {path} was trained at {ckptVoxelSize} "
            f"but --voxelSize is {voxelSize}. Re-run with --voxelSize {ckptVoxelSize}."
        )
    model.load_state_dict(ckpt["model"])
    # NOTE: BN running stats are deliberately NOT reset here. Resetting on every
    # resume wiped the statistics, so the first validation after a resume ran on
    # garbage. NaN recovery has its own reset in the training loop.
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
    parser.add_argument("--datasetDir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR)
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
    parser.add_argument("--ampDtype", choices=["bf16", "fp16"], default="bf16",
                        help="autocast dtype; bf16 avoids the fp16 overflow/underflow class "
                             "of bugs and is well supported on Ampere+")
    parser.add_argument("--accumSteps", type=int, default=8,
                        help="gradient accumulation micro-batches per optimizer step; "
                             "batch_size is fixed at 1 so this sets the effective batch")
    parser.add_argument("--loss", choices=["deflow", "epe"], default="deflow",
                        help="phase-1 loss; deflow is the speed-binned DeFlow loss")
    parser.add_argument("--removeGround", action=argparse.BooleanOptionalAction, default=True,
                        help="drop map-derived ground points before voxelization")
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
    trainBase = DiskCachedDataset("train", cacheDir)
    valBase = DiskCachedDataset("val", cacheDir)
    trainIdx = loadSplitIndices("train")
    valIdx = loadSplitIndices("val")
    if args.trainSamples > 0:
        trainIdx = trainIdx[:args.trainSamples]
    if args.valSamples > 0:
        valIdx = valIdx[:args.valSamples]
    trainDs = Subset(trainBase, trainIdx)
    valDs = Subset(trainBase, trainIdx) if args.overfit else Subset(valBase, valIdx)
    valSource = "train (overfit probe)" if args.overfit else "val"
    print(f"using {len(trainDs)} train / {len(valDs)} {valSource} samples from cache")

    trainDl = DataLoader(trainDs, batch_size=1, shuffle=True, num_workers=6,
                         persistent_workers=True, pin_memory=True, prefetch_factor=4,
                         collate_fn=identityCollate)
    valDl = DataLoader(valDs, batch_size=1, shuffle=False, num_workers=2,
                       persistent_workers=True, pin_memory=True, prefetch_factor=4,
                       collate_fn=identityCollate)

    model = SparseFlowNet(inC=8).to(device)

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
            if not name.startswith(("head.", "refineHead.", "uncertaintyHead.")):
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
        # refineHead is part of flow prediction, so it moves at flowLr with head.
        flowParams = [p for name, p in model.named_parameters()
                      if name.startswith(("head.", "refineHead.")) and p.requires_grad]
        unchParams = [p for name, p in model.named_parameters() if name.startswith("uncertaintyHead.") and p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": flowParams, "lr": args.flowLr},
            {"params": unchParams, "lr": args.lr},
        ], weight_decay=args.weightDecay)
    else:
        trainableParams = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainableParams, lr=args.lr, weight_decay=args.weightDecay)

    # The scheduler steps once per optimizer step, not per sample.
    stepsPerEpoch = math.ceil(len(trainDl) / args.accumSteps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * stepsPerEpoch, 1))
    # GradScaler only matters for fp16; under bf16 it is a no-op passthrough.
    ampDtype = torch.bfloat16 if args.ampDtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and ampDtype is torch.float16)

    startEpoch = 0
    globalStep = 0
    bestVal = float("inf")
    bestNll = float("inf")

    resumePath = resolveResumePath(args.resume, args.outDir)
    if resumePath is not None:
        startEpoch, globalStep, bestVal = loadCheckpoint(
            resumePath, model, opt, sched, scaler, device, voxelSize=args.voxelSize)

    for epoch in range(startEpoch, args.epochs):
        setTrainMode(model, args.phase)
        trainSumDev = torch.zeros((), device=device)
        trainN = 0
        t0 = time.time()
        # globalStep counts optimizer steps, not micro-batches.
        opt.zero_grad(set_to_none=True)
        pendingMicro = 0
        for microIdx, sample in enumerate(trainDl):
            with torch.autocast("cuda", dtype=ampDtype, enabled=args.amp):
                loss = runStep(model, sample, device, args.voxelSize, pointRange,
                               phase=args.phase, beta=args.beta,
                               removeGround=args.removeGround, loss=args.loss)
            if not torch.isfinite(loss):
                if args.phase == 1:
                    for m in model.modules():
                        if isinstance(m, torch.nn.BatchNorm1d):
                            m.reset_running_stats()
                continue  # skip before backward so accumulated grads stay clean
            scaler.scale(loss / args.accumSteps).backward()
            trainSumDev += loss.detach()
            trainN += 1
            pendingMicro += 1

            lastMicro = (microIdx + 1) == len(trainDl)
            if not (((microIdx + 1) % args.accumSteps == 0 or lastMicro) and pendingMicro > 0):
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            pendingMicro = 0
            globalStep += 1

            if args.checkpointEverySteps > 0 and globalStep % args.checkpointEverySteps == 0:
                avgLoss = trainSumDev.item() / max(trainN, 1)
                print(f"  step {globalStep} (epoch {epoch}, {trainN}/{len(trainDl)}): loss={avgLoss:.4f}", flush=True)
                saveCheckpoint(
                    args.outDir / "step_latest.pt",
                    model, opt, sched, scaler, epoch, globalStep, bestVal, float("nan"), args,
                )

        model.eval()
        valEpeSum, valDynSum, valStaticSum, valNllSum = 0.0, 0.0, 0.0, 0.0
        valEpeN, valDynN, valStaticN, valNllN = 0, 0, 0, 0
        with torch.no_grad():
            for sample in valDl:
                with torch.autocast("cuda", dtype=ampDtype, enabled=args.amp):
                    epe, dynEpe, staticEpe, nll = runStep(
                        model, sample, device, args.voxelSize, pointRange,
                        phase=args.phase, beta=args.beta, returnDynamic=True,
                        removeGround=args.removeGround, loss=args.loss)
                if torch.isfinite(epe):
                    valEpeSum += epe.item(); valEpeN += 1
                if torch.isfinite(dynEpe):
                    valDynSum += dynEpe.item(); valDynN += 1
                if torch.isfinite(staticEpe):
                    valStaticSum += staticEpe.item(); valStaticN += 1
                if torch.isfinite(nll):
                    valNllSum += nll.item(); valNllN += 1

        trainLoss = trainSumDev.item() / max(trainN, 1)
        valEpe = valEpeSum / max(valEpeN, 1)
        valDynEpe = valDynSum / max(valDynN, 1)
        valStaticEpe = valStaticSum / max(valStaticN, 1)
        valNll = valNllSum / max(valNllN, 1) if valNllN > 0 else float("nan")
        dt = time.time() - t0

        if args.phase in (2, 3):
            print(f"epoch {epoch}: trainNLL={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  valStaticEPE={valStaticEpe:.4f}  "
                  f"valNLL={valNll:.4f}  dt={dt:.1f}s")
        else:
            print(f"epoch {epoch}: trainLoss={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  valStaticEPE={valStaticEpe:.4f}  dt={dt:.1f}s")

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
