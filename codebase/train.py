import faulthandler
faulthandler.enable()

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import buildCachedSubset, identityCollate, loadIndexMap
from model import (SparseFlowNet, loadModelWeights, runFailureHead, runForward,
                   runForwardFeatures, runForwardHeads)
from paths import DEFAULT_CACHE_DIR, DEFAULT_DATASET, DEFAULT_DATASET_DIR


from failureDetection import rocAuc

SWEEP_DT = 0.1  # seconds between AV2 sweeps; converts flow magnitude to m/s

# chi2(0.90, df=3): squared error over variance below this is a covered point under
# an isotropic 3-D Gaussian. Matches analysis.py's coverage definition.
_CHI2_90_DF3 = 6.251388631170325

# Points kept per val sweep for the pooled ROC-AUC.
_COLLECT_PER_SAMPLE = 600

# One-time diagnostic dumps for the first non-finite loss / gradient, gated by
# --debugAnomaly. Keyed so each kind prints exactly once per run.
_REPORTED_NONFINITE = set()


def reportNonFinite(kind, **items):
    if kind in _REPORTED_NONFINITE:
        return
    _REPORTED_NONFINITE.add(kind)
    print(f"  [debugAnomaly] first non-finite {kind}:", flush=True)
    for name, t in items.items():
        if t is None:
            continue
        if not torch.is_tensor(t):
            print(f"    {name}={t}", flush=True)
            continue
        t = t.detach().float()
        nNan = int(torch.isnan(t).sum())
        nInf = int(torch.isinf(t).sum())
        finite = t[torch.isfinite(t)]
        if finite.numel() > 0:
            lo, hi, amax = finite.min().item(), finite.max().item(), finite.abs().max().item()
        else:
            lo = hi = amax = float("nan")
        print(f"    {name}: shape={tuple(t.shape)} nan={nNan} inf={nInf} "
              f"min={lo:.4g} max={hi:.4g} absmax={amax:.4g}", flush=True)


def reportNonFiniteGrad(model, totalNorm):
    if "grad" in _REPORTED_NONFINITE:
        return
    _REPORTED_NONFINITE.add("grad")
    print(f"  [debugAnomaly] non-finite grad norm (totalNorm={float(totalNorm):.4g}):", flush=True)
    bad = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            g = p.grad.detach()
            bad.append((name, int(torch.isnan(g).sum()), int(torch.isinf(g).sum())))
    for name, nNan, nInf in bad[:20]:
        print(f"    grad {name}: nan={nNan} inf={nInf}", flush=True)
    print(f"    ({len(bad)} params with non-finite grad)", flush=True)


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


# Phase 2 fits only the uncertainty head, so the flow path stays bit-identical to
# the phase-1 checkpoint; phase 3 fine-tunes the flow heads alongside it.
def trainablePrefixes(phase, objective="nll"):
    if objective == "failure":
        return ("failureHead.",)          # flow AND sigma frozen; only pi trains
    if phase == 2:
        return ("uncertaintyHead.",)
    return ("head.", "refineHead.", "uncertaintyHead.")


def failureLoss(logit, pred, gt, valid, tau):
    """BCE on P(||e|| > tau). No clamp, no sqrt(3), no chi-squared, and no 1/var term for
    the tail to dominate -- the pathologies of the scalar-Gaussian formulation are
    properties of that likelihood, not of the problem."""
    err = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    target = (err > tau).float()[valid]
    z = logit.float()[valid]
    if z.numel() == 0:
        return z.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(z, target)


def sigmaRegressionLoss(pred, predLogVar, gt, valid, floor=1e-4):
    """Oracle-style ceiling: regress log sigma straight onto the per-point NLL optimum.

    log(|e|/sqrt(3)) is exactly the sigma that minimises the isotropic 3-D Gaussian
    NLL for a single point, so fitting it under plain MSE *in log space* asks the head
    for the same target as NLL but without the 1/var term that lets the error tail own
    the gradient. Whatever this reaches therefore bounds what any head built on these
    frozen features can achieve -- which is the test for whether the near-constant
    sigma over BACKGROUND is a head failure or an information limit.
    """
    err = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1).clamp(min=floor)
    diff = (0.5 * predLogVar.float() - torch.log(err / math.sqrt(3.0)))[valid]
    if diff.numel() == 0:
        return diff.sum() * 0.0
    return (diff ** 2).mean()


def setTrainMode(model, phase, objective="nll"):
    model.train()
    if phase not in (2, 3):
        return
    trainable = tuple(p.rstrip(".") for p in trainablePrefixes(phase, objective))
    for name, m in model.named_modules():
        if not name:
            continue
        if name.split(".")[0] not in trainable:
            m.eval()


def runStep(model, sample, device, voxelSize, pointRange, phase=1, beta=0.5,
            returnDynamic=False, removeGround=True, loss="deflow", debug=False,
            sigmaTarget="nll", failureTau=0.10, failureStrata="dynamic", collect=None):
    pc0, pc1 = sample["pc0"], sample["pc1"]
    gtAll = sample["flow"]
    validAll = sample["isValid"]
    dynamicAll = sample["isDynamic"]
    categoryAll = sample["categoryIndices"]

    # Ground removal happens before voxelization, so the per-point GT must be
    # subset the same way: mask0 (the in-range mask) indexes the kept points.
    if removeGround:
        keep0 = ~sample["isGround0"]
        keep1 = ~sample["isGround1"]
        pc0, pc1 = pc0[keep0], pc1[keep1]
        gtAll, validAll, dynamicAll = gtAll[keep0], validAll[keep0], dynamicAll[keep0]
        categoryAll = categoryAll[keep0]

    # Phases 2 and 3 use partial no_grad: backbone under no_grad, heads with
    # autograd. The frozen heads need no special handling -- requires_grad=False on
    # their parameters plus a grad-free d0 already keeps them out of the graph.
    if phase in (2, 3):
        with torch.no_grad():
            d0, pc0ToUnion, inv0Point, mask0, rel0Point, xyz0Point = runForwardFeatures(
                model, pc0, pc1, voxelSize, pointRange, device)
        pred, predLogVar = runForwardHeads(model, d0, pc0ToUnion, inv0Point, rel0Point, xyz0Point)
        failLogit = (runFailureHead(model, d0, pc0ToUnion, inv0Point, rel0Point, xyz0Point)
                     if sigmaTarget == "failure" else None)
    else:
        pred, predLogVar, mask0 = runForward(model, pc0, pc1, voxelSize, pointRange, device)
        failLogit = None

    gt = gtAll.to(device, non_blocking=True)[mask0]
    valid = validAll.to(device, non_blocking=True)[mask0]

    if returnDynamic:
        nan = torch.tensor(float("nan"), device=device)
        dyn = dynamicAll.to(device, non_blocking=True)[mask0]
        dynValid, staticValid = valid & dyn, valid & ~dyn
        metrics = {
            "epe": epeLoss(pred, gt, valid),
            "dynEpe": epeLoss(pred, gt, dynValid) if dynValid.any() else nan,
            "staticEpe": epeLoss(pred, gt, staticValid) if staticValid.any() else nan,
        }
        if phase not in (2, 3) or not valid.any():
            metrics.update(nll=nan, nllMed=nan, sigma=nan, cover90=nan)
            return metrics
        logVar = predLogVar.float()[valid]
        var = torch.exp(logVar)
        sqErr = ((pred.float() - gt.float()) ** 2).sum(dim=1)[valid]
        # beta=0 on purpose: a reported score must be *proper*. The beta-weighted
        # value is not a progress measure at all, because its detached var**beta
        # weight shrinks as sigma converges -- for a static point moving var
        # 1e-3 -> its optimum 3e-6 the weighted loss RISES -0.327 -> -0.030 while
        # the proper NLL correctly falls -10.36 -> -17.56.
        pointNll = 0.5 * sqErr / var + 1.5 * logVar
        # nllMed, not nll, is what best_nll.pt selects on. The *mean* proper NLL is
        # unbounded above and a handful of catastrophically overconfident points
        # dominate it (sqErr 1 m^2 at the sigma floor contributes ~5e6), so it
        # cannot tell "well calibrated" from "well calibrated plus three bad
        # points" and rises monotonically even while calibration improves. The
        # median is robust and still rewards sharpness, unlike cover90, which a
        # constant sigma tuned to the target would score perfectly on.
        metrics["nll"] = pointNll.mean()
        metrics["nllMed"] = pointNll.median()
        metrics["sigma"] = var.sqrt().mean()
        metrics["cover90"] = (sqErr / var <= _CHI2_90_DF3).float().mean()
        if collect is not None and sigmaTarget == "failure":
            errV = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)[valid]
            rows = torch.stack([failLogit.float()[valid], errV, dyn[valid].float()], dim=1)
            # Cap per sample: keeping every val point is ~4.6 GB an epoch, and a few
            # hundred per sweep already pins the AUC to well under its bootstrap width.
            if rows.shape[0] > _COLLECT_PER_SAMPLE:
                sel = torch.randperm(rows.shape[0], device=rows.device)[:_COLLECT_PER_SAMPLE]
                rows = rows[sel]
            collect.append(rows.cpu())
        return metrics

    if phase in (2, 3):
        if sigmaTarget == "failure":
            # Foreground only. Background is 80% of points and its failure label is
            # tautological -- its GT flow is identically zero, so ||e|| = ||pred|| and
            # "will this fail" is just "is the prediction large". Training on it drags
            # the head toward the easy majority: a smoke run improved pooled AUC
            # 0.683 -> 0.761 while FG_DYNAMIC AUC FELL 0.467 -> 0.385.
            cat = categoryAll.to(device, non_blocking=True)[mask0]
            dynM = dynamicAll.to(device, non_blocking=True)[mask0]
            if failureStrata == "dynamic":
                sel = valid & (cat != 0) & dynM
            elif failureStrata == "foreground":
                sel = valid & (cat != 0)
            else:
                sel = valid
            lossVal = failureLoss(failLogit, pred, gt, sel, tau=failureTau)
        elif sigmaTarget == "logErr":
            lossVal = sigmaRegressionLoss(pred, predLogVar, gt, valid)
        else:
            lossVal = betaNllLoss(pred, predLogVar, gt, valid, beta=beta)
    else:
        lossVal = deflowLoss(pred, gt, valid) if loss == "deflow" else epeLoss(pred, gt, valid)
    if debug and not torch.isfinite(lossVal):
        # Localize the first non-finite loss: is it pred (activation/weight blow-up)
        # or a degenerate target? Reported once, then training continues.
        reportNonFinite("loss", pred=pred, predLogVar=predLogVar, gt=gt,
                        nValid=int(valid.sum()), nPoints=int(valid.numel()))
    return lossVal


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
    parser.add_argument("--fitSet", default="train",
                        help="frozen index set to train on: train | uncFit | uncFold<k>Fit. "
                             "uncFit is the val+pseudoTest carve the uncertainty head is "
                             "calibrated against; uncFold<k>Fit is a cross-fitting fold")
    parser.add_argument("--evalSet", default="val",
                        help="frozen index set to evaluate on each epoch: "
                             "val | pseudoTest | uncHoldout | uncFold<k>Eval")
    parser.add_argument("--trainSamples", type=int, default=-1,
                        help="cap number of training samples; -1 uses the full cached set")
    parser.add_argument("--valSamples", type=int, default=-1,
                        help="cap number of validation samples; -1 uses the full cached set")
    parser.add_argument("--overfit", action="store_true",
                        help="correctness probe: evaluate on the training subset itself")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="peak LR; 1e-3 was empirically too hot at effective batch 8 "
                             "(recurrent non-finite losses)")
    parser.add_argument("--warmupEpochs", type=float, default=1.0,
                        help="linear LR warmup length in epochs before cosine decay")
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--voxelSize", type=float, default=0.2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="OFF by default (fp32). bf16 autocast overflowed spconv's "
                             "TRAIN-mode implicit-GEMM convolutions once weights grew "
                             "(max|w| ~7->20 by epoch 9): the decoder convs hit inf, every "
                             "micro-batch loss went non-finite and was skipped, and the run "
                             "silently froze (retrain_v3). fp32 is provably stable on those "
                             "same weights and is nearly free here — training is dataloader-"
                             "bound and the GPU sits at ~33%. Re-enable at your own risk.")
    parser.add_argument("--ampDtype", choices=["bf16", "fp16"], default="bf16",
                        help="autocast dtype when --amp is set. NOTE: bf16 shares fp32's "
                             "exponent range, so the epoch-9 overflow was spconv accumulating "
                             "in a narrower range internally, not a true bf16 range overflow")
    parser.add_argument("--accumSteps", type=int, default=8,
                        help="gradient accumulation micro-batches per optimizer step; "
                             "batch_size is fixed at 1 so this sets the effective batch")
    parser.add_argument("--loss", choices=["deflow", "epe"], default="deflow",
                        help="phase-1 loss; deflow is the speed-binned DeFlow loss")
    parser.add_argument("--numWorkers", type=int, default=6,
                        help="dataloader workers. NOT the throughput knob: cache reads "
                             "cost ~4 ms/sample and 4/6/8/12 all land within 10%. What "
                             "mattered was torch.set_num_threads(1) below")
    parser.add_argument("--removeGround", action=argparse.BooleanOptionalAction, default=True,
                        help="drop map-derived ground points before voxelization")
    parser.add_argument("--outDir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--resume", type=str, default=None, metavar="PATH|auto")
    parser.add_argument("--checkpointEveryEpochs", type=int, default=5)
    parser.add_argument("--checkpointEverySteps", type=int, default=500)
    parser.add_argument("--maxSkipFrac", type=float, default=0.5,
                        help="if the fraction of micro-batches skipped in an epoch "
                             "(non-finite loss or non-finite grad) exceeds this, treat "
                             "the run as diverged: reload clean weights, and hard-stop "
                             "if it recurs for 2 consecutive epochs")
    parser.add_argument("--debugAnomaly", action="store_true",
                        help="enable torch.autograd anomaly detection and one-time "
                             "first-non-finite loss/grad diagnostics; slow, for "
                             "diagnostic runs only")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                        help="1 = EPE training; 2 = beta-NLL uncertainty-only training, "
                             "flow path frozen bit-identical; "
                             "3 = joint flow+uncertainty training (Option B)")
    parser.add_argument("--failureStrata", choices=["all", "foreground", "dynamic"],
                        default="dynamic",
                        help="which points the failure loss is computed on. Default dynamic: "
                             "background's label is tautological (|e| = ||pred|| there) and "
                             "fg-static is the easy half of foreground, so training on either "
                             "drags the head off the stratum the ceiling was measured on -- "
                             "smoke runs raised pooled AUC while FG_DYNAMIC AUC fell")
    parser.add_argument("--failureTau", type=float, default=0.10,
                        help="failure threshold in metres for --sigmaTarget failure")
    parser.add_argument("--sigmaTarget", choices=["nll", "logErr", "failure"], default="nll",
                        help="phase-2/3 sigma objective. 'logErr' regresses log sigma onto "
                             "log(|e|/sqrt(3)) under MSE in log space instead of using NLL: "
                             "an oracle-style ceiling on what any head can extract from the "
                             "frozen features, with no 1/var term for the tail to dominate")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="beta for beta-NLL loss (phases 2 and 3). The weight is "
                             "var.detach()**beta, and being detached it leaves the "
                             "per-point optimum at the proper-NLL var=sqErr/3 and only "
                             "reweights points: at 0.5 the weight is proportional to "
                             "sigma, which up-weights dynamic points ~16x over static, "
                             "the same rebalancing deflowLoss does for phase 1")
    parser.add_argument("--phase1Ckpt", type=Path, default=None,
                        help="checkpoint to load weights from for phases 2 and 3; "
                             "uncertaintyHead.* is dropped so it keeps its init")
    parser.add_argument("--flowLr", type=float, default=5e-5,
                        help="learning rate for flow head in phase 3 (default: 5e-5)")
    parser.add_argument("--epeBase", type=float, default=None,
                        help="baseline val EPE for phase 3 guardrail; "
                             "defaults to valEpe stored in --phase1Ckpt")
    args = parser.parse_args()

    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # One intra-op thread, set before the DataLoaders so workers inherit it. torch
    # defaults to one thread per core (18 here); with batch_size=1 sparse convs
    # every kernel takes microseconds, so the CUDA launch thread is the critical
    # path, and 18-way contention across the main process and every worker starved
    # it. Measured 15.6 -> 67 samp/s, a 4.3x speedup. See memory.md.
    torch.set_num_threads(1)
    if args.debugAnomaly:
        # Names the offending op the first time a NaN/Inf is produced in the
        # forward or backward graph. Raises, so use only for diagnostic runs.
        torch.autograd.set_detect_anomaly(True)
        print("debugAnomaly: torch.autograd anomaly detection ENABLED (slow)", flush=True)
    import spconv
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"sm={torch.cuda.get_device_capability()} spconv={spconv.__version__}",
        flush=True,
    )
    args.outDir.mkdir(parents=True, exist_ok=True)

    cacheDir = args.cacheDir
    trainMap = loadIndexMap(args.fitSet)
    trainDs = buildCachedSubset(trainMap, cacheDir, args.trainSamples)
    if args.overfit:
        valDs = buildCachedSubset(trainMap, cacheDir, args.trainSamples)
        valSource = f"{args.fitSet} (overfit probe)"
    else:
        valDs = buildCachedSubset(loadIndexMap(args.evalSet), cacheDir, args.valSamples)
        valSource = args.evalSet
    print(f"using {len(trainDs)} {args.fitSet} / {len(valDs)} {valSource} samples from cache")

    # Training is dataloader-bound, not GPU-bound: each sample is a ~2.2 MB
    # torch.load + unpickle, and the GPU idles without enough workers feeding it.
    trainDl = DataLoader(trainDs, batch_size=1, shuffle=True, num_workers=args.numWorkers,
                         persistent_workers=True, pin_memory=True, prefetch_factor=4,
                         collate_fn=identityCollate)
    valDl = DataLoader(valDs, batch_size=1, shuffle=False, num_workers=max(2, args.numWorkers // 3),
                       persistent_workers=True, pin_memory=True, prefetch_factor=4,
                       collate_fn=identityCollate)

    model = SparseFlowNet(inC=8).to(device)

    epeBase = None
    consecutiveBadEpochs = 0

    if args.phase in (2, 3):
        if args.phase1Ckpt is None:
            raise SystemExit(f"--phase {args.phase} requires --phase1Ckpt PATH")
        ckpt = torch.load(args.phase1Ckpt, map_location=device, weights_only=False)
        dropped, missing, unexpected = loadModelWeights(model, ckpt["model"])
        print(f"loaded phase-1 weights from {args.phase1Ckpt}")
        print(f"  dropped on shape mismatch (expected uncertaintyHead.* only): {dropped}")
        print(f"  missing keys (expected uncertaintyHead.*): {missing}")
        print(f"  unexpected keys (should be empty): {unexpected}")
        for name, p in model.named_parameters():
            if not name.startswith(trainablePrefixes(args.phase, args.sigmaTarget)):
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
    totalSteps = max(args.epochs * stepsPerEpoch, 1)
    warmupSteps = min(int(args.warmupEpochs * stepsPerEpoch), totalSteps - 1)
    if warmupSteps > 0:
        # Linear warmup then cosine decay over the remaining steps. Warmup gives
        # AdamW's second-moment estimate time to settle before the LR is at peak,
        # which is what tamed the early-training divergence.
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=warmupSteps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(totalSteps - warmupSteps, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[warmupSteps])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=totalSteps)
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
        setTrainMode(model, args.phase, args.sigmaTarget)
        trainSumDev = torch.zeros((), device=device)
        trainN = 0
        t0 = time.time()
        # globalStep counts optimizer steps, not micro-batches.
        opt.zero_grad(set_to_none=True)
        pendingMicro = 0
        skipped = 0
        gradSkips = 0
        for microIdx, sample in enumerate(trainDl):
            with torch.autocast("cuda", dtype=ampDtype, enabled=args.amp):
                loss = runStep(model, sample, device, args.voxelSize, pointRange,
                               phase=args.phase, beta=args.beta,
                               removeGround=args.removeGround, loss=args.loss,
                               sigmaTarget=args.sigmaTarget, failureStrata=args.failureStrata,
                               debug=args.debugAnomaly)
            if not torch.isfinite(loss):
                # Skip the bad micro-batch before backward so accumulated grads
                # stay clean. Do NOT reset BatchNorm running stats here: validation
                # runs in eval() mode and depends on those stats, and wiping them on
                # every NaN was silently corrupting val (running_var pinned at init).
                skipped += 1
                continue
            scaler.scale(loss / args.accumSteps).backward()
            trainSumDev += loss.detach()
            trainN += 1
            pendingMicro += 1

            lastMicro = (microIdx + 1) == len(trainDl)
            if not (((microIdx + 1) % args.accumSteps == 0 or lastMicro) and pendingMicro > 0):
                continue

            scaler.unscale_(opt)
            # Defense in depth: a finite loss can still produce a non-finite
            # gradient. Under bf16 (or fp32) the GradScaler is a no-op and gives
            # no inf/nan step-skip, so without this guard clip_grad_norm_ turns one
            # NaN/Inf grad into NaN over every parameter and AdamW writes NaN into
            # the weights. Dropping the step keeps the weights clean and lets the
            # accumulation restart fresh. (clip_grad_norm_ returns the pre-clip norm.)
            totalNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(totalNorm):
                if args.debugAnomaly:
                    reportNonFiniteGrad(model, totalNorm)
                opt.zero_grad(set_to_none=True)
                pendingMicro = 0
                gradSkips += 1
                continue
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

        # Circuit-breaker: mass-skipping means the run went numerically unstable.
        # v3 silently ground through 30 epochs at ~99.6% skip because the only guard
        # was on val EPE (which stayed finite on a frozen model). Catch it on the
        # training side: reload clean weights and, if it recurs, hard-stop.
        skipFrac = (skipped + gradSkips) / max(len(trainDl), 1)
        if skipFrac > args.maxSkipFrac:
            consecutiveBadEpochs += 1
            print(f"  WARNING: epoch {epoch} skipFrac={skipFrac:.3f} "
                  f"(skipped={skipped} gradSkips={gradSkips} of {len(trainDl)}) "
                  f"> maxSkipFrac={args.maxSkipFrac} ({consecutiveBadEpochs}/2)", flush=True)
            if consecutiveBadEpochs >= 2:
                print("  Hard stop: excessive micro-batch skipping for 2 consecutive epochs.", flush=True)
                saveCheckpoint(args.outDir / "last.pt", model, opt, sched, scaler,
                               epoch, globalStep, bestVal, float("nan"), args)
                break
            recoverPath = next((c for c in [args.outDir / "best.pt", args.outDir / "last.pt"]
                                if c.exists()), None)
            if recoverPath is not None:
                print(f"  Recovery: reloading clean weights/optimizer from {recoverPath}", flush=True)
                loadCheckpoint(recoverPath, model, opt, sched, scaler, device, voxelSize=args.voxelSize)
            else:
                print("  Recovery: no checkpoint to reload; continuing on current weights.", flush=True)
            continue  # skip validation/checkpointing on a poisoned epoch

        model.eval()
        # Per-metric counts, because a sample can be finite in EPE but have no
        # dynamic points at all; averaging over its own finite count keeps one
        # missing stratum from poisoning the others.
        valSums, valCounts = {}, {}
        collected = [] if args.sigmaTarget == "failure" else None
        with torch.no_grad():
            for sample in valDl:
                with torch.autocast("cuda", dtype=ampDtype, enabled=args.amp):
                    metrics = runStep(
                        model, sample, device, args.voxelSize, pointRange,
                        phase=args.phase, beta=args.beta, returnDynamic=True,
                        removeGround=args.removeGround, loss=args.loss,
                        sigmaTarget=args.sigmaTarget, failureTau=args.failureTau,
                        failureStrata=args.failureStrata, collect=collected)
                for name, value in metrics.items():
                    if torch.isfinite(value):
                        valSums[name] = valSums.get(name, 0.0) + value.item()
                        valCounts[name] = valCounts.get(name, 0) + 1

        # ROC-AUC over the WHOLE split. A mean of per-sample AUCs is not an AUC, and
        # per-sample base rates vary enormously here, so it has to be pooled.
        valAuc = valAucDyn = float("nan")
        if collected:
            C = torch.cat(collected)
            score, errv, dynv = C[:, 0].numpy(), C[:, 1].numpy(), C[:, 2].numpy() > 0.5
            lab = errv > args.failureTau
            if lab.any() and (~lab).any():
                valAuc = rocAuc(lab, score)
            if dynv.any() and lab[dynv].any() and (~lab[dynv]).any():
                valAucDyn = rocAuc(lab[dynv], score[dynv])
            del C

        def valMean(name):
            # nan (not 0) when nothing was finite, so divergence stays loud (see §9).
            return valSums[name] / valCounts[name] if valCounts.get(name) else float("nan")

        trainLoss = trainSumDev.item() / max(trainN, 1)
        valEpe, valDynEpe, valStaticEpe = valMean("epe"), valMean("dynEpe"), valMean("staticEpe")
        valNll, valSigma, valCover90 = valMean("nll"), valMean("sigma"), valMean("cover90")
        valNllMed = valMean("nllMed")
        valFinite = valCounts.get("epe", 0)
        dt = time.time() - t0

        peakGb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        skipMsg = f"  SKIPPED={skipped}" if skipped else ""
        if gradSkips:
            skipMsg += f"  GRADSKIP={gradSkips}"
        finMsg = f"  valFinite={valFinite}/{len(valDl)}" if valFinite < len(valDl) else ""
        if args.phase in (2, 3):
            print(f"epoch {epoch}: trainNLL={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  valStaticEPE={valStaticEpe:.4f}  "
                  f"valNLLmed={valNllMed:.4f}  valNLL={valNll:.4f}  "
                  f"valSigma={valSigma:.5f}  valCover90={valCover90:.4f}  "
                  f"valAUC={valAuc:.4f}  valAUCdyn={valAucDyn:.4f}  "
                  f"dt={dt:.1f}s  peakVram={peakGb:.2f}GB{skipMsg}{finMsg}", flush=True)
        else:
            print(f"epoch {epoch}: trainLoss={trainLoss:.4f}  valEPE={valEpe:.4f}  "
                  f"valDynEPE={valDynEpe:.4f}  valStaticEPE={valStaticEpe:.4f}  "
                  f"dt={dt:.1f}s  peakVram={peakGb:.2f}GB{skipMsg}{finMsg}", flush=True)

        # Divergence guard: a non-finite val EPE means the model blew up. Abort
        # after two in a row rather than burning the rest of the run.
        if not math.isfinite(valEpe):
            consecutiveBadEpochs += 1
            print(f"  WARNING: valEPE is non-finite ({valFinite}/{len(valDl)} val samples finite) "
                  f"({consecutiveBadEpochs}/2)", flush=True)
            if consecutiveBadEpochs >= 2:
                print("  Hard stop: validation diverged (non-finite) for 2 consecutive epochs.", flush=True)
                saveCheckpoint(args.outDir / "last.pt", model, opt, sched, scaler,
                               epoch, globalStep, bestVal, valEpe, args)
                break
        elif args.phase != 3:
            consecutiveBadEpochs = 0

        saveCheckpoint(args.outDir / "last.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if args.phase == 3:
            if valEpe < bestVal:
                bestVal = valEpe
            if not math.isnan(valNllMed) and valNllMed < bestNll and valEpe <= 1.05 * epeBase:
                bestNll = valNllMed
                saveCheckpoint(args.outDir / "best.pt", model, opt, sched, scaler, epoch, globalStep, bestNll, valEpe, args)
                print(f"  saved best.pt (valNLLmed={bestNll:.4f}, valEPE={valEpe:.4f})")
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
            if math.isfinite(valEpe) and valEpe < bestVal:
                bestVal = valEpe
                saveCheckpoint(args.outDir / "best_epe.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)
            # Median, not mean: see the note in runStep.
            if math.isfinite(valNllMed) and valNllMed < bestNll:
                bestNll = valNllMed
                saveCheckpoint(args.outDir / "best_nll.pt", model, opt, sched, scaler, epoch, globalStep, bestNll, valEpe, args)
        else:
            # isfinite guard so a diverged-but-finite EPE (e.g. 2e8) can't be saved as best.
            if math.isfinite(valEpe) and valEpe < bestVal:
                bestVal = valEpe
                saveCheckpoint(args.outDir / "best.pt", model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args)

        if args.checkpointEveryEpochs > 0 and (epoch + 1) % args.checkpointEveryEpochs == 0:
            saveCheckpoint(
                args.outDir / f"epoch_{epoch}.pt",
                model, opt, sched, scaler, epoch, globalStep, bestVal, valEpe, args,
            )

    if args.phase in (2, 3):
        print(f"best valEPE: {bestVal:.4f}  best valNLLmed: {bestNll:.4f}")
    else:
        print(f"best valEPE: {bestVal:.4f}")


if __name__ == "__main__":
    print("Starting...")
    main()
