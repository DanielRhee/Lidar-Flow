"""Freeze the train/val/pseudo-test split to JSON, partitioned by log UUID.

AV2's real test split ships no annotations, so it cannot be scored locally. We
carve a pseudo-test set out of the train logs instead and leave AV2's val split
intact for model selection.

Partitioning on log UUID rather than sample index is the point: consecutive
sweeps within a log are near-duplicates, so an index-level split would leak
almost-identical frames across the boundary and flatter the pseudo-test score.

Two independent operations:

  makeSplits.py               regenerate the frozen train/val/pseudoTest split.
                              Needs the dataset, since only av2's loader knows
                              the log_id -> loader-index mapping.
  makeSplits.py --uncCarve    add ONLY the uncFit/uncHoldout keys to an existing
                              split file, reading log ids out of the populated
                              cache. Needs no dataset, and cannot touch a frozen
                              key because it never recomputes one.

uncFit/uncHoldout is the log-level carve of val + pseudoTest that the uncertainty
head is fitted on: the flow weights only ever saw train, so val/pseudoTest errors
are the generalization errors sigma has to model, and the holdout exists so its
calibration can be reported out of sample.

Run once; commit the result. populate.py and train.py both read it, so the cache
and every future run share one definition.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from paths import DEFAULT_CACHE_DIR, DEFAULT_DATASET, DEFAULT_DATASET_DIR, SPLITS_FILE

# The cache is keyed on these, so a re-run that changed any of them would silently
# invalidate ~290 GB of cached samples.
FROZEN_KEYS = ("trainLogs", "pseudoTestLogs", "valLogs",
               "trainIndices", "pseudoTestIndices", "valIndices")


def logToIndices(loader):
    """Map log_id -> sorted list of *loader* indices, using the cheap file index.

    A loader index is not a file_index row: the loader yields sweep *pairs*, so
    index_map[i] gives the file_index row for loader index i, skipping each log's
    final sweep (which has no successor). get_log_id() takes a file_index row,
    not a loader index, so it is deliberately not used here.
    """
    logIds = loader.file_index["log_id"].tolist()
    mapping = {}
    for loaderIdx, backendIdx in enumerate(loader.index_map):
        mapping.setdefault(logIds[backendIdx], []).append(loaderIdx)
    return mapping


def logToIndicesFromCache(cacheDir, split, indices, numThreads=16):
    """Same mapping as logToIndices, read from the populated cache instead.

    Every cached sample stores uuid = (logId, timestamp), so the log partition is
    recoverable without the dataset. mmap=True unpickles only the header and never
    faults the tensor pages: ~0.2 ms/sample against ~2.2 MB on disk.
    """
    splitDir = Path(cacheDir) / split

    def logIdOf(idx):
        return torch.load(splitDir / f"{idx}.pt", weights_only=False, mmap=True)["uuid"][0]

    with ThreadPoolExecutor(max_workers=numThreads) as pool:
        logIds = list(pool.map(logIdOf, indices))

    mapping = {}
    for idx, logId in zip(indices, logIds):
        mapping.setdefault(logId, []).append(idx)
    return mapping


def uncertaintyCarve(logMap, holdoutFrac, seed):
    """Log-level split of {(sourceSplit, logId): [loaderIdx]} into fit / holdout.

    Log level, not index level, for the same reason as the pseudoTest carve.
    Keyed by (sourceSplit, logId) because pseudoTest indices are *train*-loader
    indices and collide numerically with val's.
    """
    logs = sorted(logMap)
    random.Random(seed).shuffle(logs)
    nHold = round(holdoutFrac * len(logs))
    holdoutKeys, fitKeys = sorted(logs[:nHold]), sorted(logs[nHold:])

    def groupBySplit(keys):
        out = {}
        for split, log in keys:
            out.setdefault(split, []).extend(logMap[(split, log)])
        return {split: sorted(idxs) for split, idxs in out.items()}

    fitIndices, holdoutIndices = groupBySplit(fitKeys), groupBySplit(holdoutKeys)
    for split in set(fitIndices) | set(holdoutIndices):
        assert not (set(fitIndices.get(split, ())) & set(holdoutIndices.get(split, ()))), \
            f"uncFit/uncHoldout index overlap in {split}"
    return fitKeys, holdoutKeys, fitIndices, holdoutIndices


# Adds only the uncFit/uncHoldout keys, from the cache. Every other key is copied
# through untouched, so this cannot invalidate the cache the way a full rebuild can.
def addUncertaintyCarve(args):
    outPath = Path(args.outFile)
    if not outPath.exists():
        raise SystemExit(f"--uncCarve needs an existing {outPath}; run makeSplits.py first")
    splits = json.loads(outPath.read_text())

    logMap = {}
    for split in ("val", "pseudoTest"):
        indices = splits[f"{split}Indices"]
        perLog = logToIndicesFromCache(args.cacheDir, split, indices)
        nMapped = sum(len(v) for v in perLog.values())
        assert nMapped == len(indices), f"{split}: mapped {nMapped} of {len(indices)}"
        print(f"  {split}: {len(perLog)} logs over {len(indices)} cached samples")
        logMap.update({(split, log): idxs for log, idxs in perLog.items()})

    fitKeys, holdoutKeys, fitIndices, holdoutIndices = uncertaintyCarve(
        logMap, args.uncHoldoutFrac, args.uncSeed)
    nFit = sum(len(v) for v in fitIndices.values())
    nHoldout = sum(len(v) for v in holdoutIndices.values())
    assert nFit + nHoldout == len(splits["valIndices"]) + len(splits["pseudoTestIndices"])

    splits.update({
        "uncHoldoutFrac": args.uncHoldoutFrac,
        "uncSeed": args.uncSeed,
        # JSON has no tuple keys, so logs are [sourceSplit, logId] pairs.
        "uncFitLogs": [list(k) for k in fitKeys],
        "uncHoldoutLogs": [list(k) for k in holdoutKeys],
        "uncFitIndices": fitIndices,
        "uncHoldoutIndices": holdoutIndices,
    })
    with open(outPath, "w") as f:
        json.dump(splits, f)

    print(f"uncFit:     {len(fitKeys):4d} logs  {nFit:6d} samples")
    print(f"uncHoldout: {len(holdoutKeys):4d} logs  {nHoldout:6d} samples")
    print(f"updated {outPath} (uncertainty keys only)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--outFile", default=str(SPLITS_FILE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pseudoTestFrac", type=float, default=0.10,
                        help="fraction of train LOGS held out as pseudo-test")
    parser.add_argument("--uncHoldoutFrac", type=float, default=0.15,
                        help="fraction of val+pseudoTest LOGS held back to report "
                             "uncertainty calibration on, out of sample")
    parser.add_argument("--uncSeed", type=int, default=1)
    parser.add_argument("--uncCarve", action="store_true",
                        help="add only the uncFit/uncHoldout keys to an existing split "
                             "file, reading log ids from the cache. Needs no dataset and "
                             "leaves every other key byte-identical")
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="populated cache to read log ids from, for --uncCarve")
    parser.add_argument("--force", action="store_true",
                        help="write even if a frozen train/val/pseudoTest index list "
                             "would change; invalidates the cache, so mean it")
    args = parser.parse_args()

    if args.uncCarve:
        addUncertaintyCarve(args)
        return

    from extractSceneflow import buildLoader

    trainLoader = buildLoader(args.datasetDir, args.dataset, "train")
    valLoader = buildLoader(args.datasetDir, args.dataset, "val")

    trainLogMap = logToIndices(trainLoader)
    valLogMap = logToIndices(valLoader)

    logs = sorted(trainLogMap)
    rng = random.Random(args.seed)
    rng.shuffle(logs)
    nHold = round(args.pseudoTestFrac * len(logs))
    pseudoTestLogs = sorted(logs[:nHold])
    trainLogs = sorted(logs[nHold:])

    assert not (set(trainLogs) & set(pseudoTestLogs)), "train/pseudoTest log overlap"

    trainIndices = sorted(i for log in trainLogs for i in trainLogMap[log])
    pseudoTestIndices = sorted(i for log in pseudoTestLogs for i in trainLogMap[log])
    valIndices = sorted(i for log in valLogMap for i in valLogMap[log])

    assert not (set(trainIndices) & set(pseudoTestIndices)), "train/pseudoTest index overlap"
    assert len(trainIndices) + len(pseudoTestIndices) == len(trainLoader)

    splits = {
        "seed": args.seed,
        "pseudoTestFrac": args.pseudoTestFrac,
        "dataset": args.dataset,
        "note": "pseudoTest is carved from AV2 train logs; val is AV2's own val split, untouched.",
        "trainLogs": trainLogs,
        "pseudoTestLogs": pseudoTestLogs,
        "valLogs": sorted(valLogMap),
        "trainIndices": trainIndices,
        "pseudoTestIndices": pseudoTestIndices,
        "valIndices": valIndices,
    }

    outPath = Path(args.outFile)
    merged = splits
    if outPath.exists():
        old = json.loads(outPath.read_text())
        changed = [k for k in FROZEN_KEYS if k in old and old[k] != splits[k]]
        if changed and not args.force:
            raise SystemExit(
                f"refusing to overwrite {outPath}: frozen key(s) {changed} would change, "
                "which invalidates the populated cache. Pass --force only if intended."
            )
        # Carry forward keys this path does not own (the --uncCarve output), but
        # drop them if the frozen indices they were derived from just moved.
        carried = {k: v for k, v in old.items() if k not in splits}
        if changed:
            stale = [k for k in carried if k.startswith("unc")]
            if stale:
                print(f"WARNING: frozen keys changed; dropping stale {stale}. Re-run --uncCarve.")
            carried = {k: v for k, v in carried.items() if not k.startswith("unc")}
        merged = {**carried, **splits}
    with open(outPath, "w") as f:
        json.dump(merged, f)

    print(f"train:      {len(trainLogs):4d} logs  {len(trainIndices):6d} samples")
    print(f"pseudoTest: {len(pseudoTestLogs):4d} logs  {len(pseudoTestIndices):6d} samples")
    print(f"val:        {len(valLogMap):4d} logs  {len(valIndices):6d} samples")
    print(f"wrote {outPath}  (run --uncCarve to add uncFit/uncHoldout)")


if __name__ == "__main__":
    main()
