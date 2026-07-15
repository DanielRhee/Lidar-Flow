"""Freeze the train/val/pseudo-test split to JSON, partitioned by log UUID.

AV2's real test split ships no annotations, so it cannot be scored locally. We
carve a pseudo-test set out of the train logs instead and leave AV2's val split
intact for model selection.

Partitioning on log UUID rather than sample index is the point: consecutive
sweeps within a log are near-duplicates, so an index-level split would leak
almost-identical frames across the boundary and flatter the pseudo-test score.

Run once; commit the result. populate.py and train.py both read it, so the cache
and every future run share one definition.
"""

import argparse
import json
import random

from extractSceneflow import buildLoader
from paths import DEFAULT_DATASET, DEFAULT_DATASET_DIR, SPLITS_FILE


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--outFile", default=str(SPLITS_FILE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pseudoTestFrac", type=float, default=0.10,
                        help="fraction of train LOGS held out as pseudo-test")
    args = parser.parse_args()

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
    with open(args.outFile, "w") as f:
        json.dump(splits, f)

    print(f"train:      {len(trainLogs):4d} logs  {len(trainIndices):6d} samples")
    print(f"pseudoTest: {len(pseudoTestLogs):4d} logs  {len(pseudoTestIndices):6d} samples")
    print(f"val:        {len(valLogMap):4d} logs  {len(valIndices):6d} samples")
    print(f"wrote {args.outFile}")


if __name__ == "__main__":
    main()
