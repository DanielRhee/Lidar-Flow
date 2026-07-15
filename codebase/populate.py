import argparse
import gc
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from extractSceneflow import buildLoader, loadAnnotation
from paths import DEFAULT_CACHE_DIR, DEFAULT_DATASET, DEFAULT_DATASET_DIR, SPLITS_FILE


def populate(split, indices, loader, cacheDir, numThreads):
    splitCache = Path(cacheDir) / split
    splitCache.mkdir(parents=True, exist_ok=True)

    todo = [i for i in indices if not (splitCache / f"{i}.pt").exists()]
    cached = len(indices) - len(todo)
    if not todo:
        print(f"  [{split}] already complete: {cached}/{len(indices)} cached, nothing to do")
        return

    print(f"  [{split}] {cached}/{len(indices)} already cached, fetching {len(todo)}")

    lock = threading.Lock()
    counter = [0]
    wallStart = time.time()

    def fetchOne(idx):
        sample = loadAnnotation(loader, idx)
        # Write to a temp name then rename, so an interrupted run never leaves a
        # truncated .pt that the existence check above would treat as complete.
        tmp = splitCache / f".{idx}.pt.tmp"
        torch.save(sample, tmp)
        tmp.rename(splitCache / f"{idx}.pt")
        del sample  # the loader leaks badly without this; see documentation.md

        with lock:
            counter[0] += 1
            n = counter[0]
            if n % 50 == 0:
                gc.collect()
            if n % 200 == 0 or n == len(todo):
                elapsed = time.time() - wallStart
                rate = n / elapsed
                eta = (len(todo) - n) / rate / 60 if rate > 0 else 0
                print(f"  [{split}] {n}/{len(todo)}  rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    with ThreadPoolExecutor(max_workers=numThreads) as pool:
        for _ in pool.map(fetchOne, todo):
            pass

    wallElapsed = time.time() - wallStart
    print(f"  [{split}] FINISHED {len(todo)} samples in {wallElapsed / 60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cacheDir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--splitsFile", type=Path, default=SPLITS_FILE)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "pseudoTest"],
                        help="which frozen splits to populate")
    parser.add_argument("--trainSamples", type=int, default=-1,
                        help="cap samples per split; -1 populates the whole frozen split")
    parser.add_argument("--valSamples", type=int, default=-1)
    parser.add_argument("--numThreads", type=int, default=12)
    args = parser.parse_args()

    Path(args.cacheDir).mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(args.splitsFile).read_text())

    # pseudoTest is carved from AV2's train logs, so it reads the train loader.
    sourceSplit = {"train": "train", "pseudoTest": "train", "val": "val"}
    indicesKey = {"train": "trainIndices", "pseudoTest": "pseudoTestIndices", "val": "valIndices"}
    cap = {"train": args.trainSamples, "pseudoTest": args.trainSamples, "val": args.valSamples}

    overallStart = time.time()
    for split in args.splits:
        indices = splits[indicesKey[split]]
        if cap[split] > 0:
            indices = indices[:cap[split]]
        print(f"{split} cache: {len(indices)} samples")
        loader = buildLoader(args.datasetDir, args.dataset, sourceSplit[split])
        populate(split, indices, loader, args.cacheDir, args.numThreads)
        del loader
        gc.collect()

    print(f"TOTAL TIME {(time.time() - overallStart) / 60:.1f}min")


if __name__ == "__main__":
    main()
