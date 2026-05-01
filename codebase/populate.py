import argparse
import gc
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from extractSceneflow import buildLoader, loadAnnotation


def populate(split, indices, loader, cacheDir, numThreads):
    splitCache = Path(cacheDir) / split
    splitCache.mkdir(parents=True, exist_ok=True)

    todo = [i for i in indices if not (splitCache / f"{i}.pt").exists()]
    cached = len(indices) - len(todo)
    if not todo:
        # cancel
        return

    lock = threading.Lock()
    counter = [0]
    perSampleTimes = []   # network read + torch.save, wall time per sample
    fetchTimes = []       # network read only
    saveTimes = []        # torch.save only
    wallStart = time.time()

    def fetchOne(idx):
        tFetchStart = time.time()
        sample = loadAnnotation(loader, idx)
        tFetchEnd = time.time()
        torch.save(sample, splitCache / f"{idx}.pt")
        tSaveEnd = time.time()
        del sample  # stop memory leak

        with lock:
            counter[0] += 1
            n = counter[0]
            perSampleTimes.append(tSaveEnd - tFetchStart)
            fetchTimes.append(tFetchEnd - tFetchStart)
            saveTimes.append(tSaveEnd - tFetchEnd)
            if n % 50 == 0:
                gc.collect()
            if n % 25 == 0 or n == len(todo):
                elapsed = time.time() - wallStart
                rate = n / elapsed
                eta = (len(todo) - n) / rate / 60 if rate > 0 else 0
                print(f"  [{split}] {n}/{len(todo)}  wall_rate={rate:.2f} samp/s  eta={eta:.1f}min", flush=True)

    with ThreadPoolExecutor(max_workers=numThreads) as pool:
        for _ in pool.map(fetchOne, todo):
            pass

    wallElapsed = time.time() - wallStart

    def pct(xs, p):
        if not xs:
            return float("nan")
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
        return s[k]

    nFetched = len(todo)

    print("FINISHED", nFetched, wallElapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetDir", type=Path, default=Path.home() / "persistent")
    parser.add_argument("--dataset", default="data")
    parser.add_argument(
        "--cacheDir",
        type=Path,
        default=Path.home() / "persistent" / "djrhee" / "lidarflow_cache",
    )
    parser.add_argument("--trainSamples", type=int, default=5000)
    parser.add_argument("--valSamples", type=int, default=500)
    parser.add_argument("--numThreads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    overallStart = time.time()

    if args.trainSamples > 0:
        print("train cache...")
        tBuild = time.time()
        trainLoader = buildLoader(args.datasetDir, args.dataset, "train")
        trainIdx = random.sample(range(len(trainLoader)), min(args.trainSamples, len(trainLoader)))
        populate("train", trainIdx, trainLoader, args.cacheDir, args.numThreads)
        del trainLoader
        gc.collect()

    if args.valSamples > 0:
        print("val cache...")
        tBuild = time.time()
        valLoader = buildLoader(args.datasetDir, args.dataset, "val")
        valIdx = random.sample(range(len(valLoader)), min(args.valSamples, len(valLoader)))
        populate("val", valIdx, valLoader, args.cacheDir, args.numThreads)
        del valLoader
        gc.collect()

    print("TOTAL TIME", time.time()-overallStart)


if __name__ == "__main__":
    main()
