from collections import Counter

import torch
from av2.datasets.sensor.constants import AnnotationCategories
from dataset import loadSplitIndices
from paths import DEFAULT_CACHE_DIR

def main():
    cacheDir = DEFAULT_CACHE_DIR

    catList = list(AnnotationCategories)
    print("=== AnnotationCategories list-position → name ===")
    for i, cat in enumerate(catList):
        print(f"  {i:3d}: {cat.value}")
    print(f"  (total {len(catList)} categories; sentinel = {len(catList)})\n")

    valIdx = loadSplitIndices("val")
    sweepIdx = valIdx[0]

    sample = torch.load(cacheDir / "val" / f"{sweepIdx}.pt", weights_only=False)

    raw    = sample["categoryIndices"].numpy()
    valid  = sample["isValid"].numpy()

    raw_counts = Counter(raw.tolist())
    print(f"=== Sweep {sweepIdx}: raw category_indices (before is_valid) ===")
    for idx in sorted(raw_counts):
        name    = catList[idx].value if idx < len(catList) else f"<sentinel {idx}>"
        n_total = raw_counts[idx]
        n_valid = int((raw == idx)[valid].sum())
        print(f"  idx {idx:3d} ({name}): {n_total:>8,} total  {n_valid:>8,} valid")

    ped_idx = next((i for i, c in enumerate(catList) if c.value == "PEDESTRIAN"), None)
    print(f"\nPEDESTRIAN list-position index: {ped_idx}")
    if ped_idx is not None:
        n = raw_counts.get(ped_idx, 0)
        print(f"  raw count in sweep {sweepIdx}: {n}")

if __name__ == "__main__":
    main()
