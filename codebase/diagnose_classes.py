from collections import Counter

import torch
from av2.datasets.sensor.constants import AnnotationCategories
from av2.evaluation.scene_flow.constants import CATEGORY_TO_INDEX

from classes import className, metaName
from dataset import loadSplitIndices
from paths import DEFAULT_CACHE_DIR

def main():
    cacheDir = DEFAULT_CACHE_DIR

    # Enum *position* is not the category index: av2 reserves 0 for background and
    # shifts everything up by one, so index = position + 1. Resolving a name with
    # catList[idx] is off by one and renames every class silently.
    catList = list(AnnotationCategories)
    print("=== AnnotationCategories position → category INDEX → name ===")
    for i, cat in enumerate(catList):
        print(f"  position {i:3d} → index {CATEGORY_TO_INDEX[cat.value]:3d}: {cat.value}")
    print(f"  index 0 is background (NONE); {len(catList)} categories → indices 1..{len(catList)}\n")

    valIdx = loadSplitIndices("val")
    sweepIdx = valIdx[0]

    sample = torch.load(cacheDir / "val" / f"{sweepIdx}.pt", weights_only=False)

    raw    = sample["categoryIndices"].numpy()
    valid  = sample["isValid"].numpy()

    raw_counts = Counter(raw.tolist())
    print(f"=== Sweep {sweepIdx}: raw category_indices (before is_valid) ===")
    for idx in sorted(raw_counts):
        n_total = raw_counts[idx]
        n_valid = int((raw == idx)[valid].sum())
        print(f"  idx {idx:3d} ({className(idx)} / {metaName(idx)}): {n_total:>8,} total  {n_valid:>8,} valid")

    pedIdx = CATEGORY_TO_INDEX["PEDESTRIAN"]
    print(f"\nPEDESTRIAN category index: {pedIdx}  (enum position {pedIdx - 1})")
    print(f"  raw count in sweep {sweepIdx}: {raw_counts.get(pedIdx, 0)}")

if __name__ == "__main__":
    main()
