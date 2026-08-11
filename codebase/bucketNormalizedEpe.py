"""Bucket-Normalized EPE, reimplemented from the reference package's definition.

Read out of `bucketed_scene_flow_eval/eval/bucketed_epe.py`. This metric aggregates
almost nothing like three-way EPE, and the difference is the point of it:

  * Speed buckets: `linspace(0, 2.0, 51)` then `inf` -> 51 buckets of width 0.04 m/frame.
    Bucket 0 = [0, 0.04) is the STATIC bucket. Note 0.04, not the 0.05 m/frame that
    three-way EPE and av2's `is_dynamic` use -- the two metrics disagree about which
    points are static.
  * Within a (class, speed-bucket) cell: count-weighted mean EPE and mean speed.
  * Normalization: dynamic cells are divided by that cell's own mean speed, so the units
    become "fraction of the object's motion missed".
  * Per class: `dynamic_error = nanmean` over the 50 dynamic buckets -- UNWEIGHTED, so a
    bucket holding 40 points counts as much as one holding 40 million.
  * **Fine classes are then MERGED INTO META-CLASSES** (`merge_matrix_classes`,
    count-weighted per bucket) and the headline mean is over the 5 meta-classes -- NOT over
    the ~30 fine classes. This is not a detail: the fine-class mean is 0.3706 here against
    the published aggregation's 0.2312, because it hands CONSTRUCTION_CONE the same weight
    as CAR.
  * `ROAD_SIGNS` (BOLLARD, CONSTRUCTION_BARREL, CONSTRUCTION_CONE,
    MOBILE_PEDESTRIAN_CROSSING_SIGN, SIGN, STOP_SIGN, MESSAGE_BOARD_TRAILER,
    TRAFFIC_LIGHT_TRAILER) belongs to no meta-class and is dropped entirely; the reference's
    own comment is "ignored because of labeling oddities". These classes never move, so
    their few dynamic points are annotation jitter landing in the 0.04-0.05 band where the
    model correctly predicts zero. A normalized error of ~1.0 is the CORRECT answer for a
    stationary object mislabelled as moving -- not evidence of model failure.

The unweighted bucket mean is deliberate: it stops the metric being a report on background
and cars. It also makes sparse cells high-variance, so per-cell counts are printed. Note
`--minCount` filters CELLS, not classes, so it cannot remove a jitter class whose single
dynamic cell is large -- use the meta-class output for that.

Two protocol gaps are flagged rather than hidden: this dump stores Euclidean range, not
the reference's L-infinity over (x, y), and it has ground removed from the input.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

BUCKET_MAX_SPEED = 20.0 / 10.0
NUM_BUCKETS = 51


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, default=Path("runs/unc_cv/oof_all.arrow"))
    parser.add_argument("--classNames", type=Path, default=Path("runs/classNames.json"))
    parser.add_argument("--maxRange", type=float, default=35.0)
    parser.add_argument("--minCount", type=int, default=1,
                        help="drop cells with fewer points than this before averaging. NOTE "
                             "this filters CELLS, not classes, so it does not remove a "
                             "jitter class whose single dynamic cell is large.")
    parser.add_argument("--fineClasses", action="store_true",
                        help="report the unweighted fine-class mean instead of the "
                             "published meta-class mean. Not the published quantity.")
    args = parser.parse_args()

    edges = np.concatenate([np.linspace(0, BUCKET_MAX_SPEED, NUM_BUCKETS), [np.inf]])
    names = {}
    if args.classNames.exists():
        import json
        names = {int(k): v for k, v in json.loads(args.classNames.read_text()).items()}

    nCls = 32
    sumEpe = np.zeros((nCls, NUM_BUCKETS))
    sumSpeed = np.zeros((nCls, NUM_BUCKETS))
    count = np.zeros((nCls, NUM_BUCKETS), dtype=np.int64)

    with ipc.open_file(str(args.dump)) as r:
        for i in range(r.num_record_batches):
            b = r.get_batch(i)
            g = lambda c: b.column(c).to_numpy(zero_copy_only=False)
            gx, gy, gz = g("gtFlowX"), g("gtFlowY"), g("gtFlowZ")
            speed = np.sqrt(gx * gx + gy * gy + gz * gz)
            epe = np.sqrt((g("predFlowX") - gx) ** 2 + (g("predFlowY") - gy) ** 2
                          + (g("predFlowZ") - gz) ** 2)
            cls = g("classIdx").astype(np.int64)
            sel = g("rangeMeters") < args.maxRange
            cls, speed, epe = cls[sel], speed[sel], epe[sel]
            bkt = np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, NUM_BUCKETS - 1)
            flat = cls * NUM_BUCKETS + bkt
            np.add.at(sumEpe.reshape(-1), flat, epe)
            np.add.at(sumSpeed.reshape(-1), flat, speed)
            np.add.at(count.reshape(-1), flat, 1)
            print(f"  batch {i+1}/{r.num_record_batches}", end="\r", flush=True)

    ok = count >= args.minCount
    cellEpe = np.where(ok, sumEpe / np.maximum(count, 1), np.nan)
    cellSpeed = np.where(ok, sumSpeed / np.maximum(count, 1), np.nan)
    norm = cellEpe.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        norm[:, 1:] = cellEpe[:, 1:] / cellSpeed[:, 1:]

    staticEpe = norm[:, 0]
    with np.errstate(invalid="ignore"):
        dynErr = np.nanmean(norm[:, 1:], axis=1)

    present = np.flatnonzero(count.sum(1) > 0)
    print(f"\n{'class':22s} {'n':>14s} {'staticEPE':>10s} {'dynErr':>9s} {'#dyn cells':>11s}")
    for c in present:
        nd = int(np.isfinite(norm[c, 1:]).sum())
        print(f"{names.get(c, f'class{c}'):22s} {count[c].sum():>14,} "
              f"{staticEpe[c]:10.5f} {dynErr[c]:9.4f} {nd:>11d}")

    print(f"\n  range < {args.maxRange} m (EUCLIDEAN; reference uses L-inf over x,y)")
    if args.fineClasses:
        with np.errstate(invalid="ignore"):
            mStatic, mDyn = np.nanmean(staticEpe[present]), np.nanmean(dynErr[present])
        print(f"  FINE-CLASS mean (NOT the published quantity)")
        print(f"    mean static EPE    = {mStatic:.5f} m")
        print(f"    mean dynamic error = {mDyn:.4f} over {len(present)} classes, unweighted")
        return

    from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import (
        BUCKETED_METACATAGORIES)
    byName = {v: k for k, v in names.items()}
    metaStatic, metaDyn = [], []
    print(f"\n  {'meta-class':16s} {'n':>15s} {'staticEPE':>10s} {'dynErr':>9s} {'cells':>6s}")
    for meta in sorted(BUCKETED_METACATAGORIES):
        idx = [byName[c] for c in BUCKETED_METACATAGORIES[meta] if c in byName]
        if not idx:
            continue
        mE, mS, mC = sumEpe[idx].sum(0), sumSpeed[idx].sum(0), count[idx].sum(0)
        ce = np.where(mC > 0, mE / np.maximum(mC, 1), np.nan)
        cs = np.where(mC > 0, mS / np.maximum(mC, 1), np.nan)
        with np.errstate(invalid="ignore"):
            de = np.nanmean(ce[1:] / cs[1:])
        metaStatic.append(ce[0])
        metaDyn.append(de)
        print(f"  {meta:16s} {mC.sum():>15,} {ce[0]:10.5f} {de:9.4f} {int((mC[1:] > 0).sum()):>6d}")
    dropped = [names[c] for c in present
               if names.get(c) not in {x for v in BUCKETED_METACATAGORIES.values() for x in v}]
    with np.errstate(invalid="ignore"):
        print(f"\n  mean static EPE      = {np.nanmean(metaStatic):.5f} m")
        print(f"  MEAN DYNAMIC ERROR   = {np.nanmean(metaDyn):.4f}   <- the published quantity")
    print(f"  dropped (no meta-class): {', '.join(sorted(dropped))}")


if __name__ == "__main__":
    main()
