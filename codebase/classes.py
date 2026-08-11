"""Scene-flow class indices, derived from av2 rather than hand-written.

av2 keys flow.category_indices as

    CATEGORY_TO_INDEX = {"NONE": 0, **{cat: i + 1 for i, cat in AnnotationCategories}}

so the index is the AnnotationCategories position **plus one**, and 0 is background
(av2/torch/structures/flow.py: "0 is background"). Indexing the enum list directly
with a category index is therefore off by one, and it fails silently: it renames
every class rather than raising. That is exactly what happened here -- bollards came
out 86% dynamic and WHEELCHAIR came out as 2.5% of all LiDAR points.

Meta-categories are av2's own four groups, which partition AnnotationCategories
exactly (6 + 4 + 8 + 12 = 30); the assert below is the invariant that would have
caught the off-by-one immediately.
"""

from av2.datasets.sensor.constants import AnnotationCategories
from av2.evaluation.scene_flow.constants import (
    CATEGORY_TO_INDEX,
    InanimateCategories,
    LeggedCategories,
    SmallVehicleCategories,
    VehicleCategories,
)

BACKGROUND = "BACKGROUND"

INDEX_TO_NAME = {0: BACKGROUND, **{idx: name for name, idx in CATEGORY_TO_INDEX.items() if idx != 0}}

META_GROUPS = (
    ("INANIMATE", InanimateCategories),
    ("LEGGED", LeggedCategories),
    ("SMALL_VEHICLE", SmallVehicleCategories),
    ("VEHICLE", VehicleCategories),
)

INDEX_TO_META = {0: BACKGROUND}
for _meta, _group in META_GROUPS:
    INDEX_TO_META.update({CATEGORY_TO_INDEX[c.value]: _meta for c in _group})

assert len(INDEX_TO_META) == len(AnnotationCategories) + 1, (
    f"av2's meta-groups cover {len(INDEX_TO_META) - 1} of {len(AnnotationCategories)} "
    "categories; they are supposed to partition the enum exactly"
)
assert INDEX_TO_NAME.keys() == INDEX_TO_META.keys()

INANIMATE_INDICES = frozenset(CATEGORY_TO_INDEX[c.value] for c in InanimateCategories)


# Unknown indices (a uint8 wrap, say) stay visible as UNKNOWN_<n>. The old hand-written
# map folded them into background, which is how WHEELED_RIDER (index 30, 71% dynamic)
# went missing.
def className(idx):
    return INDEX_TO_NAME.get(int(idx), f"UNKNOWN_{int(idx)}")


def metaName(idx):
    return INDEX_TO_META.get(int(idx), f"UNKNOWN_{int(idx)}")


# ── Evaluation strata ────────────────────────────────────────────────────────
#
# Any aggregate sigma metric over all points is 80.3% BACKGROUND, and BACKGROUND is
# a degenerate stratum for uncertainty: AV2 sets a background point's gtFlow to
# rigid_flow, so in the ego0 frame
#     flowEgo0 = ego0_SE3_ego1 . ego1_SE3_ego0 . pc0 - pc0 = 0
# **identically** (measured max |gt| = 3.08e-05 m on the holdout, i.e. fp16 storage
# noise, against a 0.0019 m mean error). Three consequences:
#   * there is no aleatoric uncertainty there -- the target is a constant, so sigma
#     is predicting the model's own residual, not noise;
#   * |err| == ||pred|| exactly, so sigma proportional to ||predFlow|| is the
#     Bayes-optimal rule on that stratum rather than a shortcut, and near-constant
#     sigma there is not a pathology;
#   * Spearman(||predFlow||, |err|) == 1 there by identity, so any "static" or
#     "all-points" rank correlation is tautological to that extent.
# Report per stratum; do not quote the aggregate. Same argument Chodosh et al. make
# for flow accuracy, which is why three-way EPE separates these.
BACKGROUND_INDEX = 0
STRATA = ("BACKGROUND", "FG_STATIC", "FG_DYNAMIC")
BACKGROUND_MAX_GT = 1e-4   # asserted by callers so the zero-target fact cannot regress


def stratumOf(classIdx, isDynamic):
    """Vectorised BACKGROUND / FG_STATIC / FG_DYNAMIC codes, indexing into STRATA."""
    import numpy as np
    background = np.asarray(classIdx) == BACKGROUND_INDEX
    dynamic = np.asarray(isDynamic).astype(bool)
    # Background is 0.0% dynamic in the data, so FG_DYNAMIC is exactly isDynamic.
    return np.where(background, 0, np.where(dynamic, 2, 1)).astype(np.int8)
