import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from scipy import stats as spStats

_MAX_ROWS = 20_000_000


def loadSampled(path, maxRows=_MAX_ROWS, seed=42):
    # Read the IPC file one batch at a time, sampling proportionally to stay under maxRows.
    # Peak memory = one decompressed batch (~1-2 GB) + sampled output (~800 MB).
    with ipc.open_file(str(path)) as reader:
        nBatches = reader.num_record_batches
        firstBatch = reader.get_batch(0)
        keepRate = min(1.0, maxRows / max(firstBatch.num_rows * nBatches, 1))
        rng = np.random.default_rng(seed)

        def sampleBatch(batch):
            if keepRate >= 1.0:
                return batch.to_pandas()
            n = max(1, round(keepRate * batch.num_rows))
            idx = np.sort(rng.choice(batch.num_rows, min(n, batch.num_rows), replace=False))
            return batch.take(idx).to_pandas()

        dfs = [sampleBatch(firstBatch)]
        del firstBatch
        for i in range(1, nBatches):
            batch = reader.get_batch(i)
            dfs.append(sampleBatch(batch))
            del batch

    df = pd.concat(dfs, ignore_index=True)
    print(f"  sampled {len(df):,} / ~{round(maxRows / keepRate):,} points from {Path(path).name}")
    return df


plt.rcParams.update({
    "figure.facecolor": "#111",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#555",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "text.color": "#ddd",
    "grid.color": "#333",
    "grid.linestyle": "--",
    "legend.facecolor": "#222",
    "legend.edgecolor": "#444",
})

_COL_A = "#4FC3F7"
_COL_B = "#EF9A9A"


def buildBucketMap():
    from av2.datasets.sensor.constants import AnnotationCategories
    # Explicit fine-grained map — VRU subtypes kept separate for per-bucket analysis.
    nameToMeta = {
        "PEDESTRIAN": "PEDESTRIAN",
        "BICYCLIST": "BICYCLIST",
        "MOTORCYCLIST": "MOTORCYCLIST",
        "WHEELED_RIDER": "WHEELED_RIDER",
        "REGULAR_VEHICLE": "VEHICLE", "BOX_TRUCK": "VEHICLE",
        "BUS": "VEHICLE", "LARGE_VEHICLE": "VEHICLE",
        "SCHOOL_BUS": "VEHICLE", "ARTICULATED_BUS": "VEHICLE",
        "TRUCK": "VEHICLE", "TRUCK_CAB": "VEHICLE",
        "VEHICULAR_TRAILER": "VEHICLE",
    }

    idxToMeta = {}
    catList = list(AnnotationCategories)
    for i, cat in enumerate(catList):
        idxToMeta[i] = nameToMeta.get(cat.value, "OTHER_FOREGROUND")

    idxToMeta[len(catList)] = "BACKGROUND"
    idxToMeta[255] = "BACKGROUND"

    print("class index → bucket (verify background sentinel):")
    for idx in sorted(idxToMeta.keys()):
        name = catList[idx].value if idx < len(catList) else f"<sentinel {idx}>"
        print(f"  {idx:3d}: {name} → {idxToMeta[idx]}")

    return idxToMeta


def addDerivedCols(df):
    df["sqErr"] = ((df["predFlowX"] - df["gtFlowX"]) ** 2 +
                   (df["predFlowY"] - df["gtFlowY"]) ** 2 +
                   (df["predFlowZ"] - df["gtFlowZ"]) ** 2)
    df["errMag"] = np.sqrt(df["sqErr"])
    df["var"] = df["predSigma"] ** 2


def bucketStats(sub):
    n = len(sub)
    if n == 0:
        return {k: float("nan") for k in ("n", "meanSigma", "rmse", "bnEpe", "bnEpeStatic", "bnEpeDynamic")}
    dynMask = sub["isDynamic"]
    statMask = ~dynMask
    return {
        "n": n,
        "meanSigma": sub["predSigma"].mean(),
        "rmse": np.sqrt(sub["sqErr"].mean()),
        "bnEpe": sub["errMag"].mean(),
        "bnEpeStatic": sub.loc[statMask, "errMag"].mean() if statMask.any() else float("nan"),
        "bnEpeDynamic": sub.loc[dynMask, "errMag"].mean() if dynMask.any() else float("nan"),
    }


def sparsifyCurve(errMag, rankKey, nSteps=200):
    # cumsum approach — O(N log N) not O(N²); critical for large datasets
    order = np.argsort(rankKey)
    sortedErr = errMag[order]
    cumErr = np.cumsum(sortedErr)
    N = len(errMag)
    fracs = np.linspace(0.01, 1.0, nSteps)
    ns = np.maximum(1, np.round(fracs * N).astype(int))
    curve = cumErr[ns - 1] / ns
    return curve, fracs


# ── σ Histogram + Quantiles ──────────────────────────────────────────────────

def sigmaHistogram(dfA, dfB, outDir):
    print("\n── σ Histogram + Quantiles ──")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Predicted σ distribution", color="#ddd", fontsize=13)

    quantRows = []
    pcts = [1, 5, 25, 50, 75, 95, 99]
    for df, col, label in ((dfA, _COL_A, "A"), (dfB, _COL_B, "B")):
        sigma = df["predSigma"].values
        ax.hist(sigma, bins=300, color=col, alpha=0.55, density=True, label=f"Checkpoint {label}")
        qs = np.percentile(sigma, pcts)
        row = {"ckpt": label}
        for p, q in zip(pcts, qs):
            row[f"p{p}"] = float(q)
        quantRows.append(row)
        print(f"  ckpt {label}: " + "  ".join(f"p{p}={q:.4f}" for p, q in zip(pcts, qs)))

    ax.set_xlabel("σ (m)", fontsize=10)
    ax.set_ylabel("Density (log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(outDir / "sigma_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(quantRows).to_csv(outDir / "sigma_quantiles.csv", index=False)
    print("  → sigma_histogram.png, sigma_quantiles.csv")


# ── Analysis 1: Sparsification + AUSE ────────────────────────────────────────

def sparsificationAndAuse(dfA, dfB, outDir):
    print("\n── Analysis 1: Sparsification + AUSE ──")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sparsification Error Curves", color="#ddd", fontsize=13)
    titles = ["All valid", "Dynamic", "Static"]
    auseRows = []

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="#ddd", fontsize=10)
        ax.set_xlabel("Fraction retained", fontsize=9)
        ax.set_ylabel("Mean EPE (m)", fontsize=9)
        ax.grid(True)

        for df, col, label in ((dfA, _COL_A, "A"), (dfB, _COL_B, "B")):
            if title == "Dynamic":
                sub = df[df["isDynamic"]]
            elif title == "Static":
                sub = df[~df["isDynamic"]]
            else:
                sub = df

            if len(sub) == 0:
                continue

            errMag = sub["errMag"].values
            sigma = sub["predSigma"].values

            predCurve, fracs = sparsifyCurve(errMag, sigma)
            oracleCurve, _ = sparsifyCurve(errMag, errMag)
            ause = float(np.trapz(predCurve - oracleCurve, fracs))

            ax.plot(fracs, predCurve, color=col, linewidth=1.8, label=f"{label} (AUSE={ause:.4f})")
            ax.plot(fracs, oracleCurve, color=col, linewidth=1.0, linestyle="--", alpha=0.5)
            auseRows.append({"subset": title, "ckpt": label, "AUSE": ause,
                             "n": len(sub), "meanEpe": float(errMag.mean())})
            print(f"  {title} / ckpt {label}: AUSE={ause:.4f}  n={len(sub):,}")

        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outDir / "sparsification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    auseDf = pd.DataFrame(auseRows)
    auseDf.to_csv(outDir / "ause.csv", index=False)
    print(f"  → sparsification.png, ause.csv")


# ── Analysis 2: Reliability + ENCE ───────────────────────────────────────────

def reliabilityAndEnce(dfA, dfB, outDir):
    print("\n── Analysis 2: Reliability + ENCE ──")
    nBins = 15
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Reliability Diagram  (σ vs RMSE per percentile bin)", color="#ddd", fontsize=13)
    enceRows = []

    for ax, df, col, label in zip(axes, (dfA, dfB), (_COL_A, _COL_B), ("A", "B")):
        sigma = df["predSigma"].values
        sqErr = df["sqErr"].values

        edges = np.quantile(sigma, np.linspace(0, 1, nBins + 1))
        binIds = np.digitize(sigma, edges[1:-1])

        meanSigmas, rmses = [], []
        for b in range(nBins):
            mask = binIds == b
            if not mask.any():
                continue
            meanSigmas.append(sigma[mask].mean())
            rmses.append(np.sqrt(sqErr[mask].mean()))

        meanSigmas = np.array(meanSigmas)
        rmses = np.array(rmses)
        ence = float(np.mean(np.abs(rmses - meanSigmas) / (meanSigmas + 1e-8)))
        enceRows.append({"ckpt": label, "ENCE": ence, "nBins": len(meanSigmas)})
        print(f"  ckpt {label}: ENCE={ence:.4f}")

        diag = np.array([meanSigmas.min(), meanSigmas.max()])
        ax.plot(diag, diag, color="#888", linewidth=1.2, linestyle="--", label="ideal")
        ax.scatter(meanSigmas, rmses, color=col, s=60, zorder=5, label=f"bins  ENCE={ence:.4f}")
        ax.plot(meanSigmas, rmses, color=col, linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Mean predicted σ (m)", fontsize=9)
        ax.set_ylabel("RMSE (m)", fontsize=9)
        ax.set_title(f"Checkpoint {label}", color="#ddd", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(enceRows).to_csv(outDir / "ence.csv", index=False)
    print("  → reliability.png, ence.csv")


# ── Analysis 3: Coverage at α-levels ─────────────────────────────────────────

def coverageAtAlpha(dfA, dfB, outDir):
    print("\n── Analysis 3: Coverage at α-levels ──")
    alphas = [0.50, 0.75, 0.90, 0.95]
    thresholds = [float(spStats.chi2.ppf(a, df=3)) for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Coverage at α-levels  (χ² df=3)", color="#ddd", fontsize=13)
    covRows = []

    for ax, df, col, label in zip(axes, (dfA, dfB), (_COL_A, _COL_B), ("A", "B")):
        sqErr = df["sqErr"].values
        var = df["var"].values + 1e-8

        empirical = []
        for thresh in thresholds:
            emp = float((sqErr / var <= thresh).mean())
            empirical.append(emp)

        for a, emp in zip(alphas, empirical):
            covRows.append({"ckpt": label, "target": a, "empirical": emp})
            print(f"  ckpt {label}: α={a:.2f}  empirical={emp:.4f}")

        ax.plot([0, 1], [0, 1], color="#888", linewidth=1.2, linestyle="--", label="ideal")
        ax.scatter(alphas, empirical, color=col, s=80, zorder=5)
        ax.plot(alphas, empirical, color=col, linewidth=1.8, label=f"checkpoint {label}")
        ax.set_xlabel("Target coverage α", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title(f"Checkpoint {label}", color="#ddd", fontsize=10)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(covRows).to_csv(outDir / "coverage.csv", index=False)
    print("  → coverage.png, coverage.csv")


# ── Temperature scaling ───────────────────────────────────────────────────────

def temperatureScaling(dfA, dfB, outDir):
    print("\n── Temperature Scaling ──")
    alphas = [0.50, 0.75, 0.90, 0.95]
    thresholds = [float(spStats.chi2.ppf(a, df=3)) for a in alphas]
    nBins = 15
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Coverage before / after temperature scaling  (χ² df=3)", color="#ddd", fontsize=13)
    tsRows = []

    for ax, df, col, label in zip(axes, (dfA, dfB), (_COL_A, _COL_B), ("A", "B")):
        sigma = df["predSigma"].values
        sqErr = df["sqErr"].values
        n = len(sigma)

        # 20 % calibration / 80 % test
        calIdx = rng.choice(n, int(0.2 * n), replace=False)
        calMask = np.zeros(n, dtype=bool)
        calMask[calIdx] = True
        testMask = ~calMask

        # Analytic optimum: minimises Gaussian NLL w.r.t. scalar T
        T = float(np.sqrt(np.mean(sqErr[calMask] / (sigma[calMask] ** 2 + 1e-12))))
        sigT = sigma[testMask] * T

        # NLL before / after on test split
        nllBefore = float(np.mean(0.5 * np.log(2 * np.pi * sigma[testMask] ** 2)
                                  + sqErr[testMask] / (2 * sigma[testMask] ** 2 + 1e-12)))
        nllAfter  = float(np.mean(0.5 * np.log(2 * np.pi * sigT ** 2)
                                  + sqErr[testMask] / (2 * sigT ** 2 + 1e-12)))

        # ENCE before / after
        def _ence(sig, sq):
            edges = np.quantile(sig, np.linspace(0, 1, nBins + 1))
            binIds = np.digitize(sig, edges[1:-1])
            ms, rs = [], []
            for b in range(nBins):
                m = binIds == b
                if not m.any():
                    continue
                ms.append(sig[m].mean())
                rs.append(np.sqrt(sq[m].mean()))
            ms, rs = np.array(ms), np.array(rs)
            return float(np.mean(np.abs(rs - ms) / (ms + 1e-8)))

        enceBefore = _ence(sigma[testMask], sqErr[testMask])
        enceAfter  = _ence(sigT, sqErr[testMask])

        # Coverage before / after
        varBefore = sigma[testMask] ** 2 + 1e-8
        varAfter  = sigT ** 2 + 1e-8
        row = {"ckpt": label, "T": T,
               "nllBefore": nllBefore, "nllAfter": nllAfter,
               "enceBefore": enceBefore, "enceAfter": enceAfter}
        empBefore, empAfter = [], []
        for a, thresh in zip(alphas, thresholds):
            eb = float((sqErr[testMask] / varBefore <= thresh).mean())
            ea = float((sqErr[testMask] / varAfter  <= thresh).mean())
            row[f"coverBefore_{a:.2f}"] = eb
            row[f"coverAfter_{a:.2f}"]  = ea
            empBefore.append(eb)
            empAfter.append(ea)
        tsRows.append(row)
        print(f"  ckpt {label}: T={T:.4f}  NLL {nllBefore:.4f}→{nllAfter:.4f}"
              f"  ENCE {enceBefore:.4f}→{enceAfter:.4f}")

        ax.plot([0, 1], [0, 1], color="#888", linewidth=1.2, linestyle="--", label="ideal")
        ax.plot(alphas, empBefore, color=col, linewidth=1.8, linestyle="-",  label="before")
        ax.plot(alphas, empAfter,  color=col, linewidth=1.8, linestyle=":",  label=f"after T={T:.2f}")
        ax.scatter(alphas, empBefore, color=col, s=60, zorder=5)
        ax.scatter(alphas, empAfter,  color=col, s=60, zorder=5, marker="^")
        ax.set_xlabel("Target coverage α", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title(f"Checkpoint {label}", color="#ddd", fontsize=10)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "coverage_temp_scaled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(tsRows).to_csv(outDir / "temperature_scaling.csv", index=False)
    print("  → coverage_temp_scaled.png, temperature_scaling.csv")


# ── Per-bucket calibration ────────────────────────────────────────────────────

def perBucketCalibration(dfA, dfB, outDir, bucketMap):
    print("\n── Per-bucket calibration ──")

    for dfX in (dfA, dfB):
        if "bucket" not in dfX.columns:
            dfX["bucket"] = dfX["classIdx"].map(lambda x: bucketMap.get(int(x), "OTHER_FOREGROUND"))

    buckets = sorted(set(dfA["bucket"].unique()) | set(dfB["bucket"].unique()))
    rows = []

    for label, df in (("A", dfA), ("B", dfB)):
        for bucket in buckets:
            sub = df[df["bucket"] == bucket]
            if len(sub) < 10:
                continue
            sigma  = sub["predSigma"].values
            errMag = sub["errMag"].values
            sqErr  = sub["sqErr"].values
            rmse   = float(np.sqrt(sqErr.mean()))
            meanSig = float(sigma.mean())
            rho, _ = spStats.spearmanr(sigma, errMag)
            rows.append({
                "ckpt": label, "bucket": bucket, "n": len(sub),
                "meanSigma": meanSig,
                "meanErrMag": float(errMag.mean()),
                "rmse": rmse,
                "rmseOverSigma": rmse / (meanSig + 1e-9),
                "sigmaP5":  float(np.percentile(sigma, 5)),
                "sigmaP50": float(np.percentile(sigma, 50)),
                "sigmaP95": float(np.percentile(sigma, 95)),
                "spearmanRho": float(rho),
            })

    calDf = pd.DataFrame(rows)
    calDf.to_csv(outDir / "per_bucket_calibration.csv", index=False)
    print(calDf.to_string(float_format="{:.4f}".format))

    # Bar chart: RMSE/σ per bucket
    calA = calDf[calDf["ckpt"] == "A"].set_index("bucket")
    calB = calDf[calDf["ckpt"] == "B"].set_index("bucket")
    allBuckets = list(dict.fromkeys(r["bucket"] for r in rows))

    fig, ax = plt.subplots(figsize=(max(8, len(allBuckets) * 1.3), 5))
    x = np.arange(len(allBuckets))
    w = 0.35
    valsA = [calA.loc[b, "rmseOverSigma"] if b in calA.index else float("nan") for b in allBuckets]
    valsB = [calB.loc[b, "rmseOverSigma"] if b in calB.index else float("nan") for b in allBuckets]
    ax.bar(x - w / 2, valsA, w, color=_COL_A, alpha=0.8, label="A")
    ax.bar(x + w / 2, valsB, w, color=_COL_B, alpha=0.8, label="B")
    ax.axhline(float(np.nanmean(valsA)), color=_COL_A, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(float(np.nanmean(valsB)), color=_COL_B, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(allBuckets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE / mean σ", fontsize=10)
    ax.set_title("RMSE/σ per bucket  (flat → temperature scaling sufficient)", color="#ddd", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(outDir / "rmse_over_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → per_bucket_calibration.csv, rmse_over_sigma.png")


# ── Analysis 4: Stratified table ─────────────────────────────────────────────

def _buildStrataRows(df, groups, groupCol):
    rows = []
    for label, mask in groups:
        sub = df[mask] if mask is not None else df
        s = bucketStats(sub)
        s[groupCol] = label
        rows.append(s)
    return rows


def stratifiedTable(dfA, dfB, outDir, bucketMap):
    print("\n── Analysis 4: Stratified table ──")

    for dfX, label in ((dfA, "A"), (dfB, "B")):
        dfX["bucket"] = dfX["classIdx"].map(lambda x: bucketMap.get(int(x), "OTHER_FOREGROUND"))

    # discover bucket names dynamically so we match whatever BUCKETED_METACATAGORIES uses
    buckets = sorted(set(dfA["bucket"].unique()) | set(dfB["bucket"].unique()))
    rangeBins = [("0–20m", (0, 20)), ("20–40m", (20, 40)), ("40–70m", (40, 70))]
    q25A, q50A, q75A = np.quantile(dfA["density"], [0.25, 0.5, 0.75])
    q25B, q50B, q75B = np.quantile(dfB["density"], [0.25, 0.5, 0.75])

    axes_def = [
        ("classBuckets", "bucket", [(b, dfA["bucket"] == b) for b in buckets],
         [(b, dfB["bucket"] == b) for b in buckets]),
        ("range", "range", [(lbl, (dfA["rangeMeters"] >= lo) & (dfA["rangeMeters"] < hi)) for lbl, (lo, hi) in rangeBins],
         [(lbl, (dfB["rangeMeters"] >= lo) & (dfB["rangeMeters"] < hi)) for lbl, (lo, hi) in rangeBins]),
        ("dynamic", "dynamic", [("dynamic", dfA["isDynamic"]), ("static", ~dfA["isDynamic"])],
         [("dynamic", dfB["isDynamic"]), ("static", ~dfB["isDynamic"])]),
        ("density", "density",
         [("Q1", dfA["density"] <= q25A), ("Q2", (dfA["density"] > q25A) & (dfA["density"] <= q50A)),
          ("Q3", (dfA["density"] > q50A) & (dfA["density"] <= q75A)), ("Q4", dfA["density"] > q75A)],
         [("Q1", dfB["density"] <= q25B), ("Q2", (dfB["density"] > q25B) & (dfB["density"] <= q50B)),
          ("Q3", (dfB["density"] > q50B) & (dfB["density"] <= q75B)), ("Q4", dfB["density"] > q75B)]),
    ]

    metricCols = ["meanSigma", "rmse", "bnEpe", "bnEpeStatic", "bnEpeDynamic"]

    for axisName, groupCol, groupsA, groupsB in axes_def:
        rowsA = _buildStrataRows(dfA, groupsA, groupCol)
        rowsB = _buildStrataRows(dfB, groupsB, groupCol)
        tbA = pd.DataFrame(rowsA).set_index(groupCol)
        tbB = pd.DataFrame(rowsB).set_index(groupCol)

        merged = pd.concat(
            [tbA[metricCols].add_suffix("_A"), tbB[metricCols].add_suffix("_B")],
            axis=1
        )
        outFile = outDir / f"stratified_{axisName}.csv"
        merged.to_csv(outFile)
        print(f"\n  [{axisName}]")
        print(merged.to_string(float_format="{:.4f}".format))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumpA", type=Path, required=True)
    parser.add_argument("--dumpB", type=Path, required=True)
    parser.add_argument("--outDir", type=Path,
                        default=Path.home() / "persistent" / "djrhee" / "Lidar-Flow" / "runs" / "evaluation")
    args = parser.parse_args()

    args.outDir.mkdir(parents=True, exist_ok=True)

    print("loading dumps...", flush=True)
    dfA = loadSampled(args.dumpA)
    dfB = loadSampled(args.dumpB)
    print(f"  A: {len(dfA):,} points   B: {len(dfB):,} points")

    addDerivedCols(dfA)
    addDerivedCols(dfB)

    bucketMap = buildBucketMap()

    sigmaHistogram(dfA, dfB, args.outDir)
    sparsificationAndAuse(dfA, dfB, args.outDir)
    reliabilityAndEnce(dfA, dfB, args.outDir)
    coverageAtAlpha(dfA, dfB, args.outDir)
    temperatureScaling(dfA, dfB, args.outDir)
    perBucketCalibration(dfA, dfB, args.outDir, bucketMap)
    stratifiedTable(dfA, dfB, args.outDir, bucketMap)

    print(f"\ndone. outputs at {args.outDir}", flush=True)


if __name__ == "__main__":
    main()
