import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from scipy import stats as spStats
from scipy.special import erf

from classes import BACKGROUND_MAX_GT, STRATA, className, metaName, stratumOf
from paths import DEFAULT_RUNS_DIR

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
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "black",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "text.color": "black",
    "grid.color": "#ccc",
    "grid.linestyle": "--",
    "legend.facecolor": "white",
    "legend.edgecolor": "#888",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

_COL_A = "#1f77b4"
_COL_B = "#d62728"

# Dimensionality of the flow error. sqErr sums three components, so the isotropic
# Gaussian log-determinant is 3*log(sigma^2) and the optimal temperature is
# sqrt(mean(sqErr/var)/3). Both facts were previously wrong in a mutually consistent
# way, which is why neither showed up.
_NLL_DIM = 3.0


def stratifiedHeadline(dfA, dfB, outDir):
    """Per-stratum scores. No all-points aggregate: it is 80.3% BACKGROUND, whose
    target is identically zero, so it measures the residual of a deterministic map
    rather than calibrated uncertainty."""
    print("\n── Stratified scores (no aggregate: 80.3% of points are BACKGROUND) ──", flush=True)
    rows = []
    for label, df in (("A", dfA), ("B", dfB)):
        for stratum in STRATA:
            sub = df[df["stratum"] == stratum]
            if len(sub) == 0:
                continue
            rows.append({"ckpt": label, "stratum": stratum, "n": len(sub),
                         "epe": sub["errMag"].mean(), "rmse": np.sqrt(sub["sqErr"].mean()),
                         "meanSigma": sub["predSigma"].mean(), "crps": sub["crps"].mean(),
                         "energyScore": sub["energyScore"].mean()})
            r = rows[-1]
            print(f"  {label} {stratum:11s} n={r['n']:>10,}  EPE={r['epe']:.5f}  RMSE={r['rmse']:.5f}"
                  f"  sigma={r['meanSigma']:.5f}  CRPS={r['crps']:.6f}  ES={r['energyScore']:.6f}")
    pd.DataFrame(rows).to_csv(outDir / "stratified_scores.csv", index=False)

    # Paired over logs: marginal CIs are far too wide to separate close rankings.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    print("\n  paired bootstrap over logs, AUSE(sigma) - AUSE(|predFlow|); negative = sigma wins",
          flush=True)
    # Per stratum only. The all-points aggregate is 80% BACKGROUND, which is exactly
    # the number this whole section exists to stop quoting.
    for stratum in STRATA:
        sub = dfA[dfA["stratum"] == stratum]
        if len(sub) < 1000:
            continue
        err = sub["errMag"].values
        sigCol = sub["predSigma"].values
        pnCol = sub["predNorm"].values

        def auseDiff(idx, err=err, sigCol=sigCol, pnCol=pnCol):
            e = err[idx]
            oc, fr = sparsifyCurve(e, e)
            return (trapz(sparsifyCurve(e, sigCol[idx])[0] - oc, fr),
                    trapz(sparsifyCurve(e, pnCol[idx])[0] - oc, fr))

        a, b = auseDiff(np.arange(len(err)))
        lo, hi = pairedBootstrap(sub["logIdx"].values, auseDiff)
        verdict = "SIGNIFICANT" if hi < 0 or lo > 0 else "not distinguishable"
        print(f"    {stratum:11s} sigma={a:.6f} predFlow={b:.6f}  diff={a-b:+.6f} "
              f"[{lo:+.6f}, {hi:+.6f}]  {verdict}", flush=True)
    print("  → stratified_scores.csv")


def rawClassDistribution(dfA, dfB, outDir):
    print("\n── Raw classIdx distribution ──")
    rows = []
    for label, df in (("A", dfA), ("B", dfB)):
        vc = df["classIdx"].value_counts(sort=False).sort_index()
        total = len(df)
        for idx, n in vc.items():
            print(f"  [{label}] idx {int(idx):3d} {className(idx):<32}: {n:>12,}  ({100*n/total:.3f}%)")
            rows.append({"ckpt": label, "classIdx": int(idx), "className": className(idx),
                         "meta": metaName(idx), "n": int(n), "pct": round(100*n/total, 4)})
    pd.DataFrame(rows).to_csv(outDir / "class_index_distribution.csv", index=False)
    print("  → class_index_distribution.csv")


def addDerivedCols(df):
    df["sqErr"] = ((df["predFlowX"] - df["gtFlowX"]) ** 2 +
                   (df["predFlowY"] - df["gtFlowY"]) ** 2 +
                   (df["predFlowZ"] - df["gtFlowZ"]) ** 2)
    df["errMag"] = np.sqrt(df["sqErr"])
    df["var"] = df["predSigma"] ** 2
    # Predicted flow magnitude: a free ranking baseline that sigma has to beat to be
    # worth its parameters, and the feature sigma was found to have collapsed onto.
    df["predNorm"] = np.sqrt(df["predFlowX"] ** 2 + df["predFlowY"] ** 2 + df["predFlowZ"] ** 2)
    # CRPS per component, averaged. Proper and bounded-influence, so unlike mean NLL
    # (whose 1/var term lets a handful of overconfident points own the average) it is
    # safe to quote as a headline and to select on.
    sig = np.maximum(df["predSigma"].values, 1e-12)
    crps = np.zeros(len(df))
    for axis in "XYZ":
        z = (df[f"predFlow{axis}"].values - df[f"gtFlow{axis}"].values) / sig
        crps += sig * (z * (2 * spStats.norm.cdf(z) - 1) + 2 * spStats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    df["crps"] = crps / 3.0
    df["energyScore"] = energyScoreGaussian(
        df["predSigma"].values,
        np.stack([df[f"predFlow{a}"].values - df[f"gtFlow{a}"].values for a in "XYZ"], axis=1))
    df["stratum"] = [STRATA[c] for c in stratumOf(df["classIdx"].values, df["isDynamic"].values)]
    # The zero-target fact is load-bearing for every stratified conclusion, so assert
    # it rather than trusting the note in classes.py.
    bg = df["stratum"] == "BACKGROUND"
    if bg.any():
        gtMag = np.sqrt(df.loc[bg, "gtFlowX"] ** 2 + df.loc[bg, "gtFlowY"] ** 2
                        + df.loc[bg, "gtFlowZ"] ** 2).max()
        assert gtMag < BACKGROUND_MAX_GT, (
            f"background |gt| reached {gtMag:.3e} m, above the {BACKGROUND_MAX_GT:.0e} fp16-noise "
            "bound: the ego0-frame convention that makes background targets identically zero "
            "may have regressed (see memory.md 4)")


def energyScoreGaussian(sigma, residual):
    """Energy score of N(0, sigma^2 I3) against an observed 3-vector residual.

    ES(F,y) = E||Y - y|| - 0.5*E||Y - Y'||, the proper multivariate generalisation of
    CRPS. Averaging three per-component CRPS values (as `crps` does) is also proper
    but blind to cross-component dependence, which matters here because the error
    covariance is strongly anisotropic (35x within fg-dynamic, 437x on background).

    Both terms are closed form for an isotropic Gaussian. With lambda = ||y||/sigma,
    ||Y-y||/sigma is noncentral chi with 3 dof, giving
        E||Y-y|| = sigma * [ (lambda + 1/lambda) erf(lambda/sqrt2)
                             + sqrt(2/pi) exp(-lambda^2/2) ]
    and E||Y-Y'|| = sqrt(2)*sigma*E||Z|| = 4 sigma/sqrt(pi), so the second term is
    2*sigma/sqrt(pi). As lambda -> 0 the first term -> 2 sigma sqrt(2/pi) = E||Y||.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    r = np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=1)
    lam = np.maximum(r / sigma, 1e-12)          # the 1/lambda term is removable, not singular
    first = sigma * ((lam + 1.0 / lam) * erf(lam / np.sqrt(2.0))
                     + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * lam ** 2))
    return first - 2.0 * sigma / np.sqrt(np.pi)


def pairedBootstrap(logIdx, statFn, nBoot=200, seed=0):
    """Bootstrap the DIFFERENCE between two statistics on the same resampled logs.

    Marginal CIs are far too wide to separate close rankings -- sigma's AUSE CI
    contains ||predFlow||'s point estimate -- because they carry the between-log
    variance of the metric itself. Resampling logs once and computing both statistics
    on that same resample cancels it.

    statFn takes an integer index array and returns (a, b). Index-based rather than
    DataFrame-based on purpose: every draw sorts its resample, so taking a .iloc copy
    of a 17M-row frame per iteration turns this from seconds into hours.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(logIdx)
    byLog = {l: np.flatnonzero(logIdx == l) for l in uniq}
    diffs = []
    for _ in range(nBoot):
        idx = np.concatenate([byLog[l] for l in rng.choice(uniq, len(uniq), replace=True)])
        a, b = statFn(idx)
        diffs.append(a - b)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bucketStats(sub):
    n = len(sub)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "meanSigma", "rmse", "crps", "energyScore", "bnEpe",
                 "bnEpeStatic", "bnEpeDynamic")}
    dynMask = sub["isDynamic"]
    statMask = ~dynMask
    return {
        "n": n,
        "meanSigma": sub["predSigma"].mean(),
        "rmse": np.sqrt(sub["sqErr"].mean()),
        "crps": sub["crps"].mean(),
        "energyScore": sub["energyScore"].mean(),
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
    fig.suptitle("Predicted σ distribution", fontsize=13)

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
    fig.savefig(outDir / "sigma_histogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(quantRows).to_csv(outDir / "sigma_quantiles.csv", index=False)
    print("  → sigma_histogram.png, sigma_quantiles.csv")


# ── Analysis 1: Sparsification + AUSE ────────────────────────────────────────

def sparsificationAndAuse(dfA, dfB, outDir):
    print("\n── Analysis 1: Sparsification + AUSE ──  (vs random and |predFlow| baselines)")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Sparsification Error Curves by stratum", fontsize=13)
    # Strata, not the old All/Dynamic/Static. "All" is 80.3% BACKGROUND, whose target
    # is identically zero, so |err| == ||pred|| there and any ||predFlow||-based
    # ranking is tautologically near-perfect. The old "Static" bucket was mostly
    # background and inherited the same identity. FG_DYNAMIC is the informative one.
    titles = list(STRATA) + ["All valid (80% background — not interpretable)"]
    auseRows = []
    rng = np.random.default_rng(0)
    # np.trapz was removed in numpy 2.0; np.trapezoid is the replacement.
    trapz = getattr(np, "trapezoid", None) or np.trapz

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Fraction retained", fontsize=9)
        ax.set_ylabel("Mean EPE (m)", fontsize=9)
        ax.grid(True)

        for df, col, label in ((dfA, _COL_A, "A"), (dfB, _COL_B, "B")):
            sub = df if title.startswith("All valid") else df[df["stratum"] == title]

            if len(sub) == 0:
                continue

            errMag = sub["errMag"].values
            oracleCurve, fracs = sparsifyCurve(errMag, errMag)

            # A constant sigma has no ordering, so argsort falls back to file order --
            # which is spatially correlated and scores better than chance. Quoting a
            # constant-sigma AUSE as the baseline therefore flatters nothing and
            # measures nothing; the honest floor is an explicit random key.
            keys = {"sigma": sub["predSigma"].values,
                    "predNorm": sub["predNorm"].values,
                    "random": rng.random(len(sub))}
            for keyName, key in keys.items():
                predCurve, _ = sparsifyCurve(errMag, key)
                ause = float(trapz(predCurve - oracleCurve, fracs))
                auseRows.append({"subset": title, "ckpt": label, "rankKey": keyName,
                                 "AUSE": ause, "n": len(sub), "meanEpe": float(errMag.mean())})
                if keyName == "sigma":
                    ax.plot(fracs, predCurve, color=col, linewidth=1.8,
                            label=f"{label} sigma (AUSE={ause:.4f})")
                elif label == "A":
                    style = ":" if keyName == "predNorm" else "-."
                    ax.plot(fracs, predCurve, color="0.45", linewidth=1.2, linestyle=style,
                            label=f"{keyName} (AUSE={ause:.4f})")
                print(f"  {title} / ckpt {label} / {keyName}: AUSE={ause:.4f}  n={len(sub):,}")
            ax.plot(fracs, oracleCurve, color=col, linewidth=1.0, linestyle="--", alpha=0.5)

        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(outDir / "sparsification.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    auseDf = pd.DataFrame(auseRows)
    auseDf.to_csv(outDir / "ause.csv", index=False)
    print(f"  → sparsification.png, ause.csv")


# ── Analysis 2: Reliability + ENCE ───────────────────────────────────────────

def reliabilityAndEnce(dfA, dfB, outDir):
    print("\n── Analysis 2: Reliability + ENCE ──")
    nBins = 15
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Reliability Diagram  (σ vs per-dim RMS error per percentile bin)", fontsize=13)
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
            # Per-DIMENSION RMS, i.e. sqrt(mean(sqErr)/3). Comparing the 3-component
            # sqrt(mean(sqErr)) directly against sigma is the same sqrt(3) error as the
            # temperature formula: a perfectly calibrated model has rmse = sqrt(3)*sigma,
            # so that version of ENCE floors at sqrt(3)-1 = 0.732 and can never reach 0.
            rmses.append(np.sqrt(sqErr[mask].mean() / _NLL_DIM))

        meanSigmas = np.array(meanSigmas)
        rmses = np.array(rmses)
        ence = float(np.mean(np.abs(rmses - meanSigmas) / (meanSigmas + 1e-8)))
        enceRows.append({"ckpt": label, "ENCE": ence, "nBins": len(meanSigmas)})
        print(f"  ckpt {label}: ENCE={ence:.4f}")

        diag = np.array([meanSigmas.min(), meanSigmas.max()])
        ax.plot(diag, diag, color="0.4", linewidth=1.2, linestyle="--", label="ideal")
        ax.scatter(meanSigmas, rmses, color=col, s=60, zorder=5, label=f"bins  ENCE={ence:.4f}")
        ax.plot(meanSigmas, rmses, color=col, linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Mean predicted σ (m)", fontsize=9)
        ax.set_ylabel("per-dim RMS error (m)", fontsize=9)
        ax.set_title(f"Checkpoint {label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "reliability.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(enceRows).to_csv(outDir / "ence.csv", index=False)
    print("  → reliability.png, ence.csv")


# ── Analysis 3: Coverage at α-levels ─────────────────────────────────────────

def coverageAtAlpha(dfA, dfB, outDir):
    print("\n── Analysis 3: Coverage at α-levels ──")
    alphas = [0.50, 0.75, 0.90, 0.95]
    thresholds = [float(spStats.chi2.ppf(a, df=3)) for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Coverage at α-levels  (χ² df=3)", fontsize=13)
    covRows = []

    for ax, df, col, label in zip(axes, (dfA, dfB), (_COL_A, _COL_B), ("A", "B")):
        # Per stratum: an all-points coverage number is 80% background, whose target
        # is identically zero, so it says almost nothing about the other 20%.
        for stratum in list(STRATA) + ["ALL"]:
            sub = df if stratum == "ALL" else df[df["stratum"] == stratum]
            if len(sub) == 0:
                continue
            ratio = sub["sqErr"].values / (sub["var"].values + 1e-8)
            emps = [float((ratio <= t).mean()) for t in thresholds]
            for a, emp in zip(alphas, emps):
                covRows.append({"ckpt": label, "stratum": stratum, "target": a, "empirical": emp})
            print(f"  ckpt {label} {stratum:11s} n={len(sub):>10,}  " +
                  "  ".join(f"a={a:.2f}:{e:.4f}" for a, e in zip(alphas, emps)))
            if stratum == "FG_DYNAMIC":
                empirical = emps

        ax.plot([0, 1], [0, 1], color="0.4", linewidth=1.2, linestyle="--", label="ideal")
        ax.scatter(alphas, empirical, color=col, s=80, zorder=5)
        ax.plot(alphas, empirical, color=col, linewidth=1.8, label=f"checkpoint {label}")
        ax.set_xlabel("Target coverage α", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title(f"Checkpoint {label}", fontsize=10)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "coverage.png", dpi=300, bbox_inches="tight", facecolor="white")
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
    fig.suptitle("Coverage before / after temperature scaling  (χ² df=3)", fontsize=13)
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

        # Analytic optimum for an isotropic 3-D Gaussian: T^2 = mean(sqErr/var)/3.
        # The /3 is mandatory -- sqErr sums three components, so without it T comes out
        # sqrt(3) too large and the ideal sits at sqrt(3) instead of 1.
        T = float(np.sqrt(np.mean(sqErr[calMask] / (sigma[calMask] ** 2 + 1e-12)) / _NLL_DIM))
        sigT = sigma[testMask] * T

        # NLL before / after on test split. The log-determinant coefficient is 1.5,
        # not 0.5: this is a 3-D isotropic Gaussian (log|Sigma| = 3 log sigma^2), and a
        # 0.5 here is a 1-D likelihood applied to a 3-D squared error. With 0.5 the
        # minimiser of this expression is sqrt(mean(sqErr/var)), which is exactly the
        # sqrt(3)-inflated T that used to be reported -- the two bugs were consistent
        # with each other and so hid one another.
        nllBefore = float(np.mean(1.5 * np.log(2 * np.pi * sigma[testMask] ** 2)
                                  + sqErr[testMask] / (2 * sigma[testMask] ** 2 + 1e-12)))
        nllAfter  = float(np.mean(1.5 * np.log(2 * np.pi * sigT ** 2)
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
                rs.append(np.sqrt(sq[m].mean() / _NLL_DIM))   # per-dim; see reliabilityAndEnce
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

        ax.plot([0, 1], [0, 1], color="0.4", linewidth=1.2, linestyle="--", label="ideal")
        ax.plot(alphas, empBefore, color=col, linewidth=1.8, linestyle="-",  label="before")
        ax.plot(alphas, empAfter,  color=col, linewidth=1.8, linestyle=":",  label=f"after T={T:.2f}")
        ax.scatter(alphas, empBefore, color=col, s=60, zorder=5)
        ax.scatter(alphas, empAfter,  color=col, s=60, zorder=5, marker="^")
        ax.set_xlabel("Target coverage α", fontsize=9)
        ax.set_ylabel("Empirical coverage", fontsize=9)
        ax.set_title(f"Checkpoint {label}", fontsize=10)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outDir / "coverage_temp_scaled.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(tsRows).to_csv(outDir / "temperature_scaling.csv", index=False)
    print("  → coverage_temp_scaled.png, temperature_scaling.csv")


# ── Per-bucket calibration ────────────────────────────────────────────────────

def perBucketCalibration(dfA, dfB, outDir):
    # rmseOverSigma's ideal value is sqrt(3) ~= 1.732, NOT 1.0: rmse is
    # sqrt(mean(sqErr)) over three summed components, so a perfectly calibrated
    # isotropic Gaussian gives rmse = sqrt(3)*sigma. The column's purpose is
    # flatness across buckets (flat => one global temperature suffices), but reading
    # 1.0 as the target understates the scale error by sqrt(3).
    print("\n── Per-bucket calibration ──  (rmseOverSigma: ideal = sqrt(3) = 1.732)")

    for dfX in (dfA, dfB):
        if "bucket" not in dfX.columns:
            dfX["bucket"] = dfX["classIdx"].map(metaName)

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
    ax.set_title("RMSE/σ per bucket  (flat → temperature scaling sufficient)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(outDir / "rmse_over_sigma.png", dpi=300, bbox_inches="tight", facecolor="white")
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


def stratifiedTable(dfA, dfB, outDir):
    print("\n── Analysis 4: Stratified table ──")

    for dfX, label in ((dfA, "A"), (dfB, "B")):
        dfX["bucket"] = dfX["classIdx"].map(metaName)

    # discover bucket names dynamically so we match whatever av2's meta-groups use
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

    metricCols = ["meanSigma", "rmse", "crps", "energyScore", "bnEpe", "bnEpeStatic", "bnEpeDynamic"]

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
    parser.add_argument("--outDir", type=Path, default=DEFAULT_RUNS_DIR / "evaluation")
    args = parser.parse_args()

    args.outDir.mkdir(parents=True, exist_ok=True)

    print("loading dumps...", flush=True)
    dfA = loadSampled(args.dumpA)
    dfB = loadSampled(args.dumpB)
    print(f"  A: {len(dfA):,} points   B: {len(dfB):,} points")

    addDerivedCols(dfA)
    addDerivedCols(dfB)

    stratifiedHeadline(dfA, dfB, args.outDir)

    rawClassDistribution(dfA, dfB, args.outDir)

    sigmaHistogram(dfA, dfB, args.outDir)
    sparsificationAndAuse(dfA, dfB, args.outDir)
    reliabilityAndEnce(dfA, dfB, args.outDir)
    coverageAtAlpha(dfA, dfB, args.outDir)
    temperatureScaling(dfA, dfB, args.outDir)
    perBucketCalibration(dfA, dfB, args.outDir)
    stratifiedTable(dfA, dfB, args.outDir)

    print(f"\ndone. outputs at {args.outDir}", flush=True)


if __name__ == "__main__":
    main()
