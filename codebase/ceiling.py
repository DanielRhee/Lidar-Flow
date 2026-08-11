"""What can ANY function of the head's 41 frozen inputs achieve on FG_DYNAMIC?

The trained sigma head is useless-to-harmful there by three independent measures. That
does not distinguish two very different diagnoses:

  ROC-AUC ~ 0.55  the features carry no signal. No head architecture, loss, or output
                  parameterisation fixes it; the answer is different inputs, and any
                  retrain of the current design is misdirected.
  ROC-AUC ~ 0.75  the features are fine and the OBJECTIVE was the problem. A failure
                  classifier on the same inputs is worth building.

Gradient boosting is used as a nonparametric stand-in for "any function": it is not the
best possible model, so it gives a *lower* bound on the achievable AUC -- which is the
useful direction, because a high value settles the question and a low value is the
weaker claim.

Runs in the isolated `analysis` env (numpy/pandas/pyarrow/sklearn only): no torch, no
spconv, no av2. Feature column layout, from model.uncertaintyInput:
  f0-f31 dec0 feature | f32-f34 rel-xyz | f35 ||xy||/70 | f36 z/3
  f37-f39 detached predicted flow | f40 ||predFlow||
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FG_DYNAMIC = 2          # classes.STRATA index; hard-coded so this env needs no av2
N_FEAT = 41


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outDir", type=Path, required=True)
    parser.add_argument("--tau", type=float, nargs="+", default=[0.10, 0.50])
    parser.add_argument("--maxPoints", type=int, default=1_200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5,
                        help="random log splits. One split is not enough: per-log failure "
                             "rate spans 0.0029-0.5649, so a single 50/50 draw gave halves "
                             "differing 3.1x at tau=0.10 and 30x at tau=0.50.")
    args = parser.parse_args()
    args.outDir.mkdir(parents=True, exist_ok=True)

    cols = [f"f{i}" for i in range(N_FEAT)] + ["errMag", "predSigma", "stratum", "logIdx"]
    with ipc.open_file(str(args.features)) as r:
        tbl = r.read_all().select(cols)
    stratum = tbl.column("stratum").to_numpy()
    keep = stratum == FG_DYNAMIC
    X = np.column_stack([tbl.column(f"f{i}").to_numpy()[keep] for i in range(N_FEAT)])
    err = tbl.column("errMag").to_numpy()[keep]
    sigma = tbl.column("predSigma").to_numpy()[keep]
    log = tbl.column("logIdx").to_numpy()[keep]
    print(f"FG_DYNAMIC: {len(err):,} points over {len(np.unique(log))} logs")

    rng = np.random.default_rng(args.seed)
    if len(err) > args.maxPoints:
        sel = rng.choice(len(err), args.maxPoints, replace=False)
        X, err, sigma, log = X[sel], err[sel], sigma[sel], log[sel]
        print(f"  subsampled to {len(err):,}")

    # Split by LOG. A point-level split would leak: points in one sweep are near-copies,
    # so the booster would memorise scenes and report a meaningless ceiling.
    logs = np.unique(log)

    def fitScore(tr, ev, tau, seed):
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        clf.fit(X[tr], err[tr] > tau)
        return roc_auc_score(err[ev] > tau, clf.predict_proba(X[ev])[:, 1])

    # Log-clustered bootstrap: points within a log are not independent, and the top 10%
    # of logs carry >50% of the error mass, so a point-level CI is meaningless here.
    def clusteredCi(y, score, lg, draws=200, seed=0):
        r = np.random.default_rng(seed)
        uniq = np.unique(lg)
        byLog = {l: np.flatnonzero(lg == l) for l in uniq}
        out = []
        for _ in range(draws):
            idx = np.concatenate([byLog[l] for l in r.choice(uniq, len(uniq), replace=True)])
            if y[idx].sum() < 10 or (~y[idx]).sum() < 10:
                continue
            out.append(roc_auc_score(y[idx], score[idx]))
        return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)

    for tau in args.tau:
        print(f"\n{'='*78}\n  tau={tau:.2f}   overall base rate {(err > tau).mean():.4f}\n{'='*78}")
        print(f"  {'split':>5s} {'dir':>5s} {'fit base':>9s} {'score base':>10s} "
              f"{'CEILING':>9s} {'sigma':>8s} {'||pred||':>9s}")
        ceilings = []
        for rep in range(args.repeats):
            r = np.random.default_rng(args.seed + rep)
            sh = logs.copy()
            r.shuffle(sh)
            halfA = set(sh[: len(sh) // 2].tolist())
            a = np.isin(log, list(halfA))
            # BOTH directions: the original single split trained on the hard half and
            # scored on the easy one, which biases the bound DOWNWARD.
            for tag, tr, ev in (("A->B", a, ~a), ("B->A", ~a, a)):
                if (err[tr] > tau).sum() < 50 or (err[ev] > tau).sum() < 50:
                    print(f"  {rep:>5d} {tag:>5s}   too few failures, skipped")
                    continue
                c = fitScore(tr, ev, tau, args.seed)
                ceilings.append(c)
                print(f"  {rep:>5d} {tag:>5s} {(err[tr] > tau).mean():9.4f} "
                      f"{(err[ev] > tau).mean():10.4f} {c:9.4f} "
                      f"{roc_auc_score(err[ev] > tau, sigma[ev]):8.4f} "
                      f"{roc_auc_score(err[ev] > tau, X[ev, 40]):9.4f}")
        if ceilings:
            print(f"  ceiling over {len(ceilings)} fits: mean {np.mean(ceilings):.4f}  "
                  f"min {np.min(ceilings):.4f}  max {np.max(ceilings):.4f}  sd {np.std(ceilings):.4f}")
            lo, hi = clusteredCi(err > tau, sigma, log, seed=args.seed)
            print(f"  sigma head, log-clustered 95% CI over ALL points: [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
