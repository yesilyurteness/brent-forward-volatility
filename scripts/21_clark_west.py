"""Clark-West test, HAR nested in HAR-X, four horizons (supplementary family, 4 tests).

WHY
---
HAR's regressors are a subset of HAR-X's, so the models are nested. Under the null that
the extra regressors (OVX, GPRD, GPRD_THREAT) have zero coefficients, the DM statistic
for nested models is not centred at zero: estimating the extra parameters adds noise to
the larger model's forecasts, which biases the MSE difference against it (Clark & West,
2007). The Clark-West adjustment removes that noise term:

    f_t = e_HAR,t^2 - [ e_HARX,t^2 - (yhat_HAR,t - yhat_HARX,t)^2 ]

H0: E[f] = 0; H1: E[f] > 0 (HAR-X better). ONE-SIDED.

FAMILY RULE (CLAUDE.md, "Reporting and Provenance Rules")
--------------------------------------------------------
The primary family stays fixed at 8 tests; CW is NOT added to it and does NOT replace the
DM tests. CW is a separate supplementary family of 4 tests (one per horizon) with its own
Holm / BH / BY corrections. Label: "statistic suited to the nested structure; added after
the primary DM tests were seen, before any CW result was seen". The family was declared
in CLAUDE.md in commit 72106e7 (2026-09-29 16:15:34 +0300); no CW statistic existed in
the repository before this script.

SETUP -- identical to 08_dm_test.py
-----------------------------------
  * saved (issued, floored) forecasts from hybrid_predictions_all, main folds only (the
    2026 fold at h=66/126 is excluded, as in DM);
  * pooled series across folds, sorted by date;
  * HAC long-run variance: Newey-West, Bartlett kernel, L = h - 1 (08.newey_west_lrv);
  * Harvey-Leybourne-Newbold small-sample factor and t(n-1), as the primary DM family;
    the corrections are applied to this HLN p-value (as in 08). The uncorrected-HLN
    normal p-value is reported alongside.
A fold-level mean of f_t is given as a descriptive number only (no test).

Outputs: outputs/clark_west{_publication_aligned}.csv, .json
Runtime: a few seconds.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import alignment

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [5, 22, 66, 126]
DECLARED = {"commit": "72106e7", "date": "2026-09-29 16:15:34 +0300",
            "where": "CLAUDE.md, Reporting and Provenance Rules"}
LABEL = ("iç içe yapıya uygun istatistik; birincil DM testleri görüldükten sonra, CW "
         "sonuçları görülmeden eklendi")

_spec = importlib.util.spec_from_file_location("dm", ROOT / "scripts" / "08_dm_test.py")
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


def by_adjust(p):
    """Benjamini-Yekutieli: BH x c(m), c(m) = sum_{i<=m} 1/i; valid under any dependence."""
    m = len(p)
    return np.minimum(dm.benjamini_hochberg(p) * sum(1 / i for i in range(1, m + 1)), 1.0)


def clark_west(df, horizons=HORIZONS):
    """CW test per horizon on a prediction frame with y_true, pred_har, pred_har_x,
    test_year, include_in_main, Date; main folds only, pooled and sorted by date.
    Holm/BH/BY are applied across the rows returned (the caller's family). Also used by
    25_rollover_clark_west.py."""
    df = df[df["include_in_main"]].copy()
    df["dt"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df = df.sort_values(["horizon", "dt"]).reset_index(drop=True)
    rows = []
    for h in horizons:
        g = df[df["horizon"] == h]
        n, lag = len(g), h - 1
        y = g["y_true"].to_numpy("float64")
        p_har = g["pred_har"].to_numpy("float64")
        p_harx = g["pred_har_x"].to_numpy("float64")
        e_har, e_harx = y - p_har, y - p_harx
        adj = (p_har - p_harx) ** 2
        f = e_har ** 2 - (e_harx ** 2 - adj)
        fbar = float(f.mean())
        omega, g0 = dm.newey_west_lrv(f, lag)
        cw = fbar / np.sqrt(max(omega, 1e-300) / n)
        hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        cw_hln = cw * hln
        fold = pd.Series(f).groupby(g["test_year"].to_numpy()).mean()
        rows.append({
            "horizon": h, "n": n, "lag": lag, "n_folds": int(len(fold)),
            "mse_har": float(np.mean(e_har ** 2)), "mse_har_x": float(np.mean(e_harx ** 2)),
            "mean_mse_diff": float(np.mean(e_har ** 2 - e_harx ** 2)),
            "mean_adjustment": float(adj.mean()), "f_mean": fbar,
            "CW": float(cw), "p_normal_one_sided": float(1 - stats.norm.cdf(cw)),
            "hln_factor": float(hln), "CW_HLN": float(cw_hln),
            "p_HLN_one_sided": float(1 - stats.t.cdf(cw_hln, df=n - 1)),
            "variance_inflation": float(omega / g0),
            "f_fold_mean": float(fold.mean()),
            "folds_f_positive": int((fold > 0).sum())})
    res = pd.DataFrame(rows)
    p = res["p_HLN_one_sided"].to_numpy()
    res["p_HLN_holm"] = dm.holm(p)
    res["p_HLN_bh"] = dm.benjamini_hochberg(p)
    res["p_HLN_by"] = by_adjust(p)
    return res


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment

    df = pd.read_csv(alignment.out("hybrid_predictions_all.csv", al),
                     float_precision="round_trip")
    res = clark_west(df)
    res["family"] = "ek: Clark-West (4 test)"

    out = alignment.out("clark_west.csv", al)
    res.to_csv(out, index=False)
    summary = {"gpr_alignment": al, "family": "supplementary, 4 tests (HAR nested in HAR-X)",
               "label": LABEL, "declared": DECLARED,
               "test": "one-sided, H1: HAR-X better; f_t = e_HAR^2 - (e_HARX^2 - "
                       "(yhat_HAR - yhat_HARX)^2)",
               "hac": "Newey-West, Bartlett, L = h-1 (same as 08)",
               "inference": "HLN factor, t(n-1); Holm/BH/BY applied to p_HLN_one_sided",
               "forecasts": "saved issued (floored) forecasts, hybrid_predictions_all, "
                            "main folds, pooled",
               "results": res.to_dict(orient="records")}
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250)
    print(res[["horizon", "n", "f_mean", "CW", "CW_HLN", "p_HLN_one_sided", "p_HLN_holm",
               "p_HLN_bh", "p_HLN_by", "folds_f_positive", "n_folds"]].to_string(index=False))
    print(f"Written: {out.name}, {out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
