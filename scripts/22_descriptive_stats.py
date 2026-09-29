"""Table 1: descriptive statistics of the model sample (no model, no fit()).

VARIABLES
---------
  * Brent daily log return, log(P_t / P_{t-1});
  * the four targets target_vol_h (h = 5, 22, 66, 126), from outputs/targets.csv;
  * OVX, level;
  * GPRD and GPRD_THREAT AS THE MODEL SEES THEM: gprd_lag1 / gprd_threat_lag1 of the
    feature file of the chosen alignment. With --gpr-alignment publication this is the
    latest observation published by t-1 (as-of alignment), so the series stays constant
    between releases.

SAMPLE: the model's sample, 4641 rows on the trading calendar (02.01.2008 - 01.09.2026).
Returns lose the first row; target_vol_h is undefined for the last h rows (skipna=False);
the publication-aligned GPR is undefined until the first release.

STATISTICS: N, mean, std (ddof=1), min, max, skewness and EXCESS kurtosis (pandas,
bias-corrected), ADF with a constant and the lag chosen by AIC (statsmodels default
maximum lag 12(n/100)^(1/4)), Ljung-Box Q(20).

Ljung-Box on the targets rejects MECHANICALLY: consecutive target_vol_h values share h-1
of their h returns, so they are autocorrelated by construction. The same holds, to a
lesser degree, for the step-shaped publication-aligned GPR. Neither rejection is evidence
of persistence; the table note says so.

FOOTNOTE ROWS (GPR on other calendars, for comparison only)
-----------------------------------------------------------
  * observation-dated, trading days: the unshifted GPRD / GPRD_THREAT columns of
    data/veriseti.xlsx;
  * the index's own daily calendar (every calendar day, weekends included), read from the
    archived 2026-09-01 vintage data_gpr_daily_recent_20260901.dta in the git-ignored
    vintage cache of 16_gpr_vintages.py (data/gpr_vintages/vintages/). This is the vintage
    data/veriseti.xlsx was built from: the script asserts that it reproduces the dataset's
    GPR values on every trading day (to floating-point rounding) before using it. If the
    file is absent these rows are skipped; run 16_gpr_vintages.py without --cleanup, or
    download that one file from the archive, to restore them. Calendar days on which a
    GPR series is exactly 0 are listed in the JSON.

Outputs: outputs/descriptive_stats{_publication_aligned}.csv, .json
Runtime: a few seconds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OWN_CAL_PATH = (ROOT / "data" / "gpr_vintages" / "vintages"
                / "data_gpr_daily_recent_20260901.dta")
MATCH_TOL = 1e-9  # the .dta -> .xlsx round trip leaves differences of order 1e-14
HORIZONS = [5, 22, 66, 126]
LB_LAG = 20


def describe(x):
    x = pd.Series(x, dtype="float64").dropna()
    adf = adfuller(x.to_numpy(), regression="c", autolag="AIC")
    lb = acorr_ljungbox(x.to_numpy(), lags=[LB_LAG])
    return {"N": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "skew": float(x.skew()),
            "excess_kurtosis": float(x.kurt()), "adf_stat": float(adf[0]),
            "adf_p": float(adf[1]), "adf_lag": int(adf[2]),
            "lb_q20": float(lb["lb_stat"].iloc[0]), "lb_p": float(lb["lb_pvalue"].iloc[0])}


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment

    raw = pd.read_excel(DATA_PATH)
    tgt = pd.read_csv(OUT_DIR / "targets.csv", float_precision="round_trip")
    feat = pd.read_csv(alignment.features_path(al), float_precision="round_trip")
    assert len(raw) == len(tgt) == len(feat) == 4641
    assert (raw["Date"].values == tgt["Date"].values).all()
    assert (raw["Date"].values == feat["Date"].values).all()
    dates = pd.to_datetime(raw["Date"], format="%d.%m.%Y")

    series = [("brent_ret", "Brent günlük log getirisi", "main",
               np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1)))]
    series += [(f"target_vol_{h}", f"Hedef, h={h}", "main", tgt[f"target_vol_{h}"])
               for h in HORIZONS]
    series += [("OVX", "OVX", "main", raw["OVX"]),
               ("GPRD", f"GPRD (model girdisi, {al})", "main", feat["gprd_lag1"]),
               ("GPRD_THREAT", f"GPRD_THREAT (model girdisi, {al})", "main",
                feat["gprd_threat_lag1"])]
    series += [(f"{c}_obs_trading", f"{c} (gözlem tarihli, işlem günleri)", "footnote",
                raw[c]) for c in ("GPRD", "GPRD_THREAT")]

    meta = {"gpr_alignment": al, "sample": {"rows": len(raw),
            "first_date": str(dates.min().date()), "last_date": str(dates.max().date())},
            "adf": "constant, lag by AIC (statsmodels default max lag)",
            "ljung_box_lag": LB_LAG, "kurtosis": "excess (pandas, bias-corrected)"}
    if OWN_CAL_PATH.exists():
        v = pd.read_stata(OWN_CAL_PATH)
        v["d"] = pd.to_datetime(v["DAY"].astype(int).astype(str), format="%Y%m%d")
        v = v[(v["d"] >= dates.min()) & (v["d"] <= dates.max())].sort_values("d")
        span = (dates.max() - dates.min()).days + 1
        assert len(v) == v["d"].nunique() == span, "own calendar is not every calendar day"
        m = pd.DataFrame({"d": dates, "GPRD": raw["GPRD"], "GPRD_THREAT": raw["GPRD_THREAT"]}
                         ).merge(v[["d", "GPRD", "GPRD_THREAT"]], on="d",
                                 suffixes=("", "_v"), validate="1:1")
        mad = {c: float((m[c] - m[f"{c}_v"]).abs().max()) for c in ("GPRD", "GPRD_THREAT")}
        assert max(mad.values()) < MATCH_TOL, f"vintage does not match the dataset: {mad}"
        zeros = {c: [{"date": str(r.d.date()), "weekday": r.d.day_name(),
                      "GPRD": float(r.GPRD), "GPRD_THREAT": float(r.GPRD_THREAT)}
                     for r in v[v[c] == 0].itertuples()] for c in ("GPRD", "GPRD_THREAT")}
        meta["own_calendar"] = {
            "file": OWN_CAL_PATH.relative_to(ROOT).as_posix(),
            "vintage": "2026-09-01",
            "sha256": hashlib.sha256(OWN_CAL_PATH.read_bytes()).hexdigest(),
            "calendar_days": int(len(v)),
            "trading_days_checked": int(len(m)),
            "max_abs_diff_vs_dataset": mad,
            "zero_days": zeros}
        series += [(f"{c}_own_calendar", f"{c} (kendi takvimi, tüm takvim günleri)",
                    "footnote", v[c].to_numpy()) for c in ("GPRD", "GPRD_THREAT")]
    else:
        meta["own_calendar"] = None
        print(f"WARNING: {OWN_CAL_PATH.name} missing; own-calendar footnote rows skipped")

    rows = [{"variable": k, "label": lab, "role": role, **describe(s)}
            for k, lab, role, s in series]
    out = pd.DataFrame(rows)
    path = alignment.out("descriptive_stats.csv", al)
    out.to_csv(path, index=False)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["label"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"Written: {path.name}, {path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
