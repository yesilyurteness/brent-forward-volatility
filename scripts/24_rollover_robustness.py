"""Roll-over robustness of the primary family (HAR, HAR-X, XGBoost).

DESIGN AND STATUSES were written to the experiment log BEFORE this run (Stage 25.1,
commit 44b9eaf). They are not changed in light of the results, and all three variants
are reported whatever they show.

WHY
---
The Brent series is Yahoo's continuous front-month BZ=F. On the first trading day after a
contract expires the series switches to the next contract, so that day's "return" mixes
the price change with the spread between two contracts. That return enters the target
(the standard deviation of the next h returns) and the return-based features.

EXPIRY CALENDAR (ICE Brent Crude futures; source: ICE contract specification, circular
13165 Attach 6, and Circular 15/235)
  * contract months up to and including February 2016: trading ceases on the Business
    Day immediately preceding the 15th calendar day before the first calendar day of the
    contract month if that 15th day is a Business Day, otherwise the Business Day
    immediately preceding the next preceding Business Day;
  * from March 2016: the last Business Day of the second month preceding the contract
    month; if that is the Business Day preceding Christmas Day or New Year's Day, the
    next preceding Business Day.
  Business Day: a trading day that is not a public holiday in England and Wales.
  The implementation is asserted to reproduce all 88 expiries of ICE's official table
  (Circular 15/235 Attachment 1, Dec-15 to Mar-23). ASSUMPTION, not verified from a CME
  document: the NYMEX BZ contract behind BZ=F follows this calendar.
  Roll row = the first data row after an expiry day.

VARIANTS (Stage 25.1)
  A   primary robustness variant: the roll-row return is removed (NaN). The target is the
      std of the remaining returns in the same h-day window (the window must still be
      complete, as with skipna=False at the series end). Return-based features are
      recomputed on the cleaned returns: brent_ret_lag1-5 and har_daily use the most
      recent clean returns; brent_vol5/20/60/126, the three volatility ratios and the
      past-volatility baseline (XGBoost's ratio denominator) are rolling stds over the
      same windows without the removed returns, NaN wherever the original is NaN.
  A2  sensitivity (A' in the log): as A, removing the roll row and the next row.
  B   sensitivity, h=5 only: no return is removed; rows whose target window contains a
      roll row are dropped.
  Price-level features (brent_lag1-5, brent_ema5/10/20) are NOT changed. LIMITATION: the
  control cleans the roll effect in the target and the return features, not in
  XGBoost's price-level features; that would need a back-adjusted series.

VALIDATION BEFORE ANY VARIANT (all asserted)
  With an empty mask the construction reproduces bit for bit the saved targets, the
  return-based feature columns and the HAR, HAR-X and XGBoost test predictions, and it
  reproduces the stored primary-family test values (DM-HLN, p-values, fold wins).

TESTS: the 8 primary-family tests (HAR vs HAR-X, HAR-X vs XGBoost, four horizons) in each
variant, as in 08_dm_test.py (DM with Newey-West Bartlett L = h-1 and the HLN factor;
fold-level sign test), with Holm/BH/BY within the variant's own 8 tests. Variant
replications are not new families and are never pooled with the primary family
(CLAUDE.md). Absolute RMSE is not comparable with the primary results because the
target changes; only within-variant model differences are reported.

Outputs: outputs/rollover_calendar.csv, rollover_{predictions,metrics,family_tests}{sfx}.csv,
         rollover_summary{sfx}.json
Runtime: about two minutes (four XGBoost walk-forward passes).
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter
from scipy import stats

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
HORIZONS = [5, 22, 66, 126]
PAIRS = [("har", "har_x"), ("har_x", "xgboost")]  # primary family, as in 08


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wf = _load("walkforward", "03_walkforward.py")
bm = _load("benchmarks", "05_benchmarks.py")
dm = _load("dm", "08_dm_test.py")

# ICE Circular 15/235 Attachment 1: official futures expiry dates (contract, dd.mm.yyyy)
OFFICIAL = """Dec-15 13.11.2015 Jan-16 16.12.2015 Feb-16 14.01.2016 Mar-16 29.01.2016
Apr-16 29.02.2016 May-16 31.03.2016 Jun-16 29.04.2016 Jul-16 31.05.2016 Aug-16 30.06.2016
Sep-16 29.07.2016 Oct-16 31.08.2016 Nov-16 30.09.2016 Dec-16 31.10.2016 Jan-17 30.11.2016
Feb-17 29.12.2016 Mar-17 31.01.2017 Apr-17 28.02.2017 May-17 31.03.2017 Jun-17 28.04.2017
Jul-17 31.05.2017 Aug-17 30.06.2017 Sep-17 31.07.2017 Oct-17 31.08.2017 Nov-17 29.09.2017
Dec-17 31.10.2017 Jan-18 30.11.2017 Feb-18 28.12.2017 Mar-18 31.01.2018 Apr-18 28.02.2018
May-18 29.03.2018 Jun-18 30.04.2018 Jul-18 31.05.2018 Aug-18 29.06.2018 Sep-18 31.07.2018
Oct-18 31.08.2018 Nov-18 28.09.2018 Dec-18 31.10.2018 Jan-19 30.11.2018 Feb-19 28.12.2018
Mar-19 31.01.2019 Apr-19 28.02.2019 May-19 29.03.2019 Jun-19 30.04.2019 Jul-19 31.05.2019
Aug-19 28.06.2019 Sep-19 31.07.2019 Oct-19 30.08.2019 Nov-19 30.09.2019 Dec-19 31.10.2019
Jan-20 29.11.2019 Feb-20 30.12.2019 Mar-20 31.01.2020 Apr-20 28.02.2020 May-20 31.03.2020
Jun-20 30.04.2020 Jul-20 29.05.2020 Aug-20 30.06.2020 Sep-20 31.07.2020 Oct-20 28.08.2020
Nov-20 30.09.2020 Dec-20 30.10.2020 Jan-21 30.11.2020 Feb-21 30.12.2020 Mar-21 29.01.2021
Apr-21 26.02.2021 May-21 31.03.2021 Jun-21 30.04.2021 Jul-21 28.05.2021 Aug-21 30.06.2021
Sep-21 30.07.2021 Oct-21 31.08.2021 Nov-21 30.09.2021 Dec-21 29.10.2021 Jan-22 30.11.2021
Feb-22 30.12.2021 Mar-22 31.01.2022 Apr-22 28.02.2022 May-22 31.03.2022 Jun-22 29.04.2022
Jul-22 31.05.2022 Aug-22 30.06.2022 Sep-22 29.07.2022 Oct-22 31.08.2022 Nov-22 30.09.2022
Dec-22 31.10.2022 Jan-23 30.11.2022 Feb-23 29.12.2022 Mar-23 31.01.2023"""


# ---------------------------------------------------------------------------
# Expiry calendar
# ---------------------------------------------------------------------------
def ew_holidays(y0=2007, y1=2027):
    """Public holidays in England and Wales, with substitute days and the one-off ones."""
    H = set()
    ts = pd.Timestamp
    for y in range(y0, y1 + 1):
        d = ts(y, 1, 1)
        while d.dayofweek >= 5:
            d += pd.Timedelta(days=1)
        H.add(d)
        e = ts(easter(y))
        H |= {e - pd.Timedelta(days=2), e + pd.Timedelta(days=1)}
        if y == 2020:
            H.add(ts(2020, 5, 8))                    # early May moved to VE Day
        else:
            d = ts(y, 5, 1)
            while d.dayofweek != 0:
                d += pd.Timedelta(days=1)
            H.add(d)
        if y in (2012, 2022):                        # spring bank moved for jubilees
            H.add(ts(2012, 6, 4) if y == 2012 else ts(2022, 6, 2))
        else:
            d = ts(y, 5, 31)
            while d.dayofweek != 0:
                d -= pd.Timedelta(days=1)
            H.add(d)
        d = ts(y, 8, 31)
        while d.dayofweek != 0:
            d -= pd.Timedelta(days=1)
        H.add(d)
        c = ts(y, 12, 25)
        H |= {5: {ts(y, 12, 27), ts(y, 12, 28)}, 6: {ts(y, 12, 26), ts(y, 12, 27)},
              4: {c, ts(y, 12, 28)}}.get(c.dayofweek, {c, ts(y, 12, 26)})
    H |= {ts(d) for d in ("2011-04-29", "2012-06-05", "2022-06-03", "2022-09-19",
                          "2023-05-08")}
    return H


HOL = ew_holidays()


def is_bd(d):
    return d.dayofweek < 5 and d not in HOL


def prev_bd(d):
    d -= pd.Timedelta(days=1)
    while not is_bd(d):
        d -= pd.Timedelta(days=1)
    return d


def expiry(cm):
    """Last trading day of the ICE Brent futures contract for contract month cm (1st day)."""
    if cm < pd.Timestamp(2016, 3, 1):
        d = cm - pd.Timedelta(days=15)
        if not is_bd(d):
            d = prev_bd(d)
        return prev_bd(d)
    d = cm - pd.DateOffset(months=1)             # first day of the preceding month
    d = prev_bd(d)                               # last BD of the second month preceding
    before_xmas = prev_bd(pd.Timestamp(d.year, 12, 25))
    before_nyd = prev_bd(pd.Timestamp(d.year + 1, 1, 1))
    if d in (before_xmas, before_nyd):
        d = prev_bd(d)
    return d


def calendar(dates):
    cms = pd.date_range("2007-12-01", "2027-01-01", freq="MS")
    cal = pd.DataFrame({"contract": cms.strftime("%b-%y"),
                        "expiry": [expiry(c) for c in cms],
                        "rule": np.where(cms < pd.Timestamp(2016, 3, 1),
                                         "15 gun (Subat 2016'ya kadar)",
                                         "ay oncesi (Mart 2016'dan)")})
    tok = OFFICIAL.split()
    official = dict(zip(tok[0::2], pd.to_datetime(tok[1::2], format="%d.%m.%Y")))
    got = cal.set_index("contract")["expiry"]
    bad = {k: (v.date(), got[k].date()) for k, v in official.items() if got[k] != v}
    assert not bad and len(official) == 88, f"expiry calendar vs ICE table: {bad}"
    cal = cal[(cal["expiry"] >= dates.min()) & (cal["expiry"] < dates.max())].copy()
    pos = np.searchsorted(dates.values, cal["expiry"].values, side="right")
    cal["expiry_in_data"] = cal["expiry"].isin(dates).values
    cal["roll_row"] = pos
    cal["roll_row_date"] = dates.iloc[pos].dt.strftime("%Y-%m-%d").values
    return cal.reset_index(drop=True), len(official)


# ---------------------------------------------------------------------------
# Variant data
# ---------------------------------------------------------------------------
def last_clean_lags(clean, k=5):
    """lag_i at row t = the i-th most recent non-NaN return at rows <= t-1."""
    v = clean.dropna().to_numpy()
    cnt = clean.notna().cumsum().shift(1).fillna(0).astype(int).to_numpy()
    out = {}
    for i in range(1, k + 1):
        j = cnt - i
        col = np.full(len(clean), np.nan)
        ok = j >= 0
        col[ok] = v[j[ok]]
        out[i] = pd.Series(col, index=clean.index)
    return out


def rolling_std_clean(clean, removed, w, like):
    """Causal rolling std over the previous w rows without the removed returns.

    `like` is the original feature (same window, all returns). Rows whose window contains
    no removed return keep the original value exactly, so a variant changes only the rows
    it has to change; NaN wherever `like` is NaN (same warm-up)."""
    cleaned = clean.rolling(w, min_periods=2).std().shift(1)
    hit = pd.Series(removed.astype(float), index=clean.index).rolling(w).sum().shift(1) > 0
    return like.where(~hit, cleaned).where(like.notna())


def build_variant(raw_ret, feat, tgt, mask):
    """Targets, return features, har_daily and past_vol for a removal mask (bool array)."""
    clean = raw_ret.where(~mask)
    f = feat.copy()
    lags = last_clean_lags(clean)
    rmk = pd.Series(mask.astype(float), index=raw_ret.index)
    hit_lag = {i: rmk.rolling(i).sum().shift(1) > 0 for i in range(1, 6)}  # removed in t-i..t-1
    for i in range(1, 6):
        o = feat[f"brent_ret_lag{i}"]
        f[f"brent_ret_lag{i}"] = o.where(~hit_lag[i], lags[i]).where(o.notna())
    vols = {w: rolling_std_clean(clean, mask, w, feat[f"brent_vol{w}"])
            for w in (5, 20, 60, 126)}
    for w, s in vols.items():
        f[f"brent_vol{w}"] = s
    eps = 1e-9  # 02_build_features.EPS
    hit_w = {w: rmk.rolling(w).sum().shift(1) > 0 for w in (20, 60, 126)}  # longer window
    for col, a, b in (("vol_ratio", 5, 20), ("vol5_vol60", 5, 60),
                      ("vol20_vol126", 20, 126)):
        o = feat[col]
        f[col] = o.where(~hit_w[b], vols[a] / (vols[b] + eps)).where(o.notna())
    o = raw_ret.abs().shift(1)                              # as 05_benchmarks
    har_daily = o.where(~hit_lag[1], lags[1].abs()).where(o.notna())
    t = {}
    pv = {}
    rm = pd.Series(mask.astype(float), index=raw_ret.index)
    for h in HORIZONS:
        # Windows without a removed return keep the saved target exactly (01's formula);
        # the others are the std of the remaining returns in the same, complete window.
        fut = pd.concat([clean.shift(-i) for i in range(1, h + 1)], axis=1)
        hit = rm.rolling(h).sum().shift(-h) > 0          # removed row in t+1 .. t+h
        orig = tgt[f"target_vol_{h}"]
        t[h] = orig.where(~hit, fut.std(axis=1, skipna=True)).where(orig.notna())
        pv[h] = rolling_std_clean(clean, mask, h, raw_ret.rolling(h).std().shift(1))
    return f, t, har_daily, pv


# ---------------------------------------------------------------------------
# Models (HAR / HAR-X exactly as 05; XGBoost through 03.run_horizon)
# ---------------------------------------------------------------------------
def har_models(dfh, h, test_years):
    y_col = f"target_vol_{h}"
    rows = []
    for fold_id, ty in enumerate(test_years, start=1):
        tr_all = dfh.index[dfh["year"] < ty]
        te_all = dfh.index[dfh["year"] == ty]
        tr_emb = tr_all[:-h]
        assert tr_emb.max() + h < te_all.min()

        def slice_for(idx, cols):
            ok = dfh.loc[idx, list(cols) + [y_col]].notna().all(axis=1)
            return dfh.loc[idx[ok.values]]

        tr_har, tr_harx = slice_for(tr_emb, bm.HAR_COLS), slice_for(tr_emb, bm.HARX_COLS)
        te = slice_for(te_all, bm.HAR_COLS)
        assert (slice_for(te_all, bm.HARX_COLS).index == te.index).all()
        floor = float(tr_har[y_col].min())
        b = bm.ols_fit(tr_har[bm.HAR_COLS].to_numpy("float64"),
                       tr_har[y_col].to_numpy("float64"))
        bx = bm.ols_fit(tr_harx[bm.HARX_COLS].to_numpy("float64"),
                        tr_harx[y_col].to_numpy("float64"))
        rows.append(pd.DataFrame({
            "horizon": h, "Date": te["Date"].values, "fold": fold_id, "test_year": ty,
            "include_in_main": not (ty == wf.PARTIAL_YEAR and h in wf.EXCLUDE_2026_HORIZONS),
            "y_true": te[y_col].values,
            "pred_har": np.maximum(bm.ols_predict(b, te[bm.HAR_COLS].to_numpy("float64")), floor),
            "pred_har_x": np.maximum(bm.ols_predict(bx, te[bm.HARX_COLS].to_numpy("float64")),
                                     floor)}))
    return pd.concat(rows, ignore_index=True)


def run_variant(name, feat_v, targets, har_daily, pv, raw_ret, test_years, horizons):
    feature_cols = [c for c in feat_v.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in wf.LOG_FEATURES if c in feature_cols]
    df = feat_v.copy()
    df["year"] = df["Date_parsed"].dt.year
    for h in HORIZONS:
        df[f"target_vol_{h}"] = targets[h].values
    df["har_daily"] = har_daily.values
    out = []
    for h in horizons:
        hp = har_models(df, h, test_years)
        xp, _ = wf.run_horizon(h, df, feature_cols, log_cols, raw_ret, test_years,
                               verbose=False, past_vol=pv[h].values)
        m = hp.merge(xp[["horizon", "Date", "y_true", "pred_xgboost"]],
                     on=["horizon", "Date"], suffixes=("", "_x"), validate="1:1")
        assert len(m) == len(hp) == len(xp), f"{name} h={h}: test rows differ"
        assert (m["y_true"] == m["y_true_x"]).all()
        out.append(m.drop(columns="y_true_x").assign(variant=name))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# The 8 primary-family tests, as in 08_dm_test.py
# ---------------------------------------------------------------------------
def family_tests(p):
    rows = []
    for h in sorted(p["horizon"].unique()):
        g = p[(p["horizon"] == h) & p["include_in_main"]].copy()
        g["dt"] = pd.to_datetime(g["Date"], format="%d.%m.%Y")
        g = g.sort_values("dt")
        n, lag = len(g), h - 1
        y = g["y_true"].to_numpy("float64")
        yrs = g["test_year"].to_numpy()
        for m1, m2 in PAIRS:
            e1 = y - g[f"pred_{m1}"].to_numpy("float64")
            e2 = y - g[f"pred_{m2}"].to_numpy("float64")
            d = e1 ** 2 - e2 ** 2
            omega, g0 = dm.newey_west_lrv(d, lag)
            stat = d.mean() / np.sqrt(max(omega, 1e-300) / n)
            hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
            f1 = pd.Series(e1 ** 2).groupby(yrs).mean() ** 0.5
            f2 = pd.Series(e2 ** 2).groupby(yrs).mean() ** 0.5
            w1 = int((f1 < f2).sum())
            harx_wins = int((f2 < f1).sum()) if m2 == "har_x" else w1
            rows.append({
                "comparison": f"{m1} vs {m2}", "horizon": h, "n": n,
                "rmse_model1_pooled": float(np.sqrt(np.mean(e1 ** 2))),
                "rmse_model2_pooled": float(np.sqrt(np.mean(e2 ** 2))),
                "DM": float(stat), "DM_HLN": float(stat * hln),
                "p_HLN": float(2 * (1 - stats.t.cdf(abs(stat * hln), df=n - 1))),
                "harx_fold_wins": harx_wins, "n_folds": int(len(f1)),
                "p_sign": float(stats.binomtest(w1, len(f1), 0.5).pvalue)})
    r = pd.DataFrame(rows)
    c = sum(1 / i for i in range(1, len(r) + 1))
    for col in ("p_HLN", "p_sign"):
        r[f"{col}_holm"] = dm.holm(r[col].to_numpy())
        r[f"{col}_bh"] = dm.benjamini_hochberg(r[col].to_numpy())
        r[f"{col}_by"] = np.minimum(r[f"{col}_bh"] * c, 1.0)
    return r


def fold_metrics(p):
    rows = []
    for (h, ty, inc), g in p.groupby(["horizon", "test_year", "include_in_main"]):
        for m in ("har", "har_x", "xgboost"):
            e = g["y_true"] - g[f"pred_{m}"]
            rows.append({"horizon": h, "test_year": ty, "include_in_main": inc, "model": m,
                         "n": len(g), "rmse": float(np.sqrt((e ** 2).mean())),
                         "mae": float(e.abs().mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment
    t0 = time.time()

    raw = pd.read_excel(bm.DATA_PATH)
    # Read exactly as 03 and 05 read them (default float parser), so the baseline can be
    # reproduced bit for bit.
    feat = pd.read_csv(alignment.features_path(al), parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    assert (raw["Date"].values == feat["Date"].values).all()
    assert (raw["Date"].values == tgt["Date"].values).all()
    dates = pd.to_datetime(raw["Date"], format="%d.%m.%Y")
    raw_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    test_years = list(range(wf.FIRST_TEST_YEAR, wf.LAST_TEST_YEAR + 1))

    cal, n_official = calendar(dates)
    cal.assign(expiry=cal["expiry"].dt.strftime("%Y-%m-%d")).to_csv(
        OUT_DIR / "rollover_calendar.csv", index=False)
    roll = np.zeros(len(raw), bool)
    roll[cal["roll_row"].to_numpy()] = True
    roll2 = roll | np.roll(roll, 1)
    roll2[0] = False
    print(f"[takvim] ICE resmi tablosu: {n_official}/{n_official} vade birebir; "
          f"orneklemde {len(cal)} vade, {int((~cal['expiry_in_data']).sum())} vade gunu veride yok")

    # ---- validation: empty mask reproduces the primary pipeline ----------------------
    f0, t0_, hd0, pv0 = build_variant(raw_ret, feat, tgt, np.zeros(len(raw), bool))
    ret_cols = ([f"brent_ret_lag{i}" for i in range(1, 6)]
                + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
                + ["vol_ratio", "vol5_vol60", "vol20_vol126"])
    for c in ret_cols:
        assert f0[c].equals(feat[c]), f"empty-mask feature differs: {c}"
    for h in HORIZONS:
        assert np.array_equal(t0_[h].to_numpy(), tgt[f"target_vol_{h}"].to_numpy(),
                              equal_nan=True), f"empty-mask target differs: h={h}"
        assert np.array_equal(pv0[h].to_numpy(),
                              raw_ret.rolling(h).std().shift(1).to_numpy(), equal_nan=True)
    assert np.array_equal(hd0.to_numpy(), raw_ret.abs().shift(1).to_numpy(), equal_nan=True)
    base = run_variant("baseline", f0, t0_, hd0, pv0, raw_ret, test_years, HORIZONS)
    bench = pd.read_csv(alignment.out("bench_predictions_all.csv", al),
                        float_precision="round_trip")
    wfp = pd.read_csv(alignment.out("wf_predictions_all.csv", al), float_precision="round_trip")
    j = base.merge(bench[["horizon", "Date", "pred_har", "pred_har_x"]], on=["horizon", "Date"],
                   suffixes=("", "_s"), validate="1:1").merge(
        wfp[["horizon", "Date", "pred_xgboost"]], on=["horizon", "Date"],
        suffixes=("", "_s"), validate="1:1")
    assert len(j) == len(base) == len(bench)
    for m in ("har", "har_x", "xgboost"):
        assert (j[f"pred_{m}"] == j[f"pred_{m}_s"]).all(), f"baseline {m} != saved predictions"
    ft0 = family_tests(base)
    pf = pd.read_csv(alignment.out("primary_family_tests.csv", al))
    pf["comparison"] = pf["karşılaştırma"]
    k = ft0.merge(pf, on=["comparison", "horizon"], suffixes=("", "_s"), validate="1:1")
    assert len(k) == 8
    for c in ("DM_HLN", "p_HLN", "p_sign", "p_HLN_bh", "p_sign_by"):
        assert np.allclose(k[c], k[f"{c}_s"], rtol=1e-8, atol=1e-12), c
    assert (k["harx_fold_wins"] == k["harx_fold_wins_s"]).all()
    print("[kontrol] bos maske: hedefler, getiri ozellikleri, HAR/HAR-X/XGBoost tahminleri "
          "bit duzeyinde; 8 test kayitli degerlerle ayni")

    # ---- variants ---------------------------------------------------------------------
    preds = [base]
    for name, mask in (("A", roll), ("A2", roll2)):
        fv, tv, hdv, pvv = build_variant(raw_ret, feat, tgt, mask)
        preds.append(run_variant(name, fv, tv, hdv, pvv, raw_ret, test_years, HORIZONS))
        print(f"[{name}] tamam ({time.time() - t0:.0f} sn)")
    tb = {h: tgt[f"target_vol_{h}"].copy() for h in HORIZONS}
    win_has_roll = pd.Series(roll.astype(float)).rolling(5).sum().shift(-5) > 0
    tb[5] = tb[5].where(~win_has_roll.values)
    preds.append(run_variant("B", f0, tb, hd0, pv0, raw_ret, test_years, [5]))
    print(f"[B] tamam ({time.time() - t0:.0f} sn)")
    P = pd.concat(preds, ignore_index=True)

    tests = pd.concat([family_tests(g).assign(variant=v) for v, g in P.groupby("variant")],
                      ignore_index=True)
    FM = pd.concat([fold_metrics(g).assign(variant=v) for v, g in P.groupby("variant")],
                   ignore_index=True)

    gaps = pd.read_csv(OUT_DIR / "date_gap_rows.csv")
    gap_rows = set(gaps.loc[gaps["category"] == "unexplained", "row"])
    assert len(gap_rows) == 40
    overlap = {"A": sorted(gap_rows & set(np.flatnonzero(roll))),
               "A2": sorted(gap_rows & set(np.flatnonzero(roll2)))}
    b_rows = int(win_has_roll.iloc[:len(raw) - 5].sum())

    sfx = alignment.suffix(al)
    P.to_csv(OUT_DIR / f"rollover_predictions{sfx}.csv", index=False)
    FM.to_csv(OUT_DIR / f"rollover_metrics{sfx}.csv", index=False)
    tests.to_csv(OUT_DIR / f"rollover_family_tests{sfx}.csv", index=False)
    summary = {
        "gpr_alignment": al, "design_registered": "experiment log Stage 25.1, commit 44b9eaf",
        "statuses": {"A": "primary robustness variant", "A2": "sensitivity (two rows)",
                     "B": "sensitivity, h=5 only"},
        "calendar": {"official_expiries_matched": n_official, "expiries_in_sample": len(cal),
                     "expiry_days_missing_in_data": int((~cal["expiry_in_data"]).sum()),
                     "rows_removed_A": int(roll.sum()), "rows_removed_A2": int(roll2.sum())},
        "B_rows_dropped_h5": b_rows,
        "documented_gaps": 40,
        "gap_rows_removed": {k: {"n": len(v),
                                 "dates": [str(dates.iloc[r].date()) for r in v]}
                             for k, v in overlap.items()},
        "limitation": ("price-level features (brent_lag1-5, brent_ema5/10/20) unchanged; the "
                       "roll effect in them is not removed (needs a back-adjusted series)"),
        "runtime_seconds": round(time.time() - t0, 1)}
    with open(OUT_DIR / f"rollover_summary{sfx}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250)
    print(tests[["variant", "comparison", "horizon", "DM_HLN", "p_HLN", "p_HLN_holm",
                 "p_HLN_bh", "p_HLN_by", "harx_fold_wins", "n_folds", "p_sign",
                 "p_sign_bh"]].to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("calendar", "B_rows_dropped_h5")}))
    print({k: v["n"] for k, v in summary["gap_rows_removed"].items()})
    print(f"Sure {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()
