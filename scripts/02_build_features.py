"""Builds every causal feature described in the "Feature Engineering" section of CLAUDE.md.

THE CAUSALITY CONTRACT
----------------------
Every feature at row t uses only data from t-1 and earlier.

  * Lag features are produced directly with .shift(1) ... .shift(5).
  * EMA / rolling / z-score quantities are first computed on the raw series, then
    .shift(1) is applied to the result. ewm(adjust=False) and rolling already look only
    at the past; the extra .shift(1) also excludes the observation at time t.
  * Derived features (ratios, interactions, spikes, momentum) are built from inputs that
    are already shifted; they are NOT shifted a second time (that would push the
    information one further day into the past for no reason).
  * Calendar features are NOT shifted. Day of week / month / start-and-end-of-month are
    derived from the date, do not depend on the data, and are always known in advance.
    Shifting them would hide day t's own calendar information.

THIS SCRIPT CONTAINS NO FIT()
-----------------------------
There is NO scaling, winsorization or log transform here. Neither MinMaxScaler /
StandardScaler nor any other estimator is called. All normalization happens inside the
walk-forward loop and is fit only on each fold's training slice (CLAUDE.md Critical
Rule 2).

The z-scores here are NOT an exception: they are computed over a 60-day ROLLING window,
which makes them a causal transform using only each row's own history. No single
mean/std is estimated over the full series.

No rows are DROPPED; the NaNs at the start of the rolling windows are preserved.

WARNING ABOUT THE CORRELATION REPORT
------------------------------------
The high_corr_pairs_full entry in the report is computed over the full data (2008-2026)
and is for DIAGNOSTIC PURPOSES ONLY. DELETING a feature based on that list would carry
test-period information into feature selection and would be leakage (Critical Rule 5). If
features genuinely have to be dropped, use high_corr_pairs_warmup (the 2008-2011 first
training set only), or recompute the correlation on each fold's training slice.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

EXPECTED_ROWS = 4641
Z_WINDOW = 60          # z-score rolling window length (days)
SPIKE_Z = 2.0          # spike threshold
REGIME_Z = 1.0         # high/low regime threshold
EPS = 1e-9             # guard for the vol_ratio denominator
CORR_THRESHOLD = 0.95  # |r| threshold for reporting
WARMUP_END = "2012-01-01"  # end of the first training set (2008-2011 warm-up period)

SERIES = {
    "brent": "Brent_Petrol",
    "ovx": "OVX",
    "gprd": "GPRD",
    "gprd_threat": "GPRD_THREAT",
}


def _flag(cond_series, source_z):
    """Builds a threshold flag; if the source z-score is NaN, the flag stays NaN.

    Had this been written as (z > 2).astype(int), the NaN z-scores in the warm-up period
    would turn into 0 and the model would see a spurious "no spike" signal. Here the NaN
    is preserved.
    """
    return pd.Series(
        np.where(source_z.isna(), np.nan, cond_series.astype(float)),
        index=source_z.index,
        dtype="float64",
    )


def build_features(df):
    """Builds the causal feature matrix from the raw data frame.

    df: a frame containing the columns Date, Brent_Petrol, OVX, GPRD, GPRD_THREAT.
    The function is pure; it holds no global state. The prefix-invariance test verifies
    causality by calling it again on slices of the data.
    """
    date_parsed = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    out = pd.DataFrame(
        {"Date": df["Date"].values, "Date_parsed": date_parsed.values},
        index=df.index,
    )

    # --- 1) Lags 1-5 (4 series x 5 = 20) ------------------------------------
    for name, col in SERIES.items():
        s = df[col]
        for i in range(1, 6):
            out[f"{name}_lag{i}"] = s.shift(i)

    # --- 2) EMA 5/10/20 (4 series x 3 = 12) ---------------------------------
    # The ewm result includes time t -> .shift(1) makes it causal.
    # min_periods=span: with adjust=False, ewm produces a value from the very first
    # observation, so ema20's value on row 2 is not a true 20-day average but the single
    # observation itself. That is not leakage (it does not look ahead) but it is a
    # wrongly scaled warm-up artifact. min_periods leaves those rows as NaN.
    for name, col in SERIES.items():
        s = df[col]
        for span in (5, 10, 20):
            out[f"{name}_ema{span}"] = (
                s.ewm(span=span, adjust=False, min_periods=span).mean().shift(1)
            )

    # --- 3) Brent log-return lags 1-5 (5) -----------------------------------
    # The price LEVEL is not stationary; in an expanding window the 2012 training range
    # does not overlap the 2026 test range, and tree models cannot extrapolate. The level
    # lags are kept because CLAUDE.md asks for them, and return lags are added alongside.
    daily_ret = np.log(df["Brent_Petrol"] / df["Brent_Petrol"].shift(1))
    for i in range(1, 6):
        out[f"brent_ret_lag{i}"] = daily_ret.shift(i)

    # --- 4) Realized volatility, HAR-style multi-scale (9) -------------------
    # If only the short windows (5, 20) were kept, then at long horizons (h=66, h=126)
    # the model would have no access at all to the MATCHING window information the
    # past-volatility baseline uses: the baseline uses the realized volatility of the
    # past h days, while the model's longest input window would be 20 days. The logic of
    # the HAR model (using daily, weekly, monthly, quarterly and yearly realized
    # volatility simultaneously) is carried over here; the 60/126/252 windows match the
    # h=66 and h=126 horizons.
    #
    # DATA WINDOW COST: the longest window determines the first fully populated row of
    # the feature matrix and costs every fold that many training rows. This is not
    # leakage but a data cost. A 252-day window was tried and REMOVED: it pushed the
    # first valid row to 253 and cost about 193 extra rows per fold, taking more than it
    # gave, so the longest window was left at 126. The 126 window already matches the
    # h=126 horizon.
    vol_windows = (5, 20, 60, 126)
    vols = {w: daily_ret.rolling(w).std().shift(1) for w in vol_windows}
    for w in vol_windows:
        out[f"brent_vol{w}"] = vols[w]
    # Ratios: short vs long window comparison (the volatility term structure).
    # In very calm periods the denominator can approach zero and blow the ratio up ->
    # epsilon guard.
    out["vol_ratio"] = vols[5] / (vols[20] + EPS)      # the existing short-term ratio
    out["vol5_vol60"] = vols[5] / (vols[60] + EPS)
    out["vol20_vol126"] = vols[20] / (vols[126] + EPS)

    # --- 5) OVX derivatives (5) ---------------------------------------------
    ovx = df["OVX"]
    ovx_mean = ovx.rolling(Z_WINDOW).mean()
    ovx_std = ovx.rolling(Z_WINDOW).std()
    ovx_z = ((ovx - ovx_mean) / ovx_std).shift(1)
    out["ovx_z60"] = ovx_z
    out["ovx_spike"] = _flag(ovx_z > SPIKE_Z, ovx_z)
    out["ovx_regime_high"] = _flag(ovx_z > REGIME_Z, ovx_z)
    out["ovx_regime_low"] = _flag(ovx_z < -REGIME_Z, ovx_z)
    # Mean reversion: percentage deviation from the 60-day average.
    out["ovx_mr60"] = (ovx / ovx_mean - 1.0).shift(1)

    # --- 6) GPR derivatives (7) ---------------------------------------------
    for name in ("gprd", "gprd_threat"):
        s = df[SERIES[name]]
        z = ((s - s.rolling(Z_WINDOW).mean()) / s.rolling(Z_WINDOW).std()).shift(1)
        out[f"{name}_z60"] = z
        out[f"{name}_spike"] = _flag(z > SPIKE_Z, z)
        # The EMA columns are already shifted; the difference is not shifted again.
        out[f"{name}_momentum"] = out[f"{name}_ema5"] - out[f"{name}_ema20"]
    # Threat ratio: division is safe because the GPRD base is always > 9.
    out["threat_ratio"] = (df["GPRD_THREAT"] / df["GPRD"]).shift(1)

    # --- 7) Interaction terms (4) -------------------------------------------
    # The inputs are already-shifted lag1 columns -> they are not shifted again.
    out["ovx_x_gprd"] = out["ovx_lag1"] * out["gprd_lag1"]
    out["ovx_x_gprd_threat"] = out["ovx_lag1"] * out["gprd_threat_lag1"]
    out["ovx_regime_high_x_gprd"] = out["ovx_regime_high"] * out["gprd_lag1"]
    out["ovx_regime_low_x_gprd"] = out["ovx_regime_low"] * out["gprd_lag1"]

    # --- 8) Calendar (5) -- not shifted, see the module docstring ------------
    out["dow"] = date_parsed.dt.dayofweek.values
    out["month"] = date_parsed.dt.month.values
    out["is_month_start"] = date_parsed.dt.is_month_start.astype(int).values
    out["is_month_end"] = date_parsed.dt.is_month_end.astype(int).values
    out["is_quarter_end"] = date_parsed.dt.is_quarter_end.astype(int).values

    return out


def feature_columns(feat):
    return [c for c in feat.columns if c not in ("Date", "Date_parsed")]


def high_corr_pairs(feat, cols, threshold):
    """Returns the feature pairs with |r| > threshold (upper triangle, no duplicates)."""
    sub = feat[cols].astype("float64")
    # Constant columns produce NaN in the correlation; they are reported separately below.
    corr = sub.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    abs_corr = corr.abs().to_numpy()
    hits = mask & (abs_corr > threshold)
    pairs = []
    for i, j in zip(*np.where(hits)):
        pairs.append({
            "feature_a": corr.index[i],
            "feature_b": corr.columns[j],
            "r": float(corr.iat[i, j]),
        })
    pairs.sort(key=lambda p: -abs(p["r"]))
    return pairs


def main():
    df = pd.read_excel(DATA_PATH)
    assert len(df) == EXPECTED_ROWS, (
        f"Kaynak veri satir sayisi degismis: {len(df)} (beklenen {EXPECTED_ROWS})"
    )

    feat = build_features(df)
    cols = feature_columns(feat)

    # === CHECK 1: prefix invariance (proof of causality) ====================
    # The data is truncated at row T and the pipeline is re-run. Every feature value
    # before T must come out IDENTICAL to the full-data version. Tolerance = 0
    # (check_exact=True): because ewm(adjust=False), rolling and shift do not look
    # ahead, bit-level equality is expected. A mismatch is a genuine look-ahead bug and
    # must not be hidden by loosening the tolerance.
    prefix_results = []
    for cut in (3000, 4000):
        truncated = build_features(df.iloc[:cut].copy())
        pd.testing.assert_frame_equal(
            feat.iloc[:cut],
            truncated,
            check_exact=True,
            check_dtype=True,
            obj=f"prefix-invariance (kesme={cut})",
        )
        prefix_results.append({"cut_row": cut, "tolerance": 0.0, "passed": True})
    print("=== Prefix-invariance testi ===")
    print(f"Kesme noktalari {[r['cut_row'] for r in prefix_results]}: "
          "kesilmis veriyle uretilen ozellikler tam veriyle BIREBIR ayni (tolerans=0).")
    print("Hicbir ozellik gelecege bakmiyor.\n")

    # === CHECK 2: date alignment ===========================================
    assert len(feat) == EXPECTED_ROWS, f"Satir sayisi degisti: {len(feat)}"
    targets_path = OUT_DIR / "targets.csv"
    targets_aligned = None
    if targets_path.exists():
        tg = pd.read_csv(targets_path)
        assert len(tg) == len(feat), "targets.csv satir sayisi features ile uyusmuyor"
        assert (tg["Date"].values == feat["Date"].values).all(), (
            "targets.csv ile features Date sutunlari birebir eslesmiyor"
        )
        targets_aligned = True
        print("targets.csv ile Date hizalamasi: TAM ESLESME\n")

    # === CHECK 3: scan for infinite values ==================================
    inf_counts = {}
    for c in cols:
        n_inf = int(np.isinf(feat[c].to_numpy(dtype="float64")).sum())
        if n_inf:
            inf_counts[c] = n_inf
    if inf_counts:
        print(f"UYARI: sonsuz deger bulundu -> NaN'a cevriliyor: {inf_counts}\n")
        feat[cols] = feat[cols].replace([np.inf, -np.inf], np.nan)
    else:
        print("Sonsuz deger taramasi: temiz (0 adet)\n")

    # === CHECK 4: scan for constant columns ================================
    constant_cols = [c for c in cols if feat[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"UYARI: sabit (tek degerli) sutunlar: {constant_cols}\n")
    else:
        print("Sabit sutun taramasi: temiz (0 adet)\n")

    # === Feature inventory: index of the first valid row ====================
    inv_rows = []
    for c in cols:
        first_valid = feat[c].first_valid_index()
        inv_rows.append({
            "feature": c,
            "first_valid_row": int(first_valid) if first_valid is not None else -1,
            "first_valid_date": (feat.loc[first_valid, "Date"]
                                 if first_valid is not None else "YOK"),
            "n_nan": int(feat[c].isna().sum()),
            "dtype": str(feat[c].dtype),
        })
    inv_df = pd.DataFrame(inv_rows)

    print("=== Uretilen ozellikler (ilk gecerli satir indeksi ile) ===")
    print(inv_df.to_string(index=False))
    print()
    print(f"TOPLAM OZELLIK SAYISI: {len(cols)}")
    print(f"CSV sutun sayisi (Date + Date_parsed dahil): {feat.shape[1]}")
    late = inv_df.loc[inv_df["first_valid_row"].idxmax()]
    print(f"En gec baslayan ozellik: {late['feature']} -> satir "
          f"{late['first_valid_row']} ({late['first_valid_date']})")
    print()

    # === vol_ratio distribution (the effect of the epsilon guard) ==========
    vr = feat["vol_ratio"]
    vol20 = feat["brent_vol20"]
    vr_stats = {
        "min": float(vr.min()),
        "median": float(vr.median()),
        "max": float(vr.max()),
        "n_nan": int(vr.isna().sum()),
        "brent_vol20_min": float(vol20.min()),
        "epsilon": EPS,
    }
    print("=== vol_ratio dagilimi ===")
    print(f"  min    : {vr_stats['min']:.6f}")
    print(f"  medyan : {vr_stats['median']:.6f}")
    print(f"  maks   : {vr_stats['max']:.6f}")
    print(f"  brent_vol20 en kucuk degeri: {vr_stats['brent_vol20_min']:.8f} "
          f"(epsilon={EPS:g})")
    print()

    # === Correlation summary ===============================================
    pairs_full = high_corr_pairs(feat, cols, CORR_THRESHOLD)
    warmup_mask = feat["Date_parsed"] < pd.Timestamp(WARMUP_END)
    pairs_warmup = high_corr_pairs(feat.loc[warmup_mask], cols, CORR_THRESHOLD)

    print(f"=== |r| > {CORR_THRESHOLD} ozellik ciftleri (TANILAMA AMACLI) ===")
    print("UYARI: asagidaki 'tum veri' listesi 2008-2026'nin tamamindan hesaplandi.")
    print("Bu listeye bakarak ozellik SILMEK test donemi bilgisini secime tasir ve")
    print("sizintidir. Eleme gerekirse isinma donemi listesi ya da fold-ici train")
    print("korelasyonu kullanilmalidir. Bu asamada hicbir ozellik SILINMIYOR.")
    print(f"\nTum veri (2008-2026): {len(pairs_full)} cift")
    if pairs_full:
        print(pd.DataFrame(pairs_full).to_string(index=False))
    print(f"\nIsinma donemi (2008-2011, ilk egitim seti): {len(pairs_warmup)} cift")
    if pairs_warmup:
        print(pd.DataFrame(pairs_warmup).to_string(index=False))
    print()

    # === Writing ===========================================================
    feat.to_csv(OUT_DIR / "features.csv", index=False)

    report = {
        "row_count": int(len(feat)),
        "n_features": len(cols),
        "n_csv_columns": int(feat.shape[1]),
        "date_min": str(feat["Date_parsed"].min().date()),
        "date_max": str(feat["Date_parsed"].max().date()),
        "params": {
            "z_window": Z_WINDOW,
            "spike_z": SPIKE_Z,
            "regime_z": REGIME_Z,
            "vol_ratio_epsilon": EPS,
            "corr_threshold": CORR_THRESHOLD,
            "warmup_end": WARMUP_END,
        },
        "no_fit_calls": True,
        "no_scaling_applied": True,
        "rows_dropped": 0,
        "prefix_invariance": prefix_results,
        "targets_date_aligned": targets_aligned,
        "inf_counts": inf_counts,
        "constant_columns": constant_cols,
        "vol_ratio_stats": vr_stats,
        "features": inv_rows,
        "high_corr_pairs_full": pairs_full,
        "high_corr_pairs_warmup": pairs_warmup,
        "high_corr_note": (
            "high_corr_pairs_full tum veriden hesaplandi ve YALNIZCA tanilama "
            "amaclidir; ozellik secimi icin kullanilirsa sizinti olur. Eleme "
            "gerekirse high_corr_pairs_warmup ya da fold-ici train korelasyonu "
            "kullanilmalidir."
        ),
    }
    with open(OUT_DIR / "build_features_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Yazildi: {OUT_DIR / 'features.csv'} "
          f"({len(feat)} satir x {feat.shape[1]} sutun)")
    print(f"Rapor: {OUT_DIR / 'build_features_report.json'}")


if __name__ == "__main__":
    main()
