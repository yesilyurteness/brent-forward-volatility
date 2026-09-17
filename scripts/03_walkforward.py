"""Expanding-window walk-forward: four horizons (h=5, 22, 66, 126), XGBoost + 2 baselines.

The BiLSTM, the hybrids, GARCH and HAR are added in later stages.

FOLD STRUCTURE
--------------
15 folds, test years 2012...2026. Fold k's training set is ALL rows before 1 January of
the test year (an expanding window). The window length is NOT OPTIMIZED (CLAUDE.md
Critical Rule 5). The fold structure is THE SAME at every horizon; the only things that
change with the horizon are the embargo length and the target column.

EMBARGO
-------
The last h rows of the training slice are dropped; the embargo length always equals that
model's forecast horizon (Critical Rule 4). The reason: row t's target looks at the
interval t+1...t+h, so the target of the last h training rows spills into the test period.
The embargo is applied BEFORE the NaN cleanup and on the chronological slice, because the
leakage criterion depends on calendar position, not on whether a row is valid. At h=126
this means deliberately sacrificing about six months of training data in every fold.

PREPROCESSING -- ALL OF IT FIT ON TRAINING DATA ONLY (Critical Rule 2)
-----------------------------------------------------------------------
1. log1p: applied to a pre-declared list of strictly positive features. The list is NOT
   derived FROM THE DATA; it comes from domain knowledge, and no parameter is estimated.
2. Winsorization: the bounds are computed ONLY from the quantiles of the training slice,
   and both train and test are clipped to those bounds. It is applied to the FEATURES
   only. The target is NOT winsorized -- clipping the target would hide exactly the
   extreme periods we want to examine, such as 2020 and 2022, and would distort the
   evaluation.
3. MinMaxScaler: fit on training only, transform applied to both. Test values may fall
   outside [0,1]; that is correct and is NOT CLIPPED.

THERE IS NO EARLY STOPPING
--------------------------
Early stopping requires a validation set. Using the test year for that purpose would be a
direct violation of Critical Rule 5.

CAPACITY FOLLOWS A PRE-DECLARED RULE
-------------------------------------
Hyperparameters are NOT SEARCHED. The number of trees, depth, min_child_weight and
reg_lambda are taken from a fixed three-tier table indexed by the fold's effective number
of independent observations (training rows / h) -- see CAPACITY_TIERS. The rule is applied
per fold; as the fold grows, effective observations increase and capacity rises
automatically. Tier selection depends only on the training row count and h, and never
looks at test data.

This is a ONE-OFF diagnostic run: it measures the question "does the imbalance between
capacity and effective sample size explain the long-horizon failure?" The thresholds and
parameters will NOT BE ITERATED ON by looking at the result; doing so would tie model
selection to test performance. The real selection is done with Optuna, on a validation set
split off from the last year of the training data.

THE TARGET TRANSFORM AND DUAN SMEARING
---------------------------------------
Because volatility is strictly positive and right-skewed, the model is trained in LOG
space. Transforming back with exp() gives the MEDIAN, not the mean; this is the well-known
retransformation bias and it pulls RMSE down systematically. It is corrected with a Duan
smearing factor: factor = mean(exp(log-space residuals)), computed from training residuals
only.

There are two target parameterizations (--target-mode):

  "ratio" (THE DEFAULT): y' = log(target_vol_h) - log(past_vol_h)
      The model predicts the DEVIATION from the past-volatility baseline.
      Transforming back: pred = past_vol_h * exp(y_hat) * smearing
      The rationale: the model does not have to learn persistence from scratch. With no
      signal, the model predicts about 0 and STRUCTURALLY reproduces the baseline; it
      cannot fall below it. In addition, the differenced series is far less
      autocorrelated, which reduces the pressure to memorize.

  "log": y' = log(target_vol_h)
      The earlier version. Kept for comparison.

past_vol_h is IDENTICAL to the series the baseline uses: rolling(h).std().shift(1), i.e.
causal. This does not make the target look ahead -- the numerator (target_vol_h) looks
forward, the denominator (past_vol_h) looks back, and the denominator is already known at
time t.

THE 2026 PARTIAL-YEAR RULE
--------------------------
The 2026 data is available only through September. As the horizon lengthens, the number of
valid targets in this fold falls rapidly, and because of overlapping windows the effective
number of independent observations drops to roughly 1 at h=66 and h=126. At those two
horizons the 2026 fold is EXCLUDED from the main metric average, but it is still computed
and reported separately as a "partial year, low statistical power" footnote. At h=5 and
h=22 it is included normally.

METRICS
-------
The main metrics are RMSE and MAE. The secondary metric is R2_oos (out-of-sample R2
against the training mean). Standard R2 is reported as a footnote; its reference is the
test slice's own mean, which cannot be known ex ante, which is why it comes out negative in
calm years. For details see "Metric reporting rule" in CLAUDE.md.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012      # the 2008-2011 warm-up period is the first training set
LAST_TEST_YEAR = 2026
EXPECTED_N_FOLDS = 15

# CLAUDE.md "2026 partial-year rule" -- see the module docstring.
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
SMEARING_WEAK_THRESHOLD = 1.02

# Target parameterization -- see the module docstring.
TARGET_MODE_DEFAULT = "ratio"

# Parameters shared by every tier, none of them tuned.
XGB_COMMON = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 4,
}

# ===========================================================================
# THE TIERED CAPACITY RULE
# ===========================================================================
# Model capacity is chosen by the fold's EFFECTIVE NUMBER OF INDEPENDENT OBSERVATIONS
# (training rows / h). The rule is PRE-DECLARED and is applied separately for every
# fold; as the fold grows, effective observations increase and capacity rises
# automatically.
#
# NO LEAKAGE: the tier depends only on the training row count and h. Both are quantities
# known at forecast time; NO information from the test data, the test metric or the test
# period is CONSULTED (Critical Rule 5).
#
# THIS IS A ONE-OFF DIAGNOSTIC RUN. Its purpose is to measure the question "does the
# capacity / effective-sample imbalance explain the long-horizon failure?" The thresholds
# and parameters will NOT BE ITERATED ON in light of the result -- doing so would mean
# selecting the model by test performance. The real hyperparameter selection comes from
# Optuna, on a VALIDATION set split off from the last year of the training data.
CAPACITY_TIERS = [
    # (minimum effective observations, tier name, parameters)
    (100, "yuksek", {"n_estimators": 400, "max_depth": 4,
                     "min_child_weight": 1.0, "reg_lambda": 1.0}),
    (30, "orta", {"n_estimators": 150, "max_depth": 3,
                  "min_child_weight": 20.0, "reg_lambda": 10.0}),
    (0, "dusuk", {"n_estimators": 80, "max_depth": 2,
                  "min_child_weight": 50.0, "reg_lambda": 25.0}),
]


def select_capacity(n_effective):
    """Returns the tier name and XGBoost parameters for a given effective observation count."""
    for threshold, name, params in CAPACITY_TIERS:
        if n_effective >= threshold:
            return name, {**XGB_COMMON, **params}
    raise RuntimeError("CAPACITY_TIERS son elemani 0 esikli olmali")

MODELS = ("xgboost", "train_mean", "past_vol")

# --- Columns to apply log1p to: from domain knowledge, NOT from the data ----
_SERIES = ("brent", "ovx", "gprd", "gprd_threat")
LOG_FEATURES = (
    [f"{s}_lag{i}" for s in _SERIES for i in range(1, 6)]
    + [f"{s}_ema{n}" for s in _SERIES for n in (5, 10, 20)]
    + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
    + ["vol_ratio", "vol5_vol60", "vol20_vol126",
       "threat_ratio", "ovx_x_gprd", "ovx_x_gprd_threat"]
)


# ===========================================================================
# Preprocessing: fit on training data only
# ===========================================================================
def _apply_log(X, log_cols):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def fit_preproc(X_train, log_cols):
    """Learns the winsorization bounds and the scaler from the TRAINING SLICE ONLY."""
    Xl = _apply_log(X_train, log_cols)
    lo = Xl.quantile(WINSOR_LOWER)
    hi = Xl.quantile(WINSOR_UPPER)
    Xc = Xl.clip(lower=lo, upper=hi, axis=1)
    scaler = MinMaxScaler().fit(Xc)
    return {"lo": lo, "hi": hi, "scaler": scaler, "log_cols": log_cols}


def apply_preproc(X, params):
    """Applies the learned parameters. No fit() is called here."""
    Xl = _apply_log(X, params["log_cols"])
    Xc = Xl.clip(lower=params["lo"], upper=params["hi"], axis=1)
    return pd.DataFrame(
        params["scaler"].transform(Xc), index=X.index, columns=X.columns
    )


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(y_true, y_pred, train_mean):
    """RMSE / MAE / R2 / R2_oos in raw volatility units.

    rmse, mae : the MAIN metrics. Same unit as the target, no reference ambiguity.
    r2_oos    : the SECONDARY metric. The reference is the mean of the training target --
                a quantity genuinely known at forecast time. For the train_mean baseline
                it comes out as exactly 0 by definition; that serves as a sanity check.
    r2        : the FOOTNOTE metric. The reference is the test slice's OWN mean; that is
                an ex-post quantity, and in calm years the shrinking SST produces large
                negative values. It is not a basis for decisions.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    sst_own = float(np.sum((y_true - y_true.mean()) ** 2))
    sst_train = float(np.sum((y_true - train_mean) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2_oos": float(1.0 - sse / sst_train) if sst_train > 0 else float("nan"),
        "r2": float(1.0 - sse / sst_own) if sst_own > 0 else float("nan"),
        "sst_own": sst_own,
        "n": int(len(y_true)),
    }


# ===========================================================================
# Walk-forward for a single horizon
# ===========================================================================
def run_horizon(h, df, feature_cols, log_cols, daily_ret, test_years,
                target_mode=TARGET_MODE_DEFAULT, verbose=True):
    df = df.copy()
    df["y"] = df[f"target_vol_{h}"]
    # Past-volatility baseline: the realized volatility of the past h days, recomputed
    # from the raw price; this keeps it independent of the feature set and gives the
    # correct window for every h.
    df["pred_past_vol"] = daily_ret.rolling(h).std().shift(1).values

    pred_frames, fold_records = [], []

    for fold_id, test_year in enumerate(test_years, start=1):
        # --- (a) Chronological split --------------------------------------
        train_idx_all = df.index[df["year"] < test_year]
        test_idx_all = df.index[df["year"] == test_year]
        assert len(train_idx_all) > 0 and len(test_idx_all) > 0, \
            f"h={h} fold {fold_id}: bos dilim"
        assert train_idx_all.max() < test_idx_all.min(), \
            f"h={h} fold {fold_id}: train test'ten sonra gelen satir iceriyor"

        # --- (b) Embargo: drop the last h training rows --------------------
        embargo_idx = train_idx_all[-h:]
        train_idx = train_idx_all[:-h]
        assert len(train_idx) > 0, f"h={h} fold {fold_id}: embargo train'i tuketti"
        # The last remaining training row's target (t+1...t+h) must not reach the first test row.
        assert train_idx.max() + h < test_idx_all.min(), (
            f"h={h} fold {fold_id}: embargo yetersiz -- son train satiri "
            f"{train_idx.max()} + h={h}, ilk test satiri {test_idx_all.min()}"
        )
        assert len(set(train_idx) & set(test_idx_all)) == 0, \
            f"h={h} fold {fold_id}: train ve test kesisiyor"

        # --- NaN cleanup (AFTER the embargo) -------------------------------
        # pred_past_vol is both a baseline and the denominator of the "ratio" target, so
        # it must be valid in both modes; it is added to the list explicitly.
        needed = feature_cols + ["y", "pred_past_vol"]
        tr_ok = df.loc[train_idx, needed].notna().all(axis=1)
        te_ok = df.loc[test_idx_all, needed].notna().all(axis=1)
        tr = df.loc[train_idx[tr_ok.values]]
        te = df.loc[test_idx_all[te_ok.values]]
        assert len(tr) > 0 and len(te) > 0, \
            f"h={h} fold {fold_id}: NaN sonrasi bos dilim"

        X_tr, y_tr = tr[feature_cols], tr["y"].to_numpy(dtype="float64")
        X_te, y_te = te[feature_cols], te["y"].to_numpy(dtype="float64")
        pv_tr = tr["pred_past_vol"].to_numpy(dtype="float64")
        pv_te = te["pred_past_vol"].to_numpy(dtype="float64")
        assert (y_tr > 0).all() and (y_te > 0).all(), \
            "Hedef pozitif olmali (log donusumu icin)"
        assert (pv_tr > 0).all() and (pv_te > 0).all(), \
            "past_vol pozitif olmali (ratio hedefinin paydasi ve log argumani)"

        # --- (c) Preprocessing: fit on training data only ------------------
        pp = fit_preproc(X_tr, log_cols)
        X_tr_s = apply_preproc(X_tr, pp)
        X_te_s = apply_preproc(X_te, pp)

        # --- (d) XGBoost, in log space -------------------------------------
        # "ratio": the DEVIATION from persistence is learned. The denominator (past_vol)
        # is a causal quantity known at time t, so it does not make the target look ahead.
        # "log": the plain log target (the earlier version).
        if target_mode == "ratio":
            y_tr_fit = np.log(y_tr) - np.log(pv_tr)
        elif target_mode == "log":
            y_tr_fit = np.log(y_tr)
        else:
            raise ValueError(f"Bilinmeyen target_mode: {target_mode}")
        # Capacity tier: derived only from the training row count and h.
        n_effective = len(tr) / h
        tier_name, xgb_params = select_capacity(n_effective)
        model = XGBRegressor(**xgb_params)
        model.fit(X_tr_s, y_tr_fit)

        # --- The Duan smearing factor -------------------------------------
        # CAUTION: these residuals are IN-SAMPLE -- they come from the model's own
        # predictions on its training data. When XGBoost overfits the training set, the
        # residuals come out smaller than they really are, the factor approaches 1 and the
        # correction stays weaker than it should be. A factor < 1.02 should raise the
        # suspicion that the correction is inadequate. There is NO LEAKAGE: only training
        # data is used and the test data is never consulted.
        #
        # TODO (the Optuna stage): once the last year of training is split off as
        # validation, compute the smearing factor from THOSE VALIDATION residuals. Because
        # validation residuals are out-of-sample, that removes the in-sample bias here
        # entirely. An embargo of length h must be applied at the validation boundary as
        # well.
        resid_log = y_tr_fit - model.predict(X_tr_s)
        smearing = float(np.mean(np.exp(resid_log)))
        resid_log_std = float(np.std(resid_log, ddof=1))

        # --- (e) Prediction, in raw volatility units ----------------------
        # In "ratio" mode the denominator is multiplied back in:
        # pred = past_vol * exp(y_hat) * smearing. When y_hat is about 0 the prediction
        # equals the past-volatility baseline; if the model finds no signal it reproduces
        # the baseline and structurally cannot fall below it.
        raw_pred = np.exp(model.predict(X_te_s)) * smearing
        pred_xgb = raw_pred * pv_te if target_mode == "ratio" else raw_pred
        train_mean = float(y_tr.mean())
        preds = {
            "xgboost": pred_xgb,
            "train_mean": np.full(len(te), train_mean),
            "past_vol": pv_te,
        }

        # The 2026 partial-year rule: at h=66 and h=126 this fold is left out of the main average.
        include_main = not (test_year == PARTIAL_YEAR and h in EXCLUDE_2026_HORIZONS)

        fold_metrics = {}
        for name, p in preds.items():
            ok = np.isfinite(p)
            fold_metrics[name] = compute_metrics(y_te[ok], p[ok], train_mean)

        pred_frames.append(pd.DataFrame({
            "horizon": h,
            "Date": te["Date"].values,
            "Date_parsed": te["Date_parsed"].values,
            "fold": fold_id,
            "test_year": test_year,
            "include_in_main": include_main,
            "y_true": y_te,
            "pred_xgboost": preds["xgboost"],
            "pred_train_mean": preds["train_mean"],
            "pred_past_vol": preds["past_vol"],
        }))

        fold_records.append({
            "horizon": h,
            "fold": fold_id,
            "test_year": test_year,
            "target_mode": target_mode,
            "include_in_main": include_main,
            # The first/last training row actually used, AFTER the NaN cleanup.
            # (Had this been train_idx.min(), we would have reported a date that is never
            # used, because of the warm-up NaNs.)
            "train_date_start": str(tr["Date_parsed"].iloc[0].date()),
            "train_date_end": str(tr["Date_parsed"].iloc[-1].date()),
            "test_date_start": str(te["Date_parsed"].iloc[0].date()),
            "test_date_end": str(te["Date_parsed"].iloc[-1].date()),
            "n_train_before_embargo": int(len(train_idx_all)),
            "n_embargoed": int(len(embargo_idx)),
            "embargo_date_start": str(df.loc[embargo_idx.min(), "Date_parsed"].date()),
            "embargo_date_end": str(df.loc[embargo_idx.max(), "Date_parsed"].date()),
            "n_train_after_embargo": int(len(train_idx)),
            "n_train_dropped_nan": int(len(train_idx) - len(tr)),
            "n_train_final": int(len(tr)),
            "n_test_before_nan": int(len(test_idx_all)),
            "n_test_dropped_nan": int(len(test_idx_all) - len(te)),
            "n_test_final": int(len(te)),
            # The effective number of independent observations: because the target is
            # computed from OVERLAPPING h-day windows, consecutive rows carry almost the
            # same information in their targets. Roughly every h rows amount to one
            # independent observation. This shows that at long horizons the real sample
            # size is far smaller than the row count, and it must be read together with
            # the model's capacity.
            "n_train_effective": round(n_effective, 1),
            "capacity_tier": tier_name,
            "n_estimators": xgb_params["n_estimators"],
            "max_depth": xgb_params["max_depth"],
            "min_child_weight": xgb_params["min_child_weight"],
            "reg_lambda": xgb_params["reg_lambda"],
            "n_test_effective": round(len(te) / h, 1),
            "train_mean_target": train_mean,
            "smearing": smearing,
            "resid_log_std": resid_log_std,
            "metrics": fold_metrics,
        })

        if verbose:
            flag = "" if include_main else "  [2026 kismi yil -> ana ortalamada YOK]"
            print(f"  h={h:3d} fold {fold_id:2d} | test {test_year} | "
                  f"train {len(tr):4d} (embargo -{len(embargo_idx)}, "
                  f"NaN -{len(train_idx) - len(tr)}) | test {len(te):3d} "
                  f"(NaN -{len(test_idx_all) - len(te)}) | "
                  f"etkin {n_effective:6.1f} -> {tier_name:6s} | "
                  f"smear {smearing:.4f} | "
                  f"RMSE {fold_metrics['xgboost']['rmse']:.6f}{flag}")

    return pd.concat(pred_frames, ignore_index=True), fold_records


def metrics_table(fold_records):
    rows = []
    for rec in fold_records:
        for name, m in rec["metrics"].items():
            rows.append({
                "horizon": rec["horizon"],
                "fold": rec["fold"],
                "test_year": rec["test_year"],
                "include_in_main": rec["include_in_main"],
                "model": name,
                "n_test": m["n"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2_oos": m["r2_oos"],
                "r2": m["r2"],
                "sst_own": m["sst_own"],
            })
    return pd.DataFrame(rows)


def aggregate(metrics_df, preds_df, h):
    """The main aggregation: only the folds flagged include_in_main."""
    rows = []
    main_m = metrics_df[metrics_df["include_in_main"]]
    main_p = preds_df[preds_df["include_in_main"]]
    for name in MODELS:
        sub = main_m[main_m["model"] == name]
        y_all = main_p["y_true"].to_numpy(dtype="float64")
        p_all = main_p[f"pred_{name}"].to_numpy(dtype="float64")
        ok = np.isfinite(p_all)
        pooled = compute_metrics(y_all[ok], p_all[ok], float(y_all[ok].mean()))
        rows.append({
            "horizon": h,
            "model": name,
            "n_folds": int(len(sub)),
            "rmse_fold_mean": float(sub["rmse"].mean()),
            "mae_fold_mean": float(sub["mae"].mean()),
            "r2_oos_fold_mean": float(sub["r2_oos"].mean()),
            "r2_fold_mean": float(sub["r2"].mean()),
            "rmse_pooled": pooled["rmse"],
            "mae_pooled": pooled["mae"],
            "r2_pooled": pooled["r2"],
            "n_pooled": pooled["n"],
        })
    return rows


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None,
                    help="Duman testi icin ilk N fold (varsayilan: hepsi)")
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS,
                    help=f"Calisacak ufuklar (varsayilan: {HORIZONS})")
    ap.add_argument("--target-mode", choices=["ratio", "log"],
                    default=TARGET_MODE_DEFAULT,
                    help="ratio: past_vol'den sapmayi tahmin et (varsayilan). "
                         "log: duz log hedef (onceki surum).")
    ap.add_argument("--suffix", default="",
                    help="Cikti dosya adlarina eklenecek ek (karsilastirma icin)")
    args = ap.parse_args()

    t0 = time.time()

    # --- Data loading and alignment ----------------------------------------
    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw), "Kaynak dosyalarin satir sayisi uyusmuyor"
    assert (feat["Date"].values == tgt["Date"].values).all(), \
        "features ve targets Date sutunlari eslesmiyor"
    assert (feat["Date"].values == raw["Date"].values).all(), \
        "features ve ham veri Date sutunlari eslesmiyor"
    assert feat["Date_parsed"].is_monotonic_increasing, "Tarihler sirali degil"

    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in LOG_FEATURES if c in feature_cols]
    assert len(log_cols) == len(LOG_FEATURES), (
        "LOG_FEATURES icinde features.csv'de olmayan sutun var: "
        f"{set(LOG_FEATURES) - set(feature_cols)}"
    )

    df = feat.copy()
    for h in HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values
    df["year"] = df["Date_parsed"].dt.year
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))

    test_years = list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    assert len(test_years) == EXPECTED_N_FOLDS, \
        f"Beklenen fold sayisi {EXPECTED_N_FOLDS}, bulunan {len(test_years)}"
    if args.max_folds is not None:
        test_years = test_years[:args.max_folds]
        print(f"[DUMAN TESTI] yalnizca ilk {len(test_years)} fold\n")

    print(f"=== Walk-forward | ufuklar {args.horizons} | {len(test_years)} fold | "
          f"{len(feature_cols)} ozellik | hedef modu: {args.target_mode} ===")
    if args.target_mode == "ratio":
        print("Model past_vol'den SAPMAYI tahmin ediyor; y_hat~0 ise tahmin "
              "baseline'a esitlenir.\n")
    else:
        print("Model duz log(vol_h) tahmin ediyor (onceki surum).\n")

    all_preds, all_folds, all_agg = [], [], []
    for h in args.horizons:
        print(f"--- h={h} ---")
        preds_df, fold_records = run_horizon(
            h, df, feature_cols, log_cols, daily_ret, test_years,
            target_mode=args.target_mode,
        )
        metrics_df = metrics_table(fold_records)
        agg_rows = aggregate(metrics_df, preds_df, h)

        suffix = f"h{h}{args.suffix}"
        preds_df.to_csv(OUT_DIR / f"wf_predictions_{suffix}.csv", index=False)
        metrics_df.to_csv(OUT_DIR / f"wf_metrics_{suffix}.csv", index=False)

        all_preds.append(preds_df)
        all_folds.extend(fold_records)
        all_agg.extend(agg_rows)
        print()

    metrics_all = metrics_table(all_folds)
    preds_all = pd.concat(all_preds, ignore_index=True)
    agg_all = pd.DataFrame(all_agg)
    fold_all = pd.DataFrame([
        {k: v for k, v in r.items() if k != "metrics"} for r in all_folds
    ])

    # ===================================================================
    # Reporting
    # ===================================================================
    pd.set_option("display.width", 200)

    print("=== Fold satir sayilari (ufuk basina embargo etkisi) ===")
    piv = fold_all.pivot_table(
        index="horizon",
        values=["n_embargoed", "n_train_final", "n_test_final"],
        aggfunc={"n_embargoed": "max", "n_train_final": ["min", "max"],
                 "n_test_final": ["min", "max"]},
    )
    print(piv.to_string())
    print()
    print("=== ETKIN BAGIMSIZ GOZLEM SAYISI (ortusen hedef pencereleri) ===")
    print("Hedef, gelecek h gunun ORTUSEN penceresinden hesaplanir; ardisik satirlarin")
    print("hedefleri neredeyse ayni bilgiyi tasir. Kabaca her h satir = 1 bagimsiz")
    print("gozlem. Model kapasitesiyle birlikte okunmalidir:")
    print(f"XGBoost {len(feature_cols)} ozellik kullaniyor; agac sayisi ve derinlik "
          "kademeli kapasite kuraliyla fold basina secilir.")
    eff = fold_all.groupby("horizon").agg(
        train_satir_min=("n_train_final", "min"),
        train_satir_maks=("n_train_final", "max"),
        train_etkin_min=("n_train_effective", "min"),
        train_etkin_maks=("n_train_effective", "max"),
        test_etkin_min=("n_test_effective", "min"),
        test_etkin_maks=("n_test_effective", "max"),
    )
    eff["ozellik_sayisi"] = len(feature_cols)
    eff["ozellik_/_etkin_gozlem"] = (
        len(feature_cols) / eff["train_etkin_maks"]
    ).round(2)
    print(eff.to_string())
    crowded = eff[eff["train_etkin_maks"] < len(feature_cols)]
    if len(crowded):
        print()
        print("SINIRLILIK: asagidaki ufuklarda EN BUYUK fold'un etkin bagimsiz gozlem")
        print("sayisi bile ozellik sayisindan azdir. Bu ufuklardaki sonuclar asiri")
        print("uyuma acik kabul edilmeli ve makalede sinirlilik olarak raporlanmalidir.")
        print(f"Etkilenen ufuklar: {crowded.index.tolist()}")
    print()

    print("=== KAPASITE KADEMELERI (fold basina, onceden ilan edilmis kural) ===")
    print("Kademe = f(train satiri / h). Test verisine bakilmaz. TEK SEFERLIK teshis;")
    print("bu esikler ve parametreler uzerinde iterasyon yapilmayacak.")
    for thr, name, prm in CAPACITY_TIERS:
        print(f"  etkin gozlem >= {thr:3d} -> {name:6s}: {prm}")
    print()
    tier_piv = fold_all.pivot(index="test_year", columns="horizon",
                              values="capacity_tier")
    print("Fold x ufuk kademe haritasi:")
    print(tier_piv.to_string())
    print()
    print("Kademe dagilimi:")
    print(fold_all.groupby(["horizon", "capacity_tier"]).size()
          .rename("fold_sayisi").reset_index().to_string(index=False))
    print()

    print("2026 fold'unda gecerli test gozlemi (kismi yil):")
    p26 = fold_all[fold_all["test_year"] == PARTIAL_YEAR][
        ["horizon", "n_test_before_nan", "n_test_dropped_nan", "n_test_final",
         "include_in_main"]
    ]
    print(p26.to_string(index=False))
    print()

    print("=== Duan smearing katsayilari ===")
    sm = fold_all.pivot(index="test_year", columns="horizon", values="smearing")
    print(sm.to_string(float_format=lambda v: f"{v:.4f}"))
    weak = fold_all[fold_all["smearing"] < SMEARING_WEAK_THRESHOLD]
    if len(weak):
        print(f"\nUYARI: {len(weak)}/{len(fold_all)} fold'da katsayi < "
              f"{SMEARING_WEAK_THRESHOLD}. Ornekleme-ici artiklar modelin train'e")
        print("asiri uyumu nedeniyle kucuk cikiyor olabilir; duzeltme yetersiz kaliyor.")
        print("Optuna asamasinda katsayi validation artiklarindan hesaplanacak.")
        print("Etkilenen (ufuk, yil) ciftleri:",
              list(zip(weak["horizon"].tolist(), weak["test_year"].tolist())))
    print()

    print("=== ANA METRIKLER: RMSE ve MAE (fold ortalamasi) ===")
    show = agg_all[["horizon", "model", "n_folds", "rmse_fold_mean",
                    "mae_fold_mean", "r2_oos_fold_mean"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()
    print("XGBoost'un baseline'lara gore RMSE kazanci (fold ortalamasi, negatif=iyi):")
    gain = []
    for h in args.horizons:
        a = agg_all[agg_all["horizon"] == h].set_index("model")
        gain.append({
            "horizon": h,
            "vs_past_vol_%": 100 * (a.loc["xgboost", "rmse_fold_mean"]
                                    / a.loc["past_vol", "rmse_fold_mean"] - 1),
            "vs_train_mean_%": 100 * (a.loc["xgboost", "rmse_fold_mean"]
                                      / a.loc["train_mean", "rmse_fold_mean"] - 1),
        })
    print(pd.DataFrame(gain).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()

    print("=== DIPNOT METRIGI: standart R2 (karar dayanagi DEGIL) ===")
    print("Referansi test yilinin KENDI ortalamasidir; bu deger tahmin aninda")
    print("bilinemez. Sakin yillarda fold SST'si kuculdugu icin buyuk negatif degerler")
    print("uretir. Bkz. CLAUDE.md 'Metrik raporlama kurali'.")
    r2piv = metrics_all[metrics_all["model"] == "xgboost"].pivot(
        index="test_year", columns="horizon", values="r2")
    print(r2piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("Fold SST'lerinin ufuk basina yayilimi (standart R2'nin kararsizligi):")
    sstp = metrics_all[metrics_all["model"] == "xgboost"].groupby("horizon")["sst_own"]
    print(pd.DataFrame({"sst_min": sstp.min(), "sst_max": sstp.max(),
                        "maks/min_kat": sstp.max() / sstp.min()}).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print()

    # --- The 2026 partial-year footnote -----------------------------------
    excluded = metrics_all[~metrics_all["include_in_main"]]
    print("=== 2026 KISMI YIL DIPNOTU ===")
    if len(excluded):
        print("Asagidaki fold'lar ana metrik ortalamasina DAHIL EDILMEDI.")
        print("Kismi yil, dusuk istatistiksel guc. Yalnizca bilgi amaclidir;")
        print("model karsilastirmasi veya secimi icin KULLANILMAZ.")
        print(excluded[["horizon", "test_year", "model", "n_test", "rmse", "mae",
                        "r2_oos"]].to_string(index=False,
                                             float_format=lambda v: f"{v:.6f}"))
    else:
        print("Bu calistirmada haric tutulan fold yok "
              f"(yalnizca h={sorted(EXCLUDE_2026_HORIZONS)} icin devreye girer).")
    print()

    # ===================================================================
    # Writing
    # ===================================================================
    sfx = args.suffix
    metrics_all.to_csv(OUT_DIR / f"wf_metrics_all{sfx}.csv", index=False)
    agg_all.to_csv(OUT_DIR / f"wf_aggregate_all{sfx}.csv", index=False)
    preds_all.to_csv(OUT_DIR / f"wf_predictions_all{sfx}.csv", index=False)

    runtime = time.time() - t0
    summary = {
        "horizons": args.horizons,
        "seed": SEED,
        "n_folds_per_horizon": len(test_years),
        "test_years": test_years,
        "n_features": len(feature_cols),
        "config": {
            "expanding_window": True,
            "window_length_optimized": False,
            "embargo_equals_horizon": True,
            "winsor_bounds": [WINSOR_LOWER, WINSOR_UPPER],
            "winsor_applied_to_target": False,
            "scaler": "MinMaxScaler (train-only fit)",
            "log1p_features": log_cols,
            "target_mode": args.target_mode,
            "target_transform": (
                "ratio: y=log(vol_h)-log(past_vol_h), geri donus "
                "past_vol*exp(y_hat)*smearing"
                if args.target_mode == "ratio"
                else "log: y=log(vol_h), geri donus exp(y_hat)*smearing"
            ),
            "smearing_residuals": "in-sample (train)",
            "early_stopping": False,
            "hyperparameters_tuned": False,
            "capacity_rule": {
                "note": (
                    "Kademe yalnizca train satir sayisi ve h'den turer; test "
                    "verisine bakilmaz. TEK SEFERLIK TESHIS -- esikler ve "
                    "parametreler uzerinde iterasyon YAPILMAYACAK. Gercek secim "
                    "Optuna + validation ile gelecek."
                ),
                "tiers": [
                    {"min_effective_obs": t, "name": n, "params": p}
                    for t, n, p in CAPACITY_TIERS
                ],
                "common": XGB_COMMON,
            },
            "partial_year": PARTIAL_YEAR,
            "exclude_2026_horizons": sorted(EXCLUDE_2026_HORIZONS),
        },
        "metric_policy": {
            "primary": ["rmse", "mae"],
            "secondary": "r2_oos (referans: fold'un train hedef ortalamasi)",
            "footnote": "r2 (referans: test diliminin kendi ortalamasi, ex-post)",
        },
        "effective_sample_size": {
            "note": (
                "Hedef h gunluk ORTUSEN pencerelerden hesaplandigi icin ardisik "
                "satirlarin hedefleri neredeyse ayni bilgiyi tasir; kabaca her h "
                "satir 1 bagimsiz gozleme denk gelir. Uzun ufuklarda etkin ornek "
                "buyuklugu ozellik sayisinin altina duser -- makalede SINIRLILIK "
                "olarak raporlanmalidir."
            ),
            "n_features": len(feature_cols),
            "by_horizon": eff.reset_index().to_dict(orient="records"),
        },
        "aggregate": all_agg,
        "excluded_2026_footnote": excluded.to_dict(orient="records"),
        "smearing_weak_folds": int(len(weak)),
        "folds": all_folds,
        "runtime_seconds": round(runtime, 2),
    }
    with open(OUT_DIR / f"wf_summary_all{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Yazildi: wf_predictions_h*{sfx}.csv, wf_metrics_h*{sfx}.csv")
    print(f"Yazildi: wf_metrics_all{sfx}.csv ({len(metrics_all)} satir), "
          f"wf_aggregate_all{sfx}.csv ({len(agg_all)} satir)")
    print(f"Rapor  : wf_summary_all{sfx}.json")
    print(f"Sure   : {runtime:.1f} saniye")


if __name__ == "__main__":
    main()
