"""Optuna + validation infrastructure: within-fold validation, search, validation-based smearing.

A continuation of 03_walkforward.py. The difference: hyperparameters are no longer taken
FIXED from the pre-declared capacity table but searched with Optuna on each fold's own
VALIDATION slice.

NOTE: fit_preproc / apply_preproc / compute_metrics / CAPACITY_TIERS are verbatim copies
of the definitions in 03_walkforward.py. When either file is changed, the two MUST BE KEPT
IN SYNC.

THE WITHIN-FOLD SPLIT
---------------------
Each fold's training slice is divided into three:

    [ train-proper ][ inner embargo (h) ][ validation ]  ||  [ test year ]
                                                         ^
                                       the outer embargo (h) sits here and takes
                                       the last h rows of the training data; because
                                       those rows fall at the end of the validation
                                       slice, the protection between validation and
                                       test is provided AUTOMATICALLY and is not
                                       applied a second time.

The inner embargo: the last h rows of train-proper are dropped, because the target of
those rows spills into the validation period. This is the same logic as at the train/test
boundary, with the same asserts.

THE VALIDATION LENGTH DERIVES FROM A FORMULA
---------------------------------------------
NOT a fixed table but a declared statistical threshold is used:

    "The validation slice must contain at least VAL_MIN_EFFECTIVE (=3) effective
     independent observations."

Because the target is computed from OVERLAPPING h-day windows, roughly every h rows amount
to 1 independent observation; so the criterion is n_val >= 3*h rows. The validation window
is grown one year at a time until that condition holds (up to MAX_VAL_YEARS). The resulting
table of lengths is the OUTPUT of this formula and is written up as such.

THE OPTUNA ELIGIBILITY RULE
---------------------------
A search is only meaningful if there is enough data. Two thresholds are pre-declared:

    train-proper effective observations >= 10  AND  validation effective observations >= 3

Folds that fail this DO NOT SEARCH; they use the parameters of the tiered capacity rule
from 03. This is not an evasion but an explicit record that there was no data to search
over in that fold, and it is reported per fold. The long-horizon results are therefore a
mixture of TWO DIFFERENT SELECTION PROCEDURES and are reported split into those groups.

LEAKAGE RULES
-------------
* Hyperparameter selection looks ONLY at the validation RMSE. The test year is never
  consulted at any stage (Critical Rule 5).
* The bounds of the search space are PRE-DECLARED and are not changed in light of the test
  result. The tier derives from train-proper's effective observation count.
* The validation length and the eligibility thresholds are pre-declared as well; both
  depend only on row counts and h, and never look at test data.
* All preprocessing (log1p, winsor, MinMax) is fit at each stage only on that stage's
  training slice: on train-proper during the search, and on train-proper + validation at
  retraining.

SMEARING NOW COMES FROM VALIDATION
-----------------------------------
The TODO from 03 is closed. The Duan smearing factor is computed from the model's
VALIDATION residuals. Because validation residuals are out-of-sample, the problem seen in
03 -- "the factor collapses to 1.0000" -- disappears.

One small residue: the validation RMSE used for RANKING also incorporates the smearing
factor computed from the same slice. Since it is a single scalar and affects every trial in
the same direction, it does not distort the ranking, but it does make the validation RMSE a
slightly OPTIMISTIC estimate of the test RMSE. The val_rmse values in the report should be
read with that note in mind.

RETRAINING
----------
After selection, the model is RETRAINED with the chosen hyperparameters on the union of
train-proper + validation. The rationale: the validation year is the year closest to the
test year and the most informative one in a regime-dependent series such as volatility; at
h=126, adding it back grows the training data by about 43%.

The cost: the smearing factor comes from the residuals of the model BEFORE retraining, so
it does not match the final model exactly. The mismatch is in the CONSERVATIVE direction --
a model trained on less data has larger residuals, so the factor comes out slightly high
and, on a right-skewed target, pushes the prediction up. It is written into the report as
the smearing_source field.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
EXPECTED_N_FOLDS = 15

PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
TARGET_MODE_DEFAULT = "ratio"

# ===========================================================================
# SMEARING SHRINKAGE
# ===========================================================================
#     S_corrected = 1 + w * (S_raw - 1),   w = n_eff / (n_eff + K)
#
# The rationale: the smearing factor is a MEAN estimated over the residuals and, like any
# mean, its variance is inversely proportional to the sample size, i.e. it scales with
# about 1/n_eff. Because the target windows overlap, the real sample size here is not the
# row count but the number of EFFECTIVE INDEPENDENT OBSERVATIONS (n_eff = rows / h). At
# long horizons n_eff falls to the order of 3-7, which makes the factor excessively
# noisy; the measured value had swung between 0.60 and 1.59, and because it multiplies
# EVERY test prediction as a single scalar, it fed straight through to RMSE.
#
# The form w = n_eff/(n_eff+K) is a classic shrinkage (James-Stein style) weight: when
# n_eff >> K, w -> 1 and the factor is left at its measured value; when n_eff << K, w -> 0
# and the factor is pulled to 1, i.e. to "no correction".
#
# K = 10: at n_eff = 10 the weight is exactly 0.5. This largely preserves the measured
# factors at short horizons (n_eff ~49 and w ~0.83 at h=5; n_eff ~10 and w ~0.50 at h=22)
# while pulling the long horizons (n_eff ~6.6 and w ~0.40 at h=66; n_eff ~5 and w ~0.33 at
# h=126) strongly towards 1. K is PRE-DECLARED and will not be tuned to the test result.
SMEARING_SHRINK_K = 10.0


def shrink_smearing(s_raw, n_eff):
    """Shrinks the raw smearing factor towards 1 according to the effective observation count."""
    w = n_eff / (n_eff + SMEARING_SHRINK_K)
    return 1.0 + w * (s_raw - 1.0), w

# --- Pre-declared thresholds (see the module docstring) ---------------------
VAL_MIN_EFFECTIVE = 3.0        # the validation slice must hold at least this many effective obs.
MAX_VAL_YEARS = 6              # a safety bound
OPTUNA_MIN_TP_EFFECTIVE = 10.0  # the search threshold for train-proper
MIN_TP_ROWS = 200              # below this, no validation split can be built at all
N_TRIALS_DEFAULT = 30
# If the selection signal falls below this threshold, a warning is printed saying "the
# best trial is not meaningfully better than the median": the selection is most likely
# noise.
SELECTION_SIGNAL_WEAK_PCT = 5.0

XGB_COMMON = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 4,
}

# --- Capacity tiers: kept IN SYNC with 03_walkforward.py -------------------
CAPACITY_TIERS = [
    (100, "yuksek", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
                     "min_child_weight": 1.0, "reg_lambda": 1.0,
                     "subsample": 0.8, "colsample_bytree": 0.8}),
    (30, "orta", {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
                  "min_child_weight": 20.0, "reg_lambda": 10.0,
                  "subsample": 0.8, "colsample_bytree": 0.8}),
    (0, "dusuk", {"n_estimators": 80, "max_depth": 2, "learning_rate": 0.05,
                  "min_child_weight": 50.0, "reg_lambda": 25.0,
                  "subsample": 0.8, "colsample_bytree": 0.8}),
]

# --- Search space: mirrors the logic of the capacity rule, bounds pre-declared --
SEARCH_SPACES = {
    "yuksek": {"n_estimators": (200, 800, 50), "max_depth": (3, 6),
               "learning_rate": (0.02, 0.15), "min_child_weight": (1.0, 20.0),
               "reg_lambda": (0.1, 20.0), "subsample": (0.6, 1.0),
               "colsample_bytree": (0.6, 1.0)},
    "orta": {"n_estimators": (60, 300, 20), "max_depth": (2, 4),
             "learning_rate": (0.02, 0.10), "min_child_weight": (5.0, 50.0),
             "reg_lambda": (1.0, 50.0), "subsample": (0.6, 1.0),
             "colsample_bytree": (0.5, 0.9)},
    "dusuk": {"n_estimators": (30, 150, 10), "max_depth": (1, 3),
              "learning_rate": (0.01, 0.08), "min_child_weight": (20.0, 100.0),
              "reg_lambda": (5.0, 100.0), "subsample": (0.5, 0.9),
              "colsample_bytree": (0.4, 0.8)},
}

MODELS = ("xgboost", "train_mean", "past_vol")

_SERIES = ("brent", "ovx", "gprd", "gprd_threat")
LOG_FEATURES = (
    [f"{s}_lag{i}" for s in _SERIES for i in range(1, 6)]
    + [f"{s}_ema{n}" for s in _SERIES for n in (5, 10, 20)]
    + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
    + ["vol_ratio", "vol5_vol60", "vol20_vol126",
       "threat_ratio", "ovx_x_gprd", "ovx_x_gprd_threat"]
)


def select_capacity(n_effective):
    for threshold, name, params in CAPACITY_TIERS:
        if n_effective >= threshold:
            return name, {**XGB_COMMON, **params}
    raise RuntimeError("CAPACITY_TIERS son elemani 0 esikli olmali")


def suggest_params(trial, tier):
    """Suggests a parameter set from the tier's declared bounds."""
    sp = SEARCH_SPACES[tier]
    lo_n, hi_n, step_n = sp["n_estimators"]
    return {
        **XGB_COMMON,
        "n_estimators": trial.suggest_int("n_estimators", lo_n, hi_n, step=step_n),
        "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"],
                                             log=True),
        "min_child_weight": trial.suggest_float("min_child_weight",
                                                *sp["min_child_weight"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *sp["reg_lambda"], log=True),
        "subsample": trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                *sp["colsample_bytree"]),
    }


# ===========================================================================
# Preprocessing (in sync with 03): fit only on the given training slice
# ===========================================================================
def _apply_log(X, log_cols):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def fit_preproc(X_train, log_cols):
    Xl = _apply_log(X_train, log_cols)
    lo = Xl.quantile(WINSOR_LOWER)
    hi = Xl.quantile(WINSOR_UPPER)
    Xc = Xl.clip(lower=lo, upper=hi, axis=1)
    scaler = MinMaxScaler().fit(Xc)
    return {"lo": lo, "hi": hi, "scaler": scaler, "log_cols": log_cols}


def apply_preproc(X, params):
    Xl = _apply_log(X, params["log_cols"])
    Xc = Xl.clip(lower=params["lo"], upper=params["hi"], axis=1)
    return pd.DataFrame(
        params["scaler"].transform(Xc), index=X.index, columns=X.columns
    )


def compute_metrics(y_true, y_pred, train_mean):
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


def fit_target(y, pv, target_mode):
    return np.log(y) - np.log(pv) if target_mode == "ratio" else np.log(y)


def back_transform(raw_log_pred, pv, smearing, target_mode):
    out = np.exp(raw_log_pred) * smearing
    return out * pv if target_mode == "ratio" else out


# ===========================================================================
# The validation length: derived FROM A FORMULA, not from a fixed table
# ===========================================================================
def choose_validation_span(df, train_idx, test_year, h):
    """Grows the window one year at a time until validation holds >= VAL_MIN_EFFECTIVE effective obs.

    train_idx is the training slice with the outer embargo ALREADY APPLIED; the last h rows
    of the validation slice have therefore already been dropped, so the count here gives
    the true size.
    """
    years = df.loc[train_idx, "year"]
    for n_years in range(1, MAX_VAL_YEARS + 1):
        val_years = [test_year - i for i in range(1, n_years + 1)]
        n_val = int(years.isin(val_years).sum())
        if n_val >= VAL_MIN_EFFECTIVE * h:
            return n_years, val_years, n_val
    val_years = [test_year - i for i in range(1, MAX_VAL_YEARS + 1)]
    return MAX_VAL_YEARS, val_years, int(years.isin(val_years).sum())


# ===========================================================================
# A single fold
# ===========================================================================
def run_fold(df, feature_cols, log_cols, h, fold_id, test_year, target_mode,
             n_trials, verbose=True):
    train_idx_all = df.index[df["year"] < test_year]
    test_idx_all = df.index[df["year"] == test_year]
    assert train_idx_all.max() < test_idx_all.min()

    # --- The outer embargo (the train / test boundary) ---------------------
    train_idx = train_idx_all[:-h]
    assert train_idx.max() + h < test_idx_all.min(), \
        f"h={h} fold {fold_id}: dis embargo yetersiz"

    # --- The validation length, from the formula ---------------------------
    n_val_years, val_years, _ = choose_validation_span(df, train_idx, test_year, h)
    val_idx_all = train_idx[df.loc[train_idx, "year"].isin(val_years).values]
    tp_idx_all = train_idx[(df.loc[train_idx, "year"] < min(val_years)).values]

    needed = feature_cols + ["y", "pred_past_vol"]

    def clean(idx):
        if len(idx) == 0:
            return df.loc[idx]
        ok = df.loc[idx, needed].notna().all(axis=1)
        return df.loc[idx[ok.values]]

    te = clean(test_idx_all)
    assert len(te) > 0

    # --- Can a validation split be built at all? ---------------------------
    split_ok = len(tp_idx_all) > h + MIN_TP_ROWS
    if split_ok:
        # --- The inner embargo (train-proper / validation boundary) --------
        tp_idx = tp_idx_all[:-h]
        assert tp_idx.max() + h < val_idx_all.min(), (
            f"h={h} fold {fold_id}: ic embargo yetersiz -- son train-proper satiri "
            f"{tp_idx.max()} + h={h}, ilk validation satiri {val_idx_all.min()}"
        )
        assert val_idx_all.max() + h < test_idx_all.min(), (
            f"h={h} fold {fold_id}: validation hedefi test donemine tasiyor"
        )
        assert not (set(tp_idx) & set(val_idx_all)), "train-proper/validation kesisiyor"
        assert not (set(val_idx_all) & set(test_idx_all)), "validation/test kesisiyor"
        assert not (set(tp_idx) & set(test_idx_all)), "train-proper/test kesisiyor"

        tp = clean(tp_idx)
        va = clean(val_idx_all)
        split_ok = len(tp) >= MIN_TP_ROWS and len(va) > 0

    if not split_ok:
        # No validation can be built at all: train on the whole training set as in 03,
        # with smearing from the training residuals. The fold is flagged explicitly.
        tp = clean(train_idx)
        va = df.loc[train_idx[:0]]

    tp_eff = len(tp) / h
    val_eff = len(va) / h if len(va) else 0.0
    tier, tier_params = select_capacity(tp_eff)

    use_optuna = (split_ok and tp_eff >= OPTUNA_MIN_TP_EFFECTIVE
                  and val_eff >= VAL_MIN_EFFECTIVE)

    X_tp, y_tp, pv_tp = tp[feature_cols], tp["y"].to_numpy("float64"), \
        tp["pred_past_vol"].to_numpy("float64")
    X_te, y_te, pv_te = te[feature_cols], te["y"].to_numpy("float64"), \
        te["pred_past_vol"].to_numpy("float64")
    y_tp_fit = fit_target(y_tp, pv_tp, target_mode)

    trial_values, best_params, val_rmse = [], None, float("nan")

    if split_ok:
        X_va, y_va, pv_va = va[feature_cols], va["y"].to_numpy("float64"), \
            va["pred_past_vol"].to_numpy("float64")
        y_va_fit = fit_target(y_va, pv_va, target_mode)
        # During the search stage, preprocessing is fit ONLY on train-proper.
        pp_search = fit_preproc(X_tp, log_cols)
        X_tp_s = apply_preproc(X_tp, pp_search)
        X_va_s = apply_preproc(X_va, pp_search)
        va_mean = float(y_va.mean())

        def evaluate(params):
            m = XGBRegressor(**params)
            m.fit(X_tp_s, y_tp_fit)
            resid = y_va_fit - m.predict(X_va_s)
            sm_raw = float(np.mean(np.exp(resid)))
            # Shrinkage is applied DURING THE SEARCH as well: the ranking must evaluate
            # the same pipeline that is ultimately deployed.
            sm, _ = shrink_smearing(sm_raw, val_eff)
            pred = back_transform(m.predict(X_va_s), pv_va, sm, target_mode)
            return compute_metrics(y_va, pred, va_mean)["rmse"], sm_raw, sm

        if use_optuna:
            def objective(trial):
                rmse, _, _ = evaluate(suggest_params(trial, tier))
                return rmse

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            trial_values = [t.value for t in study.trials if t.value is not None]
            best_params = {**XGB_COMMON, **study.best_params}
        else:
            best_params = tier_params

        val_rmse, smearing_raw, smearing = evaluate(best_params)
        smearing_n_eff = val_eff
        smearing_source = "validation"
    else:
        best_params = tier_params
        m = XGBRegressor(**best_params)
        pp_tmp = fit_preproc(X_tp, log_cols)
        m.fit(apply_preproc(X_tp, pp_tmp), y_tp_fit)
        resid = y_tp_fit - m.predict(apply_preproc(X_tp, pp_tmp))
        smearing_raw = float(np.mean(np.exp(resid)))
        smearing_n_eff = tp_eff
        smearing, _ = shrink_smearing(smearing_raw, smearing_n_eff)
        smearing_source = "train (validation kurulamadi)"

    # --- Retraining: train-proper + validation ------------------------------
    # With the chosen hyperparameters; THE PREPROCESSING is also refit on this union.
    final = pd.concat([tp, va]) if len(va) else tp
    X_fin = final[feature_cols]
    y_fin = final["y"].to_numpy("float64")
    pv_fin = final["pred_past_vol"].to_numpy("float64")
    pp_final = fit_preproc(X_fin, log_cols)
    model = XGBRegressor(**best_params)
    model.fit(apply_preproc(X_fin, pp_final), fit_target(y_fin, pv_fin, target_mode))

    pred_xgb = back_transform(
        model.predict(apply_preproc(X_te, pp_final)), pv_te, smearing, target_mode
    )
    train_mean = float(y_fin.mean())
    preds = {"xgboost": pred_xgb,
             "train_mean": np.full(len(te), train_mean),
             "past_vol": pv_te}

    include_main = not (test_year == PARTIAL_YEAR and h in EXCLUDE_2026_HORIZONS)
    fold_metrics = {}
    for name, p in preds.items():
        ok = np.isfinite(p)
        fold_metrics[name] = compute_metrics(y_te[ok], p[ok], train_mean)

    # --- The trial distribution: is the selection noise? -------------------
    if trial_values:
        t_best, t_med, t_worst = (float(np.min(trial_values)),
                                  float(np.median(trial_values)),
                                  float(np.max(trial_values)))
        signal_pct = 100.0 * (t_med - t_best) / t_med if t_med > 0 else float("nan")
    else:
        t_best = t_med = t_worst = signal_pct = float("nan")

    rec = {
        "horizon": h, "fold": fold_id, "test_year": test_year,
        "include_in_main": include_main,
        "n_val_years": n_val_years,
        "val_years": ",".join(str(y) for y in sorted(val_years)),
        "n_train_proper": int(len(tp)), "n_val": int(len(va)),
        "n_test": int(len(te)),
        "tp_effective": round(tp_eff, 2), "val_effective": round(val_eff, 2),
        "capacity_tier": tier,
        "optuna_used": bool(use_optuna),
        "n_trials": int(len(trial_values)),
        "selection_procedure": "optuna" if use_optuna else "kapasite_kurali",
        "best_params": {k: v for k, v in best_params.items()
                        if k not in XGB_COMMON},
        "val_rmse": val_rmse,
        "trial_val_rmse_best": t_best,
        "trial_val_rmse_median": t_med,
        "trial_val_rmse_worst": t_worst,
        "selection_signal_pct": signal_pct,
        "smearing_raw": smearing_raw,
        "smearing": smearing,
        "smearing_shrink_w": round(
            smearing_n_eff / (smearing_n_eff + SMEARING_SHRINK_K), 4),
        "smearing_n_eff": round(smearing_n_eff, 2),
        "smearing_source": smearing_source,
        "refit_on": "train_proper+validation" if len(va) else "train (val yok)",
        "train_mean_target": train_mean,
        "metrics": fold_metrics,
    }

    pred_df = pd.DataFrame({
        "horizon": h, "Date": te["Date"].values,
        "Date_parsed": te["Date_parsed"].values,
        "fold": fold_id, "test_year": test_year,
        "include_in_main": include_main,
        "selection_procedure": rec["selection_procedure"],
        "y_true": y_te, "pred_xgboost": pred_xgb,
        "pred_train_mean": preds["train_mean"], "pred_past_vol": pv_te,
    })

    if verbose:
        proc = "optuna" if use_optuna else "KURAL "
        sig = f"{signal_pct:5.1f}%" if not math.isnan(signal_pct) else "    - "
        print(f"  h={h:3d} {test_year} | tp {len(tp):4d} ({tp_eff:6.1f}) "
              f"val {len(va):3d} ({val_eff:5.2f}, {n_val_years}y) | {tier:6s} "
              f"| {proc} sinyal {sig} | smear {smearing:.4f} "
              f"| RMSE {fold_metrics['xgboost']['rmse']:.6f}")

    return pred_df, rec


def metrics_table(records):
    rows = []
    for r in records:
        for name, m in r["metrics"].items():
            rows.append({"horizon": r["horizon"], "fold": r["fold"],
                         "test_year": r["test_year"],
                         "include_in_main": r["include_in_main"],
                         "selection_procedure": r["selection_procedure"],
                         "model": name, "n_test": m["n"], "rmse": m["rmse"],
                         "mae": m["mae"], "r2_oos": m["r2_oos"], "r2": m["r2"]})
    return pd.DataFrame(rows)


def aggregate(metrics_df, preds_df, h, label, subset_years=None):
    m = metrics_df[metrics_df["include_in_main"]]
    p = preds_df[preds_df["include_in_main"]]
    if subset_years is not None:
        m = m[m["test_year"].isin(subset_years)]
        p = p[p["test_year"].isin(subset_years)]
    rows = []
    for name in MODELS:
        sub = m[m["model"] == name]
        if len(sub) == 0:
            continue
        rows.append({
            "horizon": h, "grup": label, "model": name, "n_folds": int(len(sub)),
            "rmse_fold_mean": float(sub["rmse"].mean()),
            "mae_fold_mean": float(sub["mae"].mean()),
            "r2_oos_fold_mean": float(sub["r2_oos"].mean()),
        })
    return rows


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--test-years", type=int, nargs="+", default=None,
                    help="Belirli test yillari (sure olcumu icin)")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--target-mode", choices=["ratio", "log"],
                    default=TARGET_MODE_DEFAULT)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    t0 = time.time()

    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    assert (feat["Date"].values == raw["Date"].values).all()
    assert feat["Date_parsed"].is_monotonic_increasing

    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in LOG_FEATURES if c in feature_cols]
    assert len(log_cols) == len(LOG_FEATURES)

    df = feat.copy()
    for h in HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values
    df["year"] = df["Date_parsed"].dt.year
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))

    test_years = args.test_years or list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    if args.test_years is None:
        assert len(test_years) == EXPECTED_N_FOLDS

    print(f"=== Optuna walk-forward | ufuklar {args.horizons} | "
          f"{len(test_years)} fold | {len(feature_cols)} ozellik | "
          f"hedef {args.target_mode} | {args.n_trials} deneme/fold ===")
    print(f"Validation uzunlugu FORMULDEN turer: >= {VAL_MIN_EFFECTIVE:.0f} etkin "
          f"bagimsiz gozlem (n_val >= {VAL_MIN_EFFECTIVE:.0f}h).")
    print(f"Optuna uygunluk: train-proper etkin >= {OPTUNA_MIN_TP_EFFECTIVE:.0f} VE "
          f"validation etkin >= {VAL_MIN_EFFECTIVE:.0f}; aksi halde kapasite kurali.\n")

    all_preds, all_recs = [], []
    for h in args.horizons:
        print(f"--- h={h} ---")
        d = df.copy()
        d["y"] = d[f"target_vol_{h}"]
        d["pred_past_vol"] = daily_ret.rolling(h).std().shift(1).values
        for fold_id, ty in enumerate(test_years, start=1):
            pdf, rec = run_fold(d, feature_cols, log_cols, h, fold_id, ty,
                                args.target_mode, args.n_trials)
            all_preds.append(pdf)
            all_recs.append(rec)
        print()

    preds_all = pd.concat(all_preds, ignore_index=True)
    metrics_all = metrics_table(all_recs)
    fold_all = pd.DataFrame([{k: v for k, v in r.items() if k != "metrics"}
                             for r in all_recs])
    fold_all["best_params"] = fold_all["best_params"].apply(json.dumps)

    pd.set_option("display.width", 220)

    # --- (1) The output of the validation-length formula -------------------
    print("=== VALIDATION UZUNLUGU (formulun ciktisi, sabit tablo DEGIL) ===")
    print(f"Kriter: validation dilimi >= {VAL_MIN_EFFECTIVE:.0f} etkin bagimsiz "
          f"gozlem. Pencere bu saglanana kadar birer yil buyutulur.")
    vt = fold_all.pivot(index="test_year", columns="horizon", values="n_val_years")
    ve = fold_all.pivot(index="test_year", columns="horizon", values="val_effective")
    print("\nSecilen validation uzunlugu (yil):")
    print(vt.to_string())
    print("\nOrtaya cikan validation etkin gozlem sayisi:")
    print(ve.to_string(float_format=lambda v: f"{v:.2f}"))
    print()

    # --- (2) Distribution of the selection procedure -----------------------
    print("=== SECIM PROSEDURU (optuna / kapasite kurali) ===")
    print(fold_all.pivot(index="test_year", columns="horizon",
                         values="selection_procedure").to_string())
    print()

    # --- (3) The trial distribution: is the selection noise? ---------------
    print("=== OPTUNA DENEME DAGILIMI (validation RMSE) ===")
    print("secim_sinyali = (medyan - en_iyi) / medyan. Kucukse, 30 deneme arasindan")
    print("en iyisini secmek validation'a asiri uyum demektir ve secim gurultudur.")
    od = fold_all[fold_all["optuna_used"]][
        ["horizon", "test_year", "capacity_tier", "n_trials", "val_effective",
         "trial_val_rmse_best", "trial_val_rmse_median", "trial_val_rmse_worst",
         "selection_signal_pct"]
    ]
    if len(od):
        print(od.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        weak = od[od["selection_signal_pct"] < SELECTION_SIGNAL_WEAK_PCT]
        print(f"\nSecim sinyali < %{SELECTION_SIGNAL_WEAK_PCT:.0f} olan fold: "
              f"{len(weak)}/{len(od)}")
        if len(weak):
            print("Bu fold'larda en iyi deneme medyandan anlamli olcude iyi degil;")
            print("secim buyuk olasilikla gurultudur:")
            print(weak[["horizon", "test_year", "val_effective",
                        "selection_signal_pct"]].to_string(
                index=False, float_format=lambda v: f"{v:.2f}"))
        print("\nKademe bazinda ortalama secim sinyali (%):")
        print(od.groupby("capacity_tier")["selection_signal_pct"].agg(
            ["mean", "median", "min", "max", "size"]).to_string(
            float_format=lambda v: f"{v:.2f}"))
    else:
        print("Bu kosuda Optuna calisan fold yok.")
    print()

    # --- (4) Smearing: raw vs shrunk ---------------------------------------
    print("=== SMEARING: HAM vs BUZULMUS ===")
    print(f"S_duzeltilmis = 1 + w*(S_ham-1), w = n_eff/(n_eff+{SMEARING_SHRINK_K:.0f})")
    print("Katsayi varyansi ~1/n_eff ile olceklendigi icin, etkin gozlemi az olan")
    print("fold'larda olculen deger guvenilmezdir ve 1'e ('duzeltme yok') cekilir.")
    print("\nHam katsayi:")
    print(fold_all.pivot(index="test_year", columns="horizon",
                         values="smearing_raw").to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\nBuzulmus katsayi (nihai kullanilan):")
    print(fold_all.pivot(index="test_year", columns="horizon",
                         values="smearing").to_string(
        float_format=lambda v: f"{v:.4f}"))
    fold_all["sap_ham"] = (fold_all["smearing_raw"] - 1).abs()
    fold_all["sap_buz"] = (fold_all["smearing"] - 1).abs()
    print("\nBuzmenin mudahale olcusu (ufuk basina):")
    print(fold_all.groupby("horizon").agg(
        w_ort=("smearing_shrink_w", "mean"),
        n_eff_ort=("smearing_n_eff", "mean"),
        ham_min=("smearing_raw", "min"), ham_maks=("smearing_raw", "max"),
        ham_ort_sapma=("sap_ham", "mean"),
        buz_min=("smearing", "min"), buz_maks=("smearing", "max"),
        buz_ort_sapma=("sap_buz", "mean"),
    ).to_string(float_format=lambda v: f"{v:.4f}"))
    src = fold_all["smearing_source"].value_counts()
    print(f"\nKaynak dagilimi: {src.to_dict()}")
    print()

    # --- (5) Metrics: overall + by group -----------------------------------
    agg_rows = []
    for h in args.horizons:
        m_h = metrics_all[metrics_all["horizon"] == h]
        p_h = preds_all[preds_all["horizon"] == h]
        agg_rows += aggregate(m_h, p_h, h, "tumu")
        # At long horizons the result mixes TWO different selection procedures -> split it.
        for proc in ("optuna", "kapasite_kurali"):
            yrs = sorted(m_h.loc[m_h["selection_procedure"] == proc,
                                 "test_year"].unique())
            if yrs and len(yrs) < len(test_years):
                agg_rows += aggregate(m_h, p_h, h, proc, subset_years=yrs)
    agg_all = pd.DataFrame(agg_rows)

    print("=== ANA METRIKLER (RMSE / MAE fold ortalamasi) ===")
    print(agg_all.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()
    print("XGBoost'un baseline'lara gore RMSE kazanci (negatif = iyi):")
    gains = []
    for (h, grp), g in agg_all.groupby(["horizon", "grup"]):
        s = g.set_index("model")
        if not {"xgboost", "past_vol", "train_mean"} <= set(s.index):
            continue
        gains.append({
            "horizon": h, "grup": grp, "n_folds": int(s.loc["xgboost", "n_folds"]),
            "vs_past_vol_%": 100 * (s.loc["xgboost", "rmse_fold_mean"]
                                    / s.loc["past_vol", "rmse_fold_mean"] - 1),
            "vs_train_mean_%": 100 * (s.loc["xgboost", "rmse_fold_mean"]
                                      / s.loc["train_mean", "rmse_fold_mean"] - 1),
            "r2_oos": s.loc["xgboost", "r2_oos_fold_mean"],
        })
    print(pd.DataFrame(gains).to_string(index=False,
                                        float_format=lambda v: f"{v:.3f}"))
    print()
    mixed = [h for h in args.horizons
             if fold_all[fold_all["horizon"] == h]["selection_procedure"].nunique() > 1]
    if mixed:
        print(f"UYARI: h={mixed} sonuclari IKI FARKLI secim prosedurunun karisimidir.")
        print("Toplu metrik tek bir yontemin performansi degildir; yukaridaki grup")
        print("bazinda satirlar birlikte okunmalidir.")
    print()

    # --- Writing ------------------------------------------------------------
    sfx = args.suffix
    preds_all.to_csv(OUT_DIR / f"opt_predictions_all{sfx}.csv", index=False)
    metrics_all.to_csv(OUT_DIR / f"opt_metrics_all{sfx}.csv", index=False)
    agg_all.to_csv(OUT_DIR / f"opt_aggregate_all{sfx}.csv", index=False)
    fold_all.to_csv(OUT_DIR / f"opt_folds_all{sfx}.csv", index=False)

    runtime = time.time() - t0
    summary = {
        "horizons": args.horizons, "test_years": test_years, "seed": SEED,
        "n_trials_per_fold": args.n_trials,
        "target_mode": args.target_mode,
        "n_features": len(feature_cols),
        "validation_rule": {
            "criterion": f"validation >= {VAL_MIN_EFFECTIVE} etkin bagimsiz gozlem",
            "formula": "n_val >= VAL_MIN_EFFECTIVE * h, pencere yil yil buyutulur",
            "max_val_years": MAX_VAL_YEARS,
            "derived_lengths": vt.to_dict(),
        },
        "optuna_eligibility": {
            "min_tp_effective": OPTUNA_MIN_TP_EFFECTIVE,
            "min_val_effective": VAL_MIN_EFFECTIVE,
            "min_tp_rows": MIN_TP_ROWS,
            "fallback": "kapasite kurali (CAPACITY_TIERS)",
        },
        "search_spaces": SEARCH_SPACES,
        "capacity_tiers": [{"min_effective_obs": t, "name": n, "params": p}
                           for t, n, p in CAPACITY_TIERS],
        "smearing_source": "validation artiklari",
        "smearing_shrinkage": {
            "formula": "S = 1 + w*(S_ham-1), w = n_eff/(n_eff+K)",
            "K": SMEARING_SHRINK_K,
            "n_eff": "validation diliminin etkin bagimsiz gozlem sayisi (satir/h)",
            "rationale": (
                "Katsayi bir ortalama tahminidir; varyansi ~1/n_eff ile olceklenir. "
                "Uzun ufuklarda n_eff 3-7'ye dustugu icin olculen katsayi asiri "
                "gurultuludur ve tek skalar olarak tum test tahminlerini carpar. "
                "K onceden ilan edilmistir, test sonucuna gore ayarlanmaz."
            ),
            "applied_during_search": True,
        },
        "smearing_caveat": (
            "Siralama icin kullanilan val_rmse, ayni dilimden hesaplanan smearing "
            "katsayisini icerir; val_rmse test RMSE'sinin hafif IYIMSER tahminidir."
        ),
        "refit_policy": (
            "Secim sonrasi train_proper+validation ile yeniden egitim. Smearing "
            "yeniden egitim ONCESI modelin validation artiklarindan; uyumsuzluk "
            "muhafazakar yonde."
        ),
        "aggregate": agg_rows,
        "gains": gains,
        "folds": all_recs,
        "runtime_seconds": round(runtime, 2),
    }
    with open(OUT_DIR / f"opt_summary_all{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    n_opt = int(fold_all["optuna_used"].sum())
    print(f"Yazildi: opt_predictions_all{sfx}.csv, opt_metrics_all{sfx}.csv, "
          f"opt_aggregate_all{sfx}.csv, opt_folds_all{sfx}.csv")
    print(f"Rapor  : opt_summary_all{sfx}.json")
    print(f"Optuna calisan fold: {n_opt}/{len(fold_all)} | "
          f"toplam deneme: {int(fold_all['n_trials'].sum())}")
    print(f"Sure   : {runtime:.1f} saniye")


if __name__ == "__main__":
    main()
