"""Econometric benchmarks: GARCH(1,1) and the HAR family (a 2x2 design).

The HAR family is 2x2: {HAR regressors, HAR-X regressors} x {levels, log-log}. HAR and
HAR-X in levels are the PRIMARY specifications; their log versions are SECONDARY.

The same fold structure as XGBoost, the same expanding window, the same 2026 partial-year
rule and the same metrics. The output schema can be merged with 03_walkforward.py.

INFORMATION SET ALIGNMENT (the most critical detail)
----------------------------------------------------
The target at row t is the realized volatility of the window [t+1, t+h]. Under CLAUDE.md
Critical Rule 6, ALL features at row t use only t-1 and earlier (they are all shifted with
.shift(1)). So XGBoost forecasts the window [t+1, t+h] with information up to t-1.

The benchmarks must use THE SAME information set, otherwise the comparison is not fair:

  * The HAR / HAR-log / HAR-X regressors are already shifted (|r_{t-1}|, brent_vol5,
    brent_vol20, ovx_lag1, ...), so they are automatically on information up to t-1.
  * For GARCH the forecast ORIGIN is taken as t-1, a horizon of h+1 is requested, and
    steps 2...h+1 are used. Those steps correspond exactly to the days [t+1, t+h].
    (Had the origin been t, GARCH would also have seen r_t and would hold one day more
    information than XGBoost.)

GARCH -> TARGET TRANSFORMATION
------------------------------
    pred_h = sqrt( mean(sigma2_{t+1}, ..., sigma2_{t+h}) )

No ddof correction is NEEDED. The target is computed with pandas .std() using ddof=1; for
independent, zero-mean, heteroskedastic returns, E[s^2] = (1/h) * sum(sigma2_i), i.e. the
mean of the variances over the window. A correction would be required only if ddof=0 had
been used.

A remaining second-order bias: by Jensen's inequality E[s] <= sqrt(E[s^2]), so
sqrt(mean variance) OVERSTATES the expected standard deviation somewhat. This approach is
the standard one in the literature, so it is kept and noted here.

The mean return term is ignored: the measured mean^2 / variance = 0.000000.

THE EMBARGO: IT VARIES BY MODEL, WITH A JUSTIFICATION
------------------------------------------------------
The embargo exists because the FORWARD-LOOKING LABELS of the training rows spill into the
test period.

  * HAR, HAR-log, HAR-X: the dependent variable is target_vol_h, i.e. they use a
    forward-looking label -> the embargo IS APPLIED (with the same asserts as XGBoost).
  * GARCH: it uses no forward-looking label at all; it is estimated purely from the
    conditional variance recursion of past returns. Discarding the RETURNS of the last h
    training days prevents no leakage and only destroys information -> the embargo is NOT
    APPLIED.

This gives GARCH a structural data advantage. It is LEGITIMATE but it MUST BE VISIBLE:
for every fold, the number of extra days GARCH sees is reported, split into two components
(the embargo difference + the warm-up difference). A referee is certain to ask about it.

THE WARM-UP DIFFERENCE
----------------------
Each model is constrained only by the NaNs of the columns IT actually uses. GARCH uses
only the return series (valid from row 2); HAR from row 22; HAR-X likewise from row 22;
XGBoost finds data only once all 65 features are valid, i.e. from row 128. Giving each
model its own maximum data is the fair choice; the difference is reported.

THE SPECIFICATIONS ARE PRE-DECLARED
------------------------------------
They will not be changed in light of the results (Critical Rule 5).

  GARCH   : constant mean, GARCH(1,1), Student-t errors. The PRIMARY econometric benchmark.
  HAR     : classic Corsi, OLS IN LEVELS. |r_{t-1}| + brent_vol5 + brent_vol20.
            The PRIMARY HAR specification.
  HAR-log : the canonical LOG-LOG form -- log(target) ~ the LOGS of the regressors, with
            the back-transformation exp * train-Duan smearing. A SECONDARY specification.
            Both forms are canonical in the literature; both are reported so that the
            question "why was the log form not tried" is closed in advance. Logging the
            dependent variable while leaving the regressors in levels is a
            MISSPECIFICATION; log-HAR logs both sides.
  HAR-X   : the HAR regressors + ovx_lag1 + gprd_lag1 + gprd_threat_lag1, OLS in levels.
            The paper's main decomposition: does the gain come from the exogenous
            variables (HAR -> HAR-X) or from the non-linear model (HAR-X -> XGBoost)?

The HAR-log smearing factor is computed from the TRAINING residuals. That was problematic
for XGBoost (400 trees memorized the training set and drove the residuals to zero); an OLS
with 3-6 regressors shows no such overfit, and its residuals are a reasonable estimate of
the variance.

Because HAR and HAR-X are OLS in levels, they can predict negative volatility. Predictions
are floored at that fold's TRAINING target minimum (train-only, no leakage) and the number
of times the floor engages is reported.

There is NO scaling, winsorization or target transformation (apart from HAR-log's log).
Carrying our own preprocessing choices in here would stop the benchmark being a benchmark.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

import alignment

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
EXPECTED_N_FOLDS = 15
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

GARCH_SCALE = 100.0  # arch expects returns in percent

# The HAR-log specification takes the logs of the regressors. har_daily = |r_{t-1}| is
# exactly zero on some days (the close did not move), which makes the log undefined. The
# pre-declared floor is a daily move of 0.01%, i.e. below the smallest economically
# meaningful threshold. The number of observations that get floored is reported.
LOG_FLOOR = 1e-4

HAR_COLS = ["har_daily", "brent_vol5", "brent_vol20"]
HARX_EXTRA = ["ovx_lag1", "gprd_lag1", "gprd_threat_lag1"]
HARX_COLS = HAR_COLS + HARX_EXTRA

MODELS = ("har", "har_log", "har_x", "har_x_log", "garch", "train_mean", "past_vol")

# The 2x2 design: {HAR regressors, HAR-X regressors} x {levels, log-log}
#   HAR        : HAR regressors,   OLS in levels   -> the PRIMARY HAR
#   HAR-log    : HAR regressors,   log-log OLS     -> secondary
#   HAR-X      : HAR-X regressors, OLS in levels   -> the PRIMARY HAR-X
#   HAR-X-log  : HAR-X regressors, log-log OLS     -> secondary
# The log form also solves STRUCTURALLY the problem of HAR-X producing negative
# predictions that hit the floor (in levels this reached 2.7% of the test observations
# at h=66).


# ===========================================================================
def compute_metrics(y_true, y_pred, train_mean):
    """Kept IN SYNC with 03_walkforward.py. Main metrics RMSE/MAE; r2_oos secondary;
    r2 a footnote (its reference is the test slice's own mean, ex-post)."""
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
        "n": int(len(y_true)),
    }


def _mk_garch(series):
    """The pre-declared specification: constant mean, GARCH(1,1), Student-t."""
    return arch_model(series, mean="Constant", vol="GARCH", p=1, q=1,
                      dist="t", rescale=False)


def ols_fit(X, y):
    """OLS with an intercept. Called on the training slice only."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--test-years", type=int, nargs="+", default=None)
    ap.add_argument("--align-start-row", type=int, default=None,
                    help="Start all benchmark models from this row (data-"
                         "equalization robustness check). If XGBoost's first usable "
                         "row is given, all models see the same window.")
    ap.add_argument("--suffix", default="")
    alignment.add_argument(ap)
    args = ap.parse_args()
    align_row = args.align_start_row
    asfx = alignment.suffix(args.gpr_alignment)

    t0 = time.time()

    feat = pd.read_csv(alignment.features_path(args.gpr_alignment),
                       parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    assert (feat["Date"].values == raw["Date"].values).all()
    assert feat["Date_parsed"].is_monotonic_increasing

    df = feat[["Date", "Date_parsed"] + HAR_COLS[1:] + HARX_EXTRA].copy()
    df["year"] = df["Date_parsed"].dt.year
    for h in HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values

    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    # The HAR daily component: because we work with daily closing data (no intraday RV),
    # the absolute return is used in place of daily realized volatility -- the standard
    # substitute.
    df["har_daily"] = daily_ret.abs().shift(1).values
    # The index stays IDENTICAL to df's row numbers (it starts at 1 because the first
    # row's return is NaN). That makes label-based selection safe.
    ret_series = pd.Series((daily_ret * GARCH_SCALE).values).dropna()

    test_years = args.test_years or list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    if args.test_years is None:
        assert len(test_years) == EXPECTED_N_FOLDS

    # To report XGBoost's warm-up boundary: the first row where every feature is valid
    xgb_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    xgb_first_row = int(feat[xgb_cols].notna().all(axis=1).idxmax())
    garch_first_row = int(daily_ret.notna().idxmax())

    print(f"=== Benchmarks | horizons {args.horizons} | {len(test_years)} folds ===")
    if align_row is not None:
        print(f"[DATA-EQUALIZATION ROBUSTNESS CHECK] all benchmark models start at row "
              f"{align_row}.")
        print("Aim: to remove the HAR family's data advantage. Extending XGBoost down to HAR's")
        print("window (row 21) is NOT POSSIBLE -- brent_vol126 is NaN at that")
        print("row, and extending it would require pruning XGBoost's feature set,")
        print("which would replace the confounder being removed with a larger one.")
        print("Therefore the equalization is done IN THE OPPOSITE DIRECTION: the benchmarks")
        print("are restricted to XGBoost's window, and no model's feature set")
        print("changes. THIS IS A ROBUSTNESS CHECK; the primary specification")
        print("does NOT CHANGE according to the result.")
    print(f"Warm-up limits: GARCH row {garch_first_row}, "
          f"HAR row {int(df[HAR_COLS].notna().all(axis=1).idxmax())}, "
          f"HAR-X row {int(df[HARX_COLS].notna().all(axis=1).idxmax())}, "
          f"XGBoost row {xgb_first_row}\n")

    pred_frames, fold_records = [], []

    for fold_id, test_year in enumerate(test_years, start=1):
        train_idx_all = df.index[df["year"] < test_year]
        test_idx_all = df.index[df["year"] == test_year]
        assert train_idx_all.max() < test_idx_all.min()
        first_test = int(test_idx_all.min())

        # ==== GARCH: ONE estimation per fold (the parameters do not depend on h) ====
        # CAUTION: arch's fit(last_obs=...) parameter is POSITIONAL. Because the return
        # series is indexed from 1, using last_obs=first_test was also including the FIRST
        # DAY of the test period in the training slice (verified: the parameters came out
        # different from the explicit slice). EXPLICIT SLICING is therefore used here --
        # it leaves no ambiguity.
        ret_train = ret_series[ret_series.index < first_test]
        if align_row is not None:
            ret_train = ret_train[ret_train.index >= align_row]
        n_garch_train = int(len(ret_train))
        gres = _mk_garch(ret_train).fit(disp="off", show_warning=False)
        # The parameters come from the training data; the conditional variance recursion
        # runs over the whole series. This is causal: sigma2_t depends only on returns at
        # t-1 and earlier. fix() estimates nothing, it applies the given parameters.
        gfix = _mk_garch(ret_series).fix(gres.params)

        print(f"Fold {fold_id:2d} | test {test_year} | GARCH train {n_garch_train} days "
              f"| omega={gres.params['omega']:.4f} alpha={gres.params['alpha[1]']:.3f} "
              f"beta={gres.params['beta[1]']:.3f} nu={gres.params['nu']:.2f}")

        for h in args.horizons:
            y_col = f"target_vol_{h}"

            # ---- Per-model NaN masks: each model is constrained by its own columns ----
            def slice_for(idx, cols, need_y=True):
                req = list(cols) + ([y_col] if need_y else [])
                if align_row is not None:
                    idx = idx[idx >= align_row]
                ok = df.loc[idx, req].notna().all(axis=1)
                return df.loc[idx[ok.values]]

            # ---- Embargo: APPLIED to the models that use a label ----
            tr_emb = train_idx_all[:-h]
            assert tr_emb.max() + h < first_test, f"fold {fold_id} h={h}: embargo insufficient"
            assert not (set(tr_emb) & set(test_idx_all))

            tr_har = slice_for(tr_emb, HAR_COLS)
            tr_harx = slice_for(tr_emb, HARX_COLS)
            te_har = slice_for(test_idx_all, HAR_COLS)
            te_harx = slice_for(test_idx_all, HARX_COLS)
            # The test rows must be the same for every HAR model (there are no regressor
            # NaNs in the test period); the metrics have to be on the same observation set.
            te = te_har
            assert (te_harx.index == te.index).all(), \
                "HAR and HAR-X test observations differ -- the metrics are not comparable"

            y_te = te[y_col].to_numpy("float64")
            train_mean = float(tr_har[y_col].mean())
            floor = float(tr_har[y_col].min())  # train-only floor

            preds, clip_counts = {}, {}

            # ---- HAR (OLS in levels) : PRIMARY ----
            b = ols_fit(tr_har[HAR_COLS].to_numpy("float64"),
                        tr_har[y_col].to_numpy("float64"))
            p = ols_predict(b, te[HAR_COLS].to_numpy("float64"))
            clip_counts["har"] = int((p < floor).sum())
            preds["har"] = np.maximum(p, floor)

            # ---- HAR-log : SECONDARY. Duan smearing from training residuals ----
            # The canonical log-HAR logs BOTH SIDES: log(RV_{t+h}) on the LOGS of the
            # regressors. Logging the dependent variable while leaving the regressors in
            # levels is a misspecification.
            # har_daily = |r_{t-1}| is exactly zero in 25 observations (the price did not
            # move), which makes the log undefined. The regressors of the log
            # specification are therefore floored at the pre-declared LOG_FLOOR.
            Xl_tr = np.log(np.maximum(tr_har[HAR_COLS].to_numpy("float64"), LOG_FLOOR))
            Xl_te = np.log(np.maximum(te[HAR_COLS].to_numpy("float64"), LOG_FLOOR))
            n_floored = int((tr_har[HAR_COLS].to_numpy("float64") < LOG_FLOOR).sum())
            ylog = np.log(tr_har[y_col].to_numpy("float64"))
            bl = ols_fit(Xl_tr, ylog)
            resid = ylog - ols_predict(bl, Xl_tr)
            smear = float(np.mean(np.exp(resid)))
            preds["har_log"] = np.exp(ols_predict(bl, Xl_te)) * smear
            clip_counts["har_log"] = 0  # the log back-transformation is already positive

            # ---- HAR-X (OLS in levels) : the PRIMARY HAR-X ----
            bx = ols_fit(tr_harx[HARX_COLS].to_numpy("float64"),
                         tr_harx[y_col].to_numpy("float64"))
            px = ols_predict(bx, te[HARX_COLS].to_numpy("float64"))
            clip_counts["har_x"] = int((px < floor).sum())
            preds["har_x"] = np.maximum(px, floor)

            # ---- HAR-X-log : SECONDARY. The fourth cell of the 2x2 ----
            # The OVX and GPR series are strictly positive (min 14.5 / 9.49 / 7.89), so the
            # log is safe.
            Xxl_tr = np.log(np.maximum(tr_harx[HARX_COLS].to_numpy("float64"), LOG_FLOOR))
            Xxl_te = np.log(np.maximum(te[HARX_COLS].to_numpy("float64"), LOG_FLOOR))
            yxlog = np.log(tr_harx[y_col].to_numpy("float64"))
            bxl = ols_fit(Xxl_tr, yxlog)
            residx = yxlog - ols_predict(bxl, Xxl_tr)
            smear_x = float(np.mean(np.exp(residx)))
            preds["har_x_log"] = np.exp(ols_predict(bxl, Xxl_te)) * smear_x
            clip_counts["har_x_log"] = 0  # the log back-transformation is structurally positive

            # ---- GARCH: origin t-1, horizon h+1, steps 2..h+1 ----
            # This forecasts the window [t+1, t+h] with information up to t-1 and aligns
            # the information set exactly with XGBoost/HAR.
            origins = te.index.to_numpy() - 1
            # arch's forecast(start=...) parameter is POSITIONAL as well; the position of
            # the first desired origin is computed explicitly and the result is verified by
            # label.
            start_pos = int(ret_series.index.get_loc(int(origins.min())))
            var = gfix.forecast(horizon=h + 1, start=start_pos, reindex=True,
                                method="analytic").variance
            valid = var.dropna(how="all")
            assert int(valid.index[0]) == int(origins.min()), (
                f"GARCH forecast alignment broken: expected first origin "
                f"{origins.min()}, found {valid.index[0]}"
            )
            # Alignment check: step 1 of origin o must be sigma2_{o+1}.
            # NOTE: arch zero-pads the column names according to the horizon
            # (h.1 / h.01 / h.001), so columns are selected BY POSITION, NOT BY NAME.
            o0 = int(origins.min())
            assert np.isclose(var.loc[o0].iloc[0],
                              gfix.conditional_volatility.loc[o0 + 1] ** 2), \
                "GARCH origin/step alignment broken"
            assert var.shape[1] == h + 1, "unexpected number of horizon columns"
            # Positions 1..h -> steps 2..h+1 -> the days [t+1, t+h] (since the origin is t-1).
            vpath = var.loc[origins].iloc[:, 1:h + 1].to_numpy("float64")
            assert vpath.shape == (len(te), h)
            preds["garch"] = np.sqrt(vpath.mean(axis=1)) / GARCH_SCALE
            clip_counts["garch"] = 0

            # ---- Naive baselines (the same definitions as in 03) ----
            preds["train_mean"] = np.full(len(te), train_mean)
            preds["past_vol"] = (
                daily_ret.rolling(h).std().shift(1).to_numpy("float64")[te.index]
            )
            clip_counts["train_mean"] = clip_counts["past_vol"] = 0

            include_main = not (test_year == PARTIAL_YEAR
                                and h in EXCLUDE_2026_HORIZONS)
            fold_metrics = {}
            for name, pr in preds.items():
                ok = np.isfinite(pr)
                fold_metrics[name] = compute_metrics(y_te[ok], pr[ok], train_mean)

            # ---- The extra days GARCH sees, split into two components ----
            n_xgb_train = int(((train_idx_all >= xgb_first_row).sum()) - h)
            extra_embargo = h
            garch_vs_xgb = n_garch_train - n_xgb_train
            garch_vs_har = n_garch_train - int(len(tr_har))

            pred_frames.append(pd.DataFrame({
                "horizon": h, "Date": te["Date"].values,
                "Date_parsed": te["Date_parsed"].values,
                "fold": fold_id, "test_year": test_year,
                "include_in_main": include_main, "y_true": y_te,
                **{f"pred_{k}": v for k, v in preds.items()},
            }))

            fold_records.append({
                "horizon": h, "fold": fold_id, "test_year": test_year,
                "include_in_main": include_main,
                "n_test": int(len(te)),
                "n_train_garch": n_garch_train,
                "n_train_har": int(len(tr_har)),
                "n_train_harx": int(len(tr_harx)),
                "n_train_xgb_equiv": n_xgb_train,
                "garch_extra_vs_xgb": garch_vs_xgb,
                "garch_extra_vs_har": garch_vs_har,
                "extra_from_embargo": extra_embargo,
                "extra_from_warmup": garch_vs_xgb - extra_embargo,
                "har_smearing": smear,
                "har_x_smearing": smear_x,
                "n_log_floored": n_floored,
                "pred_floor": floor,
                "n_clipped_har": clip_counts["har"],
                "n_clipped_har_x": clip_counts["har_x"],
                "garch_omega": float(gres.params["omega"]),
                "garch_alpha": float(gres.params["alpha[1]"]),
                "garch_beta": float(gres.params["beta[1]"]),
                "garch_nu": float(gres.params["nu"]),
                "garch_persistence": float(gres.params["alpha[1]"]
                                           + gres.params["beta[1]"]),
                "train_mean_target": train_mean,
                "metrics": fold_metrics,
            })

    preds_all = pd.concat(pred_frames, ignore_index=True)
    rows = []
    for r in fold_records:
        for name, m in r["metrics"].items():
            rows.append({"horizon": r["horizon"], "fold": r["fold"],
                         "test_year": r["test_year"],
                         "include_in_main": r["include_in_main"],
                         "model": name, "n_test": m["n"], "rmse": m["rmse"],
                         "mae": m["mae"], "r2_oos": m["r2_oos"], "r2": m["r2"]})
    metrics_all = pd.DataFrame(rows)
    fold_all = pd.DataFrame([{k: v for k, v in r.items() if k != "metrics"}
                             for r in fold_records])

    pd.set_option("display.width", 240)

    # ===================================================================
    print("\n=== GARCH DATA ADVANTAGE (legitimate, but must be visible) ===")
    print("WHY LEGITIMATE: the embargo exists because the FORWARD-LOOKING LABELS of the train rows")
    print("spill into the test period. GARCH uses no forward-looking label;")
    print("it forecasts only through the conditional-variance recursion of past returns.")
    print("Dropping the returns of the last h training days prevents no leakage, it only")
    print("loses information. Also, since GARCH uses no engineered features,")
    print("its warm-up is short too (it finds data from row 2 on).\n")
    ga = fold_all.groupby("horizon").agg(
        garch_train_ort=("n_train_garch", "mean"),
        xgb_train_ort=("n_train_xgb_equiv", "mean"),
        har_train_ort=("n_train_har", "mean"),
        fazla_vs_xgb=("garch_extra_vs_xgb", "mean"),
        embargodan=("extra_from_embargo", "mean"),
        isinmadan=("extra_from_warmup", "mean"),
        fazla_vs_har=("garch_extra_vs_har", "mean"),
    )
    print(ga.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\nDays GARCH sees beyond XGBoost, per fold:")
    print(fold_all.pivot(index="test_year", columns="horizon",
                         values="garch_extra_vs_xgb").to_string())
    print()

    print("=== GARCH parameters (per fold, train-only estimation) ===")
    gp = fold_all[fold_all["horizon"] == args.horizons[0]][
        ["test_year", "n_train_garch", "garch_omega", "garch_alpha", "garch_beta",
         "garch_persistence", "garch_nu"]]
    print(gp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("=== HAR / HAR-X flooring of negative forecasts ===")
    cl = fold_all.groupby("horizon").agg(
        har_kirpilan=("n_clipped_har", "sum"), harx_kirpilan=("n_clipped_har_x", "sum"),
        test_toplam=("n_test", "sum"), har_smearing_ort=("har_smearing", "mean"),
        log_tabanlanan=("n_log_floored", "max"))
    cl["har_%"] = 100 * cl["har_kirpilan"] / cl["test_toplam"]
    cl["harx_%"] = 100 * cl["harx_kirpilan"] / cl["test_toplam"]
    print(cl.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

    # --- Aggregation -------------------------------------------------------
    agg_rows = []
    main_m = metrics_all[metrics_all["include_in_main"]]
    for h in args.horizons:
        for name in MODELS:
            sub = main_m[(main_m["horizon"] == h) & (main_m["model"] == name)]
            if not len(sub):
                continue
            agg_rows.append({"horizon": h, "model": name, "n_folds": int(len(sub)),
                             "rmse_fold_mean": float(sub["rmse"].mean()),
                             "mae_fold_mean": float(sub["mae"].mean()),
                             "r2_oos_fold_mean": float(sub["r2_oos"].mean())})
    agg_all = pd.DataFrame(agg_rows)
    print("=== BENCHMARK METRICS (fold mean) ===")
    print(agg_all.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()

    # --- The seven-model table, including XGBoost (primary specification) --
    wf_path = alignment.out("wf_metrics_all.csv", args.gpr_alignment)
    seven = None
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        xgb = wf[(wf["model"] == "xgboost") & (wf["include_in_main"])][
            ["horizon", "test_year", "rmse", "mae", "r2_oos"]].copy()
        xgb["model"] = "xgboost"
        comb = pd.concat([
            main_m[["horizon", "test_year", "model", "rmse", "mae", "r2_oos"]], xgb
        ], ignore_index=True)
        seven = comb.groupby(["horizon", "model"]).agg(
            n_folds=("rmse", "size"), rmse_fold_mean=("rmse", "mean"),
            mae_fold_mean=("mae", "mean"), r2_oos_fold_mean=("r2_oos", "mean"),
        ).reset_index()
        order = ["xgboost", "har", "har_log", "har_x", "har_x_log", "garch",
                 "train_mean", "past_vol"]
        seven["_o"] = seven["model"].map({m: i for i, m in enumerate(order)})
        seven = seven.sort_values(["horizon", "_o"]).drop(columns="_o")
        print("=== MODEL COMPARISON (fold-mean RMSE) ===")
        print(seven.pivot(index="model", columns="horizon",
                          values="rmse_fold_mean").reindex(order).to_string(
            float_format=lambda v: f"{v:.6f}"))
        print("\nR2_oos:")
        print(seven.pivot(index="model", columns="horizon",
                          values="r2_oos_fold_mean").reindex(order).to_string(
            float_format=lambda v: f"{v:+.3f}"))
        print("\nDECOMPOSITION (RMSE %, negative = improvement):")
        dec = []
        for h in args.horizons:
            s = seven[seven["horizon"] == h].set_index("model")["rmse_fold_mean"]
            if not {"har", "har_x", "xgboost"} <= set(s.index):
                continue
            dec.append({
                "horizon": h,
                "dissal_degiskenler_HAR->HAR_X": 100 * (s["har_x"] / s["har"] - 1),
                "dogrusal_olmayan_HAR_X->XGB": 100 * (s["xgboost"] / s["har_x"] - 1),
                "toplam_HAR->XGB": 100 * (s["xgboost"] / s["har"] - 1),
                "log_form_HAR->HAR_log": 100 * (s["har_log"] / s["har"] - 1),
            })
        print(pd.DataFrame(dec).to_string(index=False,
                                          float_format=lambda v: f"{v:+.2f}"))
        print("\nThe source of the gain separates into these two columns: the contribution of the exogenous variables")
        print("(HAR -> HAR-X) and the contribution of the nonlinear model (HAR-X -> XGBoost).")
    print()

    # --- Publication mode: same sample, GPR-free models bit for bit --------
    # HAR, HAR-log, GARCH, train_mean and past_vol use no GPR input, so they must
    # reproduce the timestamp run exactly; only har_x / har_x_log may change.
    if not args.suffix and align_row is None:
        gpr_free = ["har", "har_log", "garch", "train_mean", "past_vol"]
        covered = lambda d: d["horizon"].isin(args.horizons) & d["test_year"].isin(test_years)
        alignment.check_equal(preds_all, "bench_predictions_all.csv",
                              ["horizon", "Date", "fold", "test_year"],
                              ["include_in_main", "y_true"] + [f"pred_{m}" for m in gpr_free],
                              args.gpr_alignment, rows=covered,
                              what="ayni orneklem + GPR'siz model tahminleri")
        alignment.check_equal(metrics_all, "bench_metrics_all.csv",
                              ["horizon", "test_year", "model"],
                              ["n_test", "rmse", "mae", "r2_oos", "r2"],
                              args.gpr_alignment,
                              rows=lambda d: covered(d) & d["model"].isin(gpr_free),
                              what="GPR'siz model metrikleri")
        alignment.check_equal(fold_all, "bench_folds_all.csv",
                              ["horizon", "test_year"],
                              ["n_test", "n_train_garch", "n_train_har", "n_train_harx",
                               "n_train_xgb_equiv", "har_smearing", "garch_omega",
                               "garch_alpha", "garch_beta", "garch_nu",
                               "train_mean_target"],
                              args.gpr_alignment, rows=covered,
                              what="train/test satir sayilari + GARCH parametreleri")

    # --- Writing -----------------------------------------------------------
    sfx = args.suffix + asfx
    preds_all.to_csv(OUT_DIR / f"bench_predictions_all{sfx}.csv", index=False)
    metrics_all.to_csv(OUT_DIR / f"bench_metrics_all{sfx}.csv", index=False)
    agg_all.to_csv(OUT_DIR / f"bench_aggregate_all{sfx}.csv", index=False)
    fold_all.to_csv(OUT_DIR / f"bench_folds_all{sfx}.csv", index=False)
    if seven is not None:
        seven.to_csv(OUT_DIR / f"bench_model_comparison_all{sfx}.csv", index=False)

    runtime = time.time() - t0
    summary = {
        "horizons": args.horizons, "test_years": test_years, "seed": SEED,
        "gpr_alignment": args.gpr_alignment,
        "specifications": {
            "garch": "sabit ortalama, GARCH(1,1), Student-t; BIRINCIL ekonometrik",
            "har": "Corsi, seviyelerde OLS, |r_{t-1}|+vol5+vol20; BIRINCIL HAR",
            "har_log": ("kanonik log-log: log(target) ~ log(regresorler), "
                        f"regresor tabani {LOG_FLOOR}, train-Duan smearing; IKINCIL"),
            "har_x": ("HAR + ovx_lag1 + gprd_lag1 + gprd_threat_lag1, seviyelerde "
                      "OLS; BIRINCIL HAR-X"),
            "har_x_log": ("HAR-X regresorlerinin log'u, log hedef, train-Duan "
                          "smearing; IKINCIL. Negatif tahmin sorununu yapisal "
                          "olarak cozer."),
        },
        "information_set": (
            "Tum modeller [t+1,t+h] penceresini t-1'e kadarki bilgiyle tahmin eder. "
            "GARCH icin tahmin baslangici t-1, ufuk h+1, adimlar 2..h+1."
        ),
        "garch_target_conversion": (
            "pred = sqrt(mean(sigma2_{t+1..t+h})). ddof duzeltmesi gerekmez cunku "
            "hedef ddof=1 ile hesaplanir ve E[s^2] = mean(sigma2_i). Jensen geregi "
            "sqrt(ortalama varyans) beklenen std'yi bir miktar yukari tahmin eder."
        ),
        "embargo_policy": {
            "har/har_log/har_x": "UYGULANIR -- ileriye bakan etiket kullanirlar",
            "garch": "UYGULANMAZ -- ileriye bakan etiket kullanmaz",
            "legitimacy_note": (
                "Embargo, train ETIKETLERININ test donemine tasmasini onlemek icindir. "
                "GARCH etiket kullanmaz; son h gunun getirilerini atmak hicbir "
                "sizintiyi onlemez, yalnizca bilgi kaybettirir. Avantaj mesrudur "
                "ancak fold bazinda raporlanir (embargo + isinma bilesenleri)."
            ),
        },
        "warmup_first_rows": {
            "garch": garch_first_row,
            "har": int(df[HAR_COLS].notna().all(axis=1).idxmax()),
            "har_x": int(df[HARX_COLS].notna().all(axis=1).idxmax()),
            "xgboost": xgb_first_row,
        },
        "align_start_row": align_row,
        "align_note": (
            "align_start_row verildiginde tum benchmark modelleri o satirdan "
            "baslar. Veri esitleme SAGLAMLIK KONTROLUDUR; birincil spesifikasyon "
            "sonuca gore degismez."
        ),
        "aggregate": agg_rows,
        "folds": fold_records,
        "runtime_seconds": round(runtime, 2),
    }
    with open(OUT_DIR / f"bench_summary_all{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"Written: bench_predictions_all{sfx}.csv, bench_metrics_all{sfx}.csv, "
          f"bench_aggregate_all{sfx}.csv, bench_folds_all{sfx}.csv")
    if seven is not None:
        print(f"Written: bench_model_comparison_all{sfx}.csv")
    print(f"Report : bench_summary_all{sfx}.json")
    print(f"Runtime: {runtime:.1f} seconds")


if __name__ == "__main__":
    main()
