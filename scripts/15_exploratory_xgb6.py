"""XGBoost-6: an exploratory run that isolates the functional form.

STATUS -- READ THIS FIRST
-------------------------
EXPLORATORY and POST HOC, like the exogenous ablation ladder (11_ablation_exogenous.py).
It is NOT part of the primary hypothesis family, it is NOT used for model selection, and it
does NOT change the primary specification: the primary XGBoost remains the one in
03_walkforward.py (65 features, log-ratio target, Duan smearing, tiered capacity rule).
It was designed after the main results were known, and it is reported as such.

THE QUESTION
------------
The primary comparison XGBoost vs HAR-X changes three things at once:
  1. the model family      : flexible trees vs linear OLS
  2. the input set         : 65 engineered features vs HAR-X's 6 regressors
  3. the target / preproc. : log-ratio target + Duan smearing + log1p/winsorize/MinMax on
                             the features  vs  level target, no transformation at all
So "non-linear modelling adds nothing" is not isolated there. XGBoost-6 holds 2 and 3
fixed at HAR-X's choices and changes only 1:

  * inputs    : exactly HAR-X's six regressors (har_daily, brent_vol5, brent_vol20,
                ovx_lag1, gprd_lag1, gprd_threat_lag1), raw
  * target    : target_vol_h in LEVELS, as in HAR-X. No log, no log-ratio, no smearing
  * preproc.  : none, as in HAR-X. No log1p, no winsorization, no scaling
  * rows      : the same training and test rows as HAR-X (asserted against the ablation)
  * floor     : predictions floored at the training minimum of the target, exactly as
                HAR-X (train-only; trees rarely go below it, the count is reported)
  * folds, embargo (h rows), 2026 partial-year rule: unchanged
  * capacity  : the tiered capacity rule and the shared XGBoost parameters are IMPORTED
                from 03_walkforward.py, not copied, so they cannot drift. The tier is
                chosen from this run's own training row count / h, as the rule says.
                Because HAR-X's rows start at row ~22 rather than ~128, the effective
                observation count here is slightly larger than in 03 and the tier can
                differ in a few folds; the tier used is written out per fold.

Note: colsample_bytree = 0.8 is kept from the shared parameters; with 6 inputs each tree
therefore sees 4 of them. Changing it would be a second change on top of the model family.

Decomposition reported alongside:
  HAR-X -> XGBoost-6            : the effect of the functional form alone
  XGBoost-6 -> XGBoost primary  : the effect of the 65 features + log-ratio target +
                                  smearing + feature preprocessing, together

Inputs : data/veriseti.xlsx, outputs/features.csv, outputs/targets.csv,
         outputs/ablation_exogenous_folds.csv, outputs/ablation_exogenous_predictions.csv,
         outputs/wf_metrics_all.csv
Outputs: outputs/exploratory_xgb6.csv, outputs/exploratory_xgb6_folds.csv,
         outputs/exploratory_xgb6_predictions.csv, outputs/exploratory_xgb6_summary.json

Runtime: under a minute on CPU (60 fits, at most 400 trees on 6 features).
"""
import argparse
import importlib.util
import json
import random
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import alignment

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}
HARX_COLS = ["har_daily", "brent_vol5", "brent_vol20",
             "ovx_lag1", "gprd_lag1", "gprd_threat_lag1"]

# The capacity rule and shared parameters come from the primary script itself.
_spec = importlib.util.spec_from_file_location("walkforward", ROOT / "scripts" / "03_walkforward.py")
walkforward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(walkforward)
select_capacity = walkforward.select_capacity
compute_metrics = walkforward.compute_metrics


def sign_test_p(n_wins, n_trials):
    """Two-sided exact binomial test against p=0.5. Ties are excluded upstream."""
    if n_trials == 0:
        return float("nan")
    k = min(n_wins, n_trials - n_wins)
    return float(min(1.0, 2 * sum(comb(n_trials, i) for i in range(k + 1)) / 2 ** n_trials))


def load_frame(al):
    feat = pd.read_csv(alignment.features_path(al), parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    assert (feat["Date"].values == raw["Date"].values).all()
    df = feat[["Date", "Date_parsed"] + HARX_COLS[1:]].copy()
    df["year"] = df["Date_parsed"].dt.year
    for h in HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values
    # har_daily = |r_{t-1}|, built exactly as in 05_benchmarks.py / 11_ablation_exogenous.py
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    df["har_daily"] = daily_ret.abs().shift(1).values
    return df


def run(df, horizons, test_years):
    fold_rows, pred_frames = [], []
    for fold_id, test_year in enumerate(test_years, start=1):
        train_idx_all = df.index[df["year"] < test_year]
        test_idx_all = df.index[df["year"] == test_year]
        assert train_idx_all.max() < test_idx_all.min()
        for h in horizons:
            y_col = f"target_vol_{h}"
            tr_emb = train_idx_all[:-h]                      # embargo: last h rows dropped
            assert tr_emb.max() + h < test_idx_all.min(), f"fold {fold_id} h={h}: embargo"
            assert not (set(tr_emb) & set(test_idx_all))

            def slice_for(idx):
                ok = df.loc[idx, HARX_COLS + [y_col]].notna().all(axis=1)
                return df.loc[idx[ok.values]]
            tr, te = slice_for(tr_emb), slice_for(test_idx_all)

            X_tr = tr[HARX_COLS].to_numpy("float64")
            y_tr = tr[y_col].to_numpy("float64")
            X_te = te[HARX_COLS].to_numpy("float64")
            y_te = te[y_col].to_numpy("float64")
            train_mean, floor = float(y_tr.mean()), float(y_tr.min())

            n_eff = len(tr) / h
            tier, params = select_capacity(n_eff)
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)                            # level target, raw inputs
            p = model.predict(X_te).astype("float64")
            n_clip = int((p < floor).sum())
            p = np.maximum(p, floor)
            m = compute_metrics(y_te, p, train_mean)

            include_main = not (test_year == PARTIAL_YEAR and h in EXCLUDE_2026_HORIZONS)
            fold_rows.append({
                "horizon": h, "fold": fold_id, "test_year": test_year,
                "include_in_main": include_main, "n_train": int(len(tr)),
                "n_test": m["n"], "n_train_effective": round(n_eff, 1),
                "capacity_tier": tier, "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "reg_lambda": params["reg_lambda"], "n_clipped": n_clip,
                "train_mean_target": train_mean,
                "rmse": m["rmse"], "mae": m["mae"], "r2_oos": m["r2_oos"], "r2": m["r2"],
            })
            pred_frames.append(pd.DataFrame({
                "horizon": h, "Date": te["Date"].values, "fold": fold_id,
                "test_year": test_year, "include_in_main": include_main,
                "y_true": y_te, "pred_xgb6": p}))
            print(f"  h={h:3d} fold {fold_id:2d} | test {test_year} | train {len(tr):4d} "
                  f"| effective {n_eff:6.1f} -> {tier:6s} | floor {n_clip:3d} "
                  f"| RMSE {m['rmse']:.6f}")
    return pd.DataFrame(fold_rows), pd.concat(pred_frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="quick trial on the first N folds; outputs are NOT written")
    alignment.add_argument(ap)
    args = ap.parse_args()
    al = args.gpr_alignment
    t0 = time.time()

    test_years = list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    trial = args.max_folds is not None
    if trial:
        test_years = test_years[:args.max_folds]

    df = load_frame(al)
    print("=== XGBoost-6 (EXPLORATORY, post hoc) | inputs: HAR-X's 6 regressors | "
          "target: level, no transformation/smearing ===")
    folds, preds = run(df, args.horizons, test_years)

    # ---- Same rows as HAR-X: training size per fold, test rows and y_true ----
    abl = pd.read_csv(alignment.out("ablation_exogenous_folds.csv", al))
    abl = abl[abl["variant"] == "har_x"].set_index(["horizon", "test_year"])
    j = folds.set_index(["horizon", "test_year"]).join(
        abl[["n_train", "n_test", "rmse", "mae", "r2_oos"]], rsuffix="_har_x")
    assert (j["n_train"] == j["n_train_har_x"]).all(), "train rows differ from HAR-X"
    assert (j["n_test"] == j["n_test_har_x"]).all(), "test rows differ from HAR-X"
    ap_ = pd.read_csv(alignment.out("ablation_exogenous_predictions.csv", al))
    ap_ = ap_[ap_["variant"] == "har_x"]
    chk = preds.merge(ap_[["horizon", "Date", "y_true", "pred"]], on=["horizon", "Date"],
                      suffixes=("", "_harx"), validate="1:1")
    assert len(chk) == len(preds)
    assert np.allclose(chk["y_true"], chk["y_true_harx"], rtol=0, atol=0)
    print("\n[OK] XGBoost-6 and HAR-X see the same train/test rows.")
    covered = lambda d: d["horizon"].isin(args.horizons) & d["test_year"].isin(test_years)
    alignment.check_equal(folds, "exploratory_xgb6_folds.csv", ["horizon", "test_year"],
                          ["n_train", "n_test", "n_train_effective", "capacity_tier",
                           "train_mean_target"], al, rows=covered,
                          what="train/test satir sayilari + kapasite kademesi")
    if trial:
        print(f"Trial run ({args.max_folds} folds) -- outputs NOT WRITTEN. "
              f"Runtime {time.time() - t0:.1f} s.")
        return

    # ---- Side-by-side: XGBoost-6, HAR-X, primary XGBoost ----
    wf = pd.read_csv(alignment.out("wf_metrics_all.csv", al))
    wf = wf[wf["model"] == "xgboost"].set_index(["horizon", "test_year"])
    j = j.join(wf[["rmse", "mae", "r2_oos"]].add_suffix("_xgb_primary"))
    main_j = j[j["include_in_main"]].reset_index()

    rows = []
    for h in args.horizons:
        s = main_j[main_j["horizon"] == h]
        r = {"horizon": h, "n_folds": int(len(s))}
        for tag, suf in (("xgb6", ""), ("har_x", "_har_x"), ("xgb_primary", "_xgb_primary")):
            for met in ("rmse", "mae", "r2_oos"):
                r[f"{met}_{tag}"] = float(s[met + suf].mean())
        for met in ("rmse", "mae"):
            r[f"{met}_pct_xgb6_vs_har_x"] = 100 * (r[f"{met}_xgb6"] / r[f"{met}_har_x"] - 1)
            r[f"{met}_pct_xgb_primary_vs_xgb6"] = 100 * (
                r[f"{met}_xgb_primary"] / r[f"{met}_xgb6"] - 1)
            d = s[met + "_har_x"] - s[met]              # > 0 => XGBoost-6 better
            wins6, winsx = int((d > 0).sum()), int((d < 0).sum())
            r[f"{met}_wins_xgb6"] = wins6
            r[f"{met}_wins_har_x"] = winsx
            r[f"{met}_ties"] = int((d == 0).sum())
            r[f"{met}_sign_p_two_sided"] = sign_test_p(wins6, wins6 + winsx)
        r["n_folds_tier_yuksek"] = int((s["capacity_tier"] == "yuksek").sum())
        r["n_folds_tier_orta"] = int((s["capacity_tier"] == "orta").sum())
        r["n_folds_tier_dusuk"] = int((s["capacity_tier"] == "dusuk").sum())
        r["n_clipped_total"] = int(s["n_clipped"].sum())
        rows.append(r)
    out = pd.DataFrame(rows)

    out.to_csv(alignment.out("exploratory_xgb6.csv", al), index=False)
    folds.to_csv(alignment.out("exploratory_xgb6_folds.csv", al), index=False)
    preds.to_csv(alignment.out("exploratory_xgb6_predictions.csv", al), index=False)
    runtime = time.time() - t0
    with open(alignment.out("exploratory_xgb6_summary.json", al), "w",
              encoding="utf-8") as f:
        json.dump({"gpr_alignment": al,
                   "status": "exploratory, post hoc; not in the primary hypothesis family; "
                             "does not change the primary specification",
                   "inputs": HARX_COLS, "target": "target_vol_h in levels",
                   "transformations": "none (no log, no log-ratio, no smearing, no "
                                      "log1p/winsorize/scaling)",
                   "floor": "training minimum of the target, as HAR-X",
                   "capacity_rule": "imported from 03_walkforward.py",
                   "seed": SEED, "runtime_seconds": round(runtime, 1),
                   "results": out.to_dict(orient="records")}, f, indent=2)

    print("\n=== Fold mean (folds entering the main metric) ===")
    print(out[["horizon", "n_folds", "rmse_xgb6", "rmse_har_x", "rmse_xgb_primary",
               "rmse_pct_xgb6_vs_har_x", "rmse_pct_xgb_primary_vs_xgb6",
               "rmse_wins_xgb6", "rmse_wins_har_x", "rmse_sign_p_two_sided"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(out[["horizon", "mae_xgb6", "mae_har_x", "mae_pct_xgb6_vs_har_x", "mae_wins_xgb6",
               "mae_wins_har_x", "mae_sign_p_two_sided", "r2_oos_xgb6", "r2_oos_har_x",
               "n_folds_tier_yuksek", "n_folds_tier_orta", "n_folds_tier_dusuk",
               "n_clipped_total"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten: exploratory_xgb6.csv, _folds, _predictions, _summary.json "
          f"({runtime:.1f} s)")


if __name__ == "__main__":
    main()
