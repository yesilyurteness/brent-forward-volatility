"""Exogenous-variable ablation on the HAR family: does GPR add anything over OVX?

An external (GPT-based) review claimed that the GPR block not only fails to help HAR-X but
actively hurts it. This script reproduces that claim INSIDE our own pipeline, so the
comparison is made under our fold structure, our embargo and our metric definitions rather
than under someone else's reimplementation.

Four nested OLS-in-levels specifications, identical in every other respect:

  har        : har_daily + brent_vol5 + brent_vol20                      (Corsi baseline)
  har_ovx    : HAR + ovx_lag1
  har_gpr    : HAR + gprd_lag1 + gprd_threat_lag1
  har_x      : HAR + ovx_lag1 + gprd_lag1 + gprd_threat_lag1             (current HAR-X)

Everything below is COPIED, not re-derived, from 05_benchmarks.py, because the whole point
is that the numbers have to be comparable with the ones already in outputs/:

  * folds        : expanding walk-forward, test years 2012..2026, 15 folds
  * embargo      : the last h rows of the training index are dropped (train_idx_all[:-h])
  * NaN mask     : per-variant, on that variant's own regressors plus the target
  * floor        : predictions clipped at the TRAINING minimum of the target (train-only)
  * metrics      : compute_metrics(), identical to 05/03
  * 2026 rule    : at h=66 and h=126 the partial 2026 fold is excluded from the fold mean
                   (CLAUDE.md "2026 partial-year rule"); it is still written to the
                   fold-level CSV with include_in_main=False.

har_x here must reproduce the har_x column of outputs/bench_aggregate_all.csv exactly; the
script asserts that, so a silent divergence from the published benchmark cannot pass.

Sign test: har_ovx vs har_x, per horizon, over the folds that enter the main mean. The
question is not only "which wins on average" but "is the GPR damage systematic or does it
come from a couple of folds", so the per-fold deltas are written out in full and the
two-sided exact binomial p-value is reported alongside the win counts.

Runtime: a few seconds (OLS only, no GARCH, no gradient boosting).
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"

SEED = 42
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
EXPECTED_N_FOLDS = 15
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

HAR_COLS = ["har_daily", "brent_vol5", "brent_vol20"]
OVX_EXTRA = ["ovx_lag1"]
GPR_EXTRA = ["gprd_lag1", "gprd_threat_lag1"]

VARIANTS = {
    "har": HAR_COLS,
    "har_ovx": HAR_COLS + OVX_EXTRA,
    "har_gpr": HAR_COLS + GPR_EXTRA,
    "har_x": HAR_COLS + OVX_EXTRA + GPR_EXTRA,
}
VARIANT_LABEL = {
    "har": "HAR",
    "har_ovx": "HAR + OVX",
    "har_gpr": "HAR + GPR",
    "har_x": "HAR + OVX + GPR (HAR-X)",
}


def compute_metrics(y_true, y_pred, train_mean):
    """Kept IN SYNC with 05_benchmarks.py / 03_walkforward.py."""
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


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def sign_test_p(n_wins, n_trials):
    """Two-sided exact binomial test against p=0.5. Ties are excluded upstream."""
    if n_trials == 0:
        return float("nan")
    k = min(n_wins, n_trials - n_wins)
    tail = sum(comb(n_trials, i) for i in range(k + 1)) / 2 ** n_trials
    return float(min(1.0, 2 * tail))


def main():
    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    assert (feat["Date"].values == raw["Date"].values).all()
    assert feat["Date_parsed"].is_monotonic_increasing

    all_extra = OVX_EXTRA + GPR_EXTRA
    df = feat[["Date", "Date_parsed"] + HAR_COLS[1:] + all_extra].copy()
    df["year"] = df["Date_parsed"].dt.year
    for h in HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values

    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    df["har_daily"] = daily_ret.abs().shift(1).values

    test_years = list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    assert len(test_years) == EXPECTED_N_FOLDS

    print("=== Disgil degisken ablasyonu | ufuklar {} | {} fold ===".format(
        HORIZONS, len(test_years)))
    for key, cols in VARIANTS.items():
        print("  {:26s} <- {}".format(VARIANT_LABEL[key], cols))
    print()

    fold_rows, coef_rows, pred_rows = [], [], []

    for fold_id, test_year in enumerate(test_years, start=1):
        train_idx_all = df.index[df["year"] < test_year]
        test_idx_all = df.index[df["year"] == test_year]
        assert train_idx_all.max() < test_idx_all.min()
        first_test = int(test_idx_all.min())

        for h in HORIZONS:
            y_col = "target_vol_{}".format(h)

            def slice_for(idx, cols):
                req = list(cols) + [y_col]
                ok = df.loc[idx, req].notna().all(axis=1)
                return df.loc[idx[ok.values]]

            # ---- Embargo: the last h training rows are dropped (forward-looking label)
            tr_emb = train_idx_all[:-h]
            assert tr_emb.max() + h < first_test, \
                "fold {} h={}: embargo yetersiz".format(fold_id, h)
            assert not (set(tr_emb) & set(test_idx_all))

            # The test observation set must be IDENTICAL across variants, otherwise the
            # metrics are not comparable. Verified rather than assumed.
            test_slices = {k: slice_for(test_idx_all, c) for k, c in VARIANTS.items()}
            te = test_slices["har"]
            for k, s in test_slices.items():
                assert (s.index == te.index).all(), \
                    "fold {} h={}: {} test gozlemleri HAR'dan farkli".format(
                        fold_id, h, k)

            y_te = te[y_col].to_numpy("float64")
            tr_base = slice_for(tr_emb, HAR_COLS)
            train_mean = float(tr_base[y_col].mean())
            floor = float(tr_base[y_col].min())  # train-only floor

            include_main = not (test_year == PARTIAL_YEAR
                                and h in EXCLUDE_2026_HORIZONS)

            for key, cols in VARIANTS.items():
                tr = slice_for(tr_emb, cols)
                # The exogenous columns have no NaNs after warm-up, so every variant sees
                # the same training rows. Asserted, because a silent difference in n would
                # make the ablation measure sample size instead of information content.
                assert (tr.index == tr_base.index).all(), \
                    "fold {} h={}: {} train satirlari HAR'dan farkli".format(
                        fold_id, h, key)

                b = ols_fit(tr[cols].to_numpy("float64"),
                            tr[y_col].to_numpy("float64"))
                p = ols_predict(b, te[cols].to_numpy("float64"))
                n_clip = int((p < floor).sum())
                p = np.maximum(p, floor)
                m = compute_metrics(y_te, p, train_mean)

                fold_rows.append({
                    "horizon": h, "fold": fold_id, "test_year": test_year,
                    "variant": key, "variant_label": VARIANT_LABEL[key],
                    "include_in_main": include_main,
                    "n_train": int(len(tr)), "n_test": m["n"],
                    "rmse": m["rmse"], "mae": m["mae"],
                    "r2_oos": m["r2_oos"], "r2": m["r2"],
                    "n_clipped": n_clip,
                })
                rec = {"horizon": h, "fold": fold_id, "test_year": test_year,
                       "variant": key}
                for c, v in zip(["const"] + cols, b):
                    rec["beta_" + c] = float(v)
                coef_rows.append(rec)
                pred_rows.append(pd.DataFrame({
                    "horizon": h, "Date": te["Date"].values, "fold": fold_id,
                    "test_year": test_year, "include_in_main": include_main,
                    "variant": key, "y_true": y_te, "pred": p}))

    folds = pd.DataFrame(fold_rows)
    coefs = pd.DataFrame(coef_rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    # ===================================================================
    # Aggregate: fold mean over the folds that enter the main metric
    main_folds = folds[folds["include_in_main"]]
    agg = (main_folds.groupby(["horizon", "variant"], sort=False)
                     .agg(n_folds=("fold", "count"),
                          rmse_fold_mean=("rmse", "mean"),
                          mae_fold_mean=("mae", "mean"),
                          r2_oos_fold_mean=("r2_oos", "mean"))
                     .reset_index())
    agg["variant_label"] = agg["variant"].map(VARIANT_LABEL)
    order = {k: i for i, k in enumerate(VARIANTS)}
    agg = (agg.assign(_o=agg["variant"].map(order))
              .sort_values(["horizon", "_o"]).drop(columns="_o"))

    # ---- Reproduction check against the published benchmark run ----
    bench = pd.read_csv(OUT_DIR / "bench_aggregate_all.csv")
    for key in ("har", "har_x"):
        for _, r in agg[agg["variant"] == key].iterrows():
            ref = bench[(bench["horizon"] == r["horizon"]) & (bench["model"] == key)]
            assert len(ref) == 1
            for col in ("rmse_fold_mean", "mae_fold_mean", "r2_oos_fold_mean"):
                assert np.isclose(r[col], float(ref[col].iloc[0]), rtol=0, atol=1e-12), (
                    "h={} {} {}: ablasyon {!r} != benchmark {!r} -- "
                    "fold/embargo/maske ayarlari kaymis".format(
                        r["horizon"], key, col, r[col], float(ref[col].iloc[0])))
    print("[OK] har ve har_x, bench_aggregate_all.csv ile birebir ayni cikti.\n")

    # ===================================================================
    print("=== Fold ortalamasi metrikler (ana metrige giren fold'lar) ===")
    for h in HORIZONS:
        sub = agg[agg["horizon"] == h]
        print("\n-- h={} ({} fold) --".format(h, int(sub["n_folds"].iloc[0])))
        print("{:28s} {:>12s} {:>12s} {:>10s}".format(
            "varyant", "RMSE", "MAE", "R2_oos"))
        for _, r in sub.iterrows():
            print("{:28s} {:12.6f} {:12.6f} {:10.4f}".format(
                r["variant_label"], r["rmse_fold_mean"], r["mae_fold_mean"],
                r["r2_oos_fold_mean"]))

    # ===================================================================
    # Sign test: har_ovx vs har_x
    piv = main_folds.pivot_table(index=["horizon", "fold", "test_year"],
                                 columns="variant", values=["rmse", "mae"])
    sign_rows, delta_rows = [], []
    for h in HORIZONS:
        sub = piv.xs(h, level="horizon")
        for metric in ("rmse", "mae"):
            a = sub[(metric, "har_ovx")]
            b = sub[(metric, "har_x")]
            d = b - a  # >0 => HAR-X worse => HAR+OVX wins
            wins_ovx = int((d > 0).sum())
            wins_harx = int((d < 0).sum())
            ties = int((d == 0).sum())
            n_eff = wins_ovx + wins_harx
            sign_rows.append({
                "horizon": h, "metric": metric,
                "n_folds": int(len(d)), "ties": ties,
                "wins_har_ovx": wins_ovx, "wins_har_x": wins_harx,
                "p_two_sided": sign_test_p(wins_ovx, n_eff),
                "mean_delta": float(d.mean()),
                "median_delta": float(d.median()),
                "max_single_fold_delta": float(d.max()),
                "year_of_max_delta": int(d.idxmax()[1]),
                "mean_delta_excl_worst": float(d.drop(d.idxmax()).mean()),
            })
        d = sub[("rmse", "har_x")] - sub[("rmse", "har_ovx")]
        for (fold, year), v in d.items():
            delta_rows.append({
                "horizon": h, "fold": int(fold), "test_year": int(year),
                "rmse_har_ovx": float(sub.loc[(fold, year), ("rmse", "har_ovx")]),
                "rmse_har_x": float(sub.loc[(fold, year), ("rmse", "har_x")]),
                "delta_rmse_harx_minus_harovx": float(v),
                "winner": "har_ovx" if v > 0 else ("har_x" if v < 0 else "tie"),
            })
    sign = pd.DataFrame(sign_rows)
    deltas = pd.DataFrame(delta_rows)

    print("\n\n=== Isaret testi: HAR+OVX  vs  HAR-X (HAR+OVX+GPR) ===")
    print("delta = HAR-X hatasi - HAR+OVX hatasi ; delta>0 ise GPR zarar vermis.")
    for metric in ("rmse", "mae"):
        print("\n-- {} --".format(metric.upper()))
        print("{:>4s} {:>5s} {:>12s} {:>14s} {:>13s} {:>12s} {:>13s} {:>18s}".format(
            "h", "fold", "OVX kazanir", "HAR-X kazanir", "p(iki yonlu)",
            "ort. delta", "en kotu fold", "o fold haric ort."))
        for _, r in sign[sign["metric"] == metric].iterrows():
            print("{:4d} {:5d} {:12d} {:14d} {:13.4f} {:12.2e} {:13d} {:18.2e}".format(
                int(r["horizon"]), int(r["n_folds"]), int(r["wins_har_ovx"]),
                int(r["wins_har_x"]), r["p_two_sided"], r["mean_delta"],
                int(r["year_of_max_delta"]), r["mean_delta_excl_worst"]))

    print("\n\n=== Fold bazinda RMSE farki (HAR-X - HAR+OVX) ===")
    wide = deltas.pivot(index="test_year", columns="horizon",
                        values="delta_rmse_harx_minus_harovx")
    print(wide.to_string(float_format=lambda v: "{:+.2e}".format(v)))

    # ===================================================================
    agg_out = agg[["horizon", "variant", "variant_label", "n_folds",
                   "rmse_fold_mean", "mae_fold_mean", "r2_oos_fold_mean"]]
    agg_out.to_csv(OUT_DIR / "ablation_exogenous.csv", index=False)
    folds.to_csv(OUT_DIR / "ablation_exogenous_folds.csv", index=False)
    sign.to_csv(OUT_DIR / "ablation_exogenous_sign_test.csv", index=False)
    deltas.to_csv(OUT_DIR / "ablation_exogenous_fold_deltas.csv", index=False)
    coefs.to_csv(OUT_DIR / "ablation_exogenous_coefficients.csv", index=False)
    preds.to_csv(OUT_DIR / "ablation_exogenous_predictions.csv", index=False)
    with open(OUT_DIR / "ablation_exogenous_summary.json", "w", encoding="utf-8") as f:
        json.dump({"variants": {k: list(v) for k, v in VARIANTS.items()},
                   "aggregate": agg_out.to_dict(orient="records"),
                   "sign_test": sign.to_dict(orient="records")},
                  f, indent=2)
    print("\nYazildi: outputs/ablation_exogenous.csv (+ _folds, _sign_test, "
          "_fold_deltas, _coefficients, _predictions, _summary.json)")


if __name__ == "__main__":
    main()
