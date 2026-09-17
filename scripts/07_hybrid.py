"""Hybrid models: does ML add anything complementary to HAR-X?

The Gunnarsson et al. (2024) survey points to hybrid econometric-ML structures as a
promising area. This script tests that suggestion directly.

THREE HYBRIDS
-------------
  H1  simple average of XGBoost + BiLSTM   (part of the original CLAUDE.md design)
  H2  simple average of HAR-X + XGBoost    (linear + non-linear)
  H3  HAR-X residual modelling: prediction = HAR-X + XGBoost(HAR-X residuals)

THE WEIGHTS ARE FIXED AT 0.5/0.5 AND ARE NOT OPTIMIZED
-------------------------------------------------------
Weight optimization requires a validation set. In stage 4 we measured that this setup
selects noise on this problem (in 32 of 52 folds the best candidate was less than 5%
better than the median). The fixed weight is PRE-DECLARED.

H1 and H2 require no new training: they are averages of the saved predictions from the
primary runs. It has been verified (by assert) that the test rows are IDENTICAL and the
y_true values match exactly at all four horizons.

H3 -- RESIDUAL MODELLING
-------------------------
For each fold and horizon:
  1. HAR-X is fit to the training data on ITS OWN window (from row 21). This is
     EXACTLY THE SAME HAR-X as the one reported standalone -- the test predictions are
     asserted against bench_predictions_all.csv. That way the hybrid's contribution can
     be attributed directly to the residual stage.
  2. Training residuals: e_t = y_t - HAR-X(t), only on the rows XGBoost can use (from
     row 127 onwards, with the embargo applied).
  3. XGBoost is trained to predict those residuals.
  4. Prediction: HAR-X(test) + XGBoost_residual(test).

THE TARGET IS IN LEVELS AND IS SIGNED -> there is NO log transform and NO Duan smearing.
The ratio-target trick (log(vol) - log(past_vol)) does not apply here; residuals can be
negative.

A KNOWN BIAS (not hidden): the residual model is trained on HAR-X's IN-SAMPLE residuals,
which are systematically smaller than out-of-sample residuals. We met the same problem
with smearing, but there the base model was memorizing with 400 trees. Here the base
model is an OLS with 6 regressors and thousands of observations, so the in-sample and
out-of-sample residuals are very close to each other. The bias is small but it is
reported.

NO LEAKAGE: HAR-X is fit only on the training data and the residual model only on the
training residuals; the embargo is applied in both and verified with the same asserts.

TWO DIAGNOSTICS -- THEY ANSWER THE QUESTION DIRECTLY
-----------------------------------------------------
(1) DOES THE RESIDUAL STAGE ACTUALLY EXTRACT INFORMATION?
    The residual model's R2 is measured against the baseline of "not predicting the
    residual at all" (that is, zero):
        R2_residual = 1 - SSE(e - e_hat) / SSE(e)
    computed in-sample and out-of-sample separately. If the OUT-OF-SAMPLE value is <= 0,
    the residual stage extracts nothing and the hybrid can only do harm. This is the most
    direct test of the hypothesis.

(2) WHY DOES AVERAGING WORK / NOT WORK?
    A simple average pays off when the errors of the two components are not perfectly
    correlated. The correlation of the component errors is reported for each hybrid. If
    the correlation is close to 1, the average merely carries the worse component along.

THE MAIN QUESTION COLUMN
------------------------
For all three hybrids, the RMSE difference against standalone HAR-X is reported as a
percentage IN A SEPARATE COLUMN. For H2 and H3 this is the direct answer to the study's
main question.
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
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
HYBRID_WEIGHT = 0.5  # PRE-DECLARED, not optimized

HAR_COLS = ["har_daily", "brent_vol5", "brent_vol20"]
HARX_EXTRA = ["ovx_lag1", "gprd_lag1", "gprd_threat_lag1"]
HARX_COLS = HAR_COLS + HARX_EXTRA

XGB_COMMON = {"objective": "reg:squarederror", "tree_method": "hist",
              "random_state": SEED, "n_jobs": 4}

# Capacity tiers kept IN SYNC with 03_walkforward.py.
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

_SERIES = ("brent", "ovx", "gprd", "gprd_threat")
LOG_FEATURES = (
    [f"{s}_lag{i}" for s in _SERIES for i in range(1, 6)]
    + [f"{s}_ema{n}" for s in _SERIES for n in (5, 10, 20)]
    + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
    + ["vol_ratio", "vol5_vol60", "vol20_vol126",
       "threat_ratio", "ovx_x_gprd", "ovx_x_gprd_threat"]
)

HYBRIDS = ("h1_xgb_bilstm", "h2_harx_xgb", "h3_harx_resid")
REFERENCE = "har_x"


def select_capacity(n_eff):
    for t, name, p in CAPACITY_TIERS:
        if n_eff >= t:
            return name, {**XGB_COMMON, **p}
    raise RuntimeError


def _apply_log(X, log_cols):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def fit_preproc(X_train, log_cols):
    Xl = _apply_log(X_train, log_cols)
    lo, hi = Xl.quantile(WINSOR_LOWER), Xl.quantile(WINSOR_UPPER)
    Xc = Xl.clip(lower=lo, upper=hi, axis=1)
    return {"lo": lo, "hi": hi, "scaler": MinMaxScaler().fit(Xc),
            "log_cols": log_cols}


def apply_preproc(X, p):
    Xl = _apply_log(X, p["log_cols"])
    Xc = Xl.clip(lower=p["lo"], upper=p["hi"], axis=1)
    return pd.DataFrame(p["scaler"].transform(Xc), index=X.index, columns=X.columns)


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def compute_metrics(y_true, y_pred, train_mean):
    y_true = np.asarray(y_true, "float64")
    y_pred = np.asarray(y_pred, "float64")
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    sst_own = float(np.sum((y_true - y_true.mean()) ** 2))
    sst_train = float(np.sum((y_true - train_mean) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2_oos": float(1 - sse / sst_train) if sst_train > 0 else float("nan"),
            "r2": float(1 - sse / sst_own) if sst_own > 0 else float("nan"),
            "n": int(len(y_true))}


def r2_vs_zero(e, e_hat):
    """The residual stage's contribution: R2 against the baseline of not predicting it (zero).

    Positive = the residual model adds information on top of HAR-X.
    <= 0      = the residual stage extracts nothing; the hybrid can only do harm.
    """
    e = np.asarray(e, "float64")
    e_hat = np.asarray(e_hat, "float64")
    den = float(np.sum(e ** 2))
    if den <= 0:
        return float("nan")
    return float(1.0 - np.sum((e - e_hat) ** 2) / den)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--test-years", type=int, nargs="+", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    t0 = time.time()

    # --- Saved predictions -------------------------------------------------
    xgb_p = pd.concat(
        [pd.read_csv(OUT_DIR / f"wf_predictions_h{h}.csv").assign(horizon=h)
         for h in HORIZONS], ignore_index=True)
    bil_p = pd.read_csv(OUT_DIR / "bilstm_predictions_all.csv")
    ben_p = pd.read_csv(OUT_DIR / "bench_predictions_all.csv")

    key = ["horizon", "Date"]
    merged = (xgb_p[key + ["test_year", "include_in_main", "y_true",
                           "pred_xgboost", "pred_train_mean", "pred_past_vol"]]
              .merge(bil_p[key + ["pred_bilstm", "y_true"]], on=key,
                     suffixes=("", "_b"))
              .merge(ben_p[key + ["pred_har_x", "pred_har_x_log", "pred_har",
                                  "y_true"]], on=key, suffixes=("", "_n")))
    assert len(merged) == len(xgb_p), "Tahmin dosyalari test satirlarinda ortusmuyor"
    assert (merged["y_true"] - merged["y_true_b"]).abs().max() < 1e-12
    assert (merged["y_true"] - merged["y_true_n"]).abs().max() < 1e-12
    merged = merged.drop(columns=["y_true_b", "y_true_n"])
    # In partial runs (--horizons / --test-years) H3 is computed only for the requested
    # slice; merged is narrowed to the same slice so the join leaves no gaps.
    merged = merged[merged["horizon"].isin(args.horizons)]
    if args.test_years is not None:
        merged = merged[merged["test_year"].isin(args.test_years)]
    merged = merged.reset_index(drop=True)
    print(f"Tahmin dosyalari hizalandi: {len(merged)} satir, "
          f"y_true uc kaynakta da birebir ayni.\n")

    # --- H1 and H2: fixed-weight simple average ----------------------------
    w = HYBRID_WEIGHT
    merged["pred_h1_xgb_bilstm"] = w * merged["pred_xgboost"] + (1 - w) * merged["pred_bilstm"]
    merged["pred_h2_harx_xgb"] = w * merged["pred_har_x"] + (1 - w) * merged["pred_xgboost"]

    # --- H3: HAR-X residual modelling --------------------------------------
    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))

    d = feat.copy()
    d["year"] = d["Date_parsed"].dt.year
    d["har_daily"] = daily_ret.abs().shift(1).values
    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in LOG_FEATURES if c in feature_cols]
    xgb_first = int(feat[feature_cols].notna().all(axis=1).idxmax())

    test_years = args.test_years or list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    h3_rows, fold_records = [], []

    for h in args.horizons:
        print(f"--- h={h} ---")
        d["y"] = tgt[f"target_vol_{h}"].values
        for fold_id, ty in enumerate(test_years, start=1):
            tr_all = d.index[d["year"] < ty]
            te_all = d.index[d["year"] == ty]
            tr_emb = tr_all[:-h]
            assert tr_emb.max() + h < te_all.min(), f"h={h} {ty}: embargo yetersiz"
            assert not (set(tr_emb) & set(te_all))

            need_x = HARX_COLS + ["y"]
            ok_tr = d.loc[tr_emb, need_x].notna().all(axis=1)
            tr_x = tr_emb[ok_tr.values]                       # HAR-X's own window
            ok_te = d.loc[te_all, need_x].notna().all(axis=1)
            te = te_all[ok_te.values]

            # 1) HAR-X: fit and floor EXACTLY as in the standalone run
            beta = ols_fit(d.loc[tr_x, HARX_COLS].to_numpy("float64"),
                           d.loc[tr_x, "y"].to_numpy("float64"))
            floor = float(d.loc[tr_x, "y"].min())
            harx_te = np.maximum(
                ols_predict(beta, d.loc[te, HARX_COLS].to_numpy("float64")), floor)

            # Check: does the refit match bench_predictions_all.csv?
            ref = merged[(merged["horizon"] == h)
                         & (merged["Date"].isin(d.loc[te, "Date"]))]
            ref = ref.set_index("Date").loc[d.loc[te, "Date"]]["pred_har_x"].to_numpy()
            assert np.allclose(harx_te, ref, atol=1e-10), (
                f"h={h} {ty}: HAR-X refit'i standalone ile eslesmiyor"
            )

            # 2) Training residuals, only on the rows XGBoost can use
            need_f = feature_cols + ["y"] + HARX_COLS
            cand = tr_emb[tr_emb >= xgb_first]
            ok_r = d.loc[cand, need_f].notna().all(axis=1)
            tr_r = cand[ok_r.values]
            harx_tr = np.maximum(
                ols_predict(beta, d.loc[tr_r, HARX_COLS].to_numpy("float64")), floor)
            e_tr = d.loc[tr_r, "y"].to_numpy("float64") - harx_tr

            # 3) XGBoost residual model -- IN LEVELS, NO log and NO smearing
            pp = fit_preproc(d.loc[tr_r, feature_cols], log_cols)
            n_eff = len(tr_r) / h
            tier, params = select_capacity(n_eff)
            model = XGBRegressor(**params)
            model.fit(apply_preproc(d.loc[tr_r, feature_cols], pp), e_tr)

            e_hat_tr = model.predict(apply_preproc(d.loc[tr_r, feature_cols], pp))
            e_hat_te = model.predict(apply_preproc(d.loc[te, feature_cols], pp))

            # 4) Hybrid prediction; volatility must be positive -> same training floor
            h3 = np.maximum(harx_te + e_hat_te, floor)
            n_floored = int((harx_te + e_hat_te < floor).sum())

            y_te = d.loc[te, "y"].to_numpy("float64")
            e_te = y_te - harx_te  # the true test residuals

            h3_rows.append(pd.DataFrame({
                "horizon": h, "Date": d.loc[te, "Date"].values,
                "pred_h3_harx_resid": h3,
            }))
            fold_records.append({
                "horizon": h, "fold": fold_id, "test_year": ty,
                "include_in_main": not (ty == PARTIAL_YEAR
                                        and h in EXCLUDE_2026_HORIZONS),
                "n_train_harx": int(len(tr_x)), "n_train_resid": int(len(tr_r)),
                "n_test": int(len(te)),
                "n_train_effective": round(n_eff, 2), "capacity_tier": tier,
                "resid_r2_in_sample": r2_vs_zero(e_tr, e_hat_tr),
                "resid_r2_oos": r2_vs_zero(e_te, e_hat_te),
                "resid_train_std": float(np.std(e_tr)),
                "resid_test_std": float(np.std(e_te)),
                "n_floored": n_floored,
                "pred_floor": floor,
            })
        print()

    h3_all = pd.concat(h3_rows, ignore_index=True)
    merged = merged.merge(h3_all, on=key, how="left")
    assert merged["pred_h3_harx_resid"].notna().all()

    # ===================================================================
    # Metrics
    # ===================================================================
    all_models = ["har_x", "har_x_log", "har", "xgboost", "bilstm",
                  "train_mean", "past_vol"] + list(HYBRIDS)
    rows = []
    for (h, ty), g in merged.groupby(["horizon", "test_year"]):
        tm = float(g["pred_train_mean"].iloc[0])
        inc = bool(g["include_in_main"].iloc[0])
        for m in all_models:
            col = f"pred_{m}"
            if col not in g:
                continue
            mm = compute_metrics(g["y_true"], g[col], tm)
            rows.append({"horizon": h, "test_year": ty, "include_in_main": inc,
                         "model": m, **mm})
    metrics_all = pd.DataFrame(rows)
    fold_all = pd.DataFrame(fold_records)

    main_m = metrics_all[metrics_all["include_in_main"]]
    agg = main_m.groupby(["horizon", "model"]).agg(
        n_folds=("rmse", "size"), rmse_fold_mean=("rmse", "mean"),
        mae_fold_mean=("mae", "mean"), r2_oos_fold_mean=("r2_oos", "mean"),
    ).reset_index()

    # --- THE MAIN QUESTION COLUMN: difference vs standalone HAR-X ---------
    ref = agg[agg["model"] == REFERENCE].set_index("horizon")["rmse_fold_mean"]
    agg["vs_har_x_pct"] = agg.apply(
        lambda r: 100 * (r["rmse_fold_mean"] / ref[r["horizon"]] - 1), axis=1)

    pd.set_option("display.width", 240)

    print("=== ANA SORU: hibritler HAR-X standalone'u gecti mi? ===")
    print("vs_har_x_pct negatif = hibrit daha iyi. Referans: HAR-X (seviyelerde OLS).")
    hb = agg[agg["model"].isin(HYBRIDS)].pivot(
        index="model", columns="horizon", values="vs_har_x_pct").reindex(HYBRIDS)
    print(hb.to_string(float_format=lambda v: f"{v:+.2f}"))
    print()
    print("Bilesenlerin ve tum modellerin ayni sutunu:")
    print(agg.pivot(index="model", columns="horizon",
                    values="vs_har_x_pct").reindex(all_models).to_string(
        float_format=lambda v: f"{v:+.2f}"))
    print()

    print("=== RMSE (fold ortalamasi) ===")
    print(agg.pivot(index="model", columns="horizon",
                    values="rmse_fold_mean").reindex(all_models).to_string(
        float_format=lambda v: f"{v:.6f}"))
    print("\nR2_oos:")
    print(agg.pivot(index="model", columns="horizon",
                    values="r2_oos_fold_mean").reindex(all_models).to_string(
        float_format=lambda v: f"{v:+.3f}"))
    print()

    # --- DIAGNOSTIC 1: does the residual stage extract information? -------
    print("=== TANI 1: ARTIK ASAMASI BILGI CIKARIYOR MU? (H3) ===")
    print("R2_artik = 1 - SSE(e - e_hat)/SSE(e). Taban: 'artik tahmin etmemek'.")
    print("ORNEKLEME DISI deger <= 0 ise artik asamasi hicbir sey cikarmiyor demektir")
    print("ve hibrit yalnizca zarar verebilir.")
    fm = fold_all[fold_all["include_in_main"]]
    print(fm.groupby("horizon").agg(
        fold=("resid_r2_oos", "size"),
        R2_ornekleme_ici=("resid_r2_in_sample", "mean"),
        R2_ornekleme_disi=("resid_r2_oos", "mean"),
        disi_medyan=("resid_r2_oos", "median"),
        disi_pozitif_fold=("resid_r2_oos", lambda v: int((v > 0).sum())),
    ).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nArtik standart sapmasi (train vs test) -- ornekleme-ici sapmanin olcusu:")
    print(fm.groupby("horizon").agg(
        train_std=("resid_train_std", "mean"), test_std=("resid_test_std", "mean"),
        oran=("resid_train_std", "mean")).assign(
        oran=lambda t: t["train_std"] / t["test_std"]).to_string(
        float_format=lambda v: f"{v:.6f}"))
    print(f"\nTabana takilan test gozlemi: {int(fold_all['n_floored'].sum())} / "
          f"{int(fold_all['n_test'].sum())}")
    print()

    # --- DIAGNOSTIC 2: correlation of the component errors ----------------
    print("=== TANI 2: BILESEN HATALARININ KORELASYONU ===")
    print("Basit ortalama, hatalar mukemmel korele DEGILSE kazandirir. Korelasyon")
    print("1'e yakinsa ortalama yalnizca daha kotu bileseni tasir.")
    pairs = {"h1_xgb_bilstm": ("xgboost", "bilstm"),
             "h2_harx_xgb": ("har_x", "xgboost")}
    crows = []
    for hyb, (a, b) in pairs.items():
        for h in args.horizons:
            g = merged[(merged["horizon"] == h) & merged["include_in_main"]]
            ea = g["y_true"] - g[f"pred_{a}"]
            eb = g["y_true"] - g[f"pred_{b}"]
            crows.append({
                "hibrit": hyb, "horizon": h,
                "bilesen_A": a, "bilesen_B": b,
                "hata_korelasyonu": float(np.corrcoef(ea, eb)[0, 1]),
                "rmse_A": float(np.sqrt((ea ** 2).mean())),
                "rmse_B": float(np.sqrt((eb ** 2).mean())),
            })
    cdf = pd.DataFrame(crows)
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    # --- Writing -----------------------------------------------------------
    sfx = args.suffix
    keep = key + ["test_year", "include_in_main", "y_true"] + \
        [f"pred_{m}" for m in all_models if f"pred_{m}" in merged]
    merged[keep].to_csv(OUT_DIR / f"hybrid_predictions_all{sfx}.csv", index=False)
    metrics_all.to_csv(OUT_DIR / f"hybrid_metrics_all{sfx}.csv", index=False)
    agg.to_csv(OUT_DIR / f"hybrid_aggregate_all{sfx}.csv", index=False)
    fold_all.to_csv(OUT_DIR / f"hybrid_folds_all{sfx}.csv", index=False)
    cdf.to_csv(OUT_DIR / f"hybrid_error_correlation{sfx}.csv", index=False)

    runtime = time.time() - t0
    summary = {
        "horizons": args.horizons, "seed": SEED,
        "hybrid_weight": HYBRID_WEIGHT,
        "weight_note": ("Sabit 0.5/0.5, ONCEDEN ILAN EDILMIS. Agirlik optimizasyonu "
                        "validation gerektirir; Asama 4'te bu kurgunun gurultu "
                        "sectigi olculdu."),
        "reference_model": REFERENCE,
        "h3_design": {
            "base": "HAR-X (seviyelerde OLS, kendi penceresinde fit)",
            "base_verified_against": "bench_predictions_all.csv (assert, atol=1e-10)",
            "residual_target": "e_t = y_t - HAR-X(t), SEVIYELERDE, isaretli",
            "no_log_no_smearing": True,
            "residual_train_rows": ">= XGBoost isinma satiri, embargo uygulanmis",
            "known_bias": ("Artik modeli HAR-X'in ORNEKLEM-ICI artiklariyla egitilir; "
                           "bunlar ornekleme disi artiklardan sistematik olarak "
                           "kucuktur. Taban model 6 regresorlu OLS oldugu icin sapma "
                           "kucuktur, ama raporlanir (artik std train/test orani)."),
        },
        "aggregate": agg.to_dict(orient="records"),
        "error_correlation": crows,
        "folds": fold_records,
        "runtime_seconds": round(runtime, 2),
    }
    with open(OUT_DIR / f"hybrid_summary_all{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"Yazildi: hybrid_predictions_all{sfx}.csv, hybrid_metrics_all{sfx}.csv, "
          f"hybrid_aggregate_all{sfx}.csv, hybrid_folds_all{sfx}.csv, "
          f"hybrid_error_correlation{sfx}.csv")
    print(f"Rapor  : hybrid_summary_all{sfx}.json")
    print(f"Sure   : {runtime:.1f} saniye")


if __name__ == "__main__":
    main()
