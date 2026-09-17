"""SHAP analysis: do XGBoost and HAR-X use the same variables?

If both rely on the same information, then XGBoost's loss means "it extracts the same
relationship less efficiently". If they look in different places, that also explains why
the hybrids did not work (in stage 7 the residual modelling extracted nothing
out-of-sample).

THE SHAP COMPUTATION
--------------------
The `shap` package is NOT USED; XGBoost's built-in `pred_contribs=True` option gives
EXACT TreeSHAP. The additivity property (contributions + bias = prediction) is verified
with an assert in every fold.

Because the models of the primary specification were not saved, this script retrains them
itself: the tiered capacity rule, the ratio target, the same preprocessing. Since it is
deterministic, the models are IDENTICAL to those of 03_walkforward.py.

SHAP values are computed on each fold's TEST slice; importance is meaningful where the
prediction is actually made.

A UNIT WARNING: because the model is trained in log-ratio space (log(vol) -
log(past_vol)), the SHAP values are in that unit too, NOT in raw volatility units. That is
fine for interpreting rankings and shares; it is not fine for interpreting levels.

HAR-X STANDARDIZED COEFFICIENTS
--------------------------------
For each fold, beta_j * sd(X_j) / sd(y), computed on the training slice.

A LIMIT ON COMPARABILITY: mean |SHAP| and a standardized beta are NOT THE SAME QUANTITY.
One is a mean attribution magnitude, the other a scaled partial derivative. They are
comparable only at the level of RANKING and SHARE.

A WARNING ABOUT SPEARMAN
-------------------------
HAR-X's regressors fall into only THREE groups. Over three items, Spearman carries almost
no information: it can only take the values -1, -0.5, +0.5, +1, and the p-value is at best
0.33, meaning significance is mathematically IMPOSSIBLE. It is computed and reported, but
never left on its own. Three additional measures are given:
  (1) GROUP SHARES (%) -- this is the real comparison, quantitative and interpretable.
  (2) Spearman at the level of the SHARED REGRESSORS (5 items) -- higher resolution.
  (3) The weight XGBoost gives OUTSIDE HAR-X -- the most direct answer to the question
      "are these the same variables?"

THE SIGN AGREEMENT CHECK
-------------------------
For the five shared regressors, the DIRECTION of the relationship XGBoost learned: the
correlation between each feature's value and its own SHAP value (positive -> as the
feature rises, the prediction rises). This is compared against the SIGN of HAR-X's
standardized beta.

  Opposite direction -> XGBoost may be latching onto noise.
  Same direction     -> the reading "it finds the same relationship but extracts it less
                        efficiently" is strengthened.

Both outcomes are informative, but they tell DIFFERENT stories.

TWO LIMITS OF THE SIGN TEST (both are reported)
  (a) AN UNUSED FEATURE: if the model never split on a feature, its SHAP values are
      identically zero and the direction correlation is undefined. That is NOT an
      "opposite direction"; it means "the model did not use that feature". It is EXCLUDED
      from the agreement rate and reported separately. (This is common at long horizons.)
  (b) THE RATIO-TARGET CONFOUND: XGBoost is trained on the log-ratio target
      (log(vol_h) - log(past_vol_h)). brent_vol5 and brent_vol20 are MECHANICALLY
      correlated with the past_vol_h in the denominator -- at h=5, brent_vol5 IS the
      denominator itself (correlation 1.000), and at h=22 its correlation with
      brent_vol20 is 0.991. For these features a negative SHAP direction is the
      mechanical consequence of mean reversion and does not mean "an opposite
      relationship was learned". HAR-X's beta, by contrast, is on the LEVEL target. The
      comparison is INVALID for these two features; the agreement rate is therefore also
      computed for the EXOGENOUS regressors only (ovx_lag1, gprd_lag1,
      gprd_threat_lag1), which carry no mechanical relationship with the denominator.

har_daily (= |r_{t-1}|) is NOT INCLUDED in the sign check: it has no exact counterpart in
XGBoost's feature set (the closest is brent_ret_lag1, the SIGNED return). The relationship
between an absolute value and a signed value is U-shaped; a linear sign comparison would
be misleading.

Note: the preprocessing (log1p -> winsor clip -> MinMax) is monotonically INCREASING, so
the scaled and raw feature values give the same sign correlation.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"

SEED = 42
np.random.seed(SEED)

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR, LAST_TEST_YEAR = 2012, 2026
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}
WINSOR_LOWER, WINSOR_UPPER = 0.005, 0.995

XGB_COMMON = {"objective": "reg:squarederror", "tree_method": "hist",
              "random_state": SEED, "n_jobs": 4}
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

_S = ("brent", "ovx", "gprd", "gprd_threat")
LOG_FEATURES = (
    [f"{s}_lag{i}" for s in _S for i in range(1, 6)]
    + [f"{s}_ema{n}" for s in _S for n in (5, 10, 20)]
    + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
    + ["vol_ratio", "vol5_vol60", "vol20_vol126",
       "threat_ratio", "ovx_x_gprd", "ovx_x_gprd_threat"]
)

# --- Feature groups: aligned with HAR-X's regressor structure -------------
GROUPS = {
    "brent_fiyat": ([f"brent_lag{i}" for i in range(1, 6)]
                    + [f"brent_ema{n}" for n in (5, 10, 20)]
                    + [f"brent_ret_lag{i}" for i in range(1, 6)]),
    "brent_vol": ["brent_vol5", "brent_vol20", "brent_vol60", "brent_vol126",
                  "vol_ratio", "vol5_vol60", "vol20_vol126"],
    "ovx": ([f"ovx_lag{i}" for i in range(1, 6)]
            + [f"ovx_ema{n}" for n in (5, 10, 20)]
            + ["ovx_z60", "ovx_spike", "ovx_regime_high", "ovx_regime_low",
               "ovx_mr60"]),
    "gpr": ([f"gprd_lag{i}" for i in range(1, 6)]
            + [f"gprd_ema{n}" for n in (5, 10, 20)]
            + ["gprd_z60", "gprd_spike", "gprd_momentum"]
            + [f"gprd_threat_lag{i}" for i in range(1, 6)]
            + [f"gprd_threat_ema{n}" for n in (5, 10, 20)]
            + ["gprd_threat_z60", "gprd_threat_spike", "gprd_threat_momentum",
               "threat_ratio"]),
    "etkilesim": ["ovx_x_gprd", "ovx_x_gprd_threat",
                  "ovx_regime_high_x_gprd", "ovx_regime_low_x_gprd"],
    "takvim": ["dow", "month", "is_month_start", "is_month_end", "is_quarter_end"],
}

HAR_COLS = ["har_daily", "brent_vol5", "brent_vol20"]
HARX_EXTRA = ["ovx_lag1", "gprd_lag1", "gprd_threat_lag1"]
HARX_COLS = HAR_COLS + HARX_EXTRA
# HAR-X regressor -> feature group
HARX_GROUP = {"har_daily": "brent_vol", "brent_vol5": "brent_vol",
              "brent_vol20": "brent_vol", "ovx_lag1": "ovx",
              "gprd_lag1": "gpr", "gprd_threat_lag1": "gpr"}
SHARED_GROUPS = ["brent_vol", "ovx", "gpr"]
# HAR-X regressors with an EXACT counterpart in XGBoost's feature set (har_daily excluded)
SHARED_REGRESSORS = ["brent_vol5", "brent_vol20", "ovx_lag1", "gprd_lag1",
                     "gprd_threat_lag1"]


def select_capacity(n_eff):
    for t, name, p in CAPACITY_TIERS:
        if n_eff >= t:
            return name, {**XGB_COMMON, **p}
    raise RuntimeError


def _apply_log(X, log_cols):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def fit_preproc(X, log_cols):
    Xl = _apply_log(X, log_cols)
    lo, hi = Xl.quantile(WINSOR_LOWER), Xl.quantile(WINSOR_UPPER)
    return {"lo": lo, "hi": hi, "log_cols": log_cols,
            "scaler": MinMaxScaler().fit(Xl.clip(lower=lo, upper=hi, axis=1))}


def apply_preproc(X, p):
    Xl = _apply_log(X, p["log_cols"]).clip(lower=p["lo"], upper=p["hi"], axis=1)
    return pd.DataFrame(p["scaler"].transform(Xl), index=X.index, columns=X.columns)


def ols_fit(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def main():
    t0 = time.time()
    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))

    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in LOG_FEATURES if c in feature_cols]
    grouped = [c for v in GROUPS.values() for c in v]
    assert len(grouped) == len(set(grouped)) == len(feature_cols), \
        "Grup atamasi ortusuyor ya da eksik"
    assert set(grouped) == set(feature_cols), "Grup atamasi ozellik setiyle uyusmuyor"

    d = feat.copy()
    d["year"] = d["Date_parsed"].dt.year
    d["har_daily"] = daily_ret.abs().shift(1).values
    first_full = int(d[feature_cols].notna().all(axis=1).idxmax())
    test_years = list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))

    print("=== SHAP ANALIZI: XGBoost ile HAR-X ayni degiskenleri mi kullaniyor? ===")
    print("TreeSHAP, XGBoost yerlesik pred_contribs (shap paketi gerekmez).")
    print(f"{len(feature_cols)} ozellik, {len(GROUPS)} grup, "
          f"{len(SHARED_REGRESSORS)} ortak regresor.\n")

    shap_rows, beta_rows, sign_rows = [], [], []

    for h in HORIZONS:
        print(f"--- h={h} ---")
        d["y"] = tgt[f"target_vol_{h}"].values
        d["pv"] = daily_ret.rolling(h).std().shift(1).values
        for fold_id, ty in enumerate(test_years, start=1):
            include = not (ty == PARTIAL_YEAR and h in EXCLUDE_2026_HORIZONS)
            tr_all = d.index[d["year"] < ty]
            te_all = d.index[d["year"] == ty]
            tr_emb = tr_all[:-h]
            assert tr_emb.max() + h < te_all.min()

            need = feature_cols + ["y", "pv"]
            tr = tr_emb[(tr_emb >= first_full)
                        & d.loc[tr_emb, need].notna().all(axis=1).values]
            te = te_all[d.loc[te_all, need].notna().all(axis=1).values]

            X_tr, X_te = d.loc[tr, feature_cols], d.loc[te, feature_cols]
            y_tr = d.loc[tr, "y"].to_numpy("float64")
            pv_tr = d.loc[tr, "pv"].to_numpy("float64")
            pp = fit_preproc(X_tr, log_cols)
            Xs_tr, Xs_te = apply_preproc(X_tr, pp), apply_preproc(X_te, pp)

            n_eff = len(tr) / h
            tier, params = select_capacity(n_eff)
            model = XGBRegressor(**params)
            model.fit(Xs_tr, np.log(y_tr) - np.log(pv_tr))

            # --- TreeSHAP, on the test slice ------------------------------
            contribs = model.get_booster().predict(
                xgb.DMatrix(Xs_te, feature_names=list(Xs_te.columns)),
                pred_contribs=True)
            assert contribs.shape == (len(te), len(feature_cols) + 1)
            assert np.allclose(contribs.sum(1), model.predict(Xs_te), atol=1e-4), \
                f"h={h} {ty}: SHAP toplama ozelligi saglanmiyor"
            sv = contribs[:, :-1]
            mean_abs = np.abs(sv).mean(axis=0)

            for j, c in enumerate(feature_cols):
                shap_rows.append({"horizon": h, "fold": fold_id, "test_year": ty,
                                  "include_in_main": include, "ozellik": c,
                                  "grup": next(g for g, v in GROUPS.items()
                                               if c in v),
                                  "mean_abs_shap": float(mean_abs[j])})

            # --- HAR-X standardized betas ---------------------------------
            okx = d.loc[tr_emb, HARX_COLS + ["y"]].notna().all(axis=1)
            trx = tr_emb[okx.values]
            Xh = d.loc[trx, HARX_COLS].to_numpy("float64")
            yh = d.loc[trx, "y"].to_numpy("float64")
            beta = ols_fit(Xh, yh)[1:]
            std_beta = beta * Xh.std(axis=0) / yh.std()
            for j, c in enumerate(HARX_COLS):
                beta_rows.append({"horizon": h, "fold": fold_id, "test_year": ty,
                                  "include_in_main": include, "regresor": c,
                                  "grup": HARX_GROUP[c],
                                  "std_beta": float(std_beta[j]),
                                  "abs_std_beta": float(abs(std_beta[j]))})

            # --- Sign agreement: the five shared regressors ---------------
            # Because the preprocessing is monotonically INCREASING, the scaled and raw
            # values give the same sign correlation.
            for c in SHARED_REGRESSORS:
                j = feature_cols.index(c)
                xv, s = Xs_te[c].to_numpy("float64"), sv[:, j]
                unused = not (np.abs(s) > 0).any()
                r = (float(np.corrcoef(xv, s)[0, 1])
                     if (not unused and xv.std() > 0 and s.std() > 0) else np.nan)
                b = std_beta[HARX_COLS.index(c)]
                sign_rows.append({
                    "horizon": h, "fold": fold_id, "test_year": ty,
                    "include_in_main": include, "regresor": c,
                    "kullanilmadi": bool(unused),
                    "oran_karistirmasi": c in ("brent_vol5", "brent_vol20"),
                    "xgb_yon_korelasyon": r,
                    "harx_std_beta": float(b),
                    # An unused feature is NOT an "opposite direction" -> agreement is left NaN.
                    "uyum": (bool(np.sign(r) == np.sign(b))
                             if np.isfinite(r) else np.nan),
                })
        print()

    shap_df = pd.DataFrame(shap_rows)
    beta_df = pd.DataFrame(beta_rows)
    sign_df = pd.DataFrame(sign_rows)
    S = shap_df[shap_df["include_in_main"]]
    B = beta_df[beta_df["include_in_main"]]
    G = sign_df[sign_df["include_in_main"]]

    pd.set_option("display.width", 250)

    # ===== 1) Feature-level importance ====================================
    fi = S.groupby(["horizon", "ozellik", "grup"])["mean_abs_shap"].mean().reset_index()
    fi["pay_pct"] = fi.groupby("horizon")["mean_abs_shap"].transform(
        lambda v: 100 * v / v.sum())
    print("=== En onemli 10 ozellik (ufuk basina, fold ortalamasi) ===")
    for h in HORIZONS:
        top = fi[fi["horizon"] == h].nlargest(10, "mean_abs_shap")
        print(f"h={h}: " + ", ".join(
            f"{r.ozellik}({r.pay_pct:.1f}%)" for r in top.itertuples()))
    print()

    # ===== 2) Group shares (THE REAL COMPARISON) ==========================
    gi = S.groupby(["horizon", "grup"])["mean_abs_shap"].mean().reset_index()
    gsum = S.groupby(["horizon", "grup"])["mean_abs_shap"].sum().reset_index()
    gsum["pay_pct"] = gsum.groupby("horizon")["mean_abs_shap"].transform(
        lambda v: 100 * v / v.sum())
    bsum = B.groupby(["horizon", "grup"])["abs_std_beta"].sum().reset_index()
    bsum["pay_pct"] = bsum.groupby("horizon")["abs_std_beta"].transform(
        lambda v: 100 * v / v.sum())

    print("=== GRUP PAYLARI (%) -- asil karsilastirma ===")
    print("XGBoost (toplam |SHAP| payi):")
    print(gsum.pivot(index="grup", columns="horizon", values="pay_pct").to_string(
        float_format=lambda v: f"{v:.1f}"))
    print("\nHAR-X (toplam |std beta| payi, yalnizca 3 grup):")
    print(bsum.pivot(index="grup", columns="horizon", values="pay_pct").to_string(
        float_format=lambda v: f"{v:.1f}"))

    outside = gsum[~gsum["grup"].isin(SHARED_GROUPS)].groupby("horizon")[
        "pay_pct"].sum()
    print("\n(3) XGBoost'un HAR-X DISINDAKI gruplara verdigi toplam agirlik (%):")
    print(outside.to_string(float_format=lambda v: f"{v:.1f}"))
    print("Bu pay buyukse XGBoost farkli yerlere bakiyor demektir.")
    print()

    # ===== 3) Spearman comparisons ========================================
    print("=== SPEARMAN SIRA KORELASYONLARI ===")
    print("UYARI: 3 grup uzerinde Spearman yalnizca -1/-0.5/+0.5/+1 alabilir ve")
    print("p-degeri en iyi durumda 0.33'tur; anlamlilik MATEMATIKSEL OLARAK IMKANSIZ.")
    sp_rows = []
    for h in HORIZONS:
        gx = gsum[(gsum["horizon"] == h) & gsum["grup"].isin(SHARED_GROUPS)] \
            .set_index("grup")["pay_pct"].reindex(SHARED_GROUPS)
        gb = bsum[bsum["horizon"] == h].set_index("grup")["pay_pct"].reindex(
            SHARED_GROUPS)
        rho_g, p_g = stats.spearmanr(gx.values, gb.values)
        fx = fi[(fi["horizon"] == h) & fi["ozellik"].isin(SHARED_REGRESSORS)] \
            .set_index("ozellik")["mean_abs_shap"].reindex(SHARED_REGRESSORS)
        fb = B[B["horizon"] == h].groupby("regresor")["abs_std_beta"].mean() \
            .reindex(SHARED_REGRESSORS)
        rho_f, p_f = stats.spearmanr(fx.values, fb.values)
        sp_rows.append({"horizon": h, "grup_rho": rho_g, "grup_p": p_g,
                        "ortak_regresor_rho": rho_f, "ortak_regresor_p": p_f})
    print(pd.DataFrame(sp_rows).to_string(index=False,
                                          float_format=lambda v: f"{v:.3f}"))
    print()

    # ===== 4) Sign agreement ==============================================
    print("=== ISARET UYUMU (ortak 5 regresor) ===")
    print("XGBoost yonu = corr(ozellik degeri, kendi SHAP degeri).")
    print("HAR-X yonu = standartlastirilmis beta isareti.")
    print()
    print("(a) KULLANILMAYAN OZELLIK: SHAP ozdes sifir -> yon tanimsiz. Bu 'zit yon'")
    print("    DEGILDIR; uyum oranindan dislanir.")
    unused_tbl = G.groupby("horizon")["kullanilmadi"].agg(["sum", "size"])
    unused_tbl["pct"] = 100 * unused_tbl["sum"] / unused_tbl["size"]
    print(unused_tbl.to_string(float_format=lambda v: f"{v:.1f}"))
    print()
    U = G[~G["kullanilmadi"]].copy()
    U["uyum"] = U["uyum"].astype(bool)
    print("(b) ORAN-HEDEFI KARISTIRMASI: brent_vol5/brent_vol20 paydadaki past_vol ile")
    print("    mekanik ilintilidir (h=5'te brent_vol5 paydanin KENDISI, kor.=1.000).")
    print("    Bu iki ozellikte karsilastirma GECERSIZDIR.")
    print()
    print("Regresor bazinda (yalnizca kullanilan fold'lar):")
    agg = U.groupby("regresor").agg(
        kullanilan=("uyum", "size"), uyumlu=("uyum", "sum"),
        xgb_kor_ort=("xgb_yon_korelasyon", "mean"),
        harx_beta_ort=("harx_std_beta", "mean")).reset_index()
    agg["uyum_pct"] = 100 * agg["uyumlu"] / agg["kullanilan"]
    agg["gecerli"] = ~agg["regresor"].isin(["brent_vol5", "brent_vol20"])
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    EX = U[~U["oran_karistirmasi"]]
    print("GECERLI KARSILASTIRMA: yalnizca dissal regresorler "
          "(ovx_lag1, gprd_lag1, gprd_threat_lag1)")
    ext = EX.groupby("horizon").agg(karsilastirma=("uyum", "size"),
                                    uyumlu=("uyum", "sum"))
    ext["uyum_pct"] = 100 * ext["uyumlu"] / ext["karsilastirma"]
    print(ext.to_string(float_format=lambda v: f"{v:.1f}"))
    print(f"TOPLAM (dissal): {int(EX['uyum'].sum())}/{len(EX)} "
          f"(%{100*EX['uyum'].mean():.1f})")
    print()

    # ===== 5) Stability across folds ======================================
    print("=== FOLD'LAR ARASI KARARLILIK ===")
    print("UYARI: genisleyen pencerede ardisik fold'larin EGITIM VERISI buyuk olcude")
    print("ORTUSUR (fold k'nin train seti fold k+1'inkinin alt kumesidir). Bu yuzden")
    print("yuksek ardisik Spearman korelasyonu KISMEN MEKANIKTIR ve bagimsiz bir")
    print("kararlilik kaniti SAYILMAZ. En durust olcut, ortusmenin en az oldugu")
    print("ILK-SON fold ciftidir; mesafeye gore korelasyon dususu de asagida verilir.")
    print()
    stab_rows, dist_rows = [], []
    for h in HORIZONS:
        sub = S[S["horizon"] == h]
        piv = sub.pivot(index="ozellik", columns="test_year",
                        values="mean_abs_shap")
        yrs = sorted(piv.columns)
        # Training set sizes (used to compute the overlap share)
        tsize = {}
        for ty in yrs:
            tr_all = d.index[d["year"] < ty]
            tr_emb = tr_all[:-h]
            tsize[ty] = int(((tr_emb >= first_full)
                             & d.loc[tr_emb, feature_cols + ["y", "pv"]]
                             .notna().all(axis=1).values).sum())
        # Expanding window: the smaller fold's training set is a SUBSET of the larger one's
        ov_cons = [100 * tsize[a] / tsize[b] for a, b in zip(yrs[:-1], yrs[1:])]
        ov_first_last = 100 * tsize[yrs[0]] / tsize[yrs[-1]]
        rho_first_last = stats.spearmanr(piv[yrs[0]], piv[yrs[-1]]).statistic

        # Mean correlation and mean overlap by fold distance
        for k in range(1, len(yrs)):
            pairs = [(yrs[i], yrs[i + k]) for i in range(len(yrs) - k)]
            dist_rows.append({
                "horizon": h, "fold_mesafesi": k, "cift": len(pairs),
                "rho_ort": float(np.mean([stats.spearmanr(piv[a], piv[b]).statistic
                                          for a, b in pairs])),
                "ortusme_ort_pct": float(np.mean(
                    [100 * tsize[a] / tsize[b] for a, b in pairs])),
            })

        cons = [stats.spearmanr(piv[a], piv[b]).statistic
                for a, b in zip(yrs[:-1], yrs[1:])]
        pooled = piv.mean(axis=1)
        vs_pool = [stats.spearmanr(piv[y], pooled).statistic for y in yrs]
        top5 = {y: set(piv[y].nlargest(5).index) for y in yrs}
        uniq_top5 = len(set().union(*top5.values()))
        first = [piv[y].idxmax() for y in yrs]
        mode_first = max(set(first), key=first.count)
        conc = float(100 * pooled.nlargest(5).sum() / pooled.sum())
        stab_rows.append({
            "horizon": h, "n_fold": len(yrs),
            "ardisik_rho_ort": float(np.mean(cons)),
            "ardisik_rho_min": float(np.min(cons)),
            "ardisik_rho_maks": float(np.max(cons)),
            "ardisik_ortusme_pct": float(np.mean(ov_cons)),
            "ilk_son_rho": float(rho_first_last),
            "ilk_son_ortusme_pct": float(ov_first_last),
            "havuza_rho_ort": float(np.mean(vs_pool)),
            "ilk5e_giren_farkli_ozellik": uniq_top5,
            "en_sik_birinci": mode_first,
            "birinci_kalma_fold": first.count(mode_first),
            "ilk5_yogunlasma_pct": conc,
        })
    stab = pd.DataFrame(stab_rows)
    dist = pd.DataFrame(dist_rows)
    print(stab[["horizon", "n_fold", "ardisik_rho_ort", "ardisik_rho_min",
                "ardisik_ortusme_pct", "ilk_son_rho", "ilk_son_ortusme_pct",
                "havuza_rho_ort"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("EN DURUST OLCUT -- ilk fold ile son fold (ortusme en az):")
    for r in stab_rows:
        print(f"  h={r['horizon']:3d}: rho = {r['ilk_son_rho']:+.3f} "
              f"(egitim verisi ortusmesi %{r['ilk_son_ortusme_pct']:.1f}) "
              f"| ardisik rho = {r['ardisik_rho_ort']:+.3f} "
              f"(ortusme %{r['ardisik_ortusme_pct']:.1f})")
    print()
    print("Fold mesafesine gore korelasyon dususu ve egitim verisi ortusmesi:")
    print(dist.pivot(index="fold_mesafesi", columns="horizon",
                     values="rho_ort").to_string(
        float_format=lambda v: f"{v:.3f}"))
    print("\nAyni mesafelerde ortalama egitim verisi ortusmesi (%):")
    print(dist.pivot(index="fold_mesafesi", columns="horizon",
                     values="ortusme_ort_pct").to_string(
        float_format=lambda v: f"{v:.1f}"))
    print()

    fi.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)
    gsum.to_csv(OUT_DIR / "shap_group_importance.csv", index=False)
    beta_df.to_csv(OUT_DIR / "harx_standardized_betas.csv", index=False)
    stab.to_csv(OUT_DIR / "shap_stability.csv", index=False)
    dist.to_csv(OUT_DIR / "shap_stability_by_distance.csv", index=False)
    sign_df.to_csv(OUT_DIR / "shap_sign_agreement.csv", index=False)

    with open(OUT_DIR / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "TreeSHAP via xgboost pred_contribs (shap paketi kullanilmadi)",
            "shap_units": ("log-oran uzayi (log(vol)-log(past_vol)); ham volatilite "
                           "birimi DEGIL"),
            "comparability_note": ("mean|SHAP| ile standartlastirilmis beta ayni "
                                   "buyukluk degildir; yalnizca siralama ve pay "
                                   "duzeyinde karsilastirilabilir"),
            "spearman_warning": ("3 grup uzerinde Spearman -1/-0.5/+0.5/+1 disinda "
                                 "deger alamaz; p >= 0.33, anlamlilik imkansiz"),
            "groups": {k: len(v) for k, v in GROUPS.items()},
            "shared_regressors": SHARED_REGRESSORS,
            "har_daily_excluded_from_sign": (
                "XGBoost setinde tam karsiligi yok; en yakini brent_ret_lag1 "
                "(isaretli getiri), |r| ile iliskisi U-bicimli"),
            "group_shares_xgb": gsum.to_dict(orient="records"),
            "group_shares_harx": bsum.to_dict(orient="records"),
            "outside_harx_share_pct": outside.to_dict(),
            "spearman": sp_rows,
            "sign_test_limits": {
                "unused_feature": ("SHAP ozdes sifir ise yon tanimsizdir; 'zit yon' "
                                   "degildir ve uyum oranindan dislanir"),
                "ratio_target_confound": ("brent_vol5/brent_vol20 log-oran hedefinin "
                                          "paydasiyla mekanik ilintilidir; bu iki "
                                          "ozellikte isaret karsilastirmasi gecersiz"),
            },
            "sign_agreement_unused_pct": float(100 * G["kullanilmadi"].mean()),
            "sign_agreement_all_used_pct": float(100 * U["uyum"].mean()),
            "sign_agreement_exogenous_pct": float(100 * EX["uyum"].mean()),
            "stability": stab_rows,
            "stability_warning": (
                "Genisleyen pencerede ardisik fold'larin egitim verisi buyuk olcude "
                "ortusur (kucugun train seti buyugun ALT KUMESIDIR); yuksek ardisik "
                "Spearman KISMEN MEKANIKTIR ve bagimsiz kararlilik kaniti degildir. "
                "En durust olcut ilk-son fold ciftidir (ilk_son_rho)."),
            "stability_by_distance": dist_rows,
            "runtime_seconds": round(time.time() - t0, 2),
        }, f, ensure_ascii=False, indent=2, default=str)

    print("Yazildi: shap_feature_importance.csv, shap_group_importance.csv, "
          "harx_standardized_betas.csv, shap_stability.csv, shap_sign_agreement.csv")
    print("Rapor  : shap_summary.json")
    print(f"Sure   : {time.time() - t0:.1f} saniye")


if __name__ == "__main__":
    main()
