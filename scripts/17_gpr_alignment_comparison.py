"""Timestamp-aligned vs publication-aligned GPR: how much does the publication lag matter?

WHAT IS COMPARED
----------------
Every model and horizon, once with GPR features aligned to their timestamps (the original
pipeline, unsuffixed files; paper Appendix A) and once aligned to their publication dates
(the *_publication_aligned files; PRIMARY). The RMSE difference between the two is the
measured cost of respecting the publication lag -- a finding in its own right.

SAME SAMPLE, PURE ALIGNMENT EFFECT
----------------------------------
The two versions are evaluated on EXACTLY the same samples: only the GPR-derived feature
columns differ, the first fully valid feature row stays at 127, and every training and
test row count is identical (asserted here per model x horizon x fold, and asserted on
the full prediction files by each model script). The difference is therefore a pure
alignment effect, not a sample effect. (Contrast: adding the 252-day volatility window
moved the first valid row and changed the samples, which contaminated that comparison.)
Models that use no GPR input are carried along as a control: their two columns must be
bit-identical, and this script asserts it.

Main metric: fold mean of RMSE / MAE (R2_oos secondary), with the 2026 partial-year fold
excluded at h=66 and h=126 exactly as in the main tables; those folds are listed in a
separate footnote file.

Inputs : {hybrid,bench}_metrics_all, ablation_exogenous_folds, exploratory_xgb6_folds,
         opt_metrics_all, opt_rawsmearing_metrics_all -- each in both versions
Outputs: outputs/gpr_alignment_comparison.csv          (model x horizon, fold means)
         outputs/gpr_alignment_comparison_folds.csv    (model x horizon x fold)
         outputs/gpr_alignment_comparison_2026_footnote.csv
         outputs/gpr_alignment_decomposition.csv       (HAR -> HAR-X -> XGB-6 -> XGB)
         outputs/gpr_alignment_comparison_summary.json

Runtime: a few seconds (reads saved metrics only).
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

# (model id, label, role, uses GPR input)
MODELS = [
    ("xgboost", "XGBoost (primary, 65 features)", "primary", True),
    ("bilstm", "Attention BiLSTM", "primary", True),
    ("h1_xgb_bilstm", "Hybrid H1: 0.5 XGB + 0.5 BiLSTM", "hybrid", True),
    ("h2_harx_xgb", "Hybrid H2: 0.5 HAR-X + 0.5 XGB", "hybrid", True),
    ("h3_harx_resid", "Hybrid H3: HAR-X + XGB residual", "hybrid", True),
    ("har_x", "HAR-X", "econometric", True),
    ("har_x_log", "HAR-X-log", "econometric", True),
    ("har_gpr", "HAR + GPR", "ablation", True),
    ("xgb6", "XGBoost-6 (exploratory)", "exploratory", True),
    ("xgboost_optuna", "XGBoost, Optuna + shrunk smearing", "robustness", True),
    ("xgboost_optuna_raw", "XGBoost, Optuna + raw smearing", "appendix", True),
    ("har", "HAR", "econometric", False),
    ("har_log", "HAR-log", "econometric", False),
    ("har_ovx", "HAR + OVX", "ablation", False),
    ("garch", "GARCH(1,1)", "econometric", False),
    ("train_mean", "Train-mean", "naive", False),
    ("past_vol", "Past-volatility", "naive", False),
]
KEY = ["model", "horizon", "test_year"]
MET = ["n", "rmse", "mae", "r2_oos", "include_in_main"]


def sign_test_p(n_wins, n_trials):
    if n_trials == 0:
        return float("nan")
    k = min(n_wins, n_trials - n_wins)
    return float(min(1.0, 2 * sum(comb(n_trials, i) for i in range(k + 1)) / 2 ** n_trials))


def load(al):
    """Per-fold metrics of every model in one long frame (model, horizon, test_year)."""
    rd = lambda name: pd.read_csv(alignment.out(name, al), float_precision="round_trip")
    parts = []
    hyb = rd("hybrid_metrics_all.csv")        # xgb, bilstm, hybrids, har/har_x(_log), naive
    parts.append(hyb[["model", "horizon", "test_year", "n", "rmse", "mae", "r2_oos",
                      "include_in_main"]])
    ben = rd("bench_metrics_all.csv").rename(columns={"n_test": "n"})
    parts.append(ben[ben["model"].isin(["garch", "har_log"])][parts[0].columns])
    abl = rd("ablation_exogenous_folds.csv").rename(columns={"variant": "model",
                                                             "n_test": "n"})
    parts.append(abl[abl["model"].isin(["har_ovx", "har_gpr"])][parts[0].columns])
    x6 = rd("exploratory_xgb6_folds.csv").rename(columns={"n_test": "n"}).assign(model="xgb6")
    parts.append(x6[parts[0].columns])
    for name, mid in (("opt_metrics_all.csv", "xgboost_optuna"),
                      ("opt_rawsmearing_metrics_all.csv", "xgboost_optuna_raw")):
        o = rd(name).rename(columns={"n_test": "n"})
        o = o[o["model"] == "xgboost"].assign(model=mid)
        parts.append(o[parts[0].columns])
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(KEY).any(), "duplicate model/horizon/fold rows"
    return df


def main():
    ts, pub = load("timestamp"), load("publication")
    j = ts.merge(pub, on=KEY, suffixes=("_ts", "_pub"), validate="1:1", how="outer",
                 indicator=True)
    assert (j["_merge"] == "both").all(), "a model/fold exists in only one version"
    j = j.drop(columns="_merge")
    meta = pd.DataFrame(MODELS, columns=["model", "label", "role", "uses_gpr"])
    assert set(j["model"]) == set(meta["model"]), set(j["model"]) ^ set(meta["model"])
    j = j.merge(meta, on="model")

    # ---- Same sample: identical test sizes and fold inclusion ----------------
    assert (j["n_ts"] == j["n_pub"]).all(), "test sizes differ between versions"
    assert (j["include_in_main_ts"] == j["include_in_main_pub"]).all()
    j = j.rename(columns={"n_ts": "n", "include_in_main_ts": "include_in_main"}).drop(
        columns=["n_pub", "include_in_main_pub"])
    # ---- Control: GPR-free models bit for bit ---------------------------------
    ctrl = j[~j["uses_gpr"]]
    for m in ("rmse", "mae", "r2_oos"):
        assert (ctrl[f"{m}_ts"] == ctrl[f"{m}_pub"]).all(), f"GPR-free model changed: {m}"
    print(f"[check] same sample: in {len(j)} model x horizon x fold rows the test size "
          "and fold inclusion are the same in both versions")
    print(f"[check] models without GPR {sorted(ctrl['model'].unique())}: in {len(ctrl)} fold "
          "rows RMSE/MAE/R2_oos BIT-IDENTICAL")

    for m in ("rmse", "mae"):
        j[f"{m}_pct"] = 100 * (j[f"{m}_pub"] / j[f"{m}_ts"] - 1)
    j["r2_oos_diff"] = j["r2_oos_pub"] - j["r2_oos_ts"]
    order = {m: i for i, (m, *_) in enumerate(MODELS)}
    j = j.sort_values(["horizon", "model", "test_year"],
                      key=lambda s: s.map(order) if s.name == "model" else s)
    j.to_csv(OUT_DIR / "gpr_alignment_comparison_folds.csv", index=False)

    # ---- Fold means over the main folds ----------------------------------------
    rows = []
    for (mdl, h), g in j[j["include_in_main"]].groupby(["model", "horizon"]):
        d = g["rmse_ts"] - g["rmse_pub"]                  # > 0: publication better
        wp, wt = int((d > 0).sum()), int((d < 0).sum())
        r = {"model": mdl, "horizon": h, "n_folds": len(g)}
        for m in ("rmse", "mae", "r2_oos"):
            r[f"{m}_ts"] = g[f"{m}_ts"].mean()
            r[f"{m}_pub"] = g[f"{m}_pub"].mean()
        r["rmse_pct"] = 100 * (r["rmse_pub"] / r["rmse_ts"] - 1)
        r["mae_pct"] = 100 * (r["mae_pub"] / r["mae_ts"] - 1)
        r["r2_oos_diff"] = r["r2_oos_pub"] - r["r2_oos_ts"]
        r["rmse_folds_pub_better"], r["rmse_folds_ts_better"] = wp, wt
        r["rmse_folds_tied"] = int((d == 0).sum())
        r["rmse_sign_p_two_sided"] = sign_test_p(wp, wp + wt)
        rows.append(r)
    agg = pd.DataFrame(rows).merge(meta, on="model")
    agg = agg.sort_values(["horizon", "model"],
                          key=lambda s: s.map(order) if s.name == "model" else s)
    cols = ["model", "label", "role", "uses_gpr", "horizon", "n_folds",
            "rmse_ts", "rmse_pub", "rmse_pct", "mae_ts", "mae_pub", "mae_pct",
            "r2_oos_ts", "r2_oos_pub", "r2_oos_diff", "rmse_folds_pub_better",
            "rmse_folds_ts_better", "rmse_folds_tied", "rmse_sign_p_two_sided"]
    agg = agg[cols]
    agg.to_csv(OUT_DIR / "gpr_alignment_comparison.csv", index=False)

    foot = j[~j["include_in_main"]][["model", "horizon", "test_year", "n", "rmse_ts",
                                     "rmse_pub", "rmse_pct", "mae_ts", "mae_pub",
                                     "mae_pct"]]
    foot.to_csv(OUT_DIR / "gpr_alignment_comparison_2026_footnote.csv", index=False)

    # ---- Four-step decomposition, both versions ------------------------------
    piv = agg.pivot(index="horizon", columns="model", values=["rmse_ts", "rmse_pub"])
    dec = []
    for h in sorted(agg["horizon"].unique()):
        for ver, col in (("timestamp", "rmse_ts"), ("publication", "rmse_pub")):
            s = piv[col].loc[h]
            dec.append({
                "horizon": h, "gpr_alignment": ver,
                "rmse_har": s["har"], "rmse_har_x": s["har_x"], "rmse_xgb6": s["xgb6"],
                "rmse_xgboost": s["xgboost"],
                "pct_exogenous_HAR_to_HARX": 100 * (s["har_x"] / s["har"] - 1),
                "pct_functional_form_HARX_to_XGB6": 100 * (s["xgb6"] / s["har_x"] - 1),
                "pct_feature_package_XGB6_to_XGB": 100 * (s["xgboost"] / s["xgb6"] - 1),
                "pct_total_HAR_to_XGB": 100 * (s["xgboost"] / s["har"] - 1),
            })
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT_DIR / "gpr_alignment_decomposition.csv", index=False)

    note = ("The two versions are evaluated on exactly the same samples (identical "
            "training and test rows in every model x horizon x fold; first fully valid "
            "feature row 127 in both). The difference is a pure alignment effect, not a "
            "sample effect. Models without GPR input are bit-identical across versions.")
    with open(OUT_DIR / "gpr_alignment_comparison_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({"note": note,
                   "rmse_pct_definition": "100 * (RMSE_publication / RMSE_timestamp - 1); "
                                          "positive = respecting the publication lag "
                                          "costs accuracy",
                   "main_folds": "fold mean; 2026 excluded at h=66 and h=126",
                   "n_fold_rows": int(len(j)),
                   "n_control_rows_bit_identical": int(len(ctrl)),
                   "aggregate": agg.to_dict(orient="records"),
                   "decomposition": dec.to_dict(orient="records"),
                   "footnote_2026": foot.to_dict(orient="records")}, f, indent=2,
                  default=str)

    pd.set_option("display.width", 220)
    print("\n=== RMSE: timestamp vs publication-aligned (fold mean, % = pub/ts - 1) ===")
    show = agg.pivot(index="model", columns="horizon", values="rmse_pct").reindex(
        [m for m, *_ in MODELS])
    print(show.to_string(float_format=lambda v: f"{v:+.2f}%"))
    print("\nFolds where publication-aligned is better / folds with a difference (RMSE):")
    wins = agg.assign(w=agg["rmse_folds_pub_better"].astype(str) + "/" +
                      (agg["rmse_folds_pub_better"] + agg["rmse_folds_ts_better"]).astype(str))
    print(wins.pivot(index="model", columns="horizon", values="w").reindex(
        [m for m, _, _, g in MODELS if g]).to_string())
    print("\n=== Four-step decomposition (RMSE %) ===")
    print(dec.to_string(index=False, formatters={
        c: (lambda v: f"{v:.6f}") if c.startswith("rmse") else (lambda v: f"{v:+.2f}")
        for c in dec.columns if c not in ("horizon", "gpr_alignment")}))
    print("\nNote:", note)


if __name__ == "__main__":
    main()
