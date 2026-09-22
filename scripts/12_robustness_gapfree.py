"""Robustness check: repeat the main model comparison on the gap-free 2017-2026 folds only.

Date-gap diagnosis (14_date_gap_diagnostics.py): 40 rows (55 skipped trading days, not
explained by weekends or NYSE holidays) exist in the merged data, all between 2008 and
2016. From 2017 on, every consecutive-row return spans exactly one trading day. This script
re-aggregates the already-computed per-fold out-of-sample metrics over the test years
>= 2017 and compares the model ranking against the full-sample main table.

No model is refit and no new test prediction is made; the per-fold metrics are read from
the committed outputs of scripts 03, 05, 06, 07 and 11. The subsample is fixed a priori by
the data-gap diagnosis, not by any test result (Critical Rule 5). The partial-year rule is
kept: include_in_main=False folds (2026 at h=66/126) are excluded.

Caveat: the training sets of the 2017+ folds still contain the 2008-2016 rows, including
the gap returns; only the evaluated test windows are gap-free. The 2017+ ranking also
differs from the full-sample ranking for reasons unrelated to gaps (2012-2016 and 2017-2026
are different volatility regimes); 13_gap_target_test.py isolates the gap effect directly.

Inputs : outputs/{wf,bench,bilstm,hybrid}_metrics_all.csv, outputs/ablation_exogenous_folds.csv
Outputs: outputs/robustness_gapfree_2017plus.csv, outputs/robustness_gapfree_2017plus_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
GAPFREE_START = 2017
GAP_PERIOD = (2012, 2016)
HORIZONS = [5, 22, 66, 126]
MODEL_ORDER = ["har_x_log", "har_x", "har_ovx", "har", "har_log", "xgboost", "bilstm",
               "h1_xgb_bilstm", "garch", "past_vol", "train_mean"]
COLS = ["horizon", "test_year", "include_in_main", "model", "rmse", "mae", "r2_oos"]


def load_fold_metrics():
    # Same sources as all_models_comparison.csv (scripts/06_attention_bilstm.py)
    bil = pd.read_csv(OUT_DIR / "bilstm_metrics_all.csv")
    wf = pd.read_csv(OUT_DIR / "wf_metrics_all.csv")
    wf = wf[wf["model"] == "xgboost"]
    bench = pd.read_csv(OUT_DIR / "bench_metrics_all.csv")
    bench = bench[~bench["model"].isin(["train_mean", "past_vol"])]
    # HAR + OVX from the exogenous ablation (scripts/11)
    abl = pd.read_csv(OUT_DIR / "ablation_exogenous_folds.csv")
    abl = abl[abl["variant"] == "har_ovx"].rename(columns={"variant": "model"})
    # Simple-average hybrid 0.5*XGB + 0.5*BiLSTM (scripts/07)
    hyb = pd.read_csv(OUT_DIR / "hybrid_metrics_all.csv")
    hyb = hyb[hyb["model"] == "h1_xgb_bilstm"]
    m = pd.concat([d[COLS] for d in (bil, wf, bench, abl, hyb)], ignore_index=True)
    assert not m.duplicated(["horizon", "test_year", "model"]).any()
    return m[m["include_in_main"].astype(bool)]


def aggregate(m, label):
    a = m.groupby(["horizon", "model"]).agg(
        n_folds=("rmse", "size"), first_year=("test_year", "min"), last_year=("test_year", "max"),
        rmse_fold_mean=("rmse", "mean"), mae_fold_mean=("mae", "mean"),
        r2_oos_fold_mean=("r2_oos", "mean")).reset_index()
    a["rank_rmse"] = a.groupby("horizon")["rmse_fold_mean"].rank(method="min").astype(int)
    a["rank_mae"] = a.groupby("horizon")["mae_fold_mean"].rank(method="min").astype(int)
    return a.set_index(["horizon", "model"]).add_suffix(f"_{label}")


def main():
    m = load_fold_metrics()

    full = aggregate(m, "full")
    # Sanity check: the full-sample aggregation must reproduce the committed main table
    ref = pd.read_csv(OUT_DIR / "all_models_comparison.csv").set_index(["horizon", "model"])
    chk = full.join(ref, how="inner")
    assert len(chk) == len(ref), "main-table models missing"
    for c in ["rmse_fold_mean", "mae_fold_mean", "r2_oos_fold_mean"]:
        assert np.allclose(chk[f"{c}_full"], chk[c], rtol=1e-9, atol=1e-12), c
    assert (chk["n_folds_full"] == chk["n_folds"]).all()
    print(f"Kontrol: tam orneklem ana tabloyu birebir uretiyor ({len(ref)} satir).")

    gapfree = aggregate(m[m["test_year"] >= GAPFREE_START], "2017plus")
    gapper = aggregate(m[m["test_year"].between(*GAP_PERIOD)], "2012_2016")
    out = full.join(gapfree).join(gapper).reset_index()
    out["rank_rmse_change"] = out["rank_rmse_2017plus"] - out["rank_rmse_full"]
    out["rank_mae_change"] = out["rank_mae_2017plus"] - out["rank_mae_full"]
    out["_o"] = out["model"].map({k: i for i, k in enumerate(MODEL_ORDER)})
    out = out.sort_values(["horizon", "rank_rmse_full", "_o"]).drop(columns="_o")
    out.to_csv(OUT_DIR / "robustness_gapfree_2017plus.csv", index=False)

    summary = {"gapfree_start_year": GAPFREE_START, "gap_period_years": list(GAP_PERIOD),
               "note": "re-aggregation of existing per-fold OOS metrics; no refit", "horizons": {}}
    for h in HORIZONS:
        s = out[out["horizon"] == h]
        top = lambda col, k: list(s.sort_values(col)["model"].head(k))
        d = {}
        for metric in ["rmse", "mae"]:
            d[f"spearman_{metric}_full_vs_2017plus"] = float(
                s[f"rank_{metric}_full"].corr(s[f"rank_{metric}_2017plus"], method="spearman"))
            d[f"spearman_{metric}_full_vs_2012_2016"] = float(
                s[f"rank_{metric}_full"].corr(s[f"rank_{metric}_2012_2016"], method="spearman"))
            d[f"top3_{metric}_full"] = top(f"{metric}_fold_mean_full", 3)
            d[f"top3_{metric}_2017plus"] = top(f"{metric}_fold_mean_2017plus", 3)
            d[f"n_models_rank_changed_{metric}"] = int((s[f"rank_{metric}_change"] != 0).sum())
        d["n_folds_2017plus"] = int(s["n_folds_2017plus"].iloc[0])
        summary["horizons"][str(h)] = d

        print(f"\n=== h={h}  (tam: {int(s['n_folds_full'].iloc[0])} fold, "
              f"2017+: {d['n_folds_2017plus']} fold)")
        show = s[["model", "rmse_fold_mean_full", "rank_rmse_full", "rmse_fold_mean_2017plus",
                  "rank_rmse_2017plus", "rank_rmse_change", "mae_fold_mean_2017plus",
                  "rank_mae_full", "rank_mae_2017plus", "r2_oos_fold_mean_2017plus"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print(f"Spearman RMSE-sirasi tam vs 2017+: {d['spearman_rmse_full_vs_2017plus']:.3f} | "
              f"MAE: {d['spearman_mae_full_vs_2017plus']:.3f}")

    with open(OUT_DIR / "robustness_gapfree_2017plus_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nYazildi: robustness_gapfree_2017plus.csv, robustness_gapfree_2017plus_summary.json")


if __name__ == "__main__":
    main()
