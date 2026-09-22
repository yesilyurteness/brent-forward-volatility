"""Direct test of the date-gap effect: correct the target, hold the predictions fixed.

14_date_gap_diagnostics.py finds 40 returns (all 2008-2016) that span one or more skipped
trading days. The 2017+ subsample check (12_robustness_gapfree.py) cannot separate the gap
effect from the change of period, because 2012-2016 and 2017-2026 differ in volatility
regime as well. This script isolates the gap effect:

  1. Each real-gap return r spanning 1+k trading days (k = skipped trading days) is
     rescaled to r / sqrt(1+k), its one-day equivalent under the random-walk assumption
     that variance grows linearly with time.
  2. The forward-volatility target is rebuilt from the corrected returns with the same
     formula as 01_build_targets.py (std of the next h returns, skipna=False).
  3. Every model's committed out-of-sample predictions are kept UNCHANGED and the fold
     RMSE/MAE are recomputed against the corrected target.
  4. The model ranking under the original and the corrected target is compared.

Only the evaluation target changes. Features and training targets are not corrected, so
this measures how much the gaps move the reported metrics, not how a model retrained on
gap-free data would behave. No model is refit.

The number of test targets whose value changes must equal the affected test-target count
of 14_date_gap_diagnostics.py (n_affected_test in date_gap_target_exposure.csv); the
script recomputes that count with 14's own functions and asserts the match. The
unmodified fold means must reproduce the published tables; that is asserted as well.

The gap classification is imported from 14_date_gap_diagnostics.py, so this script does
not depend on 14 having been run.

Inputs : data/veriseti.xlsx, outputs/{bench,wf,bilstm,hybrid}_predictions_all.csv,
         outputs/ablation_exogenous_predictions.csv, outputs/all_models_comparison.csv,
         outputs/ablation_exogenous.csv
Outputs: outputs/gap_target_test.csv, outputs/gap_target_test_folds.csv,
         outputs/gap_target_test_summary.json

Runtime: a few seconds.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
HORIZONS = [5, 22, 66, 126]
KEY = ["horizon", "Date", "test_year", "include_in_main", "y_true"]

_spec = importlib.util.spec_from_file_location(
    "date_gap_diagnostics", ROOT / "scripts" / "14_date_gap_diagnostics.py")
gapdiag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gapdiag)


def corrected_targets(raw, g):
    """Target at each Date for each horizon, from gap-corrected returns."""
    ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    k = pd.Series(0, index=raw.index)
    k.loc[g["row"].to_numpy()] = g["n_unexplained_weekdays"].to_numpy()
    ret_c = ret / np.sqrt(1 + k)
    out = {}
    for h in HORIZONS:
        future = pd.concat([ret_c.shift(-i) for i in range(1, h + 1)], axis=1)
        out[h] = pd.Series(future.std(axis=1, skipna=False).to_numpy(), index=raw["Date"])
    return out


def load_predictions():
    parts = []
    bench = pd.read_csv(OUT_DIR / "bench_predictions_all.csv")
    for m in ["har", "har_log", "har_x", "har_x_log", "garch"]:
        parts.append(bench[KEY].assign(model=m, pred=bench[f"pred_{m}"]))
    wf = pd.read_csv(OUT_DIR / "wf_predictions_all.csv")
    parts.append(wf[KEY].assign(model="xgboost", pred=wf["pred_xgboost"]))
    bil = pd.read_csv(OUT_DIR / "bilstm_predictions_all.csv")
    for m in ["bilstm", "train_mean", "past_vol"]:
        parts.append(bil[KEY].assign(model=m, pred=bil[f"pred_{m}"]))
    hyb = pd.read_csv(OUT_DIR / "hybrid_predictions_all.csv")
    parts.append(hyb[KEY].assign(model="h1_xgb_bilstm", pred=hyb["pred_h1_xgb_bilstm"]))
    abl = pd.read_csv(OUT_DIR / "ablation_exogenous_predictions.csv")
    for m in ["har_ovx", "har_gpr"]:
        a = abl[abl["variant"] == m]
        parts.append(a[KEY].assign(model=m, pred=a["pred"]))
    p = pd.concat(parts, ignore_index=True)
    p = p[p["include_in_main"].astype(bool)]
    # every model is evaluated on the same test rows
    n = p.groupby(["horizon", "model"]).size().unstack()
    assert (n.nunique(axis=1) == 1).all(), n
    return p


def rank(s):
    return s.groupby(level="horizon").rank(method="min").astype(int)


def main():
    raw = gapdiag.load_dates()
    g, _, _ = gapdiag.classify_gaps(raw["Date_parsed"])
    tgt = corrected_targets(raw, g)

    p = load_predictions()
    p["y_corrected"] = [tgt[h].loc[d] for h, d in zip(p["horizon"], p["Date"])]
    # The corrected target must equal the original wherever no gap is in the window
    p["changed"] = ~np.isclose(p["y_corrected"], p["y_true"], rtol=1e-12, atol=0)

    one = p[p["model"] == "har"]
    flag = gapdiag.real_gap_flags(g, len(raw))
    years = raw["Date_parsed"].dt.year.to_numpy()
    expected, n_test = {}, {}
    for h in HORIZONS:
        hit, valid = gapdiag.window_hits(flag, h)
        tm = valid & gapdiag.test_fold_mask(years, h)
        expected[h], n_test[h] = int((hit & tm).sum()), int(tm.sum())
    n_changed = one.groupby("horizon")["changed"].sum()
    for h in HORIZONS:
        assert n_changed[h] == expected[h], (h, n_changed[h], expected[h])
        assert (one["horizon"] == h).sum() == n_test[h]
    rel = (one["y_corrected"] / one["y_true"] - 1)[one["changed"]]
    tchg = rel.groupby(one.loc[one["changed"], "horizon"]).agg(["mean", "min", "max"])

    def fold_metrics(s):
        e_o, e_c = s["y_true"] - s["pred"], s["y_corrected"] - s["pred"]
        return pd.Series({"n_test": len(s),
                          "rmse_original": np.sqrt(np.mean(e_o ** 2)),
                          "rmse_corrected": np.sqrt(np.mean(e_c ** 2)),
                          "mae_original": np.mean(np.abs(e_o)),
                          "mae_corrected": np.mean(np.abs(e_c)),
                          "n_targets_changed": int(s["changed"].sum())})
    folds = (p.groupby(["horizon", "model", "test_year"])
              .apply(fold_metrics, include_groups=False).reset_index())
    folds.to_csv(OUT_DIR / "gap_target_test_folds.csv", index=False)

    agg = folds.groupby(["horizon", "model"])[
        ["rmse_original", "rmse_corrected", "mae_original", "mae_corrected"]].mean()
    agg.insert(0, "n_folds", folds.groupby(["horizon", "model"]).size())

    # Sanity: the unmodified fold means reproduce the published tables
    ref = pd.read_csv(OUT_DIR / "all_models_comparison.csv").set_index(["horizon", "model"])
    abl = pd.read_csv(OUT_DIR / "ablation_exogenous.csv").rename(
        columns={"variant": "model"}).set_index(["horizon", "model"])
    ref = pd.concat([ref[["rmse_fold_mean", "mae_fold_mean"]],
                     abl.loc[abl.index.get_level_values("model").isin(["har_ovx", "har_gpr"]),
                             ["rmse_fold_mean", "mae_fold_mean"]]])
    chk = agg.join(ref, how="inner")
    assert np.allclose(chk["rmse_original"], chk["rmse_fold_mean"], rtol=1e-9)
    assert np.allclose(chk["mae_original"], chk["mae_fold_mean"], rtol=1e-9)
    print(f"Kontrol: duzeltilmemis fold ortalamalari yayimlanan tablolarla ayni ({len(chk)} satir).")

    agg["rmse_change_pct"] = 100 * (agg["rmse_corrected"] / agg["rmse_original"] - 1)
    agg["mae_change_pct"] = 100 * (agg["mae_corrected"] / agg["mae_original"] - 1)
    agg["rank_rmse_original"] = rank(agg["rmse_original"])
    agg["rank_rmse_corrected"] = rank(agg["rmse_corrected"])
    agg["rank_mae_original"] = rank(agg["mae_original"])
    agg["rank_mae_corrected"] = rank(agg["mae_corrected"])
    out = agg.reset_index().sort_values(["horizon", "rank_rmse_original", "model"])
    out.to_csv(OUT_DIR / "gap_target_test.csv", index=False)

    summary = {"correction": "gap return r -> r / sqrt(1 + k), k = skipped trading days",
               "predictions": "held fixed; no refit", "horizons": {}}
    for h in HORIZONS:
        s = out[out["horizon"] == h]
        swaps = s[s["rank_mae_original"] != s["rank_mae_corrected"]]
        summary["horizons"][str(h)] = {
            "n_test_targets_changed": int(n_changed[h]),
            "n_test_targets": n_test[h],
            "target_change_pct_mean": float(100 * tchg.loc[h, "mean"]),
            "target_change_pct_min": float(100 * tchg.loc[h, "min"]),
            "rmse_change_pct_min": float(s["rmse_change_pct"].min()),
            "rmse_change_pct_max": float(s["rmse_change_pct"].max()),
            "n_rmse_rank_changes": int((s["rank_rmse_original"] != s["rank_rmse_corrected"]).sum()),
            "n_mae_rank_changes": int(len(swaps)),
            "mae_rank_swaps": swaps["model"].tolist(),
        }
        d = summary["horizons"][str(h)]
        print(f"\n=== h={h}: degisen test hedefi {d['n_test_targets_changed']}/"
              f"{d['n_test_targets']}, ort. hedef degisimi {d['target_change_pct_mean']:+.2f}% | "
              f"RMSE degisimi [{d['rmse_change_pct_min']:+.2f}%, {d['rmse_change_pct_max']:+.2f}%] | "
              f"RMSE sira degisimi {d['n_rmse_rank_changes']}, MAE sira degisimi "
              f"{d['n_mae_rank_changes']} {d['mae_rank_swaps']}")
        print(s[["model", "rmse_original", "rmse_corrected", "rmse_change_pct",
                 "rank_rmse_original", "rank_rmse_corrected"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    with open(OUT_DIR / "gap_target_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nYazildi: gap_target_test.csv, gap_target_test_folds.csv, gap_target_test_summary.json")


if __name__ == "__main__":
    main()
