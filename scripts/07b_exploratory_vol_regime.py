"""EXPLORATORY / SECONDARY ANALYSIS: HAR-X vs H2 by volatility regime.

THIS IS AN EXPLORATORY ANALYSIS. It does NOT change the primary finding and is NOT used
for model selection. Its purpose is to illuminate a single question:

    why do the pooled RMSE and the fold average point in OPPOSITE directions for H2
    (the average of HAR-X and XGBoost)?

The difference measured in stage 7:
    pooled     -> H2 is 0.15-2.81% BETTER than HAR-X at all four horizons
    fold avg.  -> H2 is 0.67-4.35% WORSE than HAR-X at three horizons

The pooled measure sums squared errors and therefore weights high-volatility years
heavily; the fold average gives every year equal weight. Hypothesis: H2 wins in
high-volatility years and loses in low-volatility years. This analysis tests that
hypothesis.

THE SPLIT CRITERION IS MECHANICAL AND WAS NOT CHOSEN BY LOOKING AT RESULTS
--------------------------------------------------------------------------
For each horizon, the MEAN realized volatility of every test year is computed (the mean of
y_true over that year's test slice). The years are then split in two at the MEDIAN of that
value: above the median = high regime, below the median = low regime. The threshold derives
from the data, not from performance.

The median is computed separately per horizon, because the set of folds included in the
main metric varies by horizon (2026 is excluded at h=66 and h=126 under the partial-year
rule).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

HORIZONS = [5, 22, 66, 126]
MODELS = ["har_x", "xgboost", "h2_harx_xgb"]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, "float64")
                                  - np.asarray(p, "float64")) ** 2)))


def main():
    t0 = time.time()
    pred = pd.read_csv(OUT_DIR / "hybrid_predictions_all.csv")
    pred = pred[pred["include_in_main"]].copy()

    pd.set_option("display.width", 240)
    print("=== KESIFSEL / IKINCIL ANALIZ: volatilite rejimine gore HAR-X vs H2 ===")
    print("Birincil bulguyu DEGISTIRMEZ, model secimi icin KULLANILMAZ.")
    print("Bolme kriteri MEKANIK: test yilinin ortalama gerceklesen volatilitesi,")
    print("ufuk basina medyana gore ikiye ayrilir. Esik performanstan turemez.\n")

    year_rows, group_rows, fold_rows = [], [], []

    for h in HORIZONS:
        g = pred[pred["horizon"] == h]
        yr = g.groupby("test_year").agg(
            n=("y_true", "size"), vol_ort=("y_true", "mean"))
        med = float(yr["vol_ort"].median())
        yr["rejim"] = np.where(yr["vol_ort"] > med, "yuksek", "dusuk")

        # RMSE per fold (the main metric is the fold average)
        per_fold = {}
        for m in MODELS:
            per_fold[m] = g.groupby("test_year").apply(
                lambda d, mm=m: rmse(d["y_true"], d[f"pred_{mm}"]),
                include_groups=False)
        pf = pd.DataFrame(per_fold)
        pf["rejim"] = yr["rejim"]
        pf["vol_ort"] = yr["vol_ort"]
        pf["h2_vs_harx_pct"] = 100 * (pf["h2_harx_xgb"] / pf["har_x"] - 1)
        pf["horizon"] = h
        fold_rows.append(pf.reset_index())

        yr["horizon"] = h
        yr["medyan_esik"] = med
        year_rows.append(yr.reset_index())

        for rej, sub in pf.groupby("rejim"):
            sg = g[g["test_year"].isin(sub.index)]
            row = {"horizon": h, "rejim": rej, "n_fold": int(len(sub)),
                   "vol_ort": float(sub["vol_ort"].mean())}
            for m in MODELS:
                row[f"{m}_fold_ort"] = float(sub[m].mean())
                row[f"{m}_havuz"] = rmse(sg["y_true"], sg[f"pred_{m}"])
            row["h2_vs_harx_fold_ort_pct"] = 100 * (
                row["h2_harx_xgb_fold_ort"] / row["har_x_fold_ort"] - 1)
            row["h2_vs_harx_havuz_pct"] = 100 * (
                row["h2_harx_xgb_havuz"] / row["har_x_havuz"] - 1)
            row["h2_kazanan_fold"] = int((sub["h2_harx_xgb"] < sub["har_x"]).sum())
            # This group's weight in the pooled measure: its share of squared error
            row["kareli_hata_payi_pct"] = 100 * float(
                ((sg["y_true"] - sg["pred_har_x"]) ** 2).sum()
                / ((g["y_true"] - g["pred_har_x"]) ** 2).sum())
            group_rows.append(row)

    years = pd.concat(year_rows, ignore_index=True)
    groups = pd.DataFrame(group_rows)
    folds = pd.concat(fold_rows, ignore_index=True)

    print("=== Rejim atamasi (mekanik medyan bolmesi) ===")
    piv = years.pivot(index="test_year", columns="horizon", values="rejim")
    print(piv.to_string())
    print("\nUfuk basina medyan esik (gunluk log-getiri std):")
    print(years.groupby("horizon")["medyan_esik"].first().to_string(
        float_format=lambda v: f"{v:.6f}"))
    print()

    print("=== ANA TABLO: HAR-X vs H2, rejim bazinda ===")
    show = groups[["horizon", "rejim", "n_fold", "vol_ort", "har_x_fold_ort",
                   "h2_harx_xgb_fold_ort", "h2_vs_harx_fold_ort_pct",
                   "h2_vs_harx_havuz_pct", "h2_kazanan_fold",
                   "kareli_hata_payi_pct"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("H2'nin HAR-X'e gore farki (%, fold ortalamasi; negatif = H2 iyi):")
    print(groups.pivot(index="rejim", columns="horizon",
                       values="h2_vs_harx_fold_ort_pct").to_string(
        float_format=lambda v: f"{v:+.2f}"))
    print("\nAyni fark, havuzlanmis olcutle:")
    print(groups.pivot(index="rejim", columns="horizon",
                       values="h2_vs_harx_havuz_pct").to_string(
        float_format=lambda v: f"{v:+.2f}"))
    print("\nYuksek rejimin havuzlanmis olcutteki agirligi (kareli hata payi, %):")
    print(groups.pivot(index="rejim", columns="horizon",
                       values="kareli_hata_payi_pct").to_string(
        float_format=lambda v: f"{v:.1f}"))
    print()

    print("=== Fold bazinda ayrinti ===")
    print(folds[["horizon", "test_year", "rejim", "vol_ort", "har_x",
                 "xgboost", "h2_harx_xgb", "h2_vs_harx_pct"]].to_string(
        index=False, float_format=lambda v: f"{v:.6f}"))
    print()

    # ===================================================================
    # EXTREME-EVENT ROBUSTNESS CHECK
    # ===================================================================
    # The question: is it "ML adds complementary value during extreme events", or
    # "ML added complementary value during the COVID period"? These are claims of
    # different strength.
    # (a) The CALENDAR distribution of the worst 1% of observations -- are they all
    #     concentrated in 2020?
    # (b) The 2020 fold is removed ENTIRELY and the analysis is repeated from scratch.
    #     The threshold is recomputed as well (the remaining sample's own 1%), so this
    #     is more than just filtering.
    print("=== UC OLAY SAGLAMLIK KONTROLU ===")
    tail_rows, dist_rows = [], []
    for h in HORIZONS:
        g_full = pred[pred["horizon"] == h].copy()
        for label, g in (("tum yillar", g_full),
                         ("2020 haric", g_full[g_full["test_year"] != 2020])):
            g = g.copy()
            g["e_harx"] = (g["y_true"] - g["pred_har_x"]) ** 2
            g["e_h2"] = (g["y_true"] - g["pred_h2_harx_xgb"]) ** 2
            thr = g["e_harx"].quantile(0.99)
            top = g[g["e_harx"] >= thr]
            tail_rows.append({
                "horizon": h, "ornek": label, "n_gozlem": int(len(g)),
                "n_uc": int(len(top)),
                "uc_havuz_payi_pct": 100 * float(top["e_harx"].sum()
                                                 / g["e_harx"].sum()),
                "h2_vs_harx_uc_pct": 100 * (float(np.sqrt(top["e_h2"].mean()))
                                            / float(np.sqrt(top["e_harx"].mean())) - 1),
                "h2_vs_harx_geri_kalan_pct": 100 * (
                    float(np.sqrt(g[g["e_harx"] < thr]["e_h2"].mean()))
                    / float(np.sqrt(g[g["e_harx"] < thr]["e_harx"].mean())) - 1),
                "h2_vs_harx_havuz_pct": 100 * (float(np.sqrt(g["e_h2"].mean()))
                                               / float(np.sqrt(g["e_harx"].mean())) - 1),
            })
            if label == "tum yillar":
                for yy, cnt in top["test_year"].value_counts().items():
                    dist_rows.append({"horizon": h, "test_year": int(yy),
                                      "n_uc_gozlem": int(cnt),
                                      "pay_pct": 100 * cnt / len(top)})
    tail = pd.DataFrame(tail_rows)
    dist = pd.DataFrame(dist_rows)

    print("En kotu %1 gozlemin TAKVIM dagilimi (tum yillar ornegi):")
    dpiv = dist.pivot(index="test_year", columns="horizon",
                      values="n_uc_gozlem").fillna(0).astype(int)
    dpiv["toplam"] = dpiv.sum(axis=1)
    print(dpiv.sort_values("toplam", ascending=False).to_string())
    print("\n2020'nin uc gozlemler icindeki payi (%):")
    print(dist[dist["test_year"] == 2020][["horizon", "pay_pct"]].to_string(
        index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print("2020 haric tekrar (esik yeniden hesaplandi):")
    print(tail.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nH2'nin uc gozlemlerdeki farki (%, negatif = H2 iyi):")
    print(tail.pivot(index="ornek", columns="horizon",
                     values="h2_vs_harx_uc_pct").to_string(
        float_format=lambda v: f"{v:+.2f}"))
    print("\nAyni fark, uc gozlemler DISINDA kalan %99:")
    print(tail.pivot(index="ornek", columns="horizon",
                     values="h2_vs_harx_geri_kalan_pct").to_string(
        float_format=lambda v: f"{v:+.2f}"))
    print()

    tail.to_csv(OUT_DIR / "explore_tail_robustness.csv", index=False)
    dist.to_csv(OUT_DIR / "explore_tail_year_distribution.csv", index=False)

    years.to_csv(OUT_DIR / "explore_vol_regime_years.csv", index=False)
    groups.to_csv(OUT_DIR / "explore_vol_regime_groups.csv", index=False)
    folds.to_csv(OUT_DIR / "explore_vol_regime_folds.csv", index=False)
    with open(OUT_DIR / "explore_vol_regime_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "status": "KESIFSEL / IKINCIL -- birincil bulguyu degistirmez",
            "split_rule": ("Test yilinin ortalama gerceklesen volatilitesi, ufuk "
                           "basina medyana gore ikiye ayrilir. Mekanik kriter, "
                           "performanstan turemez."),
            "purpose": ("H2 icin havuzlanmis ve fold-ortalamali metriklerin neden "
                        "ters yonde sonuc verdigini aciklamak."),
            "groups": group_rows,
            "runtime_seconds": round(time.time() - t0, 2),
        }, f, ensure_ascii=False, indent=2, default=str)

    print("Yazildi: explore_vol_regime_years.csv, explore_vol_regime_groups.csv, "
          "explore_vol_regime_folds.csv")
    print("Rapor  : explore_vol_regime_summary.json")
    print(f"Sure   : {time.time() - t0:.1f} saniye")


if __name__ == "__main__":
    main()
