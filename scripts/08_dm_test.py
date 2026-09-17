"""Diebold-Mariano tests: HAC corrected, plus a distribution-free sign test.

LOSS FUNCTION
-------------
Squared error (consistent with RMSE, our main metric):
    d_t = e_{1t}^2 - e_{2t}^2

SIGN CONVENTION: if the mean of d and the DM statistic are NEGATIVE, the FIRST model has
the lower loss, i.e. it is better. The RMSE ratio reads the same way (< 1 -> the first
model is better).

THE HAC CORRECTION IS MANDATORY
-------------------------------
Because the target windows overlap, the loss-difference series is autocorrelated. Without
the correction, the DM test UNDERSTATES the standard error severely and produces spurious
significance. Newey-West (Bartlett kernel) is therefore applied, and the variance
inflation factor (HAC variance / variance under an independence assumption) is reported
separately, to quantify how misleading the uncorrected test would have been.

LAG RULE: L = h - 1, PRE-DECLARED
----------------------------------
Theoretical justification: the errors of an OPTIMAL h-step-ahead forecast follow an
MA(h-1) process. This is not an arbitrary choice; it follows from the standard result.

AN IMPORTANT CAVEAT: the MA(h-1) result holds for OPTIMAL forecasts. Ours are not
optimal; they come from misspecified models, so the autocorrelation of the loss
difference may extend BEYOND h-1. In other words h-1 is a LOWER BOUND, not an upper
bound, and the correction may STILL BE INSUFFICIENT. This is checked with an empirical
ACF diagnostic.

ASSESSMENT OF THE LAG/SAMPLE RATIO (h=126)
-------------------------------------------
After pooling, n ~ 3500 and L = 125 -> L/n ~ 3.6%. For a Bartlett kernel the optimal
bandwidth is of order O(n^(1/3)) (about 15 for n=3500); L=125 is eight times that. The
consequence: the HAC estimator remains CONSISTENT but its variance rises and the test
loses power.

AN ALTERNATIVE LAG RULE IS NOT RECOMMENDED. Shortening the lag would exclude genuine
non-zero autocovariances, pull the standard error down, and produce exactly the spurious
significance we are trying to prevent. A short lag inflates Type I error, a long lag
inflates Type II error; in this study avoiding the former takes priority. Over-truncating
is safer than under-truncating.

HARVEY-LEYBOURNE-NEWBOLD SMALL-SAMPLE CORRECTION
-------------------------------------------------
    DM* = DM * sqrt( (n + 1 - 2h + h(h-1)/n) / n ),  compared against t(n-1).
The raw DM statistic and the standard normal p-value are reported side by side as well.

MAGNITUDE vs SIGNIFICANCE
--------------------------
The DM test measures SIGNIFICANCE, not MAGNITUDE. The RMSE ratio is therefore added as a
column of the results table. "Significant but a 0.3% difference" and "significant and an
18% difference" are very different claims; the two must be read together.

SIGN TEST (BINOMIAL) -- A DISTRIBUTION-FREE COMPLEMENT
-------------------------------------------------------
For every model pair and horizon: the number of folds in which the first model BEATS the
second, and the binomial p-value for that count (H0: p = 0.5, two-sided). The sign test
uses NONE of the HAC bandwidth, kernel choice or normality assumptions.

If the two tests point the same way, the finding is IMMUNE to the bandwidth choice; if
they diverge, that must be known and is reported.

THE POOLING ASSUMPTION
----------------------
The pooled series does not come from a single FIXED model; it consists of predictions
from models retrained every year. This is standard in walk-forward DM applications, but
it is stated as an assumption. The series is continuous in calendar terms; at h=66 and
h=126 the 2026 fold is excluded from the main metric and is therefore dropped from the DM
test as well.

MULTIPLE TESTING
----------------
5 pairs x 4 horizons = 20 tests. Holm-Bonferroni corrected values are reported alongside
the raw p-values. No decision rests on a single p-value.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

HORIZONS = [5, 22, 66, 126]

# Pairs to be tested: (model1, model2). Negative DM -> model1 is better.
PAIRS = [
    ("har_x", "xgboost"),      # MAIN FINDING (the ACF diagnostic uses this pair)
    ("har_x", "past_vol"),
    ("har_x", "garch"),
    ("xgboost", "past_vol"),
    ("har_x", "bilstm"),
    # --- extended family ---
    ("har", "har_x"),          # MAIN DECOMPOSITION: the contribution of the exogenous variables
    ("har", "past_vol"),
    ("har_x", "har_x_log"),    # levels vs log-log specification
]
# ===========================================================================
# TEST FAMILIES
# ===========================================================================
# PRIMARY (confirmatory) FAMILY: 2 comparisons x 4 horizons = 8 tests.
#   (har_x, xgboost) -> "HAR-X beats XGBoost"
#   (har,   har_x)   -> "the gain comes from the exogenous variables" (the main decomposition)
#
# AN IMPORTANT CAVEAT: these two comparisons correspond to the study's two main claims as
# DECLARED in stages 5-6; they were NOT SELECTED by looking at p-values. The family
# definition was formalized AFTER the tests, however, so this is not a pre-registration,
# and the report says so.
#
# SECONDARY / EXPLORATORY FAMILY: the remaining 6 pairs x 4 horizons = 24 tests.
#
# TWO corrections are given side by side for each family:
#   Holm  -> FWER (family-wise error rate). Appropriate for confirmatory claims.
#   BH    -> FDR (false discovery rate). Appropriate for exploratory findings.
# Showing both lets the reader choose their own threshold.
PRIMARY_PAIRS = {("har_x", "xgboost"), ("har", "har_x")}


def newey_west_lrv(d, lag):
    """Long-run variance with a Bartlett kernel. gamma0 is returned as well."""
    d = np.asarray(d, "float64")
    n = len(d)
    x = d - d.mean()
    g0 = float(np.dot(x, x) / n)
    omega = g0
    for j in range(1, min(lag, n - 1) + 1):
        gj = float(np.dot(x[j:], x[:-j]) / n)
        omega += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    return omega, g0


def acf_at(d, lags):
    d = np.asarray(d, "float64")
    n = len(d)
    x = d - d.mean()
    g0 = float(np.dot(x, x) / n)
    out = {}
    for j in lags:
        if 1 <= j <= n - 1 and g0 > 0:
            out[j] = float(np.dot(x[j:], x[:-j]) / n / g0)
        else:
            out[j] = float("nan")
    return out


def benjamini_hochberg(pvals):
    """BH (FDR) adjusted p-values; monotone, within [0,1]."""
    p = np.asarray(pvals, "float64")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, m / (i + 1) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (ordered, monotone)."""
    p = np.asarray(pvals, "float64")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main():
    t0 = time.time()

    hyb = pd.read_csv(OUT_DIR / "hybrid_predictions_all.csv")
    ben = pd.read_csv(OUT_DIR / "bench_predictions_all.csv")
    key = ["horizon", "Date"]
    df = hyb.merge(ben[key + ["pred_garch", "y_true"]], on=key,
                   suffixes=("", "_b"))
    assert len(df) == len(hyb), "GARCH birlestirmesi satir sayisini degistirdi"
    assert (df["y_true"] - df["y_true_b"]).abs().max() < 1e-12, \
        "y_true kaynaklar arasinda eslesmiyor"
    df = df.drop(columns=["y_true_b"])
    df = df[df["include_in_main"]].copy()
    df["dt"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df = df.sort_values(["horizon", "dt"]).reset_index(drop=True)

    pd.set_option("display.width", 250)
    print("=== DIEBOLD-MARIANO TESTLERI ===")
    print("Kayip: karesel hata. Isaret: NEGATIF DM -> BIRINCI model daha iyi.")
    print("HAC: Newey-West/Bartlett, L = h-1 (onceden ilan edilmis).")
    print("Havuzlama: fold'lar arasinda, takvim olarak kesintisiz seri.")
    print("h=66 ve h=126'da 2026 fold'u ana metrikten haric oldugu icin DM'den de cikarildi.\n")

    rows, diag_rows = [], []
    for h in HORIZONS:
        g = df[df["horizon"] == h]
        n = len(g)
        lag = h - 1
        y = g["y_true"].to_numpy("float64")
        yrs = g["test_year"].to_numpy()

        for m1, m2 in PAIRS:
            e1 = y - g[f"pred_{m1}"].to_numpy("float64")
            e2 = y - g[f"pred_{m2}"].to_numpy("float64")
            d = e1 ** 2 - e2 ** 2
            dbar = float(d.mean())

            omega, g0 = newey_west_lrv(d, lag)
            omega = max(omega, 1e-300)
            se_hac = np.sqrt(omega / n)
            se_iid = np.sqrt(g0 / n)
            dm = dbar / se_hac
            dm_naive = dbar / se_iid

            # Harvey-Leybourne-Newbold small-sample correction
            hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
            dm_hln = dm * hln
            p_norm = 2 * (1 - stats.norm.cdf(abs(dm)))
            p_hln = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
            p_naive = 2 * (1 - stats.norm.cdf(abs(dm_naive)))

            rmse1 = float(np.sqrt(np.mean(e1 ** 2)))
            rmse2 = float(np.sqrt(np.mean(e2 ** 2)))

            # --- Sign test: at the fold level, distribution-free ---------------
            wins, folds = 0, 0
            for yy in np.unique(yrs):
                m = yrs == yy
                folds += 1
                if np.sqrt(np.mean(e1[m] ** 2)) < np.sqrt(np.mean(e2[m] ** 2)):
                    wins += 1
            p_sign = float(stats.binomtest(wins, folds, 0.5).pvalue)

            rows.append({
                "horizon": h, "model1": m1, "model2": m2, "n": n, "lag": lag,
                "rmse1": rmse1, "rmse2": rmse2, "rmse_orani": rmse1 / rmse2,
                "fark_pct": 100 * (rmse1 / rmse2 - 1),
                "d_ortalama": dbar,
                "DM_ham": dm, "p_ham": p_norm,
                "DM_HLN": dm_hln, "p_HLN": p_hln,
                "hln_carpani": float(hln),
                "DM_HACsiz": dm_naive, "p_HACsiz": p_naive,
                "varyans_sisme": float(omega / g0) if g0 > 0 else float("nan"),
                "isaret_kazanan": wins, "isaret_fold": folds, "p_isaret": p_sign,
            })

            if (m1, m2) == PAIRS[0]:
                lags = sorted({1, max(1, lag // 2), lag, h, min(2 * lag, n - 1)})
                a = acf_at(d, lags)
                diag_rows.append({"horizon": h, "cift": f"{m1} vs {m2}",
                                  "n": n, "lag": lag,
                                  **{f"acf_{k}": v for k, v in a.items()},
                                  "acf_ort_h_2h": float(np.nanmean(list(
                                      acf_at(d, range(h, min(2 * h, n - 1) + 1)
                                             ).values()))),
                                  "varyans_sisme": float(omega / g0)})

    res = pd.DataFrame(rows)
    res["aile"] = np.where(
        [(m1, m2) in PRIMARY_PAIRS for m1, m2 in zip(res["model1"], res["model2"])],
        "birincil", "ikincil")
    # The corrections are computed separately WITHIN EACH FAMILY.
    for col, base in (("p_HLN", "DM"), ("p_isaret", "isaret")):
        for meth, fn in (("holm", holm), ("bh", benjamini_hochberg)):
            res[f"{col}_{meth}"] = np.nan
            for fam, idx in res.groupby("aile").groups.items():
                res.loc[idx, f"{col}_{meth}"] = fn(res.loc[idx, col].values)

    def stars(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else
                                       ("*" if p < 0.10 else ""))

    res["anlamlilik"] = res["p_HLN"].map(stars)

    cols = ["horizon", "model1", "model2", "rmse_orani", "fark_pct",
            "DM_HLN", "p_HLN", "p_HLN_holm", "p_HLN_bh",
            "isaret_kazanan", "isaret_fold", "p_isaret",
            "p_isaret_holm", "p_isaret_bh"]

    print("=== BIRINCIL (DOGRULAYICI) AILE: 2 karsilastirma x 4 ufuk = 8 test ===")
    print("(har_x vs xgboost) ve (har vs har_x). Bu iki karsilastirma calismanin")
    print("Asama 5-6'da ilan edilmis iki ana iddiasina karsilik gelir; p-degerlerine")
    print("bakilarak secilmemistir. Ancak aile tanimi testlerden SONRA")
    print("resmilestirilmistir -- bu bir on-kayit degildir.")
    print("fark_pct negatif = model1 daha iyi. Holm = FWER, BH = FDR.")
    prim = res[res["aile"] == "birincil"].sort_values(["model1", "horizon"])
    print(prim[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for lbl, c in (("Holm (FWER)", "p_HLN_holm"), ("BH (FDR)", "p_HLN_bh")):
        k = int((prim[c] < 0.05).sum())
        print(f"  DM, {lbl}: {k}/8 test %5'te ayakta")
    for lbl, c in (("Holm (FWER)", "p_isaret_holm"), ("BH (FDR)", "p_isaret_bh")):
        k = int((prim[c] < 0.05).sum())
        print(f"  Isaret, {lbl}: {k}/8 test %5'te ayakta")
    print()

    print("=== IKINCIL / KESIFSEL AILE: 6 karsilastirma x 4 ufuk = 24 test ===")
    print("Dogrulayici degildir; hipotez uretmek icin okunur.")
    sec = res[res["aile"] == "ikincil"].sort_values(["horizon", "model1", "model2"])
    print(sec[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for lbl, c in (("Holm (FWER)", "p_HLN_holm"), ("BH (FDR)", "p_HLN_bh")):
        k = int((sec[c] < 0.05).sum())
        print(f"  DM, {lbl}: {k}/24 test %5'te ayakta")
    for lbl, c in (("Holm (FWER)", "p_isaret_holm"), ("BH (FDR)", "p_isaret_bh")):
        k = int((sec[c] < 0.05).sum())
        print(f"  Isaret, {lbl}: {k}/24 test %5'te ayakta")
    print()
    print("Ikincil ailede BH ile ayakta kalanlar (DM):")
    print(sec[sec["p_HLN_bh"] < 0.05][["horizon", "model1", "model2", "fark_pct",
                                       "p_HLN", "p_HLN_bh", "p_HLN_holm"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nIkincil ailede BH ile ayakta kalanlar (isaret):")
    print(sec[sec["p_isaret_bh"] < 0.05][["horizon", "model1", "model2",
                                          "isaret_kazanan", "isaret_fold",
                                          "p_isaret", "p_isaret_bh",
                                          "p_isaret_holm"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("=== HAC DUZELTMESININ ETKISI ===")
    print("varyans_sisme = HAC varyansi / bagimsizlik varsayimli varyans.")
    print("Duzeltmesiz test standart hatayi sqrt(sisme) kati kucuk tahmin eder.")
    print(res[["horizon", "model1", "model2", "varyans_sisme", "DM_HACsiz",
               "p_HACsiz", "DM_ham", "p_ham", "DM_HLN", "p_HLN"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    print("=== TANI: kayip farkinin ACF'si (HAR-X vs XGBoost cifti) ===")
    print("L = h-1 kuralinin gecerliligi. h-1 ve otesinde ACF sifira inmiyorsa,")
    print("duzeltme HALA YETERSIZ demektir (optimal olmayan tahminlerde beklenir).")
    print(pd.DataFrame(diag_rows).to_string(index=False,
                                            float_format=lambda v: f"{v:.4f}"))
    print()

    print("=== IKI TESTIN UYUMU ===")
    res["dm_yon"] = np.where(res["DM_HLN"] < 0, "model1", "model2")
    res["isaret_yon"] = np.where(
        res["isaret_kazanan"] > res["isaret_fold"] / 2, "model1", "model2")
    res["yon_uyumu"] = res["dm_yon"] == res["isaret_yon"]
    res["ikisi_de_anlamli"] = (res["p_HLN"] < 0.05) & (res["p_isaret"] < 0.05)
    print(res[["horizon", "model1", "model2", "dm_yon", "isaret_yon",
               "yon_uyumu", "p_HLN", "p_isaret", "ikisi_de_anlamli"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nYon uyumu: {int(res['yon_uyumu'].sum())}/{len(res)} test")
    print(f"Iki testin de %5'te anlamli oldugu: "
          f"{int(res['ikisi_de_anlamli'].sum())}/{len(res)} test")
    disagree = res[~res["yon_uyumu"]]
    if len(disagree):
        print("\nYON AYRISMASI olan testler (bilinmesi gerekir):")
        print(disagree[["horizon", "model1", "model2", "fark_pct", "DM_HLN",
                        "isaret_kazanan", "isaret_fold"]].to_string(index=False))
    print()

    res.to_csv(OUT_DIR / "dm_test_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "dm_diagnostics.csv", index=False)
    with open(OUT_DIR / "dm_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "loss": "karesel hata",
            "sign_convention": "negatif DM / rmse_orani < 1 -> model1 daha iyi",
            "hac": "Newey-West, Bartlett, L = h-1 (onceden ilan edilmis)",
            "lag_rule_note": (
                "MA(h-1) sonucu OPTIMAL tahminler icindir; bizim tahminlerimiz "
                "optimal degildir, dolayisiyla h-1 bir ALT SINIRDIR ve duzeltme hala "
                "yetersiz olabilir. Kisaltmak Tip I hatasini artiracagi icin "
                "kisaltilmadi."
            ),
            "hln": "DM* = DM * sqrt((n+1-2h+h(h-1)/n)/n), t(n-1)",
            "pooling_note": (
                "Havuzlanan seri tek bir sabit modelden gelmez; her yil yeniden "
                "egitilmis modellerin tahminlerindendir. Walk-forward DM'de standart, "
                "ancak varsayim olarak belirtilir."
            ),
            "sign_test": (
                "Fold duzeyinde binom testi (H0: p=0.5, iki yonlu). HAC bant "
                "genisligi, cekirdek ve normallik varsayimlarinin hicbirini "
                "kullanmaz."
            ),
            "families": {
                "birincil": ("2 karsilastirma x 4 ufuk = 8 test: (har_x, xgboost) ve "
                             "(har, har_x). Calismanin ilan edilmis iki ana iddiasi."),
                "ikincil": "kalan 6 karsilastirma x 4 ufuk = 24 test, kesifsel",
                "caveat": ("Aile tanimi testlerden SONRA resmilestirilmistir; bu bir "
                           "on-kayit degildir. Ancak karsilastirmalar p-degerlerine "
                           "bakilarak degil, onceki asamalarda ilan edilmis "
                           "iddialara gore secilmistir."),
            },
            "multiple_testing": ("Holm (FWER) ve Benjamini-Hochberg (FDR), HER AILE "
                                 "ICINDE ayri hesaplanir."),
            "results": res.to_dict(orient="records"),
            "diagnostics": diag_rows,
            "runtime_seconds": round(time.time() - t0, 2),
        }, f, ensure_ascii=False, indent=2, default=str)

    print("Yazildi: dm_test_results.csv, dm_diagnostics.csv")
    print("Rapor  : dm_summary.json")
    print(f"Sure   : {time.time() - t0:.1f} saniye")


if __name__ == "__main__":
    main()
