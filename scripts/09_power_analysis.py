"""Power analysis: could the 8 comparisons in the primary family have been detected with this data?

PURPOSE
-------
To give a NUMERICAL answer to the question "could this difference have been detected with
this data?" This is not an exact power calculation; the aim is an ORDER-OF-MAGNITUDE
ESTIMATE.

THE TWO TESTS HAVE DIFFERENT POWER STRUCTURES AND ARE COMPUTED SEPARATELY
--------------------------------------------------------------------------
DM test: operates on daily observations, but because the target windows overlap, the
  amount of effective information is not the number of observations but the number of
  EFFECTIVE BLOCKS:
      B = n / h
  (h=126, n=3500 -> B ~ 28). The DM statistic scales with sqrt(B).

Sign test: operates at the fold level; the "sample" is directly the number of FOLDS
  (years). One fold = one year, so the required sample size is already in units of years.

METHOD
------
1) ACHIEVED POWER
   DM  : the observed DM statistic is taken as the non-centrality parameter (ncp).
         power = P(|Z| > 1.96 | ncp = |DM|) = Phi(-1.96 + |DM|) + Phi(-1.96 - |DM|)
   Sign: the 5% two-sided rejection region under H0 (p=0.5) is found with the EXACT
         binomial, then the probability of that region is summed under the assumption
         that the observed win rate is the true rate.

   WARNING: "achieved power" computed from the observed effect is a monotone
   transformation of the p-value and CARRIES NO NEW INFORMATION beyond it. It is given
   here for communication purposes only; the real answer is the required-sample
   calculation in (2).

2) REQUIRED SAMPLE SIZE (80% power, 5% two-sided)
   DM  : standardized effect delta = |DM_observed| / sqrt(B_observed).
         Required ncp = z_{0.975} + z_{0.80} = 1.960 + 0.842 = 2.802.
         B_required = (2.802 / delta)^2 = B_observed * (2.802 / |DM|)^2
         n_required = B_required * h  (daily rows)
         years = n_required / (trading days per year)
   Sign: the observed win rate is taken as the true rate; the number of folds is increased
         until the EXACT binomial power is >= 0.80 (upper bound MAX_FOLDS).

ASSUMPTIONS (stated explicitly)
--------------------------------
* The observed effect size is taken as the TRUE effect. In reality the observed effect is
  a noisy estimate; small observed effects may understate or overstate the true effect.
* The error structure (autocorrelation, variance) is assumed NOT to change as the sample
  grows.
* The normal approximation is used for DM; the HLN small-sample correction is NOT INCLUDED
  in the power calculation (negligible for an order-of-magnitude estimate, since n is
  large).
* The effective block count B = n/h is a rough approximation; with overlapping windows,
  block independence does not hold exactly.
* "Year" means a TEST PERIOD year (the current test period is 2012-2026).

GPR ALIGNMENT
-------------
--gpr-alignment is required (scripts/alignment.py). It selects which DM results and
prediction files are read (08_dm_test.py / 07_hybrid.py outputs of that version) and
suffixes every output accordingly. The a priori DM curves are calibrated on those DM
results, so they belong to the same version.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

ALPHA = 0.05
TARGET_POWER = 0.80
MAX_FOLDS = 5000          # upper bound for the sign-test search
Z_ALPHA = stats.norm.ppf(1 - ALPHA / 2)          # 1.9600
Z_POWER = stats.norm.ppf(TARGET_POWER)           # 0.8416
NCP_REQ = Z_ALPHA + Z_POWER                      # 2.8016


def dm_power(ncp):
    """Power of a two-sided Z test at the given non-centrality parameter."""
    return float(stats.norm.cdf(-Z_ALPHA + ncp) + stats.norm.cdf(-Z_ALPHA - ncp))


def sign_rejection_region(n):
    """The 5% two-sided EXACT binomial rejection region (k values) under H0: p=0.5."""
    return [k for k in range(n + 1)
            if stats.binomtest(k, n, 0.5).pvalue <= ALPHA]


def sign_power(n, p_true):
    """Exact binomial power, taking the observed rate as the true rate."""
    region = sign_rejection_region(n)
    if not region:
        return 0.0
    return float(stats.binom.pmf(region, n, p_true).sum())


def sign_required_n(p_true, n_start):
    """Smallest fold count reaching 80% power; None if unreachable."""
    if abs(p_true - 0.5) < 1e-9:
        return None
    for n in range(max(n_start, 5), MAX_FOLDS + 1):
        if sign_power(n, p_true) >= TARGET_POWER:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment
    t0 = time.time()
    res = pd.read_csv(alignment.out("dm_test_results.csv", al))
    prim = res[res["aile"] == "birincil"].copy()
    assert len(prim) == 8, f"The primary family must have 8 tests, found {len(prim)}"

    # Trading days per year: measured from the test period
    pred = pd.read_csv(alignment.out("hybrid_predictions_all.csv", al))
    pred = pred[pred["include_in_main"]]
    days_per_year = float(
        pred[pred["horizon"] == 5].groupby("test_year").size().mean())

    rows = []
    for _, r in prim.iterrows():
        h, n = int(r["horizon"]), int(r["n"])
        blocks = n / h                       # effective independent blocks
        dm = abs(float(r["DM_HLN"]))
        delta = dm / np.sqrt(blocks)         # standardized effect per block

        dm_ach = dm_power(dm)
        if delta > 0:
            blocks_req = (NCP_REQ / delta) ** 2
            n_req = blocks_req * h
            years_req = n_req / days_per_year
        else:
            blocks_req = n_req = years_req = float("inf")

        folds = int(r["isaret_fold"])
        wins = int(r["isaret_kazanan"])
        p_hat = wins / folds
        s_ach = sign_power(folds, p_hat)
        s_req = sign_required_n(p_hat, folds)

        rows.append({
            "horizon": h,
            "karsilastirma": f"{r['model1']} vs {r['model2']}",
            "rmse_fark_pct": float(r["fark_pct"]),
            # --- DM ---
            "dm_n_gunluk": n,
            "dm_etkin_blok": round(blocks, 1),
            "dm_istatistik": round(dm, 4),
            "dm_p": float(r["p_HLN"]),
            "dm_blok_basina_etki": round(delta, 5),
            "dm_gerceklesen_guc": round(dm_ach, 4),
            "dm_gerekli_blok": round(blocks_req, 0),
            "dm_gerekli_gun": round(n_req, 0),
            "dm_gerekli_yil": round(years_req, 1),
            "dm_kat_artis": round(blocks_req / blocks, 1),
            # --- sign test ---
            "isaret_fold": folds,
            "isaret_kazanan": wins,
            "isaret_oran": round(p_hat, 4),
            "isaret_p": float(r["p_isaret"]),
            "isaret_gerceklesen_guc": round(s_ach, 4),
            "isaret_gerekli_fold_yil": s_req if s_req is not None else np.nan,
            "isaret_kat_artis": (round(s_req / folds, 1)
                                 if s_req is not None else np.nan),
        })

    out = pd.DataFrame(rows).sort_values(["karsilastirma", "horizon"])
    out.to_csv(alignment.out("power_analysis.csv", al), index=False)

    pd.set_option("display.width", 250)
    print("=== POWER ANALYSIS: primary family (2 comparisons x 4 horizons) ===")
    print(f"Target: {TARGET_POWER*100:.0f}% power, {ALPHA*100:.0f}% two-sided.")
    print(f"Trading days per year (measured from the test period): {days_per_year:.1f}")
    print("Sample for DM = EFFECTIVE BLOCK (n/h); sample for the sign test = FOLD (year).\n")

    print("--- DM test ---")
    print(out[["horizon", "karsilastirma", "rmse_fark_pct", "dm_etkin_blok",
               "dm_istatistik", "dm_gerceklesen_guc", "dm_gerekli_blok",
               "dm_gerekli_yil", "dm_kat_artis"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print("--- Sign test ---")
    print(out[["horizon", "karsilastirma", "isaret_kazanan", "isaret_fold",
               "isaret_oran", "isaret_gerceklesen_guc",
               "isaret_gerekli_fold_yil", "isaret_kat_artis"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print()

    print("=== SUMMARY: TEST PERIOD length required for 80% power (years) ===")
    piv = out.pivot(index="karsilastirma", columns="horizon",
                    values="dm_gerekli_yil")
    print("DM test:")
    print(piv.to_string(float_format=lambda v: f"{v:,.0f}"))
    print("\nSign test:")
    print(out.pivot(index="karsilastirma", columns="horizon",
                    values="isaret_gerekli_fold_yil").to_string(
        float_format=lambda v: f"{v:,.0f}"))
    print(f"\nCurrent test period: {out['isaret_fold'].max()} years "
          f"(h=5/22), {out['isaret_fold'].min()} years (h=66/126).")
    print()
    print("WARNING: the 'realized power' computed from the observed effect is a monotone")
    print("transformation of the p-value and carries no new information. The actual answer is in the")
    print("required-sample columns. The observed effect is taken as the true effect; small")
    print("observed effects may under- or overstate the true effect.")
    print()

    # ===================================================================
    # A PRIORI POWER CURVES -- INDEPENDENT of the observed effects
    # ===================================================================
    # This section does NOT USE the observed results; it answers "what could this design
    # have detected" over hypothetical effect sizes. It is not circular.
    print("=" * 70)
    print("A PRIORI POWER CURVES -- independent of the observed effects")
    print("=" * 70)
    print()

    # --- Sign test: exact binomial, hypothetical true win probabilities ---
    print("--- SIGN TEST: n folds, true win probability p ---")
    p_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    srows = []
    for n in (15, 14):
        region = sign_rejection_region(n)
        hi = min([k for k in region if k > n / 2], default=None)
        for pv in p_grid:
            srows.append({"n_fold": n, "p_gercek": pv,
                          "guc": round(sign_power(n, pv), 4),
                          "anlamlilik_icin_gereken_kazanma": hi})
    sdf = pd.DataFrame(srows)
    print("Minimum number of wins required for 5% two-sided significance: "
          f"n=15 -> {sdf[sdf.n_fold==15]['anlamlilik_icin_gereken_kazanma'].iloc[0]}, "
          f"n=14 -> {sdf[sdf.n_fold==14]['anlamlilik_icin_gereken_kazanma'].iloc[0]}")
    print()
    print(sdf.pivot(index="p_gercek", columns="n_fold", values="guc").to_string(
        float_format=lambda v: f"{v:.3f}"))
    print()

    # --- DM test: hypothetical RMSE differences ---
    # The effect -> non-centrality mapping needs a calibration coefficient k:
    #     delta_block = k * |r^2 - 1|,  ncp = sqrt(B) * delta_block
    # k derives from the NOISE structure of the data (the error correlation of the two
    # models and the loss distribution), not from the observed EFFECT size. It is
    # averaged over the two pairs in the primary family per horizon, and the range is
    # reported as well.
    res_all = pd.read_csv(alignment.out("dm_test_results.csv", al))
    res_all["B"] = res_all["n"] / res_all["horizon"]
    res_all["k"] = (res_all["DM_HLN"] / np.sqrt(res_all["B"])) / (
        res_all["rmse_orani"] ** 2 - 1)
    kprim = res_all[res_all["aile"] == "birincil"].groupby("horizon")["k"].agg(
        ["mean", "min", "max"])

    print("--- DM TEST: number of effective blocks B, hypothetical RMSE difference ---")
    print("Donusum: delta_blok = k*|r^2-1|, ncp = sqrt(B)*delta_blok.")
    print("k is calibrated from the NOISE structure of the data (error correlation and loss")
    print("distribution), NOT from the observed EFFECT size. Per horizon, the mean of the two")
    print("pairs in the primary family is used:")
    print(kprim.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    pct_grid = [0.05, 0.10, 0.20]
    drows = []
    for h, B in ((126, 28.0), (66, 53.0), (22, 166.0), (5, 732.0)):
        kmean = float(kprim.loc[h, "mean"])
        for pct in pct_grid:
            eff = abs((1 - pct) ** 2 - 1)
            row = {"horizon": h, "etkin_blok": B, "rmse_farki_pct": int(pct * 100),
                   "k": round(kmean, 3),
                   "guc": round(dm_power(np.sqrt(B) * kmean * eff), 4)}
            for lbl in ("min", "max"):
                kk = float(kprim.loc[h, lbl])
                row[f"guc_k_{lbl}"] = round(
                    dm_power(np.sqrt(B) * kk * eff), 4)
            drows.append(row)
    ddf = pd.DataFrame(drows)
    print(ddf.pivot(index=["horizon", "etkin_blok"], columns="rmse_farki_pct",
                    values="guc").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nSensitivity to the uncertainty in k (range between the two pairs of the primary family):")
    print(ddf[["horizon", "rmse_farki_pct", "guc_k_min", "guc", "guc_k_max"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("INTERPRETATION: cells above the 80% threshold show the effects this design")
    print("can detect. For the others, the result 'no significant difference' means not that there")
    print("is no difference, but that the design cannot see it.")
    print()

    sdf.to_csv(alignment.out("apriori_power_sign.csv", al), index=False)
    ddf.to_csv(alignment.out("apriori_power_dm.csv", al), index=False)

    with open(alignment.out("power_analysis_summary.json", al), "w",
              encoding="utf-8") as f:
        json.dump({
            "gpr_alignment": al,
            "alpha": ALPHA, "target_power": TARGET_POWER,
            "days_per_year": days_per_year,
            "method_dm": ("Orneklem = etkin blok B = n/h. delta = |DM|/sqrt(B). "
                          "B_gerekli = (z_0.975 + z_0.80)^2 / delta^2. "
                          "n_gerekli = B_gerekli * h; yil = n_gerekli/gun_per_yil."),
            "method_sign": ("Orneklem = fold (yil). Tam binom; H0 p=0.5 altinda %5 "
                            "iki yonlu red bolgesi bulunur, gozlenen oran gercek "
                            "oran kabul edilerek guc hesaplanir; guc >= 0.80 olana "
                            "kadar fold sayisi artirilir."),
            "assumptions": [
                "Gozlenen etki buyuklugu GERCEK etki kabul edilir.",
                "Hata yapisi orneklem buyudukce degismez.",
                "DM icin normal yaklasim; HLN duzeltmesi guc hesabina dahil degil.",
                "B = n/h kaba yaklasimdir; blok bagimsizligi tam saglanmaz.",
                "'Yil' TEST DONEMI yili anlamindadir.",
            ],
            "caveat_observed_power": ("Gozlenen etkiden hesaplanan gerceklesen guc "
                                      "p-degerinin monoton donusumudur; yeni bilgi "
                                      "tasimaz."),
            "results": out.to_dict(orient="records"),
            "apriori_note": ("Onsel egriler gozlenen etkileri KULLANMAZ. DM icin "
                             "etki->ncp donusum katsayisi k, verinin gurultu "
                             "yapisindan kalibre edilir (hata korelasyonu ve kayip "
                             "dagilimi); gozlenen etki buyuklugunden degil."),
            "apriori_sign": sdf.to_dict(orient="records"),
            "apriori_dm": ddf.to_dict(orient="records"),
            "runtime_seconds": round(time.time() - t0, 2),
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"Written: power_analysis{alignment.suffix(al)}.csv ({len(out)} rows) and "
          f"apriori_power_*/power_analysis_summary{alignment.suffix(al)}")
    print(f"Report : power_analysis_summary{alignment.suffix(al)}.json")
    print(f"Runtime: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
