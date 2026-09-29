"""Paper number package, publication-aligned GPR (the single source for manuscript numbers).

WHAT THIS DOES
--------------
Collects every number the manuscript quotes into one Markdown file, reading ONLY saved
outputs (no model is fitted, no test set is re-read for any decision). The primary results
are the *_publication_aligned files; the timestamp-aligned (unsuffixed) files appear only in
the two-version comparison section, which belongs to paper Appendix A.

Sections: main RMSE/MAE/R2_oos table (all models x four horizons, standard R2 as footnote,
2026 partial-year footnote at h=66/126), four-step decomposition, exogenous ablation ladder,
standardized betas, SHAP group shares, the primary hypothesis family (8 tests), robustness
checks (data equalization, BiLSTM convergence), the two-version comparison, README numbers,
power analysis, methodology numbers, and (Section 12) HAR prediction-floor frequency,
QLIKE (descriptive only, no test) and per-fold smearing coefficients.

CONSISTENCY CHECKS (assertions)
-------------------------------
  * per-fold RMSE recomputed from every prediction file equals the stored fold metric;
  * the fold means equal those in gpr_alignment_comparison.csv (script 17), both versions;
  * only the 2026 fold at h=66 and h=126 is excluded from the main metric;
  * train-mean R2_oos is exactly 0 in every fold (reference check from CLAUDE.md).

Aggregation: fold mean of RMSE / MAE / R2_oos (CLAUDE.md main metric). Standard R2 is a
footnote metric, given both as fold mean (not comparable across folds, see CLAUDE.md) and
pooled over all main-fold test rows.

Inputs : outputs/*_publication_aligned.* and their unsuffixed counterparts,
         outputs/gpr_alignment_comparison*.csv, outputs/gpr_alignment_decomposition.csv
Outputs: outputs/paper_numbers_publication_aligned.md
         outputs/primary_family_tests_publication_aligned.csv

Runtime: a few seconds.
"""
import json
import subprocess
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
HORIZONS = [5, 22, 66, 126]
PUB, TS = "publication", "timestamp"

# (model id, label, role) -- same ids as scripts/17_gpr_alignment_comparison.py
MODELS = [
    ("xgboost", "XGBoost (birincil, 65 özellik)", "birincil"),
    ("bilstm", "Attention BiLSTM", "birincil"),
    ("h1_xgb_bilstm", "Hibrit H1: 0.5 XGB + 0.5 BiLSTM", "hibrit"),
    ("h2_harx_xgb", "Hibrit H2: 0.5 HAR-X + 0.5 XGB", "hibrit"),
    ("h3_harx_resid", "Hibrit H3: HAR-X + XGB artığı", "hibrit"),
    ("har_x", "HAR-X", "ekonometrik"),
    ("har_x_log", "HAR-X-log", "ekonometrik"),
    ("har", "HAR", "ekonometrik"),
    ("har_log", "HAR-log", "ekonometrik"),
    ("garch", "GARCH(1,1)", "ekonometrik"),
    ("train_mean", "Train-mean", "naif"),
    ("past_vol", "Past-volatility", "naif"),
    ("har_ovx", "HAR + OVX", "ablasyon"),
    ("har_gpr", "HAR + GPR", "ablasyon"),
    ("xgb6", "XGBoost-6", "keşifsel"),
    ("xgboost_optuna", "XGBoost, Optuna + büzülmüş smearing", "sağlamlık"),
    ("xgboost_optuna_raw", "XGBoost, Optuna + ham smearing", "ek (appendix)"),
]
LABEL = {m: lab for m, lab, _ in MODELS}
ROLE = {m: r for m, _, r in MODELS}
HYBRID_MODELS = ["har_x", "har_x_log", "har", "xgboost", "bilstm", "train_mean",
                 "past_vol", "h1_xgb_bilstm", "h2_harx_xgb", "h3_harx_resid"]
MCOLS = ["model", "horizon", "test_year", "include_in_main", "n", "rmse", "mae",
         "r2_oos", "r2"]


def provenance():
    """HEAD commit, generation time, and any uncommitted change under scripts/ or outputs/
    other than this script's own two outputs (a non-empty list means the numbers do NOT
    correspond exactly to the named commit)."""
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True)
    own = {"outputs/paper_numbers_publication_aligned.md",
           "outputs/primary_family_tests_publication_aligned.csv"}
    # porcelain lines are "XY path"; the status column may start with a space, so the
    # output must not be stripped before slicing
    dirty = [ln[3:] for ln in git("status", "--porcelain", "--untracked-files=all", "--",
                                  "scripts", "outputs", "data").splitlines()
             if ln[3:] not in own]
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    return {"commit": git("rev-parse", "HEAD"),
            "subject": git("log", "-1", "--format=%s"),
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dirty": dirty}


def rd(name, al):
    return pd.read_csv(alignment.out(name, al), float_precision="round_trip")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def fold_metrics(al):
    """Per-fold metrics of all 17 models (same source mapping as script 17, plus R2)."""
    parts = [rd("hybrid_metrics_all.csv", al)[MCOLS]]
    ben = rd("bench_metrics_all.csv", al).rename(columns={"n_test": "n"})
    parts.append(ben[ben["model"].isin(["garch", "har_log"])][MCOLS])
    abl = rd("ablation_exogenous_folds.csv", al).rename(columns={"variant": "model",
                                                                 "n_test": "n"})
    parts.append(abl[abl["model"].isin(["har_ovx", "har_gpr"])][MCOLS])
    x6 = rd("exploratory_xgb6_folds.csv", al).rename(columns={"n_test": "n"})
    parts.append(x6.assign(model="xgb6")[MCOLS])
    for name, mid in (("opt_metrics_all.csv", "xgboost_optuna"),
                      ("opt_rawsmearing_metrics_all.csv", "xgboost_optuna_raw")):
        o = rd(name, al).rename(columns={"n_test": "n"})
        parts.append(o[o["model"] == "xgboost"].assign(model=mid)[MCOLS])
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(["model", "horizon", "test_year"]).any()
    assert set(df["model"]) == set(LABEL)
    # Only the 2026 fold at h=66 / h=126 is outside the main metric.
    excl = df.loc[~df["include_in_main"], ["horizon", "test_year"]].drop_duplicates()
    assert set(map(tuple, excl.values)) == {(66, 2026), (126, 2026)}, excl
    tm = df[df["model"] == "train_mean"]
    assert (tm["r2_oos"].abs() < 1e-12).all(), "train-mean R2_oos must be exactly 0"
    return df


def predictions(al):
    """Long frame: model, horizon, test_year, include_in_main, y_true, pred."""
    keep = ["horizon", "Date", "test_year", "include_in_main", "y_true"]
    parts = []
    hyb = rd("hybrid_predictions_all.csv", al)
    for m in HYBRID_MODELS:
        parts.append(hyb[keep].assign(model=m, pred=hyb[f"pred_{m}"]))
    ben = rd("bench_predictions_all.csv", al)
    for m in ("garch", "har_log"):
        parts.append(ben[keep].assign(model=m, pred=ben[f"pred_{m}"]))
    abl = rd("ablation_exogenous_predictions.csv", al)
    abl = abl[abl["variant"].isin(["har_ovx", "har_gpr"])]
    parts.append(abl[keep].assign(model=abl["variant"].values, pred=abl["pred"].values))
    x6 = rd("exploratory_xgb6_predictions.csv", al)
    parts.append(x6[keep].assign(model="xgb6", pred=x6["pred_xgb6"]))
    for name, mid in (("opt_predictions_all.csv", "xgboost_optuna"),
                      ("opt_rawsmearing_predictions_all.csv", "xgboost_optuna_raw")):
        if not alignment.out(name, al).exists():
            # The timestamp raw-smearing run kept no prediction file; its pooled R2 is
            # not needed (pooled R2 is reported for the publication version only).
            assert al == TS, f"{name} missing in the publication version"
            continue
        o = rd(name, al)
        parts.append(o[keep].assign(model=mid, pred=o["pred_xgboost"]))
    return pd.concat(parts, ignore_index=True)


def check_preds_vs_metrics(pr, fm, al):
    e = pr["y_true"] - pr["pred"]
    rec = (pr.assign(se=e ** 2).groupby(["model", "horizon", "test_year"])
           .agg(rmse_rec=("se", lambda s: np.sqrt(s.mean())), n_rec=("se", "size"))
           .reset_index())
    fm = fm[fm["model"].isin(set(pr["model"]))]
    j = fm.merge(rec, on=["model", "horizon", "test_year"], how="outer", indicator=True)
    assert (j["_merge"] == "both").all(), f"[{al}] prediction/metric folds do not match"
    assert (j["n"] == j["n_rec"]).all(), f"[{al}] test sizes differ"
    assert np.allclose(j["rmse"], j["rmse_rec"], rtol=1e-9, atol=0), \
        f"[{al}] fold RMSE from predictions != stored metric"
    print(f"[kontrol] {al}: {len(j)} model x ufuk x fold satirinda tahminlerden "
          "yeniden hesaplanan RMSE kayitli metrikle ayni")


def common_r2_oos(pr, fm):
    """R2_oos of every model against ONE reference per fold: the train-mean baseline of
    hybrid_metrics_all (mean training target of the XGBoost window). The stored r2_oos of
    bench / ablation / XGB-6 / Optuna uses each model's own training window (HAR starts
    at row 21, XGBoost at 127), so their references differ from the hybrid ones; mixing
    them in one column would compare against different constant forecasts."""
    ref = (pr[pr["model"] == "train_mean"].groupby(["horizon", "test_year"])["pred"]
           .agg(["min", "max"]))
    assert (ref["min"] == ref["max"]).all(), "train-mean prediction not constant in fold"
    ref = ref["min"].rename("tm").reset_index()
    x = pr.merge(ref, on=["horizon", "test_year"], validate="m:1")
    x["sse"] = (x["y_true"] - x["pred"]) ** 2
    x["sst"] = (x["y_true"] - x["tm"]) ** 2
    s = x.groupby(["model", "horizon", "test_year"])[["sse", "sst"]].sum()
    r = (1 - s["sse"] / s["sst"]).rename("r2_oos_common").reset_index()
    j = fm.merge(r, on=["model", "horizon", "test_year"], how="left", validate="1:1")
    hyb = j[j["model"].isin(HYBRID_MODELS)]
    assert np.allclose(hyb["r2_oos"], hyb["r2_oos_common"], rtol=1e-9, atol=1e-12), \
        "common-reference R2_oos differs from hybrid_metrics_all"
    return j


def aggregate(fm, pr):
    main = fm[fm["include_in_main"]]
    agg = (main.groupby(["model", "horizon"])
           .agg(n_folds=("rmse", "size"), rmse=("rmse", "mean"), mae=("mae", "mean"),
                r2_oos=("r2_oos", "mean"), r2_oos_common=("r2_oos_common", "mean"),
                r2_fold=("r2", "mean")).reset_index())
    pm = pr[pr["include_in_main"]]
    pooled = []
    for (m, h), g in pm.groupby(["model", "horizon"]):
        y, p = g["y_true"].to_numpy(), g["pred"].to_numpy()
        pooled.append({"model": m, "horizon": h, "n_pooled": len(g),
                       "r2_pooled": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
    return agg.merge(pd.DataFrame(pooled), on=["model", "horizon"], how="left",
                     validate="1:1")


def step_up(p, c=1.0):
    """Benjamini-Hochberg adjusted p-values; with c = sum_{i<=m} 1/i this is
    Benjamini-Yekutieli, valid under arbitrary dependence."""
    p = np.asarray(p, dtype="float64")
    m = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * m * c / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def sign_p(k, n):
    k = min(k, n - k)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def wins(fm, a, b, h):
    """Folds (main) in which model a has lower RMSE than model b."""
    x = fm[fm["include_in_main"] & (fm["horizon"] == h)].pivot(
        index="test_year", columns="model", values="rmse")
    return int((x[a] < x[b]).sum()), int((x[a] > x[b]).sum()), len(x)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def md_table(df):
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


f6 = lambda x: f"{x:.6f}"
f3 = lambda x: f"{x:+.3f}".replace("+-", "−").replace("-", "−")
pct = lambda x: f"{x:+.2f}%".replace("-", "−")
fp = lambda x: "<0.001" if x < 0.001 else f"{x:.3f}"


def signed(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


def wide(agg, col, fmt, models, bold_min=False):
    t = agg.pivot(index="model", columns="horizon", values=col).loc[models, HORIZONS]
    out = pd.DataFrame({"model": [LABEL[m] for m in models],
                        "rol": [ROLE[m] for m in models]})
    for h in HORIZONS:
        best = t[h].min()
        out[f"h={h}"] = [(f"**{fmt(v)}**" if bold_min and v == best else fmt(v))
                         for v in t[h]]
    return out


# ---------------------------------------------------------------------------
# Section 12: HAR floor frequency, QLIKE, per-fold smearing (publication version only)
# ---------------------------------------------------------------------------
def qlike(y, p):
    """Patton (2011) QLIKE on the variance scale: r - log r - 1, r = y^2 / p^2."""
    r = (y / p) ** 2
    return r - np.log(r) - 1


def section_12e(w, qm, b):
    """Exploratory: is the h=5 HAR vs HAR-X QLIKE reversal driven by low HAR-X forecasts?
    Both checks were designed AFTER seeing the QLIKE results (post hoc)."""
    x = (qm[qm["model"].isin(["har_x", "har"])]
         .pivot_table(index=["horizon", "Date", "test_year"], columns="model",
                      values=["pred", "ql"], aggfunc="first"))
    x.columns = [f"{v}_{m}" for v, m in x.columns]
    x = x.reset_index().merge(b[["horizon", "Date", "pred_floor"]], on=["horizon", "Date"],
                              validate="1:1")
    x["kx"] = x["pred_har_x"] / x["pred_floor"]
    x["kh"] = x["pred_har"] / x["pred_floor"]
    w("### 12e. Keşifsel: h=5 QLIKE yön dönmesi ve HAR-X'in düşük tahminleri")
    w("")
    w("**Statü: keşifsel ve post hoc.** Her iki kontrol de QLIKE sonuçları görüldükten sonra "
      "tasarlandı; eşikler (1.25 × taban, 0.5 × σ̂_HAR) sonuçlara bakılarak seçildi. Test "
      "yok. Havuzlanmış, ana fold'lar.")
    w("")
    w("**(1) Tabana yakınlık, h=5.** σ̂ / fold tabanı; eşik σ̂ ≤ 1.25 × taban.")
    w("")
    g = x[x["horizon"] == 5]
    k1 = max(1, int(len(g) * 0.01))
    top = g.sort_values("ql_har_x", ascending=False).iloc[:k1]
    rows = []
    for lab, s in ((f"HAR-X'in en büyük %1 QLIKE satırı", top),
                   ("bütün h=5 test gözlemleri", g)):
        nx, nh = int((s["kx"] <= 1.25).sum()), int((s["kh"] <= 1.25).sum())
        rows.append({
            "küme": lab, "n": len(s),
            "HAR-X σ̂/taban, medyan (min–maks)":
                f"{s['kx'].median():.2f} ({s['kx'].min():.2f}–{s['kx'].max():.2f})",
            "HAR-X ≤ 1.25 × taban": f"{nx} (%{100 * nx / len(s):.1f})",
            "HAR σ̂/taban, medyan (min–maks)":
                f"{s['kh'].median():.2f} ({s['kh'].min():.2f}–{s['kh'].max():.2f})",
            "HAR ≤ 1.25 × taban": f"{nh} (%{100 * nh / len(s):.1f})"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    near = g["kx"] <= 1.25
    w(f"En büyük %1 satır HAR-X'in h=5 QLIKE toplamının "
      f"%{100 * top['ql_har_x'].sum() / g['ql_har_x'].sum():.1f}'ini, σ̂ ≤ 1.25 × taban "
      f"olan {int(near.sum())} satır %{100 * g.loc[near, 'ql_har_x'].sum() / g['ql_har_x'].sum():.1f}'ini "
      f"oluşturuyor. Bu satırlar hariç ortalama QLIKE: HAR-X "
      f"{g.loc[~near, 'ql_har_x'].mean():.4f}, HAR {g.loc[~near, 'ql_har'].mean():.4f} "
      f"(tümü: {g['ql_har_x'].mean():.4f} / {g['ql_har'].mean():.4f}).")
    w("")
    tt = top.merge(qm[(qm["model"] == "har_x") & (qm["horizon"] == 5)][["Date", "y_true"]],
                   on="Date", validate="1:1")
    w(md_table(pd.DataFrame({
        "tarih": tt["Date"], "σ": tt["y_true"].map(lambda v: f"{v:.5f}"),
        "σ̂ HAR-X": tt["pred_har_x"].map(lambda v: f"{v:.6f}"),
        "HAR-X/taban": tt["kx"].map(lambda v: f"{v:.2f}"),
        "QLIKE HAR-X": tt["ql_har_x"].map(lambda v: f"{v:.1f}"),
        "σ̂ HAR": tt["pred_har"].map(lambda v: f"{v:.5f}"),
        "HAR/taban": tt["kh"].map(lambda v: f"{v:.2f}"),
        "QLIKE HAR": tt["ql_har"].map(lambda v: f"{v:.3f}")})))
    w("")
    w("**(2) Mekanik kural: \"HAR-X belirgin düşük\" = σ̂_HAR-X < 0.5 × σ̂_HAR.** Pay: "
      "kural satırlarının Σ(QLIKE_HAR-X − QLIKE_HAR) içindeki payı (toplam fark negatifse "
      "pay işaretiyle okunmalı). Çıkarma sonrası ortalamalar havuzlanmış ve fold "
      "ortalaması olarak verilir.")
    w("")
    rows = []
    for h in HORIZONS:
        g = x[x["horizon"] == h]
        rule = g["pred_har_x"] < 0.5 * g["pred_har"]
        d = g["ql_har_x"] - g["ql_har"]
        yrs = g.loc[rule, "test_year"].value_counts().sort_index()
        rest = g[~rule]
        fm_ = rest.groupby("test_year")[["ql_har_x", "ql_har"]].mean().mean()
        rows.append({
            "ufuk": f"h={h}", "kural satırı": f"{int(rule.sum())} / {len(g)}",
            "yıllar": ", ".join(f"{y} ({n})" for y, n in yrs.items()) or "—",
            "Σ fark (tümü)": f"{d.sum():+.2f}".replace("-", "−"),
            "Σ fark (kural satırları)": f"{d[rule].sum():+.2f}".replace("-", "−"),
            "pay": (f"%{100 * d[rule].sum() / d.sum():.1f}".replace("-", "−")
                    if rule.any() else "—"),
            "hariç ort. QLIKE, havuz (HAR-X / HAR)":
                f"{rest['ql_har_x'].mean():.4f} / {rest['ql_har'].mean():.4f}",
            "hariç ort. QLIKE, fold ort. (HAR-X / HAR)":
                f"{fm_['ql_har_x']:.4f} / {fm_['ql_har']:.4f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")


def section_13(w, pr, F):
    """Clark-West supplementary family (21_clark_west.py), XGB-6 vs HAR fold counts, and a
    pointer to the 2026 partial-year footnote."""
    cw = rd("clark_west.csv", PUB)
    meta = json.load(open(alignment.out("clark_west.json", PUB), encoding="utf-8"))
    p = cw["p_HLN_one_sided"].to_numpy()
    assert np.allclose(step_up(p), cw["p_HLN_bh"], rtol=1e-12, atol=0)
    assert np.allclose(step_up(p, sum(1 / i for i in range(1, len(p) + 1))),
                       cw["p_HLN_by"], rtol=1e-12, atol=0)
    # The statistic is built from the same saved forecasts the package uses.
    q = pr[pr["include_in_main"] & pr["model"].isin(["har", "har_x"])].pivot_table(
        index=["horizon", "Date"], columns="model", values=["pred", "y_true"],
        aggfunc="first")
    for h in HORIZONS:
        g = q.loc[h]
        y = g[("y_true", "har")]
        f = ((y - g[("pred", "har")]) ** 2
             - ((y - g[("pred", "har_x")]) ** 2 - (g[("pred", "har")] - g[("pred", "har_x")]) ** 2))
        r = cw[cw["horizon"] == h].iloc[0]
        assert len(f) == r["n"] and np.isclose(f.mean(), r["f_mean"], rtol=1e-10, atol=0)

    w("## 13. Ek aile: Clark–West; XGBoost-6 vs HAR; 2026 dipnotu")
    w("")
    w("### 13a. Clark–West testi, HAR ⊂ HAR-X (ek aile, 4 test)")
    w("")
    w(f"**Etiket:** {meta['label']}. Aile {meta['declared']['date']} tarihinde "
      f"CLAUDE.md'de ilan edildi (commit `{meta['declared']['commit']}`); CW istatistiği "
      "depoda bundan önce hesaplanmamıştı. **Birincil aile 8 testle sabittir; CW oraya "
      "eklenmez ve DM testlerinin yerine geçmez.** Holm/BH/BY bu 4 test içinde.")
    w("")
    w("`f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`; H1: E[f] > 0 (HAR-X daha iyi), "
      "**tek yanlı**. Kayıtlı (yayımlanan, tabanlanmış) tahminler, ana fold'lar, "
      "havuzlanmış seri (DM gibi). HAC: Newey-West, Bartlett, L = h−1. Çıkarım DM "
      "birincil ailesiyle aynı: HLN çarpanı ve t(n−1); düzeltmeler HLN p değerine "
      "uygulanır. HLN'siz normal p yan sütunda. Kaynak: `21_clark_west.py`, "
      "`clark_west_publication_aligned.csv`.")
    w("")
    sci = lambda v: f"{v:.2e}".replace("-", "−")
    t = pd.DataFrame({
        "ufuk": [f"h={h}" for h in cw["horizon"]], "n": cw["n"],
        "ort. (e²_HAR − e²_HARX)": cw["mean_mse_diff"].map(sci),
        "ort. düzeltme (ŷ_HAR − ŷ_HARX)²": cw["mean_adjustment"].map(sci),
        "ort. f": cw["f_mean"].map(sci),
        "CW": cw["CW"].map(lambda v: f"{v:.3f}"),
        "CW (HLN)": cw["CW_HLN"].map(lambda v: f"{v:.3f}"),
        "p normal (ham)": cw["p_normal_one_sided"].map(fp),
        "p HLN (ham)": cw["p_HLN_one_sided"].map(fp),
        "Holm": cw["p_HLN_holm"].map(fp), "BH": cw["p_HLN_bh"].map(fp),
        "BY": cw["p_HLN_by"].map(fp),
        "HAC şişme": cw["variance_inflation"].map(lambda v: f"{v:.2f}")})
    w(md_table(t))
    w("")
    # The note below claims the HLN choice does not change the result: check it.
    pn = cw["p_normal_one_sided"].to_numpy()
    c4 = sum(1 / i for i in range(1, len(pn) + 1))
    holm_ = lambda x: np.minimum(1, np.maximum.accumulate(
        np.sort(x) * (len(x) - np.arange(len(x)))))[np.argsort(np.argsort(x))]
    for fn_n, col in ((holm_, "p_HLN_holm"), (step_up, "p_HLN_bh"),
                      (lambda x: step_up(x, c4), "p_HLN_by")):
        assert ((fn_n(pn) < 0.05) == (cw[col].to_numpy() < 0.05)).all(), col
    w("Not: Clark & West (2007) standart normal kullanır; burada DM ailesiyle tutarlılık "
      "için HLN uygulandı, sonuç değişmiyor (normal p'lerle Holm, BH ve BY altında %5'te "
      "reddedilen testler aynı; kontrol edildi).")
    w("")
    w("**Fold düzeyinde (betimleyici, test değil):** f'nin fold ortalamalarının ortalaması "
      "ve f ortalaması pozitif olan fold sayısı: "
      + "; ".join(f"h={r.horizon}: {sci(r.f_fold_mean)}, {r.folds_f_positive}/{r.n_folds}"
                  for r in cw.itertuples()) + ".")
    w("")
    w("Okuma notu: ilk sütun DM'nin kullandığı ham MSE farkıdır; h=66 ve h=126'da negatiftir "
      "(havuzlanmış seride HAR'ın MSE'si daha düşük). CW istatistiği buna tahmin farkının "
      "karesini ekler.")
    w("")

    w("### 13b. XGBoost-6 vs HAR, fold bazında (keşifsel; test yok)")
    w("")
    w("XGBoost-6 keşifseldir; p değeri verilmez. Kazanma: fold RMSE'si HAR'ınkinden düşük. "
      "Uyuşma: kazanma çoğunluğunun yönü ile fold ortalaması RMSE farkının yönü aynı mı. "
      "`100 × (RMSE_XGB-6 / RMSE_HAR − 1)`, pozitif = XGBoost-6 daha kötü.")
    w("")
    rows = []
    for h in HORIZONS:
        x = F[F["include_in_main"] & (F["horizon"] == h)].pivot(
            index="test_year", columns="model", values="rmse")
        d = x["xgb6"] - x["har"]
        k, n = int((d < 0).sum()), len(d)
        ties = int((d == 0).sum())
        mean_pct = 100 * (x["xgb6"].mean() / x["har"].mean() - 1)
        cnt_dir = "XGB-6" if k > n - k - ties else ("HAR" if k < n - k - ties else "eşit")
        mean_dir = "XGB-6" if mean_pct < 0 else "HAR"
        r = {"ufuk": f"h={h}", "XGB-6 kazandığı yıl / fold": f"{k}/{n}",
             "fold ort. farkı": pct(mean_pct),
             "sayım yönü": cnt_dir, "ortalama yönü": mean_dir,
             "uyuşuyor mu": "evet" if cnt_dir == mean_dir else "hayır"}
        if cnt_dir != mean_dir:
            top = d.reindex(d.abs().sort_values(ascending=False).index)
            top = top[np.sign(top) == np.sign(d.mean())].head(3)
            r["ortalamayı taşıyan yıllar"] = ", ".join(
                f"{y} ({signed(v * 1e4, 2)}e−4)" for y, v in top.items())
        else:
            r["ortalamayı taşıyan yıllar"] = "—"
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 13c. 2026 kısmi yıl, h=66 ve h=126")
    w("")
    w("Bölüm 1e'de: tüm modellerin o fold'daki RMSE/MAE/R²_oos'u, n = 101 (h=66) ve 41 "
      "(h=126). Birincil toplulaştırmadan dışlanmıştır; model karşılaştırması veya seçimi "
      "için kullanılmaz.")
    w("")


def section_14(w):
    """Table 1: descriptive statistics (22_descriptive_stats.py)."""
    d = rd("descriptive_stats.csv", PUB)
    meta = json.load(open(alignment.out("descriptive_stats.json", PUB), encoding="utf-8"))
    g4 = lambda v: f"{v:.4g}".replace("-", "−")

    def table(x):
        return md_table(pd.DataFrame({
            "değişken": x["label"], "N": x["N"], "ortalama": x["mean"].map(g4),
            "std": x["std"].map(g4), "min": x["min"].map(g4), "maks": x["max"].map(g4),
            "çarpıklık": x["skew"].map(lambda v: f"{v:.2f}".replace("-", "−")),
            "fazla basıklık": x["excess_kurtosis"].map(lambda v: f"{v:.2f}".replace("-", "−")),
            "ADF (gecikme)": [f"{s:.2f} ({k})".replace("-", "−")
                              for s, k in zip(x["adf_stat"], x["adf_lag"])],
            "ADF p": x["adf_p"].map(fp),
            f"Q({meta['ljung_box_lag']})": x["lb_q20"].map(lambda v: f"{v:.1f}"),
            "Q p": x["lb_p"].map(fp)}))

    s = meta["sample"]
    w("## 14. Tablo 1: Tanımlayıcı istatistikler")
    w("")
    w(table(d[d["role"] == "main"]))
    w("")
    w("**Tablo notu.**")
    w(f"- Örneklem: modelin kullandığı örneklem, {s['rows']} satır, işlem takvimi "
      f"({s['first_date']} – {s['last_date']}). Getiri ilk satırı kaybeder; `target_vol_h` "
      "son h satırda tanımsızdır (tamamlanmamış pencere, `skipna=False`).")
    w("- Getiri: `log(P_t / P_{t−1})`. Hedef: sonraki h günlük log getirinin standart "
      "sapması. Birim: günlük log getiri. OVX: düzey, endeks puanı.")
    w("- **GPRD ve GPRD_THREAT modelin gördüğü haliyle, yani yayım-hizalı:** satır t'de, "
      "t−1'e kadar yayımlanmış en son gözlem (`gprd_lag1`, `gprd_threat_lag1`). Seri iki "
      "yayım arasında sabit kalır; ilk yayımdan önceki satırlarda tanımsızdır.")
    w(f"- Çarpıklık ve fazla basıklık pandas'ın yanlılık düzeltmeli tahmincileri (normal "
      f"dağılımda fazla basıklık 0). ADF: sabitli, gecikme AIC ile (statsmodels varsayılan "
      f"en büyük gecikme 12(n/100)^(1/4)); H0 birim kök. Ljung–Box Q({meta['ljung_box_lag']}): "
      "H0 20. gecikmeye kadar otokorelasyon yok.")
    w("- **Hedeflerde Ljung–Box reddi mekaniktir.** Ardışık `target_vol_h` değerleri h "
      "getirinin h−1'ini paylaşan örtüşen pencerelerden hesaplanır; otokorelasyon "
      "yapıdan gelir. Bu red kalıcılık kanıtı olarak okunmamalıdır. Yayım-hizalı GPR "
      "serisi de iki yayım arasında sabit kaldığından otokorelasyonunun bir kısmı "
      "yapıdandır (Q(20): GPRD "
      + f"{d.set_index('variable').loc['GPRD', 'lb_q20']:.1f}; gözlem tarihli seride "
      + f"{d.set_index('variable').loc['GPRD_obs_trading', 'lb_q20']:.1f}, dipnot).")
    w("")
    oc = meta.get("own_calendar")
    w("**Dipnot: GPR başka takvimlerde** (karşılaştırma için; model bunları görmez).")
    w("")
    w(table(d[d["role"] == "footnote"]))
    w("")
    note = ("Gözlem tarihli satırlar `data/veriseti.xlsx`'in kaydırılmamış GPR sütunlarıdır "
            "(işlem günleri).")
    if oc:
        mad = max(oc["max_abs_diff_vs_dataset"].values())
        zg, zt = oc["zero_days"]["GPRD"], oc["zero_days"]["GPRD_THREAT"]
        note += (f" Kendi takvimi satırları endeksin her takvim gününü (hafta sonları dahil, "
                 f"{oc['calendar_days']} gün) kapsar; kaynak, veri setinin üretildiği "
                 f"{oc['vintage']} arşiv sürümü `{Path(oc['file']).name}` (SHA-256 "
                 f"`{oc['sha256'][:16]}…`, git dışı). Bu sürüm veri setinin GPR değerlerini "
                 f"{oc['trading_days_checked']} işlem gününün hepsinde yeniden üretir (en "
                 f"büyük mutlak fark {mad:.1e}, kayan nokta yuvarlaması; kontrol edildi).")
        if zg:
            note += (" " + ", ".join(z["date"] for z in zg)
                     + (" tarihinde" if len(zg) == 1 else " tarihlerinde") + " GPRD 0 (aynı "
                     + ("gün" if len(zg) == 1 else "günlerde") + " GPRD_THREAT "
                     + ", ".join(f"{z['GPRD_THREAT']:g}" for z in zg) + ").")
        if zt:
            note += (f" GPRD_THREAT {len(zt)} günde 0: "
                     + ", ".join(z["date"] for z in zt) + ".")
    w(note)
    w("")


def section_12(w, pr, A):
    fkeys = ["horizon", "fold"]
    ben = rd("bench_predictions_all.csv", PUB)
    bf = rd("bench_folds_all.csv", PUB)
    b = ben.merge(bf[fkeys + ["pred_floor", "n_clipped_har", "n_clipped_har_x"]],
                  on=fkeys, validate="m:1")
    abl = rd("ablation_exogenous_predictions.csv", PUB)
    af = rd("ablation_exogenous_folds.csv", PUB)
    a = abl.merge(bf[fkeys + ["pred_floor"]], on=fkeys, validate="m:1")
    hyb = rd("hybrid_predictions_all.csv", PUB)
    hf = rd("hybrid_folds_all.csv", PUB)
    x6f = rd("exploratory_xgb6_folds.csv", PUB)

    # --- floor detection: prediction == that fold's training-target minimum -----------
    # The floor is np.maximum(pred, floor) with floor = min of the fold's TRAINING target
    # (05 / 11 / 07). The prediction files carry no flag, so a floored row is detected as
    # an exact equality with the fold's recorded floor, and the per-fold count must equal
    # the counter the fitting script recorded at fit time.
    flo = {}  # model -> frame with horizon, test_year, include_in_main, floored
    for m in ("har", "har_x"):
        eq = b[f"pred_{m}"] == b["pred_floor"]
        per = b.assign(eq=eq).groupby(fkeys)["eq"].sum()
        rec = bf.set_index(fkeys)[f"n_clipped_{m}"].reindex(per.index)
        assert (per == rec).all(), f"{m}: floor equality != recorded n_clipped"
        flo[m] = b[["horizon", "Date", "test_year", "include_in_main"]].assign(floored=eq.values)
    for m in ("har_ovx", "har_gpr"):
        g = a[a["variant"] == m]
        eq = g["pred"] == g["pred_floor"]
        per = g.assign(eq=eq).groupby(fkeys)["eq"].sum()
        rec = af[af["variant"] == m].set_index(fkeys)["n_clipped"].reindex(per.index)
        assert (per == rec).all(), f"{m}: floor equality != recorded n_clipped"
        flo[m] = g[["horizon", "Date", "test_year", "include_in_main"]].assign(floored=eq.values)
    hh = hyb.merge(hf[["horizon", "test_year", "pred_floor", "n_floored"]],
                   on=["horizon", "test_year"], validate="m:1")
    eq = hh["pred_h3_harx_resid"] == hh["pred_floor"]
    per = hh.assign(eq=eq).groupby(["horizon", "test_year"])["eq"].sum()
    rec = hf.set_index(["horizon", "test_year"])["n_floored"].reindex(per.index)
    assert (per == rec).all(), "H3: floor equality != recorded n_floored"
    flo["h3_harx_resid"] = hh[["horizon", "Date", "test_year", "include_in_main"]].assign(
        floored=eq.values)
    # The HAR-X used in the package (hybrid file) is floored on exactly the same rows.
    j = hyb.merge(b[["horizon", "Date", "pred_floor", "pred_har_x"]], on=["horizon", "Date"],
                  suffixes=("", "_b"), validate="1:1")
    assert ((j["pred_har_x_b"] == j["pred_floor"])
            == np.isclose(j["pred_har_x"], j["pred_floor"], rtol=0, atol=1e-15)).all()
    # Chance equality: models WITHOUT a prediction floor never hit the value exactly, and
    # the closest non-floored prediction of a floored model stays well above it.
    chance = {m: int((b[f"pred_{m}"] == b["pred_floor"]).sum())
              for m in ("har_log", "har_x_log", "garch", "past_vol")}
    assert not any(chance.values()), chance
    below = {m: int((b[f"pred_{m}"] < b["pred_floor"]).sum()) for m in ("har_log", "har_x_log")}
    gaps = []
    for m in ("har", "har_x"):
        d = (b[f"pred_{m}"] - b["pred_floor"])[b[f"pred_{m}"] > b["pred_floor"]]
        gaps.append((d.min(), (d / b.loc[d.index, "pred_floor"]).min()))
    for m in ("har_ovx", "har_gpr"):
        g = a[a["variant"] == m]
        d = (g["pred"] - g["pred_floor"])[g["pred"] > g["pred_floor"]]
        gaps.append((d.min(), (d / g.loc[d.index, "pred_floor"]).min()))
    gap_abs, gap_rel = min(x[0] for x in gaps), min(x[1] for x in gaps)

    w("## 12. Taban sıklığı, QLIKE ve fold başına smearing (yayım-hizalı)")
    w("")
    w("Yeniden eğitim yok; her şey kayıtlı tahmin ve fold dosyalarından. QLIKE yalnızca "
      "betimleyicidir: QLIKE kaybıyla DM veya işaret testi koşulmadı, birincil aile 8 testle "
      "sabittir.")
    w("")
    w("### 12a. Tahmin tabanının devreye girme sıklığı")
    w("")
    w("Düzey ölçekli OLS tahminleri (HAR, HAR-X, ablasyon basamakları) ve Hibrit H3, "
      "fold'un **eğitim hedefinin minimumunda** tabanlanır: `max(tahmin, min(y_train))` "
      "(train-only). Log ölçekli HAR-log ve HAR-X-log tahminleri `exp(·) × smearing` "
      "olduğu için yapısal olarak pozitiftir; tahmine taban uygulanmaz (0 tanım gereği). "
      "(Log spesifikasyonların *regresörlerine* uygulanan `LOG_FLOOR = 1e-4` ayrı bir "
      "şeydir ve burada sayılmaz.)")
    w("")
    w("**Tespit yöntemi.** Tahmin dosyalarında taban işareti yok. Tabanlanmış satır, "
      "tahminin o fold'un kayıtlı tabanına (`bench_folds_all.pred_floor`, H3 için "
      "`hybrid_folds_all.pred_floor`) **tam eşit** olduğu satır olarak tespit edildi. "
      "Kontroller (assert):")
    w("- Her fold'da eşitlik sayısı, uyum anında kaydedilen sayaçla birebir aynı "
      "(`n_clipped_har`, `n_clipped_har_x`, ablasyon `n_clipped`, H3 `n_floored`).")
    w(f"- Tesadüfi eşitlik yok: tabansız modellerde (HAR-log, HAR-X-log, GARCH, "
      f"past-volatility) fold tabanına tam eşit tahmin sayısı "
      f"{sum(chance.values())}. Oysa HAR-log {below['har_log']}, HAR-X-log "
      f"{below['har_x_log']} satırda tabanın **altında** tahmin veriyor; yani eşitlik "
      "ancak `max()` işleminden doğuyor.")
    w(f"- Tabanlı modellerde tabanlanmamış en yakın tahmin, tabanın {gap_abs:.2e} "
      f"(göreli %{100 * gap_rel:.3f}) üstünde; sürekli bir OLS tahmininin tabana bit "
      "düzeyinde tesadüfen eşit çıkması pratikte olanaksız.")
    w("- Pakette kullanılan HAR-X (hibrit dosyası) aynı satırlarda tabanlanıyor.")
    w("")
    specs = [("har", "HAR"), ("har_ovx", "HAR + OVX"), ("har_gpr", "HAR + GPR"),
             ("har_x", "HAR-X"), ("har_log", "HAR-log"), ("har_x_log", "HAR-X-log"),
             ("h3_harx_resid", "Hibrit H3 (ek)")]
    rows = []
    for m, lab in specs:
        r = {"model": lab}
        for h in HORIZONS:
            if m in flo:
                g = flo[m][(flo[m]["horizon"] == h) & flo[m]["include_in_main"]]
                k, n = int(g["floored"].sum()), len(g)
                pf = g.groupby("test_year")["floored"].mean()
                top = (f"; en yoğun {pf.idxmax()} %{100 * pf.max():.1f}" if k else "")
                r[f"h={h}"] = f"{k} / {n} (%{100 * k / n:.2f}{top})"
            else:
                g = b[(b["horizon"] == h) & b["include_in_main"]]
                r[f"h={h}"] = f"0 / {len(g)} (taban yok)"
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    tot = {m: (int(f["floored"].sum()), len(f)) for m, f in flo.items()}
    w("Ana metriğe giren fold'lar (h=66/126'da 2026 hariç). 2026 dahil tüm fold'lar: "
      + ", ".join(f"{lab} {tot[m][0]}/{tot[m][1]}" for m, lab in specs if m in tot)
      + f". XGBoost-6 aynı train-min tabanını kullanır; kayıtlı sayaç "
        f"{int(x6f['n_clipped'].sum())} (tabana takılan tahmin yok).")
    w("")

    # --- QLIKE ------------------------------------------------------------------------
    bad = pr[~(pr["pred"] > 0)]
    if len(bad):
        raise SystemExit("QLIKE durduruldu: pozitif olmayan tahmin\n"
                         + bad.groupby(["model", "horizon"]).size().to_string())
    assert (pr["y_true"] > 0).all(), "non-positive realized target"
    q = pr.assign(ql=qlike(pr["y_true"].to_numpy(), pr["pred"].to_numpy()))
    qm = q[q["include_in_main"]]
    qf = (qm.groupby(["model", "horizon", "test_year"])["ql"].mean()
          .groupby(["model", "horizon"]).mean().rename("qlike_fold"))
    qp = qm.groupby(["model", "horizon"])["ql"].mean().rename("qlike_pooled")
    Q = pd.concat([qf, qp], axis=1).reset_index().merge(
        A[["model", "horizon", "rmse"]], on=["model", "horizon"], validate="1:1")
    models = [m for m, _, _ in MODELS]
    f4 = lambda x: f"{x:.4f}"
    w("### 12b. QLIKE (Patton 2011), varyans ölçeğinde")
    w("")
    w("`QLIKE = σ²/σ̂² − log(σ²/σ̂²) − 1`, σ = gerçekleşen hedef, σ̂ = tahmin. Hedef bir "
      "standart sapma olduğu için ikisi de karelenir. Düşük = iyi; mükemmel tahminde 0. "
      "**Taban bağlanan gözlemlerde QLIKE yayımlanan (tabanlanmış) tahmin üzerinden "
      "hesaplanır** — değerlendirilen şey modelin verdiği tahmindir. Tüm tahminler ve "
      f"hedefler pozitif (en küçük tahmin {pr['pred'].min():.6f}); durdurma koşulu "
      "tetiklenmedi. Kalın = sütundaki en düşük.")
    w("")
    w("**Fold ortalaması (birincil):**")
    w("")
    w(md_table(wide(Q, "qlike_fold", f4, models, bold_min=True)))
    w("")
    w("**Havuzlanmış (ikincil):** ana metriğe giren tüm test satırları üzerinden ortalama.")
    w("")
    w(md_table(wide(Q, "qlike_pooled", f4, models, bold_min=True)))
    w("")
    w("**Tabanlanmış satırların QLIKE payı** (havuzlanmış, ana fold'lar): tabanlanmış "
      "satırların toplam QLIKE içindeki payı / satır payı. Taban, eğitim hedefinin "
      "minimumu olduğundan bu satırlarda σ̂ küçüktür ve QLIKE büyür.")
    w("")
    rows = []
    for m, lab in specs:
        if m not in flo or m in ("har_log", "har_x_log"):
            continue
        g = qm[qm["model"] == m].merge(flo[m][["horizon", "Date", "floored"]],
                                        on=["horizon", "Date"], validate="1:1")
        r = {"model": lab}
        for h in HORIZONS:
            x = g[g["horizon"] == h]
            k = int(x["floored"].sum())
            r[f"h={h}"] = ("—" if k == 0 else
                           f"%{100 * x.loc[x['floored'], 'ql'].sum() / x['ql'].sum():.1f} / "
                           f"%{100 * k / len(x):.2f}")
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 12c. QLIKE ve RMSE sıralamaları (fold ortalaması)")
    w("")
    w("Sıra 1 = en iyi. Yalnızca iki kayıpta sırası farklı olan modeller listelenir. İki "
      "tutarlı kayıp farklı sıralayabilir (Patton 2011); farklılık bir bulgu olarak "
      "raporlanır.")
    w("")
    from scipy import stats as _st
    rows, diffs = [], []
    for h in HORIZONS:
        g = Q[Q["horizon"] == h].set_index("model")
        rr = g["rmse"].rank(method="min").astype(int)
        rq = g["qlike_fold"].rank(method="min").astype(int)
        tau = float(_st.kendalltau(g["rmse"], g["qlike_fold"])[0])
        ch = [m for m in models if rr[m] != rq[m]]
        rows.append({"ufuk": f"h={h}", "Kendall τ (17 model)": f"{tau:.3f}",
                     "sırası değişen model": f"{len(ch)}/17",
                     "RMSE'de en iyi": LABEL[rr.idxmin()],
                     "QLIKE'ta en iyi": LABEL[rq.idxmin()]})
        for m in sorted(ch, key=lambda m: rr[m]):
            diffs.append({"ufuk": f"h={h}", "model": LABEL[m], "RMSE sırası": rr[m],
                          "QLIKE sırası": rq[m],
                          "fark": f"{rq[m] - rr[m]:+d}".replace("-", "−")})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w(md_table(pd.DataFrame(diffs)))
    w("")
    w("**QLIKE'ın yoğunlaşması** (havuzlanmış, ana fold'lar): QLIKE eksik tahmini "
      "(σ̂ ≪ σ) sert cezalandırır, bu yüzden ortalama birkaç gözleme dayanabilir. Hücre: "
      "en büyük %1 satırın QLIKE toplamındaki payı; parantezde en büyük tek satırın σ/σ̂ "
      "oranı ve tarihi. Betimleyicidir.")
    w("")
    rows = []
    for m in models:
        r = {"model": LABEL[m]}
        for h in HORIZONS:
            g = qm[(qm["model"] == m) & (qm["horizon"] == h)]
            s = g["ql"].sort_values(ascending=False)
            k1 = max(1, int(len(s) * 0.01))
            i0 = s.index[0]
            r[f"h={h}"] = (f"%{100 * s.iloc[:k1].sum() / s.sum():.1f} "
                           f"({g.loc[i0, 'y_true'] / g.loc[i0, 'pred']:.1f}×, "
                           f"{g.loc[i0, 'Date']})")
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("**Birincil ailenin iki karşılaştırmasında yön** (betimleyici; test değil). "
      "`100 × (kayıp_a / kayıp_b − 1)`, pozitif = a daha kötü.")
    w("")
    rows = []
    for a_, b_ in (("har", "har_x"), ("xgboost", "har_x")):
        for h in HORIZONS:
            g = Q[Q["horizon"] == h].set_index("model")
            rows.append({"a vs b": f"{LABEL[a_]} vs {LABEL[b_]}", "ufuk": f"h={h}",
                         "RMSE (fold ort.)": pct(100 * (g.loc[a_, "rmse"] / g.loc[b_, "rmse"] - 1)),
                         "QLIKE (fold ort.)": pct(100 * (g.loc[a_, "qlike_fold"]
                                                          / g.loc[b_, "qlike_fold"] - 1)),
                         "QLIKE (havuz)": pct(100 * (g.loc[a_, "qlike_pooled"]
                                                      / g.loc[b_, "qlike_pooled"] - 1))})
    w(md_table(pd.DataFrame(rows)))
    w("")
    # --- the largest single h=5 HAR-X QLIKE row: origin vs target window -------------
    g = qm[(qm["model"] == "har_x") & (qm["horizon"] == 5)]
    top = g.loc[g["ql"].idxmax()]
    raw = pd.read_excel(ROOT / "data" / "veriseti.xlsx")
    rix = int(np.flatnonzero(raw["Date"].astype(str).values == top["Date"])[0])
    ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    win = raw["Date"].iloc[rix + 1: rix + 6].astype(str).tolist()
    assert np.isclose(ret.iloc[rix + 1: rix + 6].std(ddof=1), top["y_true"], rtol=1e-12)
    fl5 = b.loc[(b["horizon"] == 5) & (b["Date"] == top["Date"])]
    assert len(fl5) == 1 and not bool(flo["har_x"].loc[fl5.index[0], "floored"])
    fl5 = float(fl5["pred_floor"].iloc[0])
    w(f"**h=5 HAR-X'in en büyük tek QLIKE satırı ({top['Date']}).** Tarih, satırın kendi "
      f"tarihi t'dir, yani **tahmin kökeni**; hedef penceresinin başı değildir. Özellikler "
      f"`.shift(1)` ile t−1'e ({raw['Date'].iloc[rix - 1]}) kadarki bilgiyi kullanır. Hedef "
      "`std(r_{t+1}, …, r_{t+5})` olduğundan pencere, t'den sonraki beş işlem gününün "
      "getirileridir: " + ", ".join(win) + f" (her getiri bir önceki işlem gününün "
      f"kapanışından; ilki {raw['Date'].iloc[rix]} kapanışından {win[0]} kapanışına). "
      f"t günü getirisi ne özelliklerde ne hedefte yer alır. Gerçekleşen σ = "
      f"{top['y_true']:.6f}, HAR-X tahmini σ̂ = {top['pred']:.6f} (σ/σ̂ = "
      f"{top['y_true'] / top['pred']:.1f}, QLIKE = {top['ql']:.1f}). Fold tabanı "
      f"{fl5:.6f}; tahmin tabanın %{100 * (top['pred'] / fl5 - 1):.2f} üstünde, "
      "tabanlanmamış.")
    w("")
    # --- smearing -----------------------------------------------------------------------
    wfs = json.load(open(alignment.out("wf_summary_all.json", PUB), encoding="utf-8"))
    xs = pd.DataFrame(wfs["folds"])[["horizon", "test_year", "include_in_main", "smearing",
                                      "resid_log_std"]]
    bls = rd("bilstm_folds_all.csv", PUB)[["horizon", "test_year", "smearing"]]
    S = (xs.rename(columns={"smearing": "xgb", "resid_log_std": "xgb_sd"})
         .merge(bls.rename(columns={"smearing": "bilstm"}), on=["horizon", "test_year"],
                validate="1:1")
         .merge(bf[["horizon", "test_year", "har_smearing", "har_x_smearing"]]
                .rename(columns={"har_smearing": "har_log", "har_x_smearing": "har_x_log"}),
                on=["horizon", "test_year"], validate="1:1"))
    lrs = rd("log_residual_std.csv", PUB)  # 20_log_residual_std.py, OLS re-estimation
    lw = lrs.pivot(index=["horizon", "test_year"], columns="model",
                   values=["smearing", "resid_log_std"])
    lw.columns = [f"{m}{'_chk' if v == 'smearing' else '_sd'}" for v, m in lw.columns]
    S = S.merge(lw.reset_index(), on=["horizon", "test_year"], validate="1:1")
    assert len(S) == 60
    assert (S["har_log_chk"] == S["har_log"]).all() and         (S["har_x_log_chk"] == S["har_x_log"]).all(), "re-estimated smearing != bench_folds"
    w("### 12d. Fold başına Duan smearing katsayısı ve log-artık std'si")
    w("")
    w("Smearing `S = mean(exp(e))`, e = eğitim setindeki log ölçekli artıklar (örneklem-içi, "
      "train-only). XGBoost ve BiLSTM'de hedef `log(σ_h / past_vol_h)`, HAR-log ve "
      "HAR-X-log'da `log(σ_h)`. Kaynak: `wf_summary_all` (XGBoost: `smearing`, "
      "`resid_log_std`), `bilstm_folds_all`, `bench_folds_all` (`har_smearing`, "
      "`har_x_smearing`). Log-artık std'si (ddof=1): XGBoost için 03'ün kaydı "
      "(`resid_log_std`); HAR-log ve HAR-X-log için 05 yalnızca katsayıyı kaydettiğinden "
      "OLS `20_log_residual_std.py` ile yeniden tahmin edildi. Yeniden tahminin **her "
      "fold'da kayıtlı test tahminlerini ve smearing katsayısını bit düzeyinde ürettiği** "
      "assert edildi (120/120). **BiLSTM için log-artık std'si kaydedilmedi:** 06 eğitilmiş "
      "ağırlıkları saklamıyor, hesaplamak yeniden eğitim gerektirir. XGBoost/BiLSTM ile "
      "HAR-log ailesinin std'leri farklı hedeflerde (log-oran vs log-düzey) olduğundan "
      "doğrudan karşılaştırılamaz.")
    w("")
    rows = []
    for m, lab in (("xgb", "XGBoost, S"), ("xgb_sd", "XGBoost, log-artık std"),
                   ("bilstm", "BiLSTM, S"), ("har_log", "HAR-log, S"),
                   ("har_log_sd", "HAR-log, log-artık std"),
                   ("har_x_log", "HAR-X-log, S"),
                   ("har_x_log_sd", "HAR-X-log, log-artık std")):
        r = {"ölçü": lab}
        for h in HORIZONS:
            g = S[(S["horizon"] == h) & S["include_in_main"]][m]
            r[f"h={h}"] = f"{g.median():.4f} ({g.min():.4f}–{g.max():.4f})"
        rows.append(r)
    w("Özet, ana metriğe giren fold'lar: medyan (en küçük–en büyük). BiLSTM log-artık "
      "std'si: kaydedilmedi.")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    for h in HORIZONS:
        g = S[S["horizon"] == h].sort_values("test_year")
        t = pd.DataFrame({
            "yıl": [f"{y}" + ("" if im else " (ana metrik dışı)")
                    for y, im in zip(g["test_year"], g["include_in_main"])],
            "XGBoost S": g["xgb"].map(f4), "XGBoost log-artık std": g["xgb_sd"].map(f4),
            "BiLSTM S": g["bilstm"].map(f4), "BiLSTM log-artık std": "kaydedilmedi",
            "HAR-log S": g["har_log"].map(f4), "HAR-log log-artık std": g["har_log_sd"].map(f4),
            "HAR-X-log S": g["har_x_log"].map(f4),
            "HAR-X-log log-artık std": g["har_x_log_sd"].map(f4)})
        w(f"**h={h}**")
        w("")
        w(md_table(t))
        w("")
    section_12e(w, qm, b)
    return Q, flo, S


# ---------------------------------------------------------------------------
def main():
    fm = {al: fold_metrics(al) for al in (TS, PUB)}
    pr = {al: predictions(al) for al in (TS, PUB)}
    for al in (TS, PUB):
        check_preds_vs_metrics(pr[al], fm[al], al)
        fm[al] = common_r2_oos(pr[al], fm[al])
    agg ={al: aggregate(fm[al], pr[al]) for al in (TS, PUB)}

    # Consistency with script 17
    comp = pd.read_csv(OUT_DIR / "gpr_alignment_comparison.csv")
    for al, sfx in ((TS, "ts"), (PUB, "pub")):
        j = agg[al].merge(comp, on=["model", "horizon"], validate="1:1")
        assert len(j) == len(comp) == 17 * 4
        for m in ("rmse", "mae", "r2_oos"):
            assert np.allclose(j[m], j[f"{m}_{sfx}"], rtol=1e-12, atol=1e-15), (al, m)
    print("[kontrol] fold ortalamalari gpr_alignment_comparison.csv ile ayni (iki surum)")

    A = agg[PUB]
    F = fm[PUB]
    L = []  # markdown lines
    w = L.append

    # ---------------- Header ----------------
    pv = provenance()
    w("# Makale sayıları — yayım-hizalı GPR sürümü (BİRİNCİL)")
    w("")
    w(f"- **Üretildiği commit:** `{pv['commit']}` ({pv['subject']})")
    w(f"- **Üretim tarihi:** {pv['generated']}")
    w("- **Çalışma ağacı:** " + (
        "temiz — girdiler bu commit'teki dosyalarla birebir aynı."
        if not pv["dirty"] else
        "**KİRLİ — sayılar bu commit'e birebir karşılık gelmez.** Commit edilmemiş "
        "değişiklikler: " + ", ".join(f"`{p}`" for p in pv["dirty"])))
    w("- Doğrulama: `git checkout <commit> && python scripts/18_paper_numbers.py` aynı "
      "sayıları üretmelidir (yalnızca bu başlık değişir).")
    w("")
    w("> Bu dosya `scripts/18_paper_numbers.py` tarafından kayıtlı çıktılardan üretilir; "
      "elle düzenlenmez. Makale yazımında sayılar **yalnızca bu dosyadan** alınır. "
      "Zaman damgalı (eski) sürümün sayıları yalnızca Bölüm 8'de, Ek A için yer alır.")
    w("")
    w("**Genel kurallar.** Ana metrik fold ortalaması RMSE ve MAE; R²_oos ikincil "
      "(referans: o fold'un train hedef ortalaması); standart R² dipnot metriği. "
      "h=5 ve h=22'de 15 fold (2012–2026), h=66 ve h=126'da 14 fold (2026 kısmi yıl "
      "ana metrikten çıkarılır, ayrıca dipnotta verilir). Birim: günlük log getirilerin "
      "standart sapması. Yüzdeler `100 × (RMSE_a / RMSE_b − 1)`; negatif = a daha iyi.")
    w("")
    w("**p değerleri.** Çıkarım için kullanılan tek test ailesi **birincil sekizlik "
      "ailedir** (Bölüm 6: HAR vs HAR-X ve HAR-X vs XGBoost, dört ufuk). Aile "
      "**testlerden sonra resmileştirildi, ön-kayıt değildir**; ancak p değerlerine "
      "bakılarak değil, Aşama 5–6'da ilan edilmiş iki iddiaya göre seçildi ve o tarihten "
      "beri sabittir: sonradan test eklenmez. Bu dosyadaki diğer tüm p değerleri "
      "(ablasyon, XGBoost-6, iki sürüm karşılaştırması, BiLSTM kontrolleri, 9. bölüm) "
      "**keşifsel ve çoklu karşılaştırma için düzeltilmemiştir**; betimleyici olarak "
      "verilir. İkincil DM ailesi (24 test) kendi içinde Holm/BH/BY ile düzeltilir ama "
      "doğrulayıcı değildir.")
    w("")
    w("**Tutarlılık kontrolleri (assert):** her fold'un RMSE'si tahmin dosyalarından "
      "yeniden hesaplanıp kayıtlı metrikle karşılaştırıldı; fold ortalamaları "
      "`gpr_alignment_comparison.csv` ile aynı; train-mean R²_oos her fold'da tam 0; "
      "ana metrikten yalnızca 2026 fold'u h=66/126'da dışlanıyor.")
    w("")

    # ---------------- 1. Main table ----------------
    order = [m for m, _, _ in MODELS]
    w("## 1. Ana sonuç tablosu (fold ortalaması)")
    w("")
    w("Kaynak: `hybrid_metrics_all` (XGBoost, BiLSTM, hibritler, HAR, HAR-X, HAR-X-log, "
      "naif), `bench_metrics_all` (GARCH, HAR-log), `ablation_exogenous_folds` "
      "(HAR+OVX, HAR+GPR), `exploratory_xgb6_folds`, `opt_*metrics_all` — hepsi "
      "`_publication_aligned`. Ortak örneklem: her fold'da tüm modellerin test satırı "
      "sayısı aynı (assert); train pencereleri modele göre farklı olabilir (HAR ailesi "
      "satır 21'den, XGBoost 127'den başlar; Aşama 5 veri eşitleme kontrolü, Bölüm 7a). "
      "Kalın = sütundaki en düşük değer.")
    w("")
    w("### 1a. RMSE")
    w("")
    w(md_table(wide(A, "rmse", f6, order, bold_min=True)))
    w("")
    w("### 1b. MAE")
    w("")
    w(md_table(wide(A, "mae", f6, order, bold_min=True)))
    w("")
    w("### 1c. R²_oos (ikincil metrik)")
    w("")
    w("`R²_oos = 1 − SSE_model / Σ(y_test − train_mean)²`. **Referans tüm modellerde "
      "aynıdır:** her fold'da train-mean baseline'ının tahmini (XGBoost train "
      "penceresinin hedef ortalaması, `hybrid_metrics_all`). Bu yüzden train-mean satırı "
      "tam 0'dır ve sütun içindeki değerler aynı sabit tahmine göre ölçülür. Tahmin "
      "dosyalarından yeniden hesaplanmıştır; hibrit kaynaklı modellerde kayıtlı değerle "
      "birebir aynıdır (assert).")
    w("")
    w(md_table(wide(A, "r2_oos_common", f3, order)))
    w("")
    own = [m for m in order if m not in HYBRID_MODELS]
    t = pd.DataFrame({"model": [LABEL[m] for m in own]})
    for h in HORIZONS:
        s = A[A["horizon"] == h].set_index("model").loc[own]
        t[f"h={h}"] = [f3(v) for v in s["r2_oos"]]
    w("Not — kaynak dosyalardaki değerler farklı bir referans kullanır: `bench`, "
      "`ablation`, `exploratory_xgb6` ve `opt_*` her modelin **kendi** train "
      "penceresinin ortalamasını referans alır (HAR ailesi satır 21'den, XGBoost 127'den "
      "başlar). Aşağıdaki değerler o dosyalardadır; makalede kullanılmaz, yalnızca "
      "kaynak dosyalarla karşılaştırma yapılırsa farkın nedenini göstermek için "
      "verilir:")
    w("")
    w(md_table(t))
    w("")
    neg = A[(A["r2_oos_common"] < 0)].sort_values(["horizon", "model"])
    w("Negatif R²_oos (model sabit train-mean tahmininden kötü): " + "; ".join(
        f"h={h}: " + ", ".join(LABEL[m] for m in g["model"])
        for h, g in neg.groupby("horizon")) + ".")
    w("")
    w("### 1d. Standart R² (dipnot metriği, karar için kullanılmaz)")
    w("")
    w("Fold ortalaması ve havuzlanmış değer. Referansı test diliminin kendi ortalamasıdır "
      "(ex-post). Sakin yıllarda fold SST'si çok küçüldüğü için fold R²'leri aynı ölçekte "
      "değildir; fold ortalaması bu uyarıyla verilir. Havuzlanmış R² yıllar arası varyansı "
      "da içerdiği için sistematik olarak daha yüksektir.")
    w("")
    t = pd.DataFrame({"model": [LABEL[m] for m in order]})
    for h in HORIZONS:
        s = A[A["horizon"] == h].set_index("model").loc[order]
        t[f"h={h} fold ort."] = [f3(v) for v in s["r2_fold"]]
        t[f"h={h} havuz"] = [f3(v) for v in s["r2_pooled"]]
    w(md_table(t))
    w("")
    npool = A.groupby("horizon")["n_pooled"].agg(["min", "max"])
    assert (npool["min"] == npool["max"]).all(), "pooled sample differs across models"
    w("Havuzlanmış gözlem sayısı: " + ", ".join(
        f"h={h}: {int(r['min'])}" for h, r in npool.iterrows()) + ".")
    w("")
    w("### 1e. Dipnot: 2026 kısmi yıl, h=66 ve h=126 (düşük istatistiksel güç)")
    w("")
    w("Model karşılaştırması veya seçimi için kullanılmaz; yalnızca bilgi amaçlı.")
    w("")
    f26 = F[(F["test_year"] == 2026) & F["horizon"].isin([66, 126])]
    t = pd.DataFrame({"model": [LABEL[m] for m in order]})
    for h in (66, 126):
        s = f26[f26["horizon"] == h].set_index("model").loc[order]
        t[f"h={h} RMSE"] = [f6(v) for v in s["rmse"]]
        t[f"h={h} MAE"] = [f6(v) for v in s["mae"]]
        t[f"h={h} R²_oos"] = [f3(v) for v in s["r2_oos_common"]]
    w(md_table(t))
    w("")
    w("n: " + ", ".join(f"h={h}: {int(f26[f26['horizon'] == h]['n'].iloc[0])}"
                        for h in (66, 126)) + " test gözlemi.")
    w("")
    opt = rd("opt_metrics_all.csv", PUB)
    fb = (opt[(opt["model"] == "xgboost") & (opt["selection_procedure"] != "optuna")]
          .groupby("horizon").size())
    w("Not (Optuna satırları): validation dilimi kurulamayan erken fold'larda Optuna "
      "koşusu kapasite kuralına düşer: " + ", ".join(
          f"h={h}: {n} fold" for h, n in fb.items()) + " (tüm fold'lar, 2026 dahil).")
    w("")

    # ---------------- 2. Decomposition ----------------
    dec = pd.read_csv(OUT_DIR / "gpr_alignment_decomposition.csv")
    dp = dec[dec["gpr_alignment"] == PUB].set_index("horizon").loc[HORIZONS]
    w("## 2. Dört basamaklı ayrıştırma (HAR → HAR-X → XGBoost-6 → XGBoost)")
    w("")
    w("Her basamak tek bir şeyi değiştirir: (1) dışsal değişkenler (OVX, GPR) eklenir; "
      "(2) aynı altı regresör, düzey hedef, ön işleme yok — yalnızca fonksiyonel form "
      "doğrusaldan ağaca; (3) 65 özellik + log-oran hedef + smearing + ön işleme paketi. "
      "Kaynak: `gpr_alignment_decomposition.csv`. XGBoost-6 keşifsel ve post hoc'tur.")
    w("")
    t = pd.DataFrame({
        "ufuk": [f"h={h}" for h in HORIZONS],
        "RMSE HAR": [f6(v) for v in dp["rmse_har"]],
        "RMSE HAR-X": [f6(v) for v in dp["rmse_har_x"]],
        "RMSE XGB-6": [f6(v) for v in dp["rmse_xgb6"]],
        "RMSE XGB": [f6(v) for v in dp["rmse_xgboost"]],
        "(1) dışsal HAR→HAR-X": [pct(v) for v in dp["pct_exogenous_HAR_to_HARX"]],
        "(2) fonksiyonel form HAR-X→XGB-6": [pct(v) for v in
                                             dp["pct_functional_form_HARX_to_XGB6"]],
        "(3) özellik paketi XGB-6→XGB": [pct(v) for v in
                                         dp["pct_feature_package_XGB6_to_XGB"]],
        "toplam HAR→XGB": [pct(v) for v in dp["pct_total_HAR_to_XGB"]],
    })
    w(md_table(t))
    w("")
    rows = []
    for h in HORIZONS:
        a, b, n = wins(F, "xgb6", "har_x", h)
        c, d, _ = wins(F, "xgboost", "xgb6", h)
        rows.append({"ufuk": f"h={h}", "XGB-6 < HAR-X (fold)": f"{a}/{n}",
                     "işaret p": fp(sign_p(a, a + b)),
                     "XGB < XGB-6 (fold)": f"{c}/{n}", "işaret p ": fp(sign_p(c, c + d))})
    w("Fold bazında (RMSE, düzeltmesiz iki yönlü işaret testi):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")

    # ---------------- 3. Ablation ladder ----------------
    w("## 3. Ablasyon merdiveni (HAR, HAR+OVX, HAR+GPR, HAR-X)")
    w("")
    w("Kaynak: `ablation_exogenous_folds_publication_aligned.csv`. Keşifsel/post hoc; "
      "birincil hipotez ailesine dahil değil. HAR ve HAR+OVX GPR kullanmaz, iki sürümde "
      "bit düzeyinde aynıdır. RMSE/MAE ablasyon dosyasıyla aynı (assert); R²_oos Bölüm "
      "1c'deki ortak referansla.")
    w("")
    abl_models = ["har", "har_ovx", "har_gpr", "har_x"]
    ab = rd("ablation_exogenous_folds.csv", PUB).rename(columns={"variant": "model"})
    chk = ab.merge(F, on=["model", "horizon", "test_year"], suffixes=("", "_F"))
    assert len(chk) == len(ab) and np.allclose(chk["rmse"], chk["rmse_F"], rtol=1e-12)
    ab = F[F["include_in_main"] & F["model"].isin(abl_models)]
    aa = (ab.groupby(["model", "horizon"])[["rmse", "mae", "r2_oos_common"]].mean()
          .rename(columns={"r2_oos_common": "r2_oos"}))
    rows = []
    for h in HORIZONS:
        base = aa.loc[("har", h), "rmse"]
        for m in abl_models:
            r = aa.loc[(m, h)]
            rows.append({"ufuk": f"h={h}", "varyant": LABEL[m] if m != "har_x" else
                         "HAR-X (HAR + OVX + GPR)", "RMSE": f6(r["rmse"]),
                         "MAE": f6(r["mae"]), "R²_oos": f3(r["r2_oos"]),
                         "RMSE vs HAR": "—" if m == "har" else pct(100 * (r["rmse"] / base - 1))})
    w(md_table(pd.DataFrame(rows)))
    w("")
    abf = ab.pivot_table(index=["horizon", "test_year"], columns="model", values="rmse")
    rows = []
    for h in HORIZONS:
        x = abf.loc[h]
        row = {"ufuk": f"h={h}", "fold": len(x)}
        for lab, a, b in (("OVX katkısı: HAR+OVX < HAR", "har_ovx", "har"),
                          ("GPR katkısı: HAR+GPR < HAR", "har_gpr", "har"),
                          ("OVX üstüne GPR: HAR-X < HAR+OVX", "har_x", "har_ovx")):
            k, l = int((x[a] < x[b]).sum()), int((x[a] > x[b]).sum())
            p = sign_p(k, k + l)
            row[lab] = f"{k}/{len(x)} (p{'<0.001' if p < 0.001 else '=' + fp(p)})"
            row[lab.split(":")[0] + " RMSE %"] = pct(100 * (x[a].mean() / x[b].mean() - 1))
        rows.append(row)
    w("Fold bazında kazanma sayıları (düzeltmesiz iki yönlü işaret testi) ve fold "
      "ortalaması RMSE farkı:")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")

    # ---------------- 4. Standardized betas ----------------
    sb = rd("ablation_exogenous_std_beta_summary.csv", PUB)
    sb = sb[sb["fold_set"] == "main"].set_index(["variant", "horizon", "regressor"])
    w("## 4. Standartlaştırılmış betalar")
    w("")
    w("beta_std = beta × sd(X) / sd(y), sd'ler o fold'un train diliminden. Fold "
      "ortalaması (pozitif / negatif fold sayısı), ana fold kümesi. Kaynak: "
      "`ablation_exogenous_std_beta_summary_publication_aligned.csv`.")
    w("")

    def beta_cell(v, h, r):
        s = sb.loc[(v, h, r)]
        return f"{signed(s['mean'])} ({int(s['n_positive'])}/{int(s['n_negative'])})"

    harx_regs = ["har_daily", "brent_vol5", "brent_vol20", "ovx_lag1", "gprd_lag1",
                 "gprd_threat_lag1"]
    w("### 4a. HAR-X, altı regresör")
    w("")
    t = pd.DataFrame({"regresör": harx_regs})
    for h in HORIZONS:
        t[f"h={h}"] = [beta_cell("har_x", h, r) for r in harx_regs]
    w(md_table(t))
    w("")
    w("### 4b. Ablasyon varyantlarında dışsal katsayılar")
    w("")
    cols = [("har_ovx", "ovx_lag1"), ("har_ovx", "brent_vol20"), ("har", "brent_vol20"),
            ("har_gpr", "gprd_lag1"), ("har_gpr", "gprd_threat_lag1")]
    t = pd.DataFrame({"ufuk": [f"h={h}" for h in HORIZONS]})
    for v, r in cols:
        t[f"{LABEL[v]}: {r}"] = [beta_cell(v, h, r) for h in HORIZONS]
    w(md_table(t))
    w("")

    # ---------------- 5. SHAP ----------------
    sh = json.load(open(alignment.out("shap_summary.json", PUB), encoding="utf-8"))
    g = pd.DataFrame(sh["group_shares_xgb"]).pivot(index="grup", columns="horizon",
                                                   values="pay_pct")
    grp_order = ["brent_vol", "gpr", "ovx", "brent_fiyat", "etkilesim", "takvim"]
    w("## 5. SHAP grup payları (birincil XGBoost)")
    w("")
    w(f"Yöntem: {sh['method']}. Birim: {sh['shap_units']}. Pay = grup mean|SHAP| / "
      "toplam. **Uyarı:** grup payı özellik sayısıyla birlikte büyür; grup başına "
      "özellik sayısı ikinci sütunda. Kaynak: `shap_summary_publication_aligned.json`.")
    w("")
    t = pd.DataFrame({"grup": grp_order,
                      "özellik sayısı": [sh["groups"][k] for k in grp_order]})
    for h in HORIZONS:
        t[f"h={h}"] = [f"{g.loc[k, h]:.1f}%" for k in grp_order]
    w(md_table(t))
    w("")
    gh = pd.DataFrame(sh["group_shares_harx"]).pivot(index="grup", columns="horizon",
                                                     values="pay_pct")
    w("Karşılaştırma: HAR-X'te |standartlaştırılmış beta| grup payları. mean|SHAP| ile "
      "standartlaştırılmış beta aynı büyüklük değildir; yalnızca sıralama ve pay "
      "düzeyinde karşılaştırılabilir.")
    w("")
    t = pd.DataFrame({"grup": ["brent_vol", "ovx", "gpr"]})
    for h in HORIZONS:
        t[f"h={h}"] = [f"{gh.loc[k, h]:.1f}%" for k in t["grup"]]
    w(md_table(t))
    w("")
    w("XGBoost SHAP'ında HAR-X'in altı regresörü dışındaki özelliklerin payı: " + ", ".join(
        f"h={h}: {sh['outside_harx_share_pct'][str(h)]:.1f}%" for h in HORIZONS) + ".")
    w("")

    # ---------------- 6. Primary family ----------------
    dm = {al: pd.read_csv(alignment.out("dm_test_results.csv", al)) for al in (TS, PUB)}
    dms = json.load(open(alignment.out("dm_summary.json", PUB), encoding="utf-8"))
    # Benjamini-Yekutieli within each family. BH (as stored by 08) is recomputed with the
    # same step-up function first, so BY differs from it only by the factor c(m).
    for al in (TS, PUB):
        for fam, idx in dm[al].groupby("aile").groups.items():
            m = len(idx)
            c_m = sum(1 / i for i in range(1, m + 1))
            for p, col in (("p_HLN", "p_HLN"), ("p_isaret", "p_isaret")):
                bh = step_up(dm[al].loc[idx, p])
                assert np.allclose(bh, dm[al].loc[idx, f"{col}_bh"], rtol=1e-12, atol=0), \
                    (al, fam, p, "BH reimplementation differs from 08")
                dm[al].loc[idx, f"{col}_by"] = step_up(dm[al].loc[idx, p], c_m)
    C8 = sum(1 / i for i in range(1, 9))
    d = dm[PUB][dm[PUB]["aile"] == "birincil"].copy()
    assert len(d) == 8
    # Fold-level RMSE differences behind the two h=22 sign tests (HAR-X better > 0)
    hp = rd("hybrid_predictions_all.csv", PUB)
    hp = hp[(hp["horizon"] == 22) & hp["include_in_main"]]
    fr = hp.groupby("test_year").apply(lambda x: pd.Series(
        {m: np.sqrt(((x["y_true"] - x[f"pred_{m}"]) ** 2).mean())
         for m in ("har", "har_x", "xgboost")}), include_groups=False)
    d1, d2 = fr["har"] - fr["har_x"], fr["xgboost"] - fr["har_x"]
    assert not np.allclose(d1, d2) and (d1 != 0).all() and (d2 != 0).all()
    rho22 = float(np.corrcoef(d1, d2)[0, 1])
    lose1 = [int(y) for y in fr.index[d1 < 0]]
    lose2 = [int(y) for y in fr.index[d2 < 0]]
    d["claim"] = np.where(d["model1"] == "har", "HAR-X, HAR'ı geçer",
                          "HAR-X, XGBoost'u geçer")
    # HAR-X fold wins (isaret_kazanan counts model1's wins)
    d["harx_wins"] = np.where(d["model1"] == "har_x", d["isaret_kazanan"],
                              d["isaret_fold"] - d["isaret_kazanan"])
    d["harx_better_pooled"] = np.where(d["model1"] == "har_x", d["fark_pct"] < 0,
                                       d["fark_pct"] > 0)
    d = d.sort_values(["model1", "horizon"])
    out = pd.DataFrame({
        "karşılaştırma": [f"{a} vs {b}" for a, b in zip(d["model1"], d["model2"])],
        "horizon": d["horizon"], "n": d["n"], "hac_lag": d["lag"],
        "rmse_model1_pooled": d["rmse1"], "rmse_model2_pooled": d["rmse2"],
        "diff_pct_pooled": d["fark_pct"], "DM": d["DM_ham"], "p_raw": d["p_ham"],
        "DM_HLN": d["DM_HLN"], "p_HLN": d["p_HLN"], "p_HLN_holm": d["p_HLN_holm"],
        "p_HLN_bh": d["p_HLN_bh"], "p_HLN_by": d["p_HLN_by"],
        "harx_fold_wins": d["harx_wins"],
        "n_folds": d["isaret_fold"], "p_sign": d["p_isaret"],
        "p_sign_holm": d["p_isaret_holm"], "p_sign_bh": d["p_isaret_bh"],
        "p_sign_by": d["p_isaret_by"]})
    out.to_csv(OUT_DIR / "primary_family_tests_publication_aligned.csv", index=False)

    w("## 6. Birincil hipotez ailesi (8 test) — merkez çıkarım sonucu")
    w("")
    w("İki iddia × dört ufuk: **HAR vs HAR-X** (dışsal değişkenler katkı sağlar mı) ve "
      "**HAR-X vs XGBoost** (doğrusal olmayan model katkı sağlar mı). Holm (FWER), "
      "Benjamini-Hochberg (FDR, pozitif bağımlılık/PRDS altında geçerli) ve "
      "Benjamini-Yekutieli (FDR, her bağımlılık yapısında geçerli; BH × c(m), "
      f"c(8) = {C8:.3f}) aile içinde, 8 test üzerinden. "
      f"Kayıp: {dms['loss']}. HAC: {dms['hac']}. HLN: `{dms['hln']}`. "
      "DM işareti: negatif = ilk model daha iyi. **Holm, BH ve BY düzeltmeleri HLN p "
      "değerine uygulanır** (ham DM p'sine değil). BY bu dosyada hesaplanır; BH, "
      "`08_dm_test.py`'nin kayıtlı değeriyle aynı fonksiyonla yeniden üretilip "
      "doğrulanır (assert). İşaret testi fold düzeyinde "
      "binom (H0: p=0.5, iki yönlü); HAC/normallik varsayımı kullanmaz. DM havuzlanmış "
      "seri üzerindedir (her yıl yeniden eğitilmiş modellerin tahminleri), fold "
      "ortalaması değil; bu yüzden havuzlanmış RMSE farkı Bölüm 1'deki fold ortalaması "
      "farkından farklıdır ve h=66/126'da HAR vs HAR-X'te işaret değiştirir.")
    w("")
    w("Şeffaflık: aile tanımı testlerden sonra resmileştirilmiştir (ön-kayıt değildir); "
      "karşılaştırmalar p değerine göre değil, Aşama 5–6'da ilan edilmiş iddialara göre "
      "seçilmiştir.")
    w("")
    t = pd.DataFrame({
        "ufuk": [f"h={h}" for h in d["horizon"]],
        "karşılaştırma": [("HAR vs HAR-X" if a == "har" else "HAR-X vs XGBoost")
                          for a in d["model1"]],
        "havuz RMSE farkı": [pct(v) for v in d["fark_pct"]],
        "DM": [signed(v, 3) for v in d["DM_ham"]],
        "DM (HLN)": [signed(v, 3) for v in d["DM_HLN"]],
        "ham p": [fp(v) for v in d["p_ham"]],
        "HLN p": [fp(v) for v in d["p_HLN"]],
        "Holm p": [fp(v) for v in d["p_HLN_holm"]],
        "BH p": [(f"**{fp(v)}**" if v < 0.05 else fp(v)) for v in d["p_HLN_bh"]],
        "BY p": [(f"**{fp(v)}**" if v < 0.05 else fp(v)) for v in d["p_HLN_by"]],
        "işaret: HAR-X kazanır": [f"{k}/{n}" for k, n in zip(d["harx_wins"],
                                                             d["isaret_fold"])],
        "işaret ham p": [fp(v) for v in d["p_isaret"]],
        "işaret Holm p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                          for v in d["p_isaret_holm"]],
        "işaret BH p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                        for v in d["p_isaret_bh"]],
        "işaret BY p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                        for v in d["p_isaret_by"]],
    })
    w(md_table(t))
    w("")
    w("Makine okunur kopya: `primary_family_tests_publication_aligned.csv`.")
    w("")

    def survivors(x):
        return {"DM Holm": int((x["p_HLN_holm"] < .05).sum()),
                "DM BH": int((x["p_HLN_bh"] < .05).sum()),
                "DM BY": int((x["p_HLN_by"] < .05).sum()),
                "işaret Holm": int((x["p_isaret_holm"] < .05).sum()),
                "işaret BH": int((x["p_isaret_bh"] < .05).sum()),
                "işaret BY": int((x["p_isaret_by"] < .05).sum())}

    rows = []
    for al, name in ((PUB, "yayım-hizalı (birincil)"), (TS, "zaman damgalı (Ek A)")):
        for fam in ("birincil", "ikincil"):
            x = dm[al][dm[al]["aile"] == fam]
            rows.append({"sürüm": name, "aile": f"{fam} ({len(x)} test)",
                         **{k: f"{v}/{len(x)}" for k, v in survivors(x).items()}})
    w("Ayakta kalan test sayısı (%5 eşiği):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    surv = d[d["p_isaret_bh"] < .05]
    assert list(surv["horizon"]) == [22, 22], "survivor text below assumes the two h=22 tests"
    assert (d["p_isaret_by"] >= .05).all() and (d["p_HLN_by"] >= .05).all()
    w("**Birincil aile sonucu (yayım-hizalı):**")
    w("")
    w("- **Holm (FWER):** hiçbir test ayakta kalmıyor.")
    w("- **BH (FDR, PRDS varsayımıyla):** h=22'de iki hipotez reddediliyor: " + "; ".join(
        f"{r.claim} (işaret {r.harx_wins}/{r.isaret_fold}, BH p = {r.p_isaret_bh:.3f})"
        for r in surv.itertuples()) + ". **Bunlar iki ayrı hipotez, ama birbirinden "
      "bağımsız iki kanıt değil:** iki fold farkı vektörü (HAR − HAR-X ve XGBoost − "
      f"HAR-X) {rho22:.2f} korelasyonlu. İkisi de HAR-X'i içeriyor ve 2020 ortak kayıp "
      f"yılı (HAR-X'in kaybettiği yıllar: HAR'a karşı {', '.join(map(str, lose1))}; "
      f"XGBoost'a karşı {', '.join(map(str, lose2))}). Aynı 13/15 ve aynı ham p, binom "
      "testinin yalnızca kazanma sayısına bağlı olmasından geliyor; vektörler farklı "
      "(assert).")
    w("- **BY (FDR, bağımlılık yapısından bağımsız geçerli):** hiçbir test ayakta "
      f"kalmıyor. En küçük BY p = {d['p_isaret_by'].min():.3f} (h=22 işaret testleri; "
      f"BH p {surv['p_isaret_bh'].iloc[0]:.4f} × c(8) = {C8:.3f}).")
    w(f"- **DM:** hiçbir düzeltmede anlamlılık yok (HLN p aralığı "
      f"{d['p_HLN'].min():.3f}–{d['p_HLN'].max():.3f}).")
    w("")
    ok = (d["dm_yon"] == d["isaret_yon"])
    w(f"Yön uyumu: DM (havuz) ve işaret testi {int(ok.sum())}/8 testte aynı modeli "
      "işaret ediyor; uyuşmayanlar: " + (", ".join(
          f"h={r.horizon} {r.model1} vs {r.model2}" for r in d[~ok].itertuples()) or "yok")
      + ".")
    w("")

    # ---------------- 7. Robustness ----------------
    w("## 7. Sağlamlık kontrolleri (yayım modunda yeniden koşuldu)")
    w("")
    w("### 7a. Veri eşitleme: benchmark'lar XGBoost'un penceresine (satır 127) indirildi")
    w("")
    w("XGBoost'a göre RMSE farkı (%, negatif = benchmark daha iyi). Kaynak: "
      "`bench_model_comparison_all{,_aligned}_publication_aligned.csv`.")
    w("")
    rows = []
    eq = {}
    for al in (PUB, TS):
        for win, name in (("normal", "bench_model_comparison_all.csv"),
                          ("eşitlenmiş", "bench_model_comparison_all_aligned.csv")):
            p = rd(name, al).pivot_table(index="model", columns="horizon",
                                         values="rmse_fold_mean")
            eq[(al, win)] = 100 * (p.loc[["har", "har_x", "har_x_log"]] / p.loc["xgboost"] - 1)
    for m in ("har", "har_x", "har_x_log"):
        rows.append({"model": LABEL[m],
                     "normal pencere": " / ".join(pct(eq[(PUB, 'normal')].loc[m, h])
                                                  for h in HORIZONS),
                     "eşitlenmiş pencere": " / ".join(pct(eq[(PUB, 'eşitlenmiş')].loc[m, h])
                                                      for h in HORIZONS)})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("(h=5 / h=22 / h=66 / h=126.)")
    w("")
    w("### 7b. BiLSTM yakınsama kontrolü (yalnızca yüksek kademe; son %10 epoch "
      "diliminde kayıp düşüşü < %2, üst sınır 200 epoch)")
    w("")
    conv_ok = alignment.out("bilstm_aggregate_all_conv.csv", PUB).exists()
    if conv_ok:
        rows = []
        for h in HORIZONS:
            row = {"ufuk": f"h={h}"}
            for al, tag in ((PUB, "yayım"), (TS, "zaman d.")):
                pri = rd("bilstm_folds_all.csv", al)
                cf = rd("bilstm_folds_all_conv.csv", al)
                prim = rd("bilstm_metrics_all.csv", al)
                conv = rd("bilstm_metrics_all_conv.csv", al)
                sel = lambda x: x[(x["model"] == "bilstm") & (x["horizon"] == h) &
                                  x["include_in_main"]].set_index("test_year")["rmse"]
                a1, a2 = sel(prim), sel(conv)
                row[f"{tag}: birincil"] = f6(a1.mean())
                row[f"{tag}: yakınsama"] = f6(a2.mean())
                row[f"{tag}: değişim"] = pct(100 * (a2.mean() / a1.mean() - 1))
                row[f"{tag}: yakınsama daha iyi"] = f"{int((a2 < a1).sum())}/{len(a1)}"
                if al == PUB:
                    c = cf[(cf["horizon"] == h) & (cf["arch_tier"] == "yuksek")]
                    row["yayım: ort. epoch (yüksek kademe)"] = (
                        f"{c['epochs_run'].mean():.0f}" if len(c) else "—")
                    pc = pri[(pri["horizon"] == h) & (pri["arch_tier"] == "yuksek")]
                    lr = (pc.set_index("test_year")["loss_final"] /
                          c.set_index("test_year")["loss_final"])
                    row["yayım: kayıp oranı birincil/yakınsama, medyan"] = (
                        f"{lr.median():.2f}" if len(c) else "—")
                    row["yayım: yetersiz eğitilmiş (birincil→yakınsama)"] = (
                        f"{int((pri[pri['horizon'] == h]['convergence'] == 'yetersiz_egitilmis').sum())}"
                        f"→{int((cf[cf['horizon'] == h]['convergence'] == 'yetersiz_egitilmis').sum())}")
            rows.append(row)
        w(md_table(pd.DataFrame(rows)))
        w("")
        w("Değişim = yakınsama / birincil − 1 (fold ortalaması RMSE, ana fold'lar). "
          "h=66 ve h=126'da yüksek kademe fold yok, sonuçlar tanım gereği aynı.")
        w("")
        w("**Yorum sınırı (bkz. deney günlüğü 16.4):** yayım sürümünde durdurma kriteri "
          "erken tetiklendi, eğitim kaybı yalnızca ~1.4 kat düştü (zaman damgalı "
          "sürümde geç fold'larda 4–6 kat). Bu kontrol tek başına \"yetersiz eğitim "
          "değil aşırı uyum\" iddiasını desteklemez; iddianın dayanağı 7c'deki sabit "
          "200 epoch kontrolüdür.")
    else:
        w("_`bilstm_*_conv_publication_aligned` henüz üretilmedi._")
    w("")
    fx_path = alignment.out("bilstm_fixed200_folds.csv", PUB)
    w("### 7c. BiLSTM sabit 200 epoch (keşifsel, post hoc; erken durdurma yok)")
    w("")
    if fx_path.exists():
        fx = rd("bilstm_fixed200_folds.csv", PUB)
        fx = fx[fx["include_in_main"]]
        w("Yüksek kademe fold'lar, h=5 ve h=22. Kosinüs programı yakınsama koşusuyla aynı "
          "(`T_max=200`); k'ıncı epoch yakınsama koşusunun kendisidir (kayıp ve test "
          "tahminleri bit düzeyinde aynı, assert). k = yakınsama kuralının durduğu "
          "epoch. Kaynak: `bilstm_fixed200_folds_publication_aligned.csv`; epoch bazında "
          "kayıp `bilstm_fixed200_loss_history_publication_aligned.csv`. Günlük 16.5.")
        w("")
        rows = []
        cmv = rd("bilstm_metrics_all_conv.csv", PUB)
        for h, gx in fx.groupby("horizon"):
            k = int((gx["rmse_200"] < gx["rmse_k"]).sum())
            full = cmv[(cmv["model"] == "bilstm") & (cmv["horizon"] == h)
                       & cmv["include_in_main"]].set_index("test_year")["rmse"]
            base = full.mean()
            full.loc[gx["test_year"]] = gx.set_index("test_year")["rmse_200"]
            rng = lambda c: f"{gx[c].median():.2f}× ({gx[c].min():.2f}–{gx[c].max():.2f})"
            rows.append({
                "ufuk": f"h={h}", "fold": len(gx), "ort. k": f"{gx['k_conv'].mean():.0f}",
                "eğitim kaybı k→200, medyan (aralık)": rng("loss_ratio_k_to_200"),
                "eval-modu eğitim MSE k→200": rng("train_mse_eval_ratio_k_to_200"),
                "kayıp birincil(60)→200, medyan":
                    f"{gx['loss_ratio_primary_to_200'].median():.2f}×",
                "test RMSE birincil / k / 200": f"{gx['rmse_primary'].mean():.6f} / "
                                                f"{gx['rmse_k'].mean():.6f} / "
                                                f"{gx['rmse_200'].mean():.6f}",
                "RMSE 200 vs k": pct(100 * (gx["rmse_200"].mean() / gx["rmse_k"].mean() - 1)),
                "200 daha iyi (işaret p; keşifsel, düzeltmesiz)":
                    f"{k}/{len(gx)} (p={fp(sign_p(k, len(gx)))})",
                "MAE 200 vs k": pct(100 * (gx["mae_200"].mean() / gx["mae_k"].mean() - 1)),
                "sd oranı k→200": f"{gx['pred_std_ratio_k'].mean():.2f} → "
                                  f"{gx['pred_std_ratio_200'].mean():.2f}",
                "ufuk ortalaması RMSE (tüm fold'lar)":
                    f"{base:.6f} → {full.mean():.6f} ({pct(100 * (full.mean() / base - 1))})"})
        w(md_table(pd.DataFrame(rows)))
        w("")
        w("**p değerlerinin statüsü:** buradaki işaret testi p'leri (h=5: 0.007) keşifsel "
          "bir teşhisten gelir, **birincil sekizlik aileye dahil değildir ve "
          "düzeltilmemiştir**; statüsü HAR+OVX vs HAR-X'in düzeltmesiz p = 0.035'iyle "
          "aynıdır. Birincil aile sabittir (testlerden sonra resmileştirildi, ön-kayıt "
          "değil; bkz. Bölüm 6); sonradan test eklenmez.")
        w("")
        w("**Yorum:** eğitim kaybı durdurma kuralı olmadan ciddi düşüyor ve test hatası "
          "iyileşmiyor, kötüleşiyor. \"Yetersiz eğitim değil aşırı uyum\" bulgusu yayım "
          "sürümünde bu kontrolle destekleniyor. Zaman damgalı sürümdeki \"~4×\" rakamı "
          "Ek A'ya aittir; yayım sürümünün rakamı yukarıdaki medyanlardır.")
    else:
        w("_`bilstm_fixed200_*_publication_aligned` henüz üretilmedi._")
    w("")

    # ---------------- 8. Two-version comparison ----------------
    w("## 8. İki sürüm karşılaştırması (Ek A)")
    w("")
    w("Zaman damgalı sürüm: GPR her gün bir gün gecikmeyle kullanılıyordu (satır t, "
      "t−1 tarihli gözlemi görüyordu); GPR ise haftalık yayımlandığı için bu gözlemler "
      "tahmin anında çoğu zaman yayımlanmamıştı (zaman damgalı satırların "
      f"%{100 * json.load(open(OUT_DIR / 'build_features_publication_aligned_report.json', encoding='utf-8'))['effective_lag']['share_rows_timestamp_uses_unpublished_obs']:.1f}'i "
      "yayımlanmamış gözlem kullanıyor). İki sürüm tamamen aynı örneklemde "
      "değerlendirilir (aynı train/test satırları); GPR kullanmayan modeller bit "
      "düzeyinde aynıdır. Fark saf hizalama etkisidir. Kaynak: "
      "`gpr_alignment_comparison*.csv`.")
    w("")
    w("### 8a. RMSE, GPR kullanan modeller (fold ortalaması)")
    w("")
    w("% = 100 × (RMSE_yayım / RMSE_zaman damgalı − 1); pozitif = yayım gecikmesine "
      "uymanın doğruluk maliyeti. Fold sayımı: yayım sürümünün daha iyi olduğu fold "
      "sayısı / toplam; p düzeltmesiz iki yönlü işaret testi.")
    w("")
    cg = comp[comp["uses_gpr"]]
    gmodels = [m for m in order if m in set(cg["model"])]
    t = pd.DataFrame({"model": [LABEL[m] for m in gmodels]})
    for h in HORIZONS:
        s = cg[cg["horizon"] == h].set_index("model").loc[gmodels]
        t[f"h={h}"] = [f"{f6(a)} → {f6(b)} ({pct(c)}; {int(k)}/{int(n)}, p={fp(p)})"
                       for a, b, c, k, n, p in zip(s["rmse_ts"], s["rmse_pub"],
                                                   s["rmse_pct"],
                                                   s["rmse_folds_pub_better"],
                                                   s["n_folds"],
                                                   s["rmse_sign_p_two_sided"])]
    w(md_table(t))
    w("")
    w("GPR kullanmayan modeller (HAR, HAR-log, HAR+OVX, GARCH, train-mean, "
      "past-volatility) iki sürümde bit düzeyinde aynı; tablo dışı.")
    w("")
    w("### 8b. Ayrıştırma, iki sürüm")
    w("")
    rows = []
    for h in HORIZONS:
        for al, name in ((TS, "zaman damgalı"), (PUB, "yayım-hizalı")):
            r = dec[(dec["horizon"] == h) & (dec["gpr_alignment"] == al)].iloc[0]
            rows.append({"ufuk": f"h={h}", "sürüm": name,
                         "(1) dışsal": pct(r["pct_exogenous_HAR_to_HARX"]),
                         "(2) fonksiyonel form": pct(r["pct_functional_form_HARX_to_XGB6"]),
                         "(3) özellik paketi": pct(r["pct_feature_package_XGB6_to_XGB"]),
                         "toplam": pct(r["pct_total_HAR_to_XGB"])})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8c. Birincil aile, iki sürüm")
    w("")
    rows = []
    dt = dm[TS][dm[TS]["aile"] == "birincil"].set_index(["model1", "model2", "horizon"])
    for r in d.itertuples():
        q = dt.loc[(r.model1, r.model2, r.horizon)]
        tw = q["isaret_kazanan"] if r.model1 == "har_x" else q["isaret_fold"] - q["isaret_kazanan"]
        rows.append({"ufuk": f"h={r.horizon}",
                     "karşılaştırma": "HAR vs HAR-X" if r.model1 == "har" else "HAR-X vs XGBoost",
                     "HLN p (z.d. → yayım)": f"{fp(q['p_HLN'])} → {fp(r.p_HLN)}",
                     "DM BH p": f"{fp(q['p_HLN_bh'])} → {fp(r.p_HLN_bh)}",
                     "HAR-X kazanır": f"{int(tw)}/{int(q['isaret_fold'])} → "
                                      f"{r.harx_wins}/{r.isaret_fold}",
                     "işaret ham p": f"{fp(q['p_isaret'])} → {fp(r.p_isaret)}",
                     "işaret Holm p": f"{fp(q['p_isaret_holm'])} → {fp(r.p_isaret_holm)}",
                     "işaret BH p": f"{fp(q['p_isaret_bh'])} → {fp(r.p_isaret_bh)}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8d. GPR'ın ağırlığı, iki sürüm")
    w("")
    sht = json.load(open(alignment.out("shap_summary.json", TS), encoding="utf-8"))
    gt = pd.DataFrame(sht["group_shares_xgb"]).pivot(index="grup", columns="horizon",
                                                     values="pay_pct")
    sbt = rd("ablation_exogenous_std_beta_summary.csv", TS)
    sbt = sbt[sbt["fold_set"] == "main"].set_index(["variant", "horizon", "regressor"])
    rows = []
    for h in HORIZONS:
        rows.append({
            "ufuk": f"h={h}",
            "SHAP GPR payı": f"{gt.loc['gpr', h]:.1f}% → {g.loc['gpr', h]:.1f}%",
            "SHAP OVX payı": f"{gt.loc['ovx', h]:.1f}% → {g.loc['ovx', h]:.1f}%",
            "HAR-X β gprd_lag1": f"{signed(sbt.loc[('har_x', h, 'gprd_lag1'), 'mean'])} → "
                                 f"{signed(sb.loc[('har_x', h, 'gprd_lag1'), 'mean'])}",
            "HAR-X β gprd_threat_lag1":
                f"{signed(sbt.loc[('har_x', h, 'gprd_threat_lag1'), 'mean'])} → "
                f"{signed(sb.loc[('har_x', h, 'gprd_threat_lag1'), 'mean'])}",
            "HAR-X β ovx_lag1": f"{signed(sbt.loc[('har_x', h, 'ovx_lag1'), 'mean'])} → "
                                f"{signed(sb.loc[('har_x', h, 'ovx_lag1'), 'mean'])}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8e. Sağlamlık kontrolleri, zaman damgalı sürüm (karşılaştırma için)")
    w("")
    rows = []
    for m in ("har", "har_x", "har_x_log"):
        rows.append({"model": LABEL[m],
                     "normal pencere": " / ".join(pct(eq[(TS, 'normal')].loc[m, h])
                                                  for h in HORIZONS),
                     "eşitlenmiş pencere": " / ".join(pct(eq[(TS, 'eşitlenmiş')].loc[m, h])
                                                      for h in HORIZONS)})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8f. 2026 kısmi yıl dipnotu, iki sürüm (h=66, h=126; bilgi amaçlı)")
    w("")
    f26c = pd.read_csv(OUT_DIR / "gpr_alignment_comparison_2026_footnote.csv")
    f26c = f26c[f26c["model"].isin(gmodels)]
    t = pd.DataFrame({"model": [LABEL[m] for m in gmodels]})
    for h in (66, 126):
        s = f26c[f26c["horizon"] == h].set_index("model").loc[gmodels]
        t[f"h={h} RMSE z.d. → yayım"] = [f"{f6(a)} → {f6(b)} ({pct(c)})" for a, b, c in
                                         zip(s["rmse_ts"], s["rmse_pub"], s["rmse_pct"])]
    w(md_table(t))
    w("")

    # ---------------- 9. Additional numbers used in the root README ----------------
    w("## 9. README'de kullanılan ek sayılar (yayım-hizalı)")
    w("")
    w("Kök `README.md`'deki her sayı ya Bölüm 1–8'den ya da bu bölümden gelir.")
    w("")
    R = A.set_index(["model", "horizon"])
    rr = lambda a, b, h: 100 * (R.loc[(a, h), "rmse"] / R.loc[(b, h), "rmse"] - 1)
    w("### 9a. Başlıca RMSE karşılaştırmaları (fold ortalaması, %)")
    w("")
    rows = []
    for lab, a, b in (("XGBoost vs past-volatility", "xgboost", "past_vol"),
                      ("XGBoost vs HAR", "xgboost", "har"),
                      ("XGBoost (Optuna) vs HAR", "xgboost_optuna", "har"),
                      ("BiLSTM vs HAR", "bilstm", "har"),
                      ("XGBoost vs HAR-X", "xgboost", "har_x"),
                      ("BiLSTM vs HAR-X", "bilstm", "har_x"),
                      ("H1 (XGB+BiLSTM) vs HAR-X", "h1_xgb_bilstm", "har_x"),
                      ("H2 (HAR-X+XGB) vs HAR-X", "h2_harx_xgb", "har_x"),
                      ("H3 (HAR-X+artık) vs HAR-X", "h3_harx_resid", "har_x"),
                      ("HAR-X-log vs HAR-X", "har_x_log", "har_x"),
                      ("HAR+OVX vs HAR-X", "har_ovx", "har_x")):
        rows.append({"karşılaştırma": lab, **{f"h={h}": pct(rr(a, b, h)) for h in HORIZONS}})
    har_family = ["har", "har_log", "har_x", "har_x_log", "har_ovx", "har_gpr"]
    best = {h: A[(A["horizon"] == h) & A["model"].isin(har_family)]
            .sort_values("rmse").iloc[0]["model"] for h in HORIZONS}
    for lab, ms in (("en iyi hibrit vs en iyi HAR-ailesi",
                     ["h1_xgb_bilstm", "h2_harx_xgb", "h3_harx_resid"]),
                    ("en iyi birincil doğrusal olmayan (XGB, XGB-Optuna, BiLSTM) vs HAR",
                     ["xgboost", "xgboost_optuna", "bilstm"]),
                    ("XGBoost-6 (keşifsel; HAR-X'in girdileri, OVX dahil) vs HAR",
                     ["xgb6"])):
        row = {"karşılaştırma": lab}
        for h in HORIZONS:
            s = A[(A["horizon"] == h) & A["model"].isin(ms)].sort_values("rmse").iloc[0]
            ref = best[h] if "HAR-ailesi" in lab else "har"
            row[f"h={h}"] = f"{pct(rr(s['model'], ref, h))} ({s['model']} vs {ref})"
        rows.append(row)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("En iyi HAR-ailesi modeli (RMSE): " + ", ".join(f"h={h}: {LABEL[m]}"
                                                     for h, m in best.items()) + ".")
    w("")
    rows = []
    for h in HORIZONS:
        x = abf.loc[h]
        k = int((x["har_ovx"] < x["har_x"]).sum())
        rows.append({"ufuk": f"h={h}", "HAR+OVX, HAR-X'i geçer": f"{k}/{len(x)}",
                     "işaret p": fp(sign_p(k, len(x)))})
    w("HAR+OVX vs HAR-X, fold bazında (düzeltmesiz iki yönlü işaret testi):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    gpr_b = sb.loc[[i for i in sb.index if i[0] in ("har_x", "har_gpr")
                    and i[2] in ("gprd_lag1", "gprd_threat_lag1")], "mean"]
    ovx_b = sb.loc[[i for i in sb.index if i[0] in ("har_x", "har_ovx")
                    and i[2] == "ovx_lag1"], "mean"]
    w(f"Standartlaştırılmış beta aralığı (fold ortalamaları, HAR-X ve ablasyon, dört ufuk): "
      f"GPR {signed(gpr_b.min())} ile {signed(gpr_b.max())} arası; OVX "
      f"{signed(ovx_b.min())} ile {signed(ovx_b.max())} arası.")
    w("")
    w("### 9b. DM ikincil aile (24 test), yayım-hizalı: ayakta kalanlar")
    w("")
    sec = dm[PUB][dm[PUB]["aile"] == "ikincil"]
    rows = []
    for r in sec.itertuples():
        if min(r.p_HLN_bh, r.p_isaret_bh) < .05:
            rows.append({"ufuk": f"h={r.horizon}", "karşılaştırma": f"{r.model1} vs {r.model2}",
                         "havuz RMSE farkı": pct(r.fark_pct),
                         "DM Holm / BH / BY": f"{fp(r.p_HLN_holm)} / {fp(r.p_HLN_bh)} / "
                                              f"{fp(r.p_HLN_by)}",
                         "işaret": f"{r.isaret_kazanan}/{r.isaret_fold}",
                         "işaret Holm / BH / BY": f"{fp(r.p_isaret_holm)} / "
                                                  f"{fp(r.p_isaret_bh)} / {fp(r.p_isaret_by)}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("İşaret: model1'in kazandığı fold / toplam. Uzun ufuklarda (h=66, h=126) naif "
      "baseline'a karşı DM anlamlılığı: " + (", ".join(
          f"h={r.horizon} {r.model1} vs {r.model2} (BH {fp(r.p_HLN_bh)})"
          for r in sec[(sec["horizon"] >= 66) & (sec["model2"] == "past_vol")
                       & (sec["p_HLN_bh"] < .05)].itertuples()) or "yok") + ".")
    w("")
    w("### 9c. SHAP ek ölçüler")
    w("")
    sa = rd("shap_sign_agreement.csv", PUB)
    unused = sa.groupby("horizon")["kullanilmadi"].mean() * 100
    used = sa[~sa["kullanilmadi"]]
    ovx_ag = used[used["regresor"] == "ovx_lag1"]
    stab = {r["horizon"]: r for r in sh["stability"]}
    t = pd.DataFrame({"ölçü": [
        "HAR-X: OVX'in |std beta| payı",
        "XGBoost: HAR-X'in altı regresörü dışındaki SHAP payı",
        "kullanılmayan karşılaştırma oranı (5 ortak regresör × fold; SHAP özdeş sıfır)",
        "OVX işaret uyumu (XGBoost SHAP yönü vs HAR-X beta; kullanılan karşılaştırmalar)",
        "atıf sıralaması kararlılığı: ilk-son fold Spearman ρ"]})
    for h in HORIZONS:
        oh = ovx_ag[ovx_ag["horizon"] == h]
        t[f"h={h}"] = [f"{gh.loc['ovx', h]:.1f}%",
                       f"{sh['outside_harx_share_pct'][str(h)]:.1f}%",
                       f"{unused[h]:.1f}%",
                       f"{100 * oh['uyum'].mean():.0f}% ({int(oh['uyum'].sum())}/{len(oh)})",
                       f"{stab[h]['ilk_son_rho']:.2f}"]
    w(md_table(t))
    w("")
    w(f"OVX işaret uyumu, dört ufuk birlikte: {100 * ovx_ag['uyum'].mean():.0f}% "
      f"({int(ovx_ag['uyum'].sum())}/{len(ovx_ag)}). Tüm fold'lar (2026 dahil), "
      "`shap_sign_agreement_publication_aligned.csv`.")
    w("")
    w("### 9d. Tarih boşluğu: doğrudan hedef düzeltme testi (tahminler sabit)")
    w("")
    gt_ = rd("gap_target_test.csv", PUB).set_index(["horizon", "model"])
    gs = json.load(open(alignment.out("gap_target_test_summary.json", PUB),
                        encoding="utf-8"))["horizons"]
    rows = []
    for h in HORIZONS:
        x = gt_.loc[h]
        s = gs[str(h)]
        rows.append({
            "ufuk": f"h={h}", "model": len(x),
            "en büyük |RMSE değişimi|": f"{x['rmse_change_pct'].abs().max():.2f}%",
            "RMSE sıra değişimi": s["n_rmse_rank_changes"],
            "MAE sıra değişimi": f"{s['n_mae_rank_changes']}"
                                 + (f" ({' ↔ '.join(s['mae_rank_swaps'])})"
                                    if s["mae_rank_swaps"] else ""),
            "HAR+OVX < HAR-X (düzeltilmiş)":
                "evet" if x.loc["har_ovx", "rmse_corrected"] < x.loc["har_x", "rmse_corrected"]
                else "hayır",
            "HAR < HAR+GPR (düzeltilmiş)":
                "evet" if x.loc["har", "rmse_corrected"] < x.loc["har_gpr", "rmse_corrected"]
                else "hayır"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 9e. Boşluksuz alt örneklem (2017–2026 fold'ları)")
    w("")
    gf = rd("robustness_gapfree_2017plus.csv", PUB).set_index(["horizon", "model"])
    gfs = json.load(open(alignment.out("robustness_gapfree_2017plus_summary.json", PUB),
                         encoding="utf-8"))["horizons"]
    rows = []
    for h in HORIZONS:
        x = gf.loc[h]
        r17 = x["rmse_fold_mean_2017plus"]
        fam = r17[[m for m in har_family if m in r17.index]].min()
        top5 = r17.sort_values().iloc[:5]
        rows.append({
            "ufuk": f"h={h}", "fold (2017+)": int(x["n_folds_2017plus"].iloc[0]),
            "Spearman RMSE sırası, tüm vs 2017+":
                f"{gfs[str(h)]['spearman_rmse_full_vs_2017plus']:.2f}",
            "Spearman, tüm vs 2012–2016":
                f"{gfs[str(h)]['spearman_rmse_full_vs_2012_2016']:.2f}",
            "en iyi HAR-ailesi < XGBoost ve BiLSTM":
                "evet" if fam < min(r17["xgboost"], r17["bilstm"]) else "hayır",
            "train-mean'den düşük RMSE'li model": f"{int((r17 < r17['train_mean']).sum())}"
                                                  f"/{len(r17) - 1}",
            "ilk beş RMSE aralığı": f"{100 * (top5.max() / top5.min() - 1):.1f}%"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("\"train-mean'den düşük RMSE'li model\" RMSE üzerinden sayılır (R²_oos referans "
      "farklarından etkilenmez).")
    w("")

    # ---------------- 10. Power analysis ----------------
    w("## 10. Güç analizi (birincil aile; `09_power_analysis.py`)")
    w("")
    w("%80 güç, %5 iki yönlü. DM: örneklem birimi etkin blok B = n/h. İşaret testi: "
      "örneklem birimi fold (yıl), tam binom. Kaynak: `power_analysis_publication_"
      "aligned.csv`, `apriori_power_{sign,dm}_publication_aligned.csv`.")
    w("")
    w("**Uyarı:** gözlenen etkiden hesaplanan \"gerçekleşen güç\" p değerinin monoton bir "
      "dönüşümüdür ve p değerinin ötesinde bilgi taşımaz; bir sonucun tesadüf olup "
      "olmadığına kanıt olarak kullanılamaz. Bilgi taşıyan kısımlar gerekli örneklem "
      "(10a) ve gözlenen sonuçlardan bağımsız önsel eğrilerdir (10b).")
    w("")
    pw = {al: rd("power_analysis.csv", al) for al in (PUB, TS)}
    pw_s = json.load(open(alignment.out("power_analysis_summary.json", PUB), encoding="utf-8"))
    w(f"Yıl başına işlem günü (test döneminden ölçüldü): {pw_s['days_per_year']:.1f}.")
    w("")
    w("### 10a. Gözlenen etki gerçek kabul edilirse %80 güç için gereken test dönemi")
    w("")
    p_ = pw[PUB].copy()
    # HAR-X's fold wins, as in Section 6 (isaret_kazanan counts model1's wins)
    p_["harx_w"] = np.where(p_["karsilastirma"] == "har vs har_x",
                            p_["isaret_fold"] - p_["isaret_kazanan"], p_["isaret_kazanan"])
    p_ = p_.sort_values(["karsilastirma", "horizon"])
    t = pd.DataFrame({
        "ufuk": [f"h={h}" for h in p_["horizon"]],
        "karşılaştırma": ["HAR vs HAR-X" if c == "har vs har_x" else "HAR-X vs XGBoost"
                          for c in p_["karsilastirma"]],
        "DM: etkin blok": [f"{v:.0f}" for v in p_["dm_etkin_blok"]],
        "DM: gerçekleşen güç": [f"{v:.3f}" for v in p_["dm_gerceklesen_guc"]],
        "DM: gerekli yıl": [f"{v:,.0f}" for v in p_["dm_gerekli_yil"]],
        "DM: kat": [f"{v:.1f}×" for v in p_["dm_kat_artis"]],
        "işaret: HAR-X kazanır": [f"{k}/{n}" for k, n in zip(p_["harx_w"], p_["isaret_fold"])],
        "işaret: gerçekleşen güç": [f"{v:.3f}" for v in p_["isaret_gerceklesen_guc"]],
        "işaret: gerekli yıl": [f"{v:.0f}" for v in p_["isaret_gerekli_fold_yil"]],
        "işaret: kat": [f"{v:.1f}×" for v in p_["isaret_kat_artis"]],
    })
    w(md_table(t))
    w("")
    w(f"Aralıklar: DM için gereken uzatma mevcut test döneminin "
      f"{p_['dm_kat_artis'].min():.1f}–{p_['dm_kat_artis'].max():.0f} katı; işaret testi "
      f"için {p_['isaret_kat_artis'].min():.1f}–{p_['isaret_kat_artis'].max():.1f} katı.")
    w("")
    w("### 10b. Önsel güç eğrileri (gözlenen sonuçları kullanmaz)")
    w("")
    ps = rd("apriori_power_sign.csv", PUB)
    thr = ps.groupby("n_fold")["anlamlilik_icin_gereken_kazanma"].first()
    w("İşaret testi: %5 iki yönlü anlamlılık için gereken en az kazanma: " + ", ".join(
        f"n={n}: {int(v)}" for n, v in thr.sort_index(ascending=False).items()) + ".")
    w("")
    t = ps.pivot(index="p_gercek", columns="n_fold", values="guc")
    t = pd.DataFrame({"gerçek kazanma olasılığı": t.index,
                      **{f"n={n}": [f"{v:.3f}" for v in t[n]] for n in sorted(t.columns)}})
    w(md_table(t))
    w("")
    pdm = rd("apriori_power_dm.csv", PUB)
    w("DM testi: `δ_blok = k·|r²−1|`, `ncp = √B·δ_blok`; k verinin gürültü yapısından "
      "(birincil ailedeki iki çiftin ortalaması) kalibre edilir, gözlenen etkiden değil. "
      "Parantezde k'nın iki çift arasındaki aralığıyla güç.")
    w("")
    rows = []
    for (h, B), gq in pdm.groupby(["horizon", "etkin_blok"], sort=False):
        row = {"ufuk": f"h={h}", "etkin blok": f"{B:.0f}", "k": f"{gq['k'].iloc[0]:.3f}"}
        for r in gq.itertuples():
            row[f"%{r.rmse_farki_pct} RMSE farkı"] = (f"{r.guc:.3f} "
                                                      f"({r.guc_k_min:.3f}–{r.guc_k_max:.3f})")
        rows.append(row)
    w(md_table(pd.DataFrame(rows).iloc[::-1]))
    w("")
    w("### 10c. İki sürüm (Ek A)")
    w("")
    pt = pw[TS].set_index(["karsilastirma", "horizon"])
    rows = []
    for r in p_.itertuples():
        q = pt.loc[(r.karsilastirma, r.horizon)]
        rows.append({"ufuk": f"h={r.horizon}",
                     "karşılaştırma": "HAR vs HAR-X" if r.karsilastirma == "har vs har_x"
                     else "HAR-X vs XGBoost",
                     "DM gerekli yıl (z.d. → yayım)":
                         f"{q['dm_gerekli_yil']:,.0f} → {r.dm_gerekli_yil:,.0f}",
                     "işaret gerekli yıl (z.d. → yayım)":
                         f"{q['isaret_gerekli_fold_yil']:.0f} → {r.isaret_gerekli_fold_yil:.0f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    pst = rd("apriori_power_sign.csv", TS)
    assert ps.equals(pst), "a priori sign-test curves depend only on n and must match"
    pdt = rd("apriori_power_dm.csv", TS).set_index(["horizon", "rmse_farki_pct"])
    kk = pdm.groupby("horizon")["k"].first()
    kt = pdt.groupby(level="horizon")["k"].first()
    w("Önsel işaret testi eğrileri iki sürümde birebir aynıdır (yalnızca fold sayısına "
      "bağlı; assert). Önsel DM eğrilerinde k: " + ", ".join(
          f"h={h}: {kt[h]:.3f} → {kk[h]:.3f}" for h in HORIZONS) + ".")
    w("")

    # ---------------- 11. Methodology / Limitations numbers ----------------
    w("## 11. Yöntem ve sınırlılık sayıları (GPR yayım hizalaması, test ailesi)")
    w("")
    rev = json.load(open(OUT_DIR / "gpr_revision_summary.json", encoding="utf-8"))
    rep = json.load(open(OUT_DIR / "build_features_publication_aligned_report.json",
                         encoding="utf-8"))
    vm = pd.read_csv(OUT_DIR / "gpr_vintage_meta.csv")
    gap = rev["vintage_minus_last_obs_days_counts"]
    import re
    m78 = re.search(r"discard (\d+)% of published observations", rep["method"])
    assert m78, "forward-fill discard share not found in the feature report"
    w("### 11a. Yayım kuralı (`16_gpr_vintages.py`, erişim " + rev["access_date"] + ")")
    w("")
    exc = vm[vm["vintage_minus_last_obs_days"] != 0].copy()
    exc["vd"] = pd.to_datetime(exc["vintage_date"])
    exc["lo"] = pd.to_datetime(exc["last_obs_date"])
    stale = exc["vintage_minus_last_obs_days"] > 31
    prev_month_end = (~stale & exc["lo"].dt.is_month_end
                      & (exc["lo"].dt.to_period("M") < exc["vd"].dt.to_period("M")))
    other = exc[~stale & ~prev_month_end]
    w(f"- Arşivlenmiş sürüm: **{rev['n_vintages']}** ({rev['first_vintage']} – "
      f"{rev['last_vintage']}).")
    w(f"- Kural \"D günü yayımlanan dosya D dahil D'ye kadarki gözlemleri içerir\": "
      f"**{gap['0']}/{rev['n_vintages']}** sürüm destekliyor. İstisnalar: "
      + ", ".join(f"{v} sürüm {k} gün geride" for k, v in gap.items() if k != "0")
      + f". İstisnaların dökümü: **{int(prev_month_end.sum())}** ay başı dosyası bir "
        f"önceki ayın son gününde duruyor; **{int(stale.sum())}** bayat yükleme ("
        + ", ".join(f"{a.date()} dosyası, son gözlem {b.date()}"
                    for a, b in zip(exc.loc[stale, 'vd'], exc.loc[stale, 'lo']))
        + f"); **{len(other)}** diğer ("
        + ", ".join(f"{a.date()} dosyası, son gözlem {b.date()}"
                    for a, b in zip(other["vd"], other["lo"])) + ").")
    w("- Sürüm günleri: " + ", ".join(f"{k} {v}" for k, v in
                                      rev["vintage_weekday_counts"].items()) + ".")
    pl = rev["publication_lag_days"]
    w(f"- Gözlem başına yayım gecikmesi (takvim günü): medyan {pl['median']:.0f}, "
      f"ortalama {pl['mean']:.2f}, en fazla {pl['max']}. Gözlemin haftanın gününe göre "
      "medyan: " + ", ".join(f"{k} {v['median']:.0f}" for k, v in
                             rev["publication_lag_by_obs_weekday"].items()) + ".")
    for s in ("GPRD", "GPRD_THREAT"):
        r_ = rev["revision_rel_first_release_vs_current"][s]
        w(f"- Revizyon, {s} (ilk yayım vs güncel, göreli): ortalama "
          f"{100 * r_['mean']:+.1f}%, ortalama mutlak {100 * r_['mean_abs']:.1f}%, medyan "
          f"mutlak {100 * r_['median_abs']:.1f}% (n = {r_['n']}). Revizyonlar modellenmedi; "
          "değerler güncel sürümden.")
    w(f"- 2022-02-24 öncesi: arşiv yok, kural karşı-olgusal uygulanır ("
      f"{rep['publication_rule']['before_2022-02-24']}).")
    w(f"- Forward-fill reddi: düzey seriyi işlem takvimine ileri doldurmak yayımlanan "
      f"gözlemlerin **%{m78.group(1)}**'ini atardı.")
    el = rep["effective_lag"]
    w(f"- Zaman damgalı hizalamada satırların **%{100 * el['share_rows_timestamp_uses_unpublished_obs']:.1f}**'i "
      "tahmin anında henüz yayımlanmamış bir gözlem kullanıyordu.")
    w(f"- Değişen özellik: {rep['n_changed']}/{rep['n_features']}; ilk tam dolu satır iki "
      f"sürümde de {rep['first_fully_valid_row']['publication']}.")
    w("")
    w("### 11b. Etkin gecikme (işlem günü t ile kullanılan GPR gözleminin tarihi arası)")
    w("")
    cdp, cdt, trp = (el["calendar_days_publication"], el["calendar_days_timestamp"],
                     el["trading_rows_publication"])
    t = pd.DataFrame({
        "ölçü": ["takvim günü, medyan", "takvim günü, ortalama", "takvim günü, en fazla",
                 "işlem satırı, medyan / ortalama / en fazla"],
        "zaman damgalı": [f"{cdt['median']:.0f}", f"{cdt['mean']:.2f}", f"{cdt['max']:.0f}",
                          "—"],
        "yayım-hizalı": [f"{cdp['median']:.0f}", f"{cdp['mean']:.2f}", f"{cdp['max']:.0f}",
                         f"{trp['median']:.0f} / {trp['mean']:.2f} / {trp['max']}"]})
    w(md_table(t))
    w("")
    wd = el["calendar_days_by_trading_weekday"]
    order_wd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    w("Yayım-hizalı, işlem gününe göre takvim günü (medyan / ortalama / en fazla): "
      + "; ".join(f"{d} {wd[d]['median']:.0f} / {wd[d]['mean']:.2f} / {wd[d]['max']:.0f}"
                  for d in order_wd) + ".")
    w("")
    w("### 11c. Nedensellik doğrulamaları")
    w("")
    pi = rep["prefix_invariance"]
    assert all(x["passed"] and x["tolerance"] == 0 for x in pi)
    w("- **Prefix-invariance:** özellikler veri " + " ve ".join(
        str(x["cut_row"]) for x in pi) + ". satırda kesilerek yeniden hesaplandı; "
      "kesim öncesi tüm satırlar tam veriyle hesaplananla **sıfır toleransta** aynı "
      f"({len(pi)}/{len(pi)} geçti).")
    ps_ = pd.DataFrame(rep["publication_sensitivity_test"])
    n_s = len(ps_)
    pub_ok = int(ps_["publication_aligned_unchanged"].sum())
    ctrl = int(ps_["timestamp_aligned_row_changed"].sum())
    w(f"- **Yayım duyarlılığı:** rastgele {n_s} satırda, o satırın tarihinde henüz "
      f"yayımlanmamış tüm GPR gözlemleri bozuldu (satır başına "
      f"{ps_['n_unpublished_perturbed'].min()}–{ps_['n_unpublished_perturbed'].max()} "
      f"gözlem). Yayım-hizalı kol: **{pub_ok}/{n_s} değişmedi**. Kontrol kolu (zaman "
      f"damgalı): **{ctrl}/{n_s} değişti**.")
    cal = pd.read_csv(OUT_DIR / "gpr_publication_calendar.csv",
                      parse_dates=["date", "publication_date"]).set_index("date")
    fdates = pd.to_datetime(pd.read_csv(OUT_DIR / "features.csv", usecols=["Date"])["Date"],
                            dayfirst=True)
    pred_unch, wds = [], []
    for r in ps_.itertuples():
        t_, tm1 = fdates.iloc[r.row], fdates.iloc[r.row - 1]
        assert t_ == pd.Timestamp(r.date)
        released = cal.loc[tm1, "publication_date"] <= tm1
        pred_unch.append(released)
        if released:
            wds.append(t_.day_name())
    pred_unch = np.array(pred_unch)
    ok_pred = int((pred_unch == ~ps_["timestamp_aligned_row_changed"].values).sum())
    assert ok_pred == n_s, "release calendar must predict the control arm in every row"
    wdc = pd.Series(wds).value_counts()
    w(f"- **Kontrol kolunda değişmeyen {n_s - ctrl} satırın mekanizması:** zaman damgalı "
      "kol satır t'de t−1 tarihli gözlemi kullanır ve bozulma yalnızca t−1'e kadar "
      "yayımlanmamış gözlemlere uygulanır. Değişmeyen satırlar tam olarak t−1 gözleminin "
      "t−1'e kadar zaten yayımlanmış olduğu satırlardır ("
      + ", ".join(f"{v} {k}" for k, v in wdc.items()) + "; Salı satırlarının t−1'i aynı "
      "gün yayımlanan Pazartesi gözlemi, Çarşamba satırı İşçi Bayramı haftası). Yayım "
      f"takvimi kontrol kolunun sonucunu **{ok_pred}/{n_s}** satırda doğru öngörüyor.")
    w("")
    w("### 11d. h=22'deki iki işaret testinin bağımlılığı")
    w("")
    rel = float(np.corrcoef(d1 / fr["har_x"], d2 / fr["har_x"])[0, 1])
    from scipy import stats as _st
    sp = float(_st.spearmanr(d1, d2)[0])
    hx = float(np.corrcoef(fr["har"], fr["xgboost"])[0, 1])
    agree = int(((d1 > 0) == (d2 > 0)).sum())
    w(f"- Fold farkı vektörleri (HAR − HAR-X, XGBoost − HAR-X): Pearson "
      f"**{rho22:.2f}**; HAR-X RMSE'sine bölünmüş göreli farklarla {rel:.2f}; Spearman "
      f"{sp:.2f}. İşaret aynı olan yıl: {agree}/{len(d1)}.")
    w(f"- Mekanizma: HAR ve XGBoost'un fold RMSE profilleri neredeyse aynı (fold'lar "
      f"arası korelasyon **{hx:.3f}**). İki fark da aynı HAR-X RMSE'sini içerdiğinden, iki "
      "test büyük ölçüde HAR-X'i aynı ölçüte karşı sınıyor: HAR-X'in iyi geçirdiği yıl "
      "iki karşılaştırmada birden kazanç, kötü geçirdiği yıl (2020) iki karşılaştırmada "
      "birden kayıp olarak görünüyor. İki fark vektörü arasındaki korelasyon yıl bazlı "
      "ölçek farkından ibaret değil; göreli farklarda ve sıralamada da sürüyor.")
    w("")
    w("**0.995 kendi başına bir bulgu değildir.** Fold RMSE, yılın volatilite düzeyiyle "
      "birlikte ölçeklenir; bu yüzden hemen her model çiftinin fold RMSE'leri yüksek "
      "korelasyonludur. Aşağıdaki tablo bunun karşılaştırma değerlerini veriyor. Ölçekten "
      "arındırılmış ölçüler: fold RMSE'nin train-mean RMSE'sine oranı üzerinden "
      "korelasyon ve günlük hata korelasyonu. Keşifsel, çıkarım için değil.")
    w("")
    hp_all = rd("hybrid_predictions_all.csv", PUB)
    hp_all = hp_all[hp_all["include_in_main"]]
    ms_ = ["har", "har_x", "xgboost", "bilstm", "train_mean", "past_vol"]
    rows = []
    for h in HORIZONS:
        gq = hp_all[hp_all["horizon"] == h]
        fr_ = gq.groupby("test_year").apply(lambda x: pd.Series(
            {m: np.sqrt(((x["y_true"] - x[f"pred_{m}"]) ** 2).mean()) for m in ms_}),
            include_groups=False)
        c = fr_.corr()
        rc = fr_.div(fr_["train_mean"], axis=0).corr()
        ec = pd.DataFrame({m: gq["y_true"] - gq[f"pred_{m}"] for m in ms_}).corr()
        rows.append({
            "ufuk": f"h={h}",
            "fold RMSE: HAR~XGB": f"{c.loc['har', 'xgboost']:.3f}",
            "fold RMSE: HAR~past-vol": f"{c.loc['har', 'past_vol']:.3f}",
            "fold RMSE: HAR~train-mean": f"{c.loc['har', 'train_mean']:.3f}",
            "göreli: HAR~XGB": f"{rc.loc['har', 'xgboost']:.3f}",
            "göreli: HAR~HAR-X": f"{rc.loc['har', 'har_x']:.3f}",
            "göreli: HAR~BiLSTM": f"{rc.loc['har', 'bilstm']:.3f}",
            "günlük hata: HAR~XGB": f"{ec.loc['har', 'xgboost']:.3f}",
            "günlük hata: HAR-X~XGB": f"{ec.loc['har_x', 'xgboost']:.3f}",
            "günlük hata: HAR~HAR-X": f"{ec.loc['har', 'har_x']:.3f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("Okuma: h=22'de HAR ile naif past-volatility baseline'ının fold RMSE korelasyonu da "
      "HAR~XGB kadar yüksek. Yani 0.995, XGBoost'un HAR'ı özel olarak izlediğini değil, "
      "yılların zorluk düzeyinin bütün modellere ortak olduğunu gösteriyor; üstelik "
      "yalnızca h=22 değeri. Ölçekten arındırılmış ölçüler daha bilgilendirici ama "
      "keşifseldir ve ufka göre değişir.")
    w("")
    w("### 11e. İşaret testi eşikleri (tam binom, %5 iki yönlü)")
    w("")
    rows = []
    for n in (9, 14, 15):
        kmin = min(k for k in range(n + 1) if k > n / 2 and sign_p(k, n) <= 0.05)
        rows.append({"fold": n, "anlamlılık için en az kazanma": f"{kmin}/{n}",
                     "o eşikte p": f"{sign_p(kmin, n):.4f}",
                     "bir eksiğinde p": f"{sign_p(kmin - 1, n):.4f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("n=9 (Bölüm 7c, h=22 yüksek kademe) için anlamlılık mümkündür ama 9 fold'un en az "
      "8'inde aynı yön gerekir; gözlenen 6/9 bu eşiğin iki fold altındadır.")
    w("")

    # ---------------- 12. Floor, QLIKE, smearing ----------------
    section_12(w, pr[PUB], A)

    # ---------------- 13. Clark-West, XGB-6 vs HAR, 2026 pointer ----------------
    section_13(w, pr[PUB], F)

    # ---------------- 14. Table 1: descriptive statistics ----------------
    section_14(w)

    path = OUT_DIR / "paper_numbers_publication_aligned.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Yazildi: {path.name}, primary_family_tests_publication_aligned.csv")


if __name__ == "__main__":
    main()
