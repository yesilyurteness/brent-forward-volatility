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
checks (data equalization, BiLSTM convergence) and the two-version comparison.

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
          "sürümde geç fold'larda 4–6 kat). Bu kontrol yayım sürümünde \"yetersiz eğitim "
          "değil aşırı uyum\" iddiasını desteklemez; yalnızca ek eğitimin test hatasını "
          "iyileştirmediğini gösterir. İddianın makaledeki biçimi açık karar.")
    else:
        w("_`bilstm_*_conv_publication_aligned` henüz üretilmedi._")
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

    path = OUT_DIR / "paper_numbers_publication_aligned.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Yazildi: {path.name}, primary_family_tests_publication_aligned.csv")


if __name__ == "__main__":
    main()
