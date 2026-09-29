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

LANGUAGE: the package text is English. It was translated from Turkish after the analysis
freeze (presentation only); scripts/check_translation.py verifies that every number,
commit hash and file path is unchanged in value and order. Turkish free-text values read
from saved outputs are shown through EN_VALUES; the outputs themselves are not changed.

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
    ("xgboost", "XGBoost (primary, 65 features)", "primary"),
    ("bilstm", "Attention BiLSTM", "primary"),
    ("h1_xgb_bilstm", "Hybrid H1: 0.5 XGB + 0.5 BiLSTM", "hybrid"),
    ("h2_harx_xgb", "Hybrid H2: 0.5 HAR-X + 0.5 XGB", "hybrid"),
    ("h3_harx_resid", "Hybrid H3: HAR-X + XGB residual", "hybrid"),
    ("har_x", "HAR-X", "econometric"),
    ("har_x_log", "HAR-X-log", "econometric"),
    ("har", "HAR", "econometric"),
    ("har_log", "HAR-log", "econometric"),
    ("garch", "GARCH(1,1)", "econometric"),
    ("train_mean", "Train-mean", "naive"),
    ("past_vol", "Past-volatility", "naive"),
    ("har_ovx", "HAR + OVX", "ablation"),
    ("har_gpr", "HAR + GPR", "ablation"),
    ("xgb6", "XGBoost-6", "exploratory"),
    ("xgboost_optuna", "XGBoost, Optuna + shrunk smearing", "robustness"),
    ("xgboost_optuna_raw", "XGBoost, Optuna + raw smearing", "appendix"),
]
LABEL = {m: lab for m, lab, _ in MODELS}
ROLE = {m: r for m, _, r in MODELS}
HYBRID_MODELS = ["har_x", "har_x_log", "har", "xgboost", "bilstm", "train_mean",
                 "past_vol", "h1_xgb_bilstm", "h2_harx_xgb", "h3_harx_resid"]
MCOLS = ["model", "horizon", "test_year", "include_in_main", "n", "rmse", "mae",
         "r2_oos", "r2"]

# English display text for the Turkish free-text values this script prints from saved
# outputs. The outputs keep their Turkish text (outputs/README.md, "A note on language");
# a value missing here stops the script instead of printing untranslated text.
EN_VALUES = {
    "iç içe yapıya uygun istatistik; birincil DM testleri görüldükten sonra, CW sonuçları "
    "görülmeden eklendi":
        "statistic suited to the nested structure; added after the primary DM tests were "
        "seen, before the CW results were seen",
    "TreeSHAP via xgboost pred_contribs (shap paketi kullanilmadi)":
        "TreeSHAP via xgboost pred_contribs (shap package not used)",
    "log-oran uzayi (log(vol)-log(past_vol)); ham volatilite birimi DEGIL":
        "log-ratio space (log(vol)-log(past_vol)); NOT raw volatility units",
    "karesel hata": "squared error",
    "Newey-West, Bartlett, L = h-1 (onceden ilan edilmis)":
        "Newey-West, Bartlett, L = h-1 (declared in advance)",
    "Brent günlük log getirisi": "Brent daily log return",
    "Hedef, h=5": "Target, h=5",
    "Hedef, h=22": "Target, h=22",
    "Hedef, h=66": "Target, h=66",
    "Hedef, h=126": "Target, h=126",
    "OVX": "OVX",
    "GPRD (model girdisi, publication)": "GPRD (model input, publication)",
    "GPRD_THREAT (model girdisi, publication)": "GPRD_THREAT (model input, publication)",
    "GPRD (gözlem tarihli, işlem günleri)": "GPRD (observation-dated, trading days)",
    "GPRD_THREAT (gözlem tarihli, işlem günleri)":
        "GPRD_THREAT (observation-dated, trading days)",
    "GPRD (kendi takvimi, tüm takvim günleri)": "GPRD (own calendar, all calendar days)",
    "GPRD_THREAT (kendi takvimi, tüm takvim günleri)":
        "GPRD_THREAT (own calendar, all calendar days)",
    "birincil": "primary",
    "ikincil": "secondary",
}


def en(v):
    assert v in EN_VALUES, f"no English display text for {v!r} (add it to EN_VALUES)"
    return EN_VALUES[v]


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
    print(f"[check] {al}: in {len(j)} model x horizon x fold rows the RMSE recomputed "
          "from the predictions equals the stored metric")


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
                        "role": [ROLE[m] for m in models]})
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
    w("### 12e. Exploratory: the h=5 QLIKE direction reversal and HAR-X's low forecasts")
    w("")
    w("**Status: exploratory and post hoc.** Both checks were designed after the QLIKE "
      "results were seen; the thresholds (1.25 × floor, 0.5 × σ̂_HAR) were chosen by "
      "looking at the results. No test. Pooled, main folds.")
    w("")
    w("**(1) Closeness to the floor, h=5.** σ̂ / fold floor; threshold σ̂ ≤ 1.25 × floor.")
    w("")
    g = x[x["horizon"] == 5]
    k1 = max(1, int(len(g) * 0.01))
    top = g.sort_values("ql_har_x", ascending=False).iloc[:k1]
    rows = []
    for lab, s in ((f"HAR-X's largest 1% QLIKE rows", top),
                   ("all h=5 test observations", g)):
        nx, nh = int((s["kx"] <= 1.25).sum()), int((s["kh"] <= 1.25).sum())
        rows.append({
            "set": lab, "n": len(s),
            "HAR-X σ̂/floor, median (min–max)":
                f"{s['kx'].median():.2f} ({s['kx'].min():.2f}–{s['kx'].max():.2f})",
            "HAR-X ≤ 1.25 × floor": f"{nx} ({100 * nx / len(s):.1f}%)",
            "HAR σ̂/floor, median (min–max)":
                f"{s['kh'].median():.2f} ({s['kh'].min():.2f}–{s['kh'].max():.2f})",
            "HAR ≤ 1.25 × floor": f"{nh} ({100 * nh / len(s):.1f}%)"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    near = g["kx"] <= 1.25
    w(f"The largest 1% of rows: share of HAR-X's h=5 QLIKE total "
      f"{100 * top['ql_har_x'].sum() / g['ql_har_x'].sum():.1f}%; rows with σ̂ ≤ 1.25 × floor "
      f"({int(near.sum())} rows): {100 * g.loc[near, 'ql_har_x'].sum() / g['ql_har_x'].sum():.1f}%. "
      f"Mean QLIKE excluding these rows: HAR-X "
      f"{g.loc[~near, 'ql_har_x'].mean():.4f}, HAR {g.loc[~near, 'ql_har'].mean():.4f} "
      f"(all: {g['ql_har_x'].mean():.4f} / {g['ql_har'].mean():.4f}).")
    w("")
    tt = top.merge(qm[(qm["model"] == "har_x") & (qm["horizon"] == 5)][["Date", "y_true"]],
                   on="Date", validate="1:1")
    w(md_table(pd.DataFrame({
        "date": tt["Date"], "σ": tt["y_true"].map(lambda v: f"{v:.5f}"),
        "σ̂ HAR-X": tt["pred_har_x"].map(lambda v: f"{v:.6f}"),
        "HAR-X/floor": tt["kx"].map(lambda v: f"{v:.2f}"),
        "QLIKE HAR-X": tt["ql_har_x"].map(lambda v: f"{v:.1f}"),
        "σ̂ HAR": tt["pred_har"].map(lambda v: f"{v:.5f}"),
        "HAR/floor": tt["kh"].map(lambda v: f"{v:.2f}"),
        "QLIKE HAR": tt["ql_har"].map(lambda v: f"{v:.3f}")})))
    w("")
    w("**(2) Mechanical rule: \"HAR-X markedly low\" = σ̂_HAR-X < 0.5 × σ̂_HAR.** Share: "
      "the rule rows' share of Σ(QLIKE_HAR-X − QLIKE_HAR) (if the total difference is "
      "negative, the share must be read with its sign). The means after removal are given "
      "pooled and as fold means.")
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
            "horizon": f"h={h}", "rule rows": f"{int(rule.sum())} / {len(g)}",
            "years": ", ".join(f"{y} ({n})" for y, n in yrs.items()) or "—",
            "Σ difference (all)": f"{d.sum():+.2f}".replace("-", "−"),
            "Σ difference (rule rows)": f"{d[rule].sum():+.2f}".replace("-", "−"),
            "share": (f"{100 * d[rule].sum() / d.sum():.1f}%".replace("-", "−")
                      if rule.any() else "—"),
            "mean QLIKE excl., pooled (HAR-X / HAR)":
                f"{rest['ql_har_x'].mean():.4f} / {rest['ql_har'].mean():.4f}",
            "mean QLIKE excl., fold mean (HAR-X / HAR)":
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

    w("## 13. Supplementary family: Clark–West; XGBoost-6 vs HAR; 2026 footnote")
    w("")
    w("### 13a. Clark–West test, HAR ⊂ HAR-X (supplementary family, 4 tests)")
    w("")
    w(f"**Label:** {en(meta['label'])}. The family was declared on {meta['declared']['date']} "
      f"in CLAUDE.md (commit `{meta['declared']['commit']}`); the CW statistic had not been "
      "computed in the repository before that. **The primary family is fixed at 8 tests; CW "
      "is not added to it and does not replace the DM tests.** Holm/BH/BY within these 4 "
      "tests.")
    w("")
    w("`f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`; H1: E[f] > 0 (HAR-X better), "
      "**one-sided**. Saved (published, floored) forecasts, main folds, pooled series (as "
      "for DM). HAC: Newey-West, Bartlett, L = h−1. Inference as in the DM primary family: "
      "HLN factor and t(n−1); the corrections are applied to the HLN p-value. The normal p "
      "without HLN is in a side column. Source: `21_clark_west.py`, "
      "`clark_west_publication_aligned.csv`.")
    w("")
    sci = lambda v: f"{v:.2e}".replace("-", "−")
    t = pd.DataFrame({
        "horizon": [f"h={h}" for h in cw["horizon"]], "n": cw["n"],
        "mean (e²_HAR − e²_HARX)": cw["mean_mse_diff"].map(sci),
        "mean adjustment (ŷ_HAR − ŷ_HARX)²": cw["mean_adjustment"].map(sci),
        "mean f": cw["f_mean"].map(sci),
        "CW": cw["CW"].map(lambda v: f"{v:.3f}"),
        "CW (HLN)": cw["CW_HLN"].map(lambda v: f"{v:.3f}"),
        "p normal (raw)": cw["p_normal_one_sided"].map(fp),
        "p HLN (raw)": cw["p_HLN_one_sided"].map(fp),
        "Holm": cw["p_HLN_holm"].map(fp), "BH": cw["p_HLN_bh"].map(fp),
        "BY": cw["p_HLN_by"].map(fp),
        "HAC inflation": cw["variance_inflation"].map(lambda v: f"{v:.2f}")})
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
    w("Note: Clark & West (2007) use the standard normal; here HLN was applied for "
      "consistency with the DM family, and the result does not change (with the normal "
      "p-values the tests rejected at 5% under Holm, BH and BY are the same; checked).")
    w("")
    w("**At fold level (descriptive, not a test):** the mean of f's fold means and the "
      "number of folds with a positive mean f: "
      + "; ".join(f"h={r.horizon}: {sci(r.f_fold_mean)}, {r.folds_f_positive}/{r.n_folds}"
                  for r in cw.itertuples()) + ".")
    w("")
    w("Reading note: the first column is the raw MSE difference that DM uses; at h=66 and "
      "h=126 it is negative (HAR's MSE is lower in the pooled series). The CW statistic "
      "adds the square of the forecast difference to it.")
    w("")

    w("### 13b. XGBoost-6 vs HAR, by fold (exploratory; no test)")
    w("")
    w("XGBoost-6 is exploratory; no p-value is given. Win: fold RMSE lower than HAR's. "
      "Agreement: whether the direction of the win majority and the direction of the "
      "fold-mean RMSE difference are the same. "
      "`100 × (RMSE_XGB-6 / RMSE_HAR − 1)`, positive = XGBoost-6 worse.")
    w("")
    rows = []
    for h in HORIZONS:
        x = F[F["include_in_main"] & (F["horizon"] == h)].pivot(
            index="test_year", columns="model", values="rmse")
        d = x["xgb6"] - x["har"]
        k, n = int((d < 0).sum()), len(d)
        ties = int((d == 0).sum())
        mean_pct = 100 * (x["xgb6"].mean() / x["har"].mean() - 1)
        cnt_dir = "XGB-6" if k > n - k - ties else ("HAR" if k < n - k - ties else "tie")
        mean_dir = "XGB-6" if mean_pct < 0 else "HAR"
        r = {"horizon": f"h={h}", "years XGB-6 wins / folds": f"{k}/{n}",
             "fold-mean difference": pct(mean_pct),
             "count direction": cnt_dir, "mean direction": mean_dir,
             "agree?": "yes" if cnt_dir == mean_dir else "no"}
        if cnt_dir != mean_dir:
            top = d.reindex(d.abs().sort_values(ascending=False).index)
            top = top[np.sign(top) == np.sign(d.mean())].head(3)
            r["years carrying the mean"] = ", ".join(
                f"{y} ({signed(v * 1e4, 2)}e−4)" for y, v in top.items())
        else:
            r["years carrying the mean"] = "—"
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 13c. 2026 partial year, h=66 and h=126")
    w("")
    w("In Section 1e: every model's RMSE/MAE/R²_oos in that fold, n = 101 (h=66) and 41 "
      "(h=126). Excluded from the primary aggregation; not used for model comparison or "
      "selection.")
    w("")


def section_15(w):
    """Roll-over robustness (24_rollover_robustness.py), Appendix A."""
    s = json.load(open(alignment.out("rollover_summary.json", PUB), encoding="utf-8"))
    T = rd("rollover_family_tests.csv", PUB)
    FM = rd("rollover_metrics.csv", PUB)
    base = T[T["variant"] == "baseline"].set_index(["comparison", "horizon"])
    pf = rd("primary_family_tests.csv", PUB).rename(columns={"karşılaştırma": "comparison"})
    pf = pf.set_index(["comparison", "horizon"])
    assert np.allclose(base["p_HLN"], pf.loc[base.index, "p_HLN"], rtol=1e-8)
    for v in T["variant"].unique():   # corrections are within each variant's own tests
        g = T[T["variant"] == v]
        c = sum(1 / i for i in range(1, len(g) + 1))
        for col in ("p_HLN", "p_sign"):
            assert np.allclose(step_up(g[col].to_numpy()), g[f"{col}_bh"], rtol=1e-12)
            assert np.allclose(step_up(g[col].to_numpy(), c), g[f"{col}_by"], rtol=1e-12)
    cal = s["calendar"]
    VN = {"A": "A (primary robustness variant)", "A2": "A′ (sensitivity: two rows)",
          "B": "B (sensitivity: h=5 only)"}
    w("## 15. Appendix A: Roll-over robustness analysis (HAR, HAR-X, XGBoost)")
    w("")
    w(f"**The statuses were written before the run** ({s['design_registered']}) and were not "
      "changed according to the results; all three variants are reported whatever the "
      "result. Source: `24_rollover_robustness.py`, `rollover_*_publication_aligned`.")
    w("")
    w(f"- **Expiry calendar:** ICE Brent rule (up to February 2016 the 15-day rule, from March "
      f"2016 the month-before rule; ICE contract specification and Circular "
      f"15/235). All {cal['official_expiries_matched']} expiries in ICE's official table "
      f"(December 2015 – March 2023) are reproduced exactly (assert); no official "
      f"table was found for 2008–2015. {cal['expiries_in_sample']} expiries in the sample; "
      f"{cal['expiry_days_missing_in_data']} expiry days are absent from the data. Assumption "
      "(not verified from a CME document): the NYMEX BZ contract underlying `BZ=F` follows "
      "this calendar.")
    w("- **Roll-over row:** the first data row after the expiry day. **A:** that row's return "
      f"is removed ({cal['rows_removed_A']} rows); the target is the std of the returns "
      "remaining in the same window; the return features and the past-vol baseline use "
      f"clean returns. **A′:** two rows after expiry ({cal['rows_removed_A2']} rows). **B "
      f"(h=5 only):** no return is removed; the {s['B_rows_dropped_h5']} rows with a roll-over "
      "in their target window are dropped from the sample. In rows with no removed return "
      "in their window, target and features are identical to the primary values.")
    w("- **Verification:** with an empty mask the targets, the return features and the "
      "HAR/HAR-X/XGBoost forecasts are reproduced bit for bit, and the values of the 8 "
      "tests identical to the stored primary values (assert).")
    w("- **Limitation:** the check removes the roll-over effect in the target and in the "
      "return features; it does not remove the effect in XGBoost's price-level features "
      "(`brent_lag1-5`, `brent_ema5/10/20`), because that requires a back-adjusted series.")
    gr = s["gap_rows_removed"]
    w(f"- **Overlap with date gaps:** of the {s['documented_gaps']} documented gap "
      f"rows, **{gr['A']['n']}** are rows whose return is removed in A, **{gr['A2']['n']}** in "
      "A′. *Hypothesis, not proven:* some of the gaps may stem from roll-overs (Yahoo may "
      "be skipping the row of the expiry day or of the day after); the overlap count does "
      "not show the cause.")
    w("")
    w("### 15a. Within-variant model differences (fold mean)")
    w("")
    w("`100 × (RMSE_a / RMSE_b − 1)`, positive = a worse. **Absolute RMSE is not compared "
      "with the primary result** (the target changes); only within-variant differences "
      "are given. The reference row is the unmasked run of the same code path (= primary result).")
    w("")
    fm = FM[FM["include_in_main"]].groupby(["variant", "horizon", "model"])[
        ["rmse", "mae"]].mean().unstack("model")
    rows = []
    for v in ("baseline", "A", "A2", "B"):
        for h in HORIZONS:
            if (v, h) not in fm.index:
                continue
            r, m = fm.loc[(v, h), "rmse"], fm.loc[(v, h), "mae"]
            rows.append({"variant": VN.get(v, "primary (reference)"), "horizon": f"h={h}",
                         "HAR vs HAR-X, RMSE": pct(100 * (r["har"] / r["har_x"] - 1)),
                         "XGBoost vs HAR-X, RMSE": pct(100 * (r["xgboost"] / r["har_x"] - 1)),
                         "HAR vs HAR-X, MAE": pct(100 * (m["har"] / m["har_x"] - 1)),
                         "XGBoost vs HAR-X, MAE": pct(100 * (m["xgboost"] / m["har_x"] - 1))})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 15b. Tests of the primary family, in each variant with its own corrections")
    w("")
    w("Setup as in 08: DM (Newey-West Bartlett, L = h−1, HLN, t(n−1)) and a fold-level "
      "sign test. Holm/BH/BY within each variant's own tests (A and A′: 8, "
      "B: 2, because B is by design h=5 only). CLAUDE.md: replications on variant data "
      "are not new families and are not pooled with the primary family.")
    w("")
    for v in ("A", "A2", "B"):
        g = T[T["variant"] == v]
        w(f"**{VN[v]}**")
        w("")
        w(md_table(pd.DataFrame({
            "comparison": g["comparison"].str.replace("har_x", "HAR-X").str.replace(
                "har", "HAR").str.replace("xgboost", "XGBoost"),
            "horizon": g["horizon"].map(lambda h: f"h={h}"),
            "DM (HLN)": g["DM_HLN"].map(lambda x: signed(x, 3)),
            "p": g["p_HLN"].map(fp), "Holm": g["p_HLN_holm"].map(fp),
            "BH": g["p_HLN_bh"].map(fp), "BY": g["p_HLN_by"].map(fp),
            "HAR-X wins": [f"{a}/{b}" for a, b in zip(g["harx_fold_wins"], g["n_folds"])],
            "sign p": g["p_sign"].map(fp), "sign Holm": g["p_sign_holm"].map(fp),
            "sign BH": g["p_sign_bh"].map(fp), "sign BY": g["p_sign_by"].map(fp)})))
        w("")
        k = {c: int((g[c] < 0.05).sum()) for c in ("p_HLN_holm", "p_HLN_bh", "p_HLN_by",
                                                  "p_sign_holm", "p_sign_bh", "p_sign_by")}
        w(f"Surviving at 5%: DM Holm {k['p_HLN_holm']}, BH {k['p_HLN_bh']}, BY "
          f"{k['p_HLN_by']}; sign Holm {k['p_sign_holm']}, BH {k['p_sign_bh']}, BY "
          f"{k['p_sign_by']} (total {len(g)} tests).")
        w("")

    # ---- 15c. Clark-West family replicated on the variants (25_rollover_clark_west.py) --
    C = rd("rollover_clark_west.csv", PUB)
    w("### 15c. Replication of the Clark–West supplementary family (HAR ⊂ HAR-X) on the variants")
    w("")
    w("Design and statuses were written before the run (log Stage 26, commit `15ff613`); "
      "statuses as in the roll-over analysis. Setup as in §13a (one-sided, Newey-West "
      "Bartlett L = h−1, HLN and t(n−1); corrections to the HLN p). Holm/BH/BY within each "
      "variant's own tests: 4 in A and A′, 1 in B (with a single test, corrected p = raw p). "
      "CLAUDE.md: a replication of a declared family on variant data is not a new family "
      "and is not pooled with the primary family. The unmasked forecasts reproduce "
      "§13a (assert). Source: `25_rollover_clark_west.py`.")
    w("")
    sci = lambda v: f"{v:.2e}".replace("-", "−")
    diffs = []
    for v in ("A", "A2", "B"):
        g = C[C["variant"] == v]
        c_ = sum(1 / i for i in range(1, len(g) + 1))
        for col in ("p_HLN_bh", "p_HLN_by"):
            ref = step_up(g["p_HLN_one_sided"].to_numpy(),
                          c_ if col.endswith("by") else 1.0)
            assert np.allclose(ref, g[col], rtol=1e-12)
        w(f"**{VN[v]}**")
        w("")
        w(md_table(pd.DataFrame({
            "horizon": g["horizon"].map(lambda h: f"h={h}"), "n": g["n"],
            "mean (e²_HAR − e²_HARX)": g["mean_mse_diff"].map(sci),
            "mean adjustment": g["mean_adjustment"].map(sci),
            "CW": g["CW"].map(lambda x: f"{x:.3f}"),
            "CW (HLN)": g["CW_HLN"].map(lambda x: f"{x:.3f}"),
            "p normal (raw)": g["p_normal_one_sided"].map(fp),
            "p HLN (raw)": g["p_HLN_one_sided"].map(fp),
            "Holm": g["p_HLN_holm"].map(fp), "BH": g["p_HLN_bh"].map(fp),
            "BY": g["p_HLN_by"].map(fp),
            "folds with f > 0": [f"{a}/{b}" for a, b in zip(g["folds_f_positive"],
                                                            g["n_folds"])]})))
        w("")
        # Would the standard-normal p-values (Clark & West 2007) change a 5% decision?
        pn = g["p_normal_one_sided"].to_numpy()
        m = len(pn)
        holm_n = np.minimum(1, np.maximum.accumulate(
            np.sort(pn) * (m - np.arange(m))))[np.argsort(np.argsort(pn))]
        for lab, a_, b_ in (("raw", pn, g["p_HLN_one_sided"].to_numpy()),
                            ("Holm", holm_n, g["p_HLN_holm"].to_numpy()),
                            ("BH", step_up(pn), g["p_HLN_bh"].to_numpy()),
                            ("BY", step_up(pn, c_), g["p_HLN_by"].to_numpy())):
            for h, x, y in zip(g["horizon"], a_, b_):
                if (x < 0.05) != (y < 0.05):
                    diffs.append(f"{VN[v].split(' ')[0]}, h={h}, {lab}: normal p "
                                 f"{x:.4f}, HLN p {y:.4f}")
    w("**Effect of the HLN choice.** The counterpart on the variants of the note in §13a "
      "(\"the result does not change\"): cells where the 5% decision would differ with "
      "standard-normal p-values — "
      + ("; ".join(diffs) if diffs else "none") + ". The primary inference is HLN (the "
      "setup written before the run in Stage 26); these cells are reported according to HLN.")
    w("")


def section_17(w, pr, A):
    """Pooled R2_oos with the common reference, and the H3 residual-stage R2."""
    models = [m for m, _, _ in MODELS]
    ref = (pr[pr["model"] == "train_mean"].groupby(["horizon", "test_year"])["pred"]
           .first().rename("tm").reset_index())
    x = pr[pr["include_in_main"]].merge(ref, on=["horizon", "test_year"], validate="m:1")
    x["sse"] = (x["y_true"] - x["pred"]) ** 2
    x["sst"] = (x["y_true"] - x["tm"]) ** 2
    pooled = x.groupby(["model", "horizon"])[["sse", "sst"]].sum()
    pooled = (1 - pooled["sse"] / pooled["sst"]).rename("r2_oos_pooled").reset_index()
    P = A[["model", "horizon", "r2_oos_common"]].merge(pooled, on=["model", "horizon"],
                                                       validate="1:1")
    assert (P.loc[P["model"] == "train_mean", "r2_oos_pooled"].abs() < 1e-12).all()
    w("## 17. Pooled R²_oos (common reference) and the R² of the H3 residual stage")
    w("")
    w("### 17a. Pooled R²_oos, common reference — NOT the standard R² of §1d")
    w("")
    w("`R²_oos,pooled = 1 − Σ SSE_model / Σ (y − train_mean_fold)²`; sums over all test "
      "observations entering the main metric. The reference in each fold is the forecast of "
      "the train-mean baseline (the same common reference as §1c; the target mean of the "
      "XGBoost training window), i.e. a constant known at forecast time. The standard R² of "
      "§1d, by contrast, is relative to the test slice's own mean (ex post); this table must "
      "not be confused with it. The train-mean row is 0 by definition (assert). The "
      "fold-mean column is §1c's common-reference value. The pooled value weights "
      "high-volatility years more.")
    w("")
    rows = []
    for m in models:
        r = {"model": LABEL[m]}
        for h in HORIZONS:
            q = P[(P["model"] == m) & (P["horizon"] == h)].iloc[0]
            r[f"h={h} fold mean"] = f3(q["r2_oos_common"])
            r[f"h={h} pooled"] = f3(q["r2_oos_pooled"])
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    flip = P[(P["model"] != "train_mean")
             & (np.sign(P["r2_oos_common"]) != np.sign(P["r2_oos_pooled"]))]
    w("**Cells whose sign differs between the two measures** (fold mean → pooled): "
      + ("; ".join(f"{LABEL[r.model]}, h={r.horizon}: {f3(r.r2_oos_common)} → "
                   f"{f3(r.r2_oos_pooled)}" for r in flip.sort_values(
                       ["horizon", "model"]).itertuples()) if len(flip) else "none") + ".")
    w("")

    # ---- 17b. H3 residual stage ----
    hf = rd("hybrid_folds_all.csv", PUB)
    hp = rd("hybrid_predictions_all.csv", PUB)
    # Check the recorded out-of-sample residual R2 against the saved predictions where it is
    # exactly recoverable: in folds with no floored row, e_hat = pred_h3 - pred_har_x.
    chk = 0
    for r in hf[hf["n_floored"] == 0].itertuples():
        g = hp[(hp["horizon"] == r.horizon) & (hp["test_year"] == r.test_year)]
        e = g["y_true"].to_numpy() - g["pred_har_x"].to_numpy()
        eh = g["pred_h3_harx_resid"].to_numpy() - g["pred_har_x"].to_numpy()
        r2 = 1 - ((e - eh) ** 2).sum() / (e ** 2).sum()
        assert np.isclose(r2, r.resid_r2_oos, rtol=1e-9, atol=1e-12), (r.horizon, r.test_year)
        chk += 1
    m = hf[hf["include_in_main"]]
    w("### 17b. H3 mechanism: R² of XGBoost modelling HAR-X's residuals (descriptive)")
    w("")
    w("**Descriptive; no test.** `R²_resid = 1 − SSE(e − ê) / SSE(e)`, e = HAR-X residual, "
      "ê = XGBoost's residual forecast; the baseline is \"not forecasting the residual\" (0). "
      "Positive = the residual stage adds information to HAR-X. In-sample: on XGBoost's own "
      "training rows; out-of-sample: in the test year. The values are those that "
      "`07_hybrid.py` recorded per fold in publication mode, `resid_r2_in_sample` and "
      "`resid_r2_oos`; no refit was needed. Check: "
      f"in the {chk} folds with no floored row, the out-of-sample R² recomputed from the saved H3 "
      "and HAR-X forecasts equals the stored value (assert); in floored "
      "folds ê cannot be recovered from the forecasts. Source: "
      "`hybrid_folds_all_publication_aligned.csv`.")
    w("")
    rows = []
    for h in HORIZONS:
        g = m[m["horizon"] == h]
        rows.append({"horizon": f"h={h}", "fold": len(g),
                     "in-sample, fold mean": f3(g["resid_r2_in_sample"].mean()),
                     "out-of-sample, fold mean": f3(g["resid_r2_oos"].mean()),
                     "out-of-sample, median": f3(g["resid_r2_oos"].median()),
                     "folds with out-of-sample > 0": f"{int((g['resid_r2_oos'] > 0).sum())}/{len(g)}",
                     "out-of-sample, min – max":
                         f"{f3(g['resid_r2_oos'].min())} – {f3(g['resid_r2_oos'].max())}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    return P, flip, rows


def section_18(w, pr):
    """Pooled squared-error difference HAR-X vs XGB-6 by year, h=66 and h=126."""
    w("## 18. XGBoost-6 and HAR-X: decomposition by year of the pooled squared-error "
      "difference (h=66, h=126; descriptive)")
    w("")
    w("**Descriptive; no test.** Contribution = the fold's Σ(e²_HAR-X − e²_XGB-6); positive = "
      "in that year XGBoost-6's squared error is lower. Share = contribution / total at the "
      "horizon (total = 100%; a negative share marks years against the direction of the "
      "total). Main folds; 2026 excluded at these two horizons. From saved forecasts "
      "(`hybrid_predictions_all`, `exploratory_xgb6_predictions`).")
    w("")
    q = pr[pr["include_in_main"] & pr["model"].isin(["har_x", "xgb6"])
           & pr["horizon"].isin([66, 126])].pivot_table(
        index=["horizon", "test_year", "Date"], columns="model",
        values=["pred", "y_true"], aggfunc="first")
    assert (q[("y_true", "har_x")] == q[("y_true", "xgb6")]).all()
    q = pd.DataFrame({
        "d": (q[("y_true", "har_x")] - q[("pred", "har_x")]) ** 2
             - (q[("y_true", "har_x")] - q[("pred", "xgb6")]) ** 2}).reset_index()
    out = {}
    for h in (66, 126):
        g = q[q["horizon"] == h].groupby("test_year")["d"].agg(["sum", "size"])
        tot = g["sum"].sum()
        mse = q[q["horizon"] == h]["d"].mean()
        out[h] = g
        w(f"**h={h}.** Total Σ(e²_HAR-X − e²_XGB-6) = "
          + f"{tot:+.4e} ".replace("-", "−")
          + f"(n = {int(g['size'].sum())}; pooled MSE difference "
          + f"{mse:+.3e}".replace("-", "−") + "; total "
          f"{'positive: XGBoost-6 better in the pool' if tot > 0 else 'negative: HAR-X better in the pool'}).")
        w("")
        srt = g.reindex(g["sum"].sort_values(ascending=False).index)
        cum = (srt["sum"].cumsum() / tot * 100)
        w(md_table(pd.DataFrame({
            "year": srt.index, "n": srt["size"].values,
            "contribution": [f"{v:+.3e}".replace("-", "−") for v in srt["sum"]],
            "share": [f"{100 * v / tot:+.1f}%".replace("-", "−") for v in srt["sum"]],
            "cumulative share (largest to smallest)": [f"{v:.1f}%".replace("-", "−") for v in cum]})))
        w("")
    return out


def section_16(w):
    """Appendix A additions: A1 feature list, volatility regime analysis (07b), A8."""
    w("## 16. Appendix A additions: feature list, volatility regime analysis, training length")
    w("")
    # ---- A1 ----
    feat = pd.read_csv(alignment.features_path(PUB), nrows=1)
    cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    fi = rd("shap_feature_importance.csv", PUB)
    grp = fi[fi["horizon"] == 5].set_index("ozellik")["grup"]
    assert len(cols) == 65 and set(cols) == set(grp.index)
    GL = {"brent_fiyat": "Brent price level and return", "brent_vol": "Brent realized "
          "volatility", "ovx": "OVX", "gpr": "GPR (publication-aligned)", "etkilesim": "Interaction",
          "takvim": "Calendar"}
    w("### 16a. A1: full list of the 65 features")
    w("")
    w("Source: `features_publication_aligned.csv` (`02_build_features.py`); group mapping "
      "`shap_feature_importance_publication_aligned.csv`. All features except the calendar "
      "ones are causal via `.shift(1)`; the GPR features are additionally aligned to their publication dates.")
    w("")
    rows = []
    for gkey in ("brent_vol", "brent_fiyat", "ovx", "gpr", "etkilesim", "takvim"):
        fs = [c for c in cols if grp[c] == gkey]
        rows.append({"group": GL[gkey], "count": len(fs), "features": ", ".join(
            f"`{c}`" for c in fs)})
    w(md_table(pd.DataFrame(rows)))
    w("")
    # ---- volatility regime analysis (script 07b) ----
    vs = json.load(open(alignment.out("explore_vol_regime_summary.json", PUB), encoding="utf-8"))
    gr = rd("explore_vol_regime_groups.csv", PUB)
    w("### 16b. Volatility regime analysis (`07b_exploratory_vol_regime.py`)")
    w("")
    w("**Exploratory and post hoc.** Designed after the contradiction measured in Stage 7 (H2 "
      "better than HAR-X in the pool, worse in the fold mean) was seen, to explain it (log, "
      "clarification note 2026-09-27). It does not change the primary finding and is not "
      "used in model selection. **The regime boundary is mechanical:** at each horizon the "
      "test year's mean realized volatility is split in two at the median across years; "
      "the boundary was not chosen by looking at performance. (The package's §7b is a "
      "different thing: the BiLSTM convergence check.)")
    w("")
    assert "Mekanik" in vs["split_rule"]
    RJ = {"dusuk": "low", "yuksek": "high"}
    w(md_table(pd.DataFrame({
        "horizon": gr["horizon"].map(lambda h: f"h={h}"), "regime": gr["rejim"].map(RJ),
        "fold": gr["n_fold"], "mean volatility": gr["vol_ort"].map(f6),
        "RMSE HAR-X (fold mean)": gr["har_x_fold_ort"].map(f6),
        "RMSE H2 (fold mean)": gr["h2_harx_xgb_fold_ort"].map(f6),
        "H2 vs HAR-X, fold mean": gr["h2_vs_harx_fold_ort_pct"].map(pct),
        "H2 vs HAR-X, pooled": gr["h2_vs_harx_havuz_pct"].map(pct),
        "H2 winning folds": [f"{a}/{b}" for a, b in zip(gr["h2_kazanan_fold"], gr["n_fold"])],
        "squared-error share": gr["kareli_hata_payi_pct"].map(lambda v: f"{v:.1f}%")})))
    w("")
    w("`100 × (RMSE_H2 / RMSE_HAR-X − 1)`, positive = H2 worse. Squared-error share: that "
      "regime's share of HAR-X's squared errors in the pool. Source: "
      "`explore_vol_regime_groups_publication_aligned.csv`.")
    w("")
    # ---- A8 ----
    b = rd("bench_folds_all.csv", PUB)
    xw = pd.DataFrame(json.load(open(alignment.out("wf_summary_all.json", PUB),
                                     encoding="utf-8"))["folds"])
    bl = rd("bilstm_folds_all.csv", PUB)
    j = (b.merge(xw[["horizon", "test_year", "n_train_final"]], on=["horizon", "test_year"],
                 validate="1:1")
         .merge(bl[["horizon", "test_year", "n_train"]].rename(columns={"n_train": "n_bilstm"}),
                on=["horizon", "test_year"], validate="1:1"))
    assert (j["n_train_xgb_equiv"] == j["n_train_final"]).all()
    assert (j["n_train_harx"] == j["n_train_har"]).all()
    assert (j["garch_extra_vs_xgb"] == j["extra_from_embargo"] + j["extra_from_warmup"]).all()
    w("### 16c. A8: Training-length asymmetry")
    w("")
    w("The models' training row counts in each fold. The test rows are the same for all "
      "models; the difference is only at the start of the training window and in the "
      "embargo. The HAR family starts at row 21, XGBoost at row 127 (126-day volatility "
      "window); the BiLSTM's 20-day input sequence (lookback) costs another 19 rows once "
      "at the start of the series; GARCH starts at the start of the return series and, "
      "since it uses no labels, no embargo is applied. "
      "Source: `bench_folds_all` (`05_benchmarks.py`), `wf_summary_all` (`03_walkforward.py`), "
      "`bilstm_folds_all` (`06_attention_bilstm.py`).")
    w("")
    rows = []
    for h in HORIZONS:
        g = j[j["horizon"] == h].sort_values("test_year")
        d_har = (g["n_train_har"] - g["n_train_final"]).unique()
        d_bl = (g["n_bilstm"] - g["n_train_final"]).unique()
        assert len(d_har) == 1
        f0, f1 = g.iloc[0], g.iloc[-1]
        rows.append({
            "horizon": f"h={h}",
            "XGBoost, first – last fold": f"{f0['n_train_final']} – {f1['n_train_final']}",
            "BiLSTM − XGBoost": ", ".join(signed(int(x), 0) for x in d_bl),
            "HAR/HAR-X − XGBoost": f"+{int(d_har[0])}",
            "GARCH − XGBoost": f"+{int(g['garch_extra_vs_xgb'].iloc[0])} "
                               f"({int(g['extra_from_warmup'].iloc[0])} warm-up + "
                               f"{int(g['extra_from_embargo'].iloc[0])} embargo)",
            "HAR excess / XGBoost, first – last fold":
                f"{100 * d_har[0] / f0['n_train_final']:.1f}% – "
                f"{100 * d_har[0] / f1['n_train_final']:.1f}%"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("The effect of this asymmetry on the result is measured in §7a (data equalization).")
    w("")


def section_14(w):
    """Table 1: descriptive statistics (22_descriptive_stats.py)."""
    d = rd("descriptive_stats.csv", PUB)
    meta = json.load(open(alignment.out("descriptive_stats.json", PUB), encoding="utf-8"))
    g4 = lambda v: f"{v:.4g}".replace("-", "−")

    def table(x):
        return md_table(pd.DataFrame({
            "variable": x["label"].map(en), "N": x["N"], "mean": x["mean"].map(g4),
            "std": x["std"].map(g4), "min": x["min"].map(g4), "max": x["max"].map(g4),
            "skewness": x["skew"].map(lambda v: f"{v:.2f}".replace("-", "−")),
            "excess kurtosis": x["excess_kurtosis"].map(lambda v: f"{v:.2f}".replace("-", "−")),
            "ADF (lag)": [f"{s:.2f} ({k})".replace("-", "−")
                          for s, k in zip(x["adf_stat"], x["adf_lag"])],
            "ADF p": x["adf_p"].map(fp),
            f"Q({meta['ljung_box_lag']})": x["lb_q20"].map(lambda v: f"{v:.1f}"),
            "Q p": x["lb_p"].map(fp)}))

    s = meta["sample"]
    w("## 14. Table 1: Descriptive statistics")
    w("")
    w(table(d[d["role"] == "main"]))
    w("")
    w("**Table note.**")
    w(f"- Sample: the sample the model uses, {s['rows']} rows, trading calendar "
      f"({s['first_date']} – {s['last_date']}). The return loses the first row; `target_vol_h` "
      "is undefined in the last h rows (incomplete window, `skipna=False`).")
    w("- Return: `log(P_t / P_{t−1})`. Target: the standard deviation of the next h daily "
      "log returns. Unit: daily log return. OVX: level, index points.")
    w("- **GPRD and GPRD_THREAT as the model sees them, i.e. publication-aligned:** at row t, "
      "the latest observation published up to t−1 (`gprd_lag1`, `gprd_threat_lag1`). The "
      "series stays constant between two publications; it is undefined in the rows before "
      "the first publication.")
    w(f"- Skewness and excess kurtosis are pandas' bias-corrected estimators (excess "
      f"kurtosis 0 under the normal distribution). ADF: with constant, lag by AIC (statsmodels default "
      f"maximum lag 12(n/100)^(1/4)); H0 unit root. Ljung–Box Q({meta['ljung_box_lag']}): "
      "H0 no autocorrelation up to lag 20.")
    w("- **The Ljung–Box rejection for the targets is mechanical.** Consecutive `target_vol_h` "
      "values are computed from overlapping windows that share h−1 of their h returns; the "
      "autocorrelation comes from the construction. This rejection must not be read as "
      "evidence of persistence. Since the publication-aligned GPR series also stays constant "
      "between two publications, part of its autocorrelation is structural "
      "(Q(20): GPRD "
      + f"{d.set_index('variable').loc['GPRD', 'lb_q20']:.1f}; in the observation-dated series "
      + f"{d.set_index('variable').loc['GPRD_obs_trading', 'lb_q20']:.1f}, footnote).")
    w("")
    oc = meta.get("own_calendar")
    w("**Footnote: GPR on other calendars** (for comparison; the model does not see these).")
    w("")
    w(table(d[d["role"] == "footnote"]))
    w("")
    note = ("The observation-dated rows are the unshifted GPR columns of `data/veriseti.xlsx` "
            "(trading days).")
    if oc:
        mad = max(oc["max_abs_diff_vs_dataset"].values())
        zg, zt = oc["zero_days"]["GPRD"], oc["zero_days"]["GPRD_THREAT"]
        note += (f" The own-calendar rows cover every calendar day of the index (weekends included, "
                 f"{oc['calendar_days']} days); the source is the {oc['vintage']} archive "
                 f"vintage from which the dataset was built, `{Path(oc['file']).name}` (SHA-256 "
                 f"`{oc['sha256'][:16]}…`, outside git). This vintage reproduces the dataset's GPR values "
                 f"on all {oc['trading_days_checked']} trading days (largest "
                 f"absolute difference {mad:.1e}, floating-point rounding; checked).")
        if zg:
            note += (" " + ", ".join(z["date"] for z in zg)
                     + (": on this date" if len(zg) == 1 else ": on these dates") + " GPRD 0 (on the same "
                     + ("day" if len(zg) == 1 else "days") + " GPRD_THREAT "
                     + ", ".join(f"{z['GPRD_THREAT']:g}" for z in zg) + ").")
        if zt:
            note += (f" GPRD_THREAT on {len(zt)} days is 0: "
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

    w("## 12. Floor frequency, QLIKE and per-fold smearing (publication-aligned)")
    w("")
    w("No retraining; everything comes from the saved prediction and fold files. QLIKE is "
      "descriptive only: no DM or sign test was run with the QLIKE loss, the primary family "
      "is fixed at 8 tests.")
    w("")
    w("### 12a. How often the prediction floor binds")
    w("")
    w("The level-scale OLS forecasts (HAR, HAR-X, the ablation steps) and Hybrid H3 are "
      "floored at the **minimum of the fold's training target**: `max(forecast, min(y_train))` "
      "(train-only). The log-scale HAR-log and HAR-X-log forecasts are structurally "
      "positive because they are `exp(·) × smearing`; no floor is applied to the forecast (0 by definition). "
      "(The `LOG_FLOOR = 1e-4` applied to the *regressors* of the log specifications is a "
      "separate thing and is not counted here.)")
    w("")
    w("**Detection method.** The prediction files carry no floor flag. A floored row was "
      "detected as a row where the forecast is **exactly equal** to that fold's recorded floor "
      "(`bench_folds_all.pred_floor`, for H3 "
      "`hybrid_folds_all.pred_floor`). "
      "Checks (assert):")
    w("- In every fold the number of equalities is identical to the counter recorded at "
      "fit time (`n_clipped_har`, `n_clipped_har_x`, ablation `n_clipped`, H3 `n_floored`).")
    w(f"- No chance equality: in the models without a floor (HAR-log, HAR-X-log, GARCH, "
      f"past-volatility) the number of forecasts exactly equal to the fold floor is "
      f"{sum(chance.values())}. Yet HAR-log in {below['har_log']}, HAR-X-log in "
      f"{below['har_x_log']} rows gives forecasts **below** the floor; so the equality "
      "arises only from the `max()` operation.")
    w(f"- In the floored models the closest non-floored forecast is {gap_abs:.2e} "
      f"(relative {100 * gap_rel:.3f}%) above the floor; a continuous OLS forecast equalling "
      "the floor bit for bit by chance is practically impossible.")
    w("- The HAR-X used in the package (hybrid file) is floored on the same rows.")
    w("")
    specs = [("har", "HAR"), ("har_ovx", "HAR + OVX"), ("har_gpr", "HAR + GPR"),
             ("har_x", "HAR-X"), ("har_log", "HAR-log"), ("har_x_log", "HAR-X-log"),
             ("h3_harx_resid", "Hybrid H3 (additional)")]
    rows = []
    for m, lab in specs:
        r = {"model": lab}
        for h in HORIZONS:
            if m in flo:
                g = flo[m][(flo[m]["horizon"] == h) & flo[m]["include_in_main"]]
                k, n = int(g["floored"].sum()), len(g)
                pf = g.groupby("test_year")["floored"].mean()
                top = (f"; densest {pf.idxmax()} {100 * pf.max():.1f}%" if k else "")
                r[f"h={h}"] = f"{k} / {n} ({100 * k / n:.2f}%{top})"
            else:
                g = b[(b["horizon"] == h) & b["include_in_main"]]
                r[f"h={h}"] = f"0 / {len(g)} (no floor)"
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    tot = {m: (int(f["floored"].sum()), len(f)) for m, f in flo.items()}
    w("Folds entering the main metric (at h=66/126, 2026 excluded). All folds including 2026: "
      + ", ".join(f"{lab} {tot[m][0]}/{tot[m][1]}" for m, lab in specs if m in tot)
      + f". XGBoost-6 uses the same train-min floor; the recorded counter is "
        f"{int(x6f['n_clipped'].sum())} (no floored forecast).")
    w("")

    # --- QLIKE ------------------------------------------------------------------------
    bad = pr[~(pr["pred"] > 0)]
    if len(bad):
        raise SystemExit("QLIKE stopped: non-positive forecast\n"
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
    w("### 12b. QLIKE (Patton 2011), on the variance scale")
    w("")
    w("`QLIKE = σ²/σ̂² − log(σ²/σ̂²) − 1`, σ = realized target, σ̂ = forecast. Both are "
      "squared because the target is a standard deviation. Lower = better; 0 for a perfect forecast. "
      "**For observations where the floor binds, QLIKE is computed on the published (floored) "
      "forecast** — what is evaluated is the forecast the model gives. All forecasts and "
      f"targets are positive (smallest forecast {pr['pred'].min():.6f}); the stop condition "
      "was not triggered. Bold = lowest in the column.")
    w("")
    w("**Fold mean (primary):**")
    w("")
    w(md_table(wide(Q, "qlike_fold", f4, models, bold_min=True)))
    w("")
    w("**Pooled (secondary):** mean over all test rows entering the main metric.")
    w("")
    w(md_table(wide(Q, "qlike_pooled", f4, models, bold_min=True)))
    w("")
    w("**QLIKE share of the floored rows** (pooled, main folds): the floored rows' share of "
      "total QLIKE / their share of rows. Since the floor is the minimum of the training "
      "target, σ̂ is small in these rows and QLIKE grows.")
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
                           f"{100 * x.loc[x['floored'], 'ql'].sum() / x['ql'].sum():.1f}% / "
                           f"{100 * k / len(x):.2f}%")
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 12c. QLIKE and RMSE rankings (fold mean)")
    w("")
    w("Rank 1 = best. Only the models whose rank differs between the two losses are listed. "
      "Two consistent losses can rank differently (Patton 2011); the difference is "
      "reported as a finding.")
    w("")
    from scipy import stats as _st
    rows, diffs = [], []
    for h in HORIZONS:
        g = Q[Q["horizon"] == h].set_index("model")
        rr = g["rmse"].rank(method="min").astype(int)
        rq = g["qlike_fold"].rank(method="min").astype(int)
        tau = float(_st.kendalltau(g["rmse"], g["qlike_fold"])[0])
        ch = [m for m in models if rr[m] != rq[m]]
        rows.append({"horizon": f"h={h}", "Kendall τ (17 models)": f"{tau:.3f}",
                     "models whose rank changes": f"{len(ch)}/17",
                     "best in RMSE": LABEL[rr.idxmin()],
                     "best in QLIKE": LABEL[rq.idxmin()]})
        for m in sorted(ch, key=lambda m: rr[m]):
            diffs.append({"horizon": f"h={h}", "model": LABEL[m], "RMSE rank": rr[m],
                          "QLIKE rank": rq[m],
                          "difference": f"{rq[m] - rr[m]:+d}".replace("-", "−")})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w(md_table(pd.DataFrame(diffs)))
    w("")
    w("**Concentration of QLIKE** (pooled, main folds): QLIKE penalizes under-prediction "
      "(σ̂ ≪ σ) harshly, so the mean can rest on a few observations. Cell: "
      "the share of the largest 1% of rows in the QLIKE total; in parentheses the σ/σ̂ "
      "ratio and the date of the largest single row. Descriptive.")
    w("")
    rows = []
    for m in models:
        r = {"model": LABEL[m]}
        for h in HORIZONS:
            g = qm[(qm["model"] == m) & (qm["horizon"] == h)]
            s = g["ql"].sort_values(ascending=False)
            k1 = max(1, int(len(s) * 0.01))
            i0 = s.index[0]
            r[f"h={h}"] = (f"{100 * s.iloc[:k1].sum() / s.sum():.1f}% "
                           f"({g.loc[i0, 'y_true'] / g.loc[i0, 'pred']:.1f}×, "
                           f"{g.loc[i0, 'Date']})")
        rows.append(r)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("**Direction in the primary family's two comparisons** (descriptive; not a test). "
      "`100 × (loss_a / loss_b − 1)`, positive = a worse.")
    w("")
    rows = []
    for a_, b_ in (("har", "har_x"), ("xgboost", "har_x")):
        for h in HORIZONS:
            g = Q[Q["horizon"] == h].set_index("model")
            rows.append({"a vs b": f"{LABEL[a_]} vs {LABEL[b_]}", "horizon": f"h={h}",
                         "RMSE (fold mean)": pct(100 * (g.loc[a_, "rmse"] / g.loc[b_, "rmse"] - 1)),
                         "QLIKE (fold mean)": pct(100 * (g.loc[a_, "qlike_fold"]
                                                          / g.loc[b_, "qlike_fold"] - 1)),
                         "QLIKE (pooled)": pct(100 * (g.loc[a_, "qlike_pooled"]
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
    w(f"**h=5 HAR-X's largest single QLIKE row ({top['Date']}).** The date is the row's own "
      f"date t, i.e. the **forecast origin**; it is not the start of the target window. The features, "
      f"via `.shift(1)`, use information up to t−1 ({raw['Date'].iloc[rix - 1]}). Since the target "
      "is `std(r_{t+1}, …, r_{t+5})`, the window is the returns of the five trading days "
      "after t: " + ", ".join(win) + f" (each return from the previous trading day's "
      f"close; the first from the {raw['Date'].iloc[rix]} close to the {win[0]} close). "
      f"The return of day t enters neither the features nor the target. Realized σ = "
      f"{top['y_true']:.6f}, HAR-X forecast σ̂ = {top['pred']:.6f} (σ/σ̂ = "
      f"{top['y_true'] / top['pred']:.1f}, QLIKE = {top['ql']:.1f}). Fold floor "
      f"{fl5:.6f}; the forecast is {100 * (top['pred'] / fl5 - 1):.2f}% above the floor, "
      "not floored.")
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
    w("### 12d. Per-fold Duan smearing coefficient and log-residual std")
    w("")
    w("Smearing `S = mean(exp(e))`, e = log-scale residuals on the training set (in-sample, "
      "train-only). For XGBoost and the BiLSTM the target is `log(σ_h / past_vol_h)`, for HAR-log and "
      "HAR-X-log `log(σ_h)`. Source: `wf_summary_all` (XGBoost: `smearing`, "
      "`resid_log_std`), `bilstm_folds_all`, `bench_folds_all` (`har_smearing`, "
      "`har_x_smearing`). Log-residual std (ddof=1): for XGBoost 03's record "
      "(`resid_log_std`); for HAR-log and HAR-X-log, since 05 recorded only the coefficient, "
      "the OLS was re-estimated with `20_log_residual_std.py`. It was asserted that the "
      "re-estimation **reproduces the saved test forecasts and the smearing coefficient bit "
      "for bit in every fold** (120/120). **The log-residual std was not recorded for the "
      "BiLSTM:** 06 does not store the trained weights; computing it would require "
      "retraining. The stds of XGBoost/BiLSTM and of the HAR-log family are on different "
      "targets (log-ratio vs log-level), so they cannot be compared "
      "directly.")
    w("")
    rows = []
    for m, lab in (("xgb", "XGBoost, S"), ("xgb_sd", "XGBoost, log-residual std"),
                   ("bilstm", "BiLSTM, S"), ("har_log", "HAR-log, S"),
                   ("har_log_sd", "HAR-log, log-residual std"),
                   ("har_x_log", "HAR-X-log, S"),
                   ("har_x_log_sd", "HAR-X-log, log-residual std")):
        r = {"measure": lab}
        for h in HORIZONS:
            g = S[(S["horizon"] == h) & S["include_in_main"]][m]
            r[f"h={h}"] = f"{g.median():.4f} ({g.min():.4f}–{g.max():.4f})"
        rows.append(r)
    w("Summary, folds entering the main metric: median (min–max). BiLSTM log-residual "
      "std: not recorded.")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    for h in HORIZONS:
        g = S[S["horizon"] == h].sort_values("test_year")
        t = pd.DataFrame({
            "year": [f"{y}" + ("" if im else " (outside the main metric)")
                     for y, im in zip(g["test_year"], g["include_in_main"])],
            "XGBoost S": g["xgb"].map(f4), "XGBoost log-residual std": g["xgb_sd"].map(f4),
            "BiLSTM S": g["bilstm"].map(f4), "BiLSTM log-residual std": "not recorded",
            "HAR-log S": g["har_log"].map(f4), "HAR-log log-residual std": g["har_log_sd"].map(f4),
            "HAR-X-log S": g["har_x_log"].map(f4),
            "HAR-X-log log-residual std": g["har_x_log_sd"].map(f4)})
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
    print("[check] fold means equal gpr_alignment_comparison.csv (both versions)")

    A = agg[PUB]
    F = fm[PUB]
    L = []  # markdown lines
    w = L.append

    # ---------------- Header ----------------
    pv = provenance()
    w("# Paper numbers — publication-aligned GPR version (PRIMARY)")
    w("")
    w(f"- **Generated from commit:** `{pv['commit']}` ({pv['subject']})")
    w(f"- **Generated at:** {pv['generated']}")
    w("- **Working tree:** " + (
        "clean — the inputs are identical to the files in this commit."
        if not pv["dirty"] else
        "**DIRTY — the numbers do not correspond exactly to this commit.** Uncommitted "
        "changes: " + ", ".join(f"`{p}`" for p in pv["dirty"])))
    w("- Verification: `git checkout <commit> && python scripts/18_paper_numbers.py` must "
      "produce the same numbers (only this header changes).")
    w("")
    w("> This file is generated by `scripts/18_paper_numbers.py` from saved outputs; "
      "it is not edited by hand. When writing the paper, numbers are taken **only from this file**. "
      "The numbers of the timestamp-aligned (old) version appear only in Section 8, for Appendix A.")
    w("")
    w("**General rules.** Main metric: fold-mean RMSE and MAE; R²_oos secondary "
      "(reference: that fold's training target mean); standard R² footnote metric. "
      "At h=5 and h=22, 15 folds (2012–2026); at h=66 and h=126, 14 folds (the 2026 partial year "
      "is excluded from the main metric and given separately in a footnote). Unit: standard deviation "
      "of daily log returns. Percentages `100 × (RMSE_a / RMSE_b − 1)`; negative = a better.")
    w("")
    w("**p-values.** The only test family used for inference is **the primary family of "
      "eight** (Section 6: HAR vs HAR-X and HAR-X vs XGBoost, four horizons). The family "
      "**was formalized after the tests, it is not a pre-registration**; but it was chosen not "
      "by looking at p-values but according to two claims declared in Stages 5–6, and it has "
      "been fixed since then: no test is added afterwards. All other p-values in this file "
      "(ablation, XGBoost-6, the two-version comparison, the BiLSTM checks, Section 9) "
      "are **exploratory and not corrected for multiple comparisons**; they are given as "
      "descriptive. The secondary DM family (24 tests) is corrected within itself with Holm/BH/BY but "
      "is not confirmatory.")
    w("")
    w("**Consistency checks (assert):** each fold's RMSE was recomputed from the prediction "
      "files and compared with the stored metric; the fold means equal "
      "`gpr_alignment_comparison.csv`; train-mean R²_oos is exactly 0 in every fold; "
      "only the 2026 fold is excluded from the main metric, at h=66/126.")
    w("")

    # ---------------- 1. Main table ----------------
    order = [m for m, _, _ in MODELS]
    w("## 1. Main results table (fold mean)")
    w("")
    w("Source: `hybrid_metrics_all` (XGBoost, BiLSTM, hybrids, HAR, HAR-X, HAR-X-log, "
      "naive), `bench_metrics_all` (GARCH, HAR-log), `ablation_exogenous_folds` "
      "(HAR+OVX, HAR+GPR), `exploratory_xgb6_folds`, `opt_*metrics_all` — all "
      "`_publication_aligned`. Common sample: in every fold all models have the same number "
      "of test rows (assert); the training windows may differ by model (the HAR family "
      "starts at row 21, XGBoost at 127; Stage 5 data-equalization check, Section 7a). "
      "Bold = lowest value in the column.")
    w("")
    w("### 1a. RMSE")
    w("")
    w(md_table(wide(A, "rmse", f6, order, bold_min=True)))
    w("")
    w("### 1b. MAE")
    w("")
    w(md_table(wide(A, "mae", f6, order, bold_min=True)))
    w("")
    w("### 1c. R²_oos (secondary metric)")
    w("")
    w("`R²_oos = 1 − SSE_model / Σ(y_test − train_mean)²`. **The reference is the same for "
      "all models:** in every fold the forecast of the train-mean baseline (the target mean "
      "of the XGBoost training window, `hybrid_metrics_all`). Hence the train-mean row "
      "is exactly 0 and the values in a column are measured against the same constant forecast. "
      "Recomputed from the prediction files; for the models sourced from the hybrid file it is "
      "identical to the stored value (assert).")
    w("")
    w(md_table(wide(A, "r2_oos_common", f3, order)))
    w("")
    own = [m for m in order if m not in HYBRID_MODELS]
    t = pd.DataFrame({"model": [LABEL[m] for m in own]})
    for h in HORIZONS:
        s = A[A["horizon"] == h].set_index("model").loc[own]
        t[f"h={h}"] = [f3(v) for v in s["r2_oos"]]
    w("Note — the values in the source files use a different reference: `bench`, "
      "`ablation`, `exploratory_xgb6` and `opt_*` take the mean of each model's **own** "
      "training window as the reference (the HAR family starts at row 21, XGBoost at 127). "
      "The values below are those in these files; they are not used in the paper and "
      "are given only to show the reason for the difference if a comparison with the "
      "source files is made:")
    w("")
    w(md_table(t))
    w("")
    neg = A[(A["r2_oos_common"] < 0)].sort_values(["horizon", "model"])
    w("Negative R²_oos (model worse than the constant train-mean forecast): " + "; ".join(
        f"h={h}: " + ", ".join(LABEL[m] for m in g["model"])
        for h, g in neg.groupby("horizon")) + ".")
    w("")
    w("### 1d. Standard R² (footnote metric, not used for decisions)")
    w("")
    w("Fold mean and pooled value. The reference is the test slice's own mean "
      "(ex-post). Because the fold SST becomes very small in calm years, the fold R² values are "
      "not on the same scale; the fold mean is given with this caveat. The pooled R² is "
      "systematically higher because it also contains between-year variance.")
    w("")
    t = pd.DataFrame({"model": [LABEL[m] for m in order]})
    for h in HORIZONS:
        s = A[A["horizon"] == h].set_index("model").loc[order]
        t[f"h={h} fold mean"] = [f3(v) for v in s["r2_fold"]]
        t[f"h={h} pooled"] = [f3(v) for v in s["r2_pooled"]]
    w(md_table(t))
    w("")
    npool = A.groupby("horizon")["n_pooled"].agg(["min", "max"])
    assert (npool["min"] == npool["max"]).all(), "pooled sample differs across models"
    w("Number of pooled observations: " + ", ".join(
        f"h={h}: {int(r['min'])}" for h, r in npool.iterrows()) + ".")
    w("")
    w("### 1e. Footnote: 2026 partial year, h=66 and h=126 (low statistical power)")
    w("")
    w("Not used for model comparison or selection; informational only.")
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
                        for h in (66, 126)) + " test observations.")
    w("")
    opt = rd("opt_metrics_all.csv", PUB)
    fb = (opt[(opt["model"] == "xgboost") & (opt["selection_procedure"] != "optuna")]
          .groupby("horizon").size())
    w("Note (Optuna rows): in early folds where no validation slice can be built, the Optuna "
      "run falls back to the capacity rule: " + ", ".join(
          f"h={h}: {n} folds" for h, n in fb.items()) + " (all folds, including 2026).")
    w("")

    # ---------------- 2. Decomposition ----------------
    dec = pd.read_csv(OUT_DIR / "gpr_alignment_decomposition.csv")
    dp = dec[dec["gpr_alignment"] == PUB].set_index("horizon").loc[HORIZONS]
    w("## 2. Four-step decomposition (HAR → HAR-X → XGBoost-6 → XGBoost)")
    w("")
    w("Each step changes one thing: (1) the exogenous variables (OVX, GPR) are added; "
      "(2) the same six regressors, level target, no preprocessing — only the functional form, "
      "from linear to tree; (3) the 65 features + log-ratio target + smearing + preprocessing package. "
      "Source: `gpr_alignment_decomposition.csv`. XGBoost-6 is exploratory and post hoc.")
    w("")
    t = pd.DataFrame({
        "horizon": [f"h={h}" for h in HORIZONS],
        "RMSE HAR": [f6(v) for v in dp["rmse_har"]],
        "RMSE HAR-X": [f6(v) for v in dp["rmse_har_x"]],
        "RMSE XGB-6": [f6(v) for v in dp["rmse_xgb6"]],
        "RMSE XGB": [f6(v) for v in dp["rmse_xgboost"]],
        "(1) exogenous HAR→HAR-X": [pct(v) for v in dp["pct_exogenous_HAR_to_HARX"]],
        "(2) functional form HAR-X→XGB-6": [pct(v) for v in
                                            dp["pct_functional_form_HARX_to_XGB6"]],
        "(3) feature package XGB-6→XGB": [pct(v) for v in
                                          dp["pct_feature_package_XGB6_to_XGB"]],
        "total HAR→XGB": [pct(v) for v in dp["pct_total_HAR_to_XGB"]],
    })
    w(md_table(t))
    w("")
    rows = []
    for h in HORIZONS:
        a, b, n = wins(F, "xgb6", "har_x", h)
        c, d, _ = wins(F, "xgboost", "xgb6", h)
        rows.append({"horizon": f"h={h}", "XGB-6 < HAR-X (fold)": f"{a}/{n}",
                     "sign p": fp(sign_p(a, a + b)),
                     "XGB < XGB-6 (fold)": f"{c}/{n}", "sign p ": fp(sign_p(c, c + d))})
    w("By fold (RMSE, uncorrected two-sided sign test):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")

    # ---------------- 3. Ablation ladder ----------------
    w("## 3. Ablation ladder (HAR, HAR+OVX, HAR+GPR, HAR-X)")
    w("")
    w("Source: `ablation_exogenous_folds_publication_aligned.csv`. Exploratory/post hoc; "
      "not part of the primary hypothesis family. HAR and HAR+OVX do not use GPR and are "
      "bit-identical in the two versions. RMSE/MAE identical to the ablation file (assert); R²_oos with the "
      "common reference of Section 1c.")
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
            rows.append({"horizon": f"h={h}", "variant": LABEL[m] if m != "har_x" else
                         "HAR-X (HAR + OVX + GPR)", "RMSE": f6(r["rmse"]),
                         "MAE": f6(r["mae"]), "R²_oos": f3(r["r2_oos"]),
                         "RMSE vs HAR": "—" if m == "har" else pct(100 * (r["rmse"] / base - 1))})
    w(md_table(pd.DataFrame(rows)))
    w("")
    abf = ab.pivot_table(index=["horizon", "test_year"], columns="model", values="rmse")
    rows = []
    for h in HORIZONS:
        x = abf.loc[h]
        row = {"horizon": f"h={h}", "fold": len(x)}
        for lab, a, b in (("OVX contribution: HAR+OVX < HAR", "har_ovx", "har"),
                          ("GPR contribution: HAR+GPR < HAR", "har_gpr", "har"),
                          ("GPR on top of OVX: HAR-X < HAR+OVX", "har_x", "har_ovx")):
            k, l = int((x[a] < x[b]).sum()), int((x[a] > x[b]).sum())
            p = sign_p(k, k + l)
            row[lab] = f"{k}/{len(x)} (p{'<0.001' if p < 0.001 else '=' + fp(p)})"
            row[lab.split(":")[0] + " RMSE %"] = pct(100 * (x[a].mean() / x[b].mean() - 1))
        rows.append(row)
    w("Fold win counts (uncorrected two-sided sign test) and fold-mean "
      "RMSE difference:")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")

    # ---------------- 4. Standardized betas ----------------
    sb = rd("ablation_exogenous_std_beta_summary.csv", PUB)
    sb = sb[sb["fold_set"] == "main"].set_index(["variant", "horizon", "regressor"])
    w("## 4. Standardized betas")
    w("")
    w("beta_std = beta × sd(X) / sd(y), the sds from that fold's training slice. Fold "
      "mean (number of positive / negative folds), main fold set. Source: "
      "`ablation_exogenous_std_beta_summary_publication_aligned.csv`.")
    w("")

    def beta_cell(v, h, r):
        s = sb.loc[(v, h, r)]
        return f"{signed(s['mean'])} ({int(s['n_positive'])}/{int(s['n_negative'])})"

    harx_regs = ["har_daily", "brent_vol5", "brent_vol20", "ovx_lag1", "gprd_lag1",
                 "gprd_threat_lag1"]
    w("### 4a. HAR-X, six regressors")
    w("")
    t = pd.DataFrame({"regressor": harx_regs})
    for h in HORIZONS:
        t[f"h={h}"] = [beta_cell("har_x", h, r) for r in harx_regs]
    w(md_table(t))
    w("")
    w("### 4b. Exogenous coefficients in the ablation variants")
    w("")
    cols = [("har_ovx", "ovx_lag1"), ("har_ovx", "brent_vol20"), ("har", "brent_vol20"),
            ("har_gpr", "gprd_lag1"), ("har_gpr", "gprd_threat_lag1")]
    t = pd.DataFrame({"horizon": [f"h={h}" for h in HORIZONS]})
    for v, r in cols:
        t[f"{LABEL[v]}: {r}"] = [beta_cell(v, h, r) for h in HORIZONS]
    w(md_table(t))
    w("")

    # ---------------- 5. SHAP ----------------
    sh = json.load(open(alignment.out("shap_summary.json", PUB), encoding="utf-8"))
    g = pd.DataFrame(sh["group_shares_xgb"]).pivot(index="grup", columns="horizon",
                                                   values="pay_pct")
    grp_order = ["brent_vol", "gpr", "ovx", "brent_fiyat", "etkilesim", "takvim"]
    w("## 5. SHAP group shares (primary XGBoost)")
    w("")
    w(f"Method: {en(sh['method'])}. Unit: {en(sh['shap_units'])}. Share = group mean|SHAP| / "
      "total. **Caveat:** a group's share grows with its number of features; the number of "
      "features per group is in the second column. Source: `shap_summary_publication_aligned.json`.")
    w("")
    t = pd.DataFrame({"group": grp_order,
                      "features": [sh["groups"][k] for k in grp_order]})
    for h in HORIZONS:
        t[f"h={h}"] = [f"{g.loc[k, h]:.1f}%" for k in grp_order]
    w(md_table(t))
    w("")
    gh = pd.DataFrame(sh["group_shares_harx"]).pivot(index="grup", columns="horizon",
                                                     values="pay_pct")
    w("Comparison: group shares of |standardized beta| in HAR-X. mean|SHAP| and the "
      "standardized beta are not the same quantity; they can be compared only at the level "
      "of ranking and shares.")
    w("")
    t = pd.DataFrame({"group": ["brent_vol", "ovx", "gpr"]})
    for h in HORIZONS:
        t[f"h={h}"] = [f"{gh.loc[k, h]:.1f}%" for k in t["group"]]
    w(md_table(t))
    w("")
    w("Share of the features other than HAR-X's six regressors in the XGBoost SHAP: " + ", ".join(
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
    d["claim"] = np.where(d["model1"] == "har", "HAR-X beats HAR",
                          "HAR-X beats XGBoost")
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

    w("## 6. Primary hypothesis family (8 tests) — the central inferential result")
    w("")
    w("Two claims × four horizons: **HAR vs HAR-X** (do the exogenous variables contribute) and "
      "**HAR-X vs XGBoost** (does the nonlinear model contribute). Holm (FWER), "
      "Benjamini-Hochberg (FDR, valid under positive dependence/PRDS) and "
      "Benjamini-Yekutieli (FDR, valid under any dependence structure; BH × c(m), "
      f"c(8) = {C8:.3f}) within the family, over 8 tests. "
      f"Loss: {en(dms['loss'])}. HAC: {en(dms['hac'])}. HLN: `{dms['hln']}`. "
      "DM sign: negative = first model better. **The Holm, BH and BY corrections are applied "
      "to the HLN p-value** (not to the raw DM p). BY is computed in this file; BH is "
      "reproduced with the same function and verified against the stored value of `08_dm_test.py` "
      "(assert). The sign test is a fold-level "
      "binomial (H0: p=0.5, two-sided); it uses no HAC/normality assumption. DM is on the pooled "
      "series (forecasts of models retrained every year), not the fold "
      "mean; hence the pooled RMSE difference differs from the fold-mean difference in "
      "Section 1 and changes sign at h=66/126 for HAR vs HAR-X.")
    w("")
    w("Transparency: the family definition was formalized after the tests (it is not a "
      "pre-registration); the comparisons were chosen not according to p-values but "
      "according to the claims declared in Stages 5–6.")
    w("")
    t = pd.DataFrame({
        "horizon": [f"h={h}" for h in d["horizon"]],
        "comparison": [("HAR vs HAR-X" if a == "har" else "HAR-X vs XGBoost")
                       for a in d["model1"]],
        "pooled RMSE difference": [pct(v) for v in d["fark_pct"]],
        "DM": [signed(v, 3) for v in d["DM_ham"]],
        "DM (HLN)": [signed(v, 3) for v in d["DM_HLN"]],
        "raw p": [fp(v) for v in d["p_ham"]],
        "HLN p": [fp(v) for v in d["p_HLN"]],
        "Holm p": [fp(v) for v in d["p_HLN_holm"]],
        "BH p": [(f"**{fp(v)}**" if v < 0.05 else fp(v)) for v in d["p_HLN_bh"]],
        "BY p": [(f"**{fp(v)}**" if v < 0.05 else fp(v)) for v in d["p_HLN_by"]],
        "sign: HAR-X wins": [f"{k}/{n}" for k, n in zip(d["harx_wins"],
                                                        d["isaret_fold"])],
        "sign raw p": [fp(v) for v in d["p_isaret"]],
        "sign Holm p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                        for v in d["p_isaret_holm"]],
        "sign BH p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                      for v in d["p_isaret_bh"]],
        "sign BY p": [(f"**{fp(v)}**" if v < 0.05 else fp(v))
                      for v in d["p_isaret_by"]],
    })
    w(md_table(t))
    w("")
    w("Machine-readable copy: `primary_family_tests_publication_aligned.csv`.")
    w("")

    def survivors(x):
        return {"DM Holm": int((x["p_HLN_holm"] < .05).sum()),
                "DM BH": int((x["p_HLN_bh"] < .05).sum()),
                "DM BY": int((x["p_HLN_by"] < .05).sum()),
                "sign Holm": int((x["p_isaret_holm"] < .05).sum()),
                "sign BH": int((x["p_isaret_bh"] < .05).sum()),
                "sign BY": int((x["p_isaret_by"] < .05).sum())}

    rows = []
    for al, name in ((PUB, "publication-aligned (primary)"), (TS, "timestamp-aligned (Appendix A)")):
        for fam in ("birincil", "ikincil"):
            x = dm[al][dm[al]["aile"] == fam]
            rows.append({"version": name, "family": f"{en(fam)} ({len(x)} tests)",
                         **{k: f"{v}/{len(x)}" for k, v in survivors(x).items()}})
    w("Number of surviving tests (5% threshold):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    surv = d[d["p_isaret_bh"] < .05]
    assert list(surv["horizon"]) == [22, 22], "survivor text below assumes the two h=22 tests"
    assert (d["p_isaret_by"] >= .05).all() and (d["p_HLN_by"] >= .05).all()
    w("**Primary family result (publication-aligned):**")
    w("")
    w("- **Holm (FWER):** no test survives.")
    w("- **BH (FDR, under the PRDS assumption):** at h=22 two hypotheses are rejected: " + "; ".join(
        f"{r.claim} (sign {r.harx_wins}/{r.isaret_fold}, BH p = {r.p_isaret_bh:.3f})"
        for r in surv.itertuples()) + ". **These are two separate hypotheses, but not two "
      "independent pieces of evidence:** the two fold-difference vectors (HAR − HAR-X and XGBoost − "
      f"HAR-X) are correlated at {rho22:.2f}. Both contain HAR-X, and 2020 is a common losing "
      f"year (years HAR-X loses: against HAR {', '.join(map(str, lose1))}; "
      f"against XGBoost {', '.join(map(str, lose2))}). The same 13/15 and the same raw p come from the "
      "binomial test depending only on the win count; the vectors differ "
      "(assert).")
    w("- **BY (FDR, valid regardless of the dependence structure):** no test "
      f"survives. Smallest BY p = {d['p_isaret_by'].min():.3f} (h=22 sign tests; "
      f"BH p {surv['p_isaret_bh'].iloc[0]:.4f} × c(8) = {C8:.3f}).")
    w(f"- **DM:** no significance under any correction (HLN p range "
      f"{d['p_HLN'].min():.3f}–{d['p_HLN'].max():.3f}).")
    w("")
    ok = (d["dm_yon"] == d["isaret_yon"])
    w(f"Direction agreement: DM (pooled) and the sign test point to the same model in {int(ok.sum())}/8 tests; "
      "disagreeing: " + (", ".join(
          f"h={r.horizon} {r.model1} vs {r.model2}" for r in d[~ok].itertuples()) or "none")
      + ".")
    w("")

    # ---------------- 7. Robustness ----------------
    w("## 7. Robustness checks (re-run in publication mode)")
    w("")
    w("### 7a. Data equalization: benchmarks restricted to XGBoost's window (row 127)")
    w("")
    w("RMSE difference relative to XGBoost (%, negative = benchmark better). Source: "
      "`bench_model_comparison_all{,_aligned}_publication_aligned.csv`.")
    w("")
    rows = []
    eq = {}
    for al in (PUB, TS):
        for win, name in (("normal", "bench_model_comparison_all.csv"),
                          ("equalized", "bench_model_comparison_all_aligned.csv")):
            p = rd(name, al).pivot_table(index="model", columns="horizon",
                                         values="rmse_fold_mean")
            eq[(al, win)] = 100 * (p.loc[["har", "har_x", "har_x_log"]] / p.loc["xgboost"] - 1)
    for m in ("har", "har_x", "har_x_log"):
        rows.append({"model": LABEL[m],
                     "normal window": " / ".join(pct(eq[(PUB, 'normal')].loc[m, h])
                                                 for h in HORIZONS),
                     "equalized window": " / ".join(pct(eq[(PUB, 'equalized')].loc[m, h])
                                                    for h in HORIZONS)})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("(h=5 / h=22 / h=66 / h=126.)")
    w("")
    w("### 7b. BiLSTM convergence check (high tier only; over the final 10% epoch "
      "slice a loss decrease < 2%, upper bound 200 epochs)")
    w("")
    conv_ok = alignment.out("bilstm_aggregate_all_conv.csv", PUB).exists()
    if conv_ok:
        rows = []
        for h in HORIZONS:
            row = {"horizon": f"h={h}"}
            for al, tag in ((PUB, "publication"), (TS, "timestamp")):
                pri = rd("bilstm_folds_all.csv", al)
                cf = rd("bilstm_folds_all_conv.csv", al)
                prim = rd("bilstm_metrics_all.csv", al)
                conv = rd("bilstm_metrics_all_conv.csv", al)
                sel = lambda x: x[(x["model"] == "bilstm") & (x["horizon"] == h) &
                                  x["include_in_main"]].set_index("test_year")["rmse"]
                a1, a2 = sel(prim), sel(conv)
                row[f"{tag}: primary"] = f6(a1.mean())
                row[f"{tag}: convergence"] = f6(a2.mean())
                row[f"{tag}: change"] = pct(100 * (a2.mean() / a1.mean() - 1))
                row[f"{tag}: convergence better"] = f"{int((a2 < a1).sum())}/{len(a1)}"
                if al == PUB:
                    c = cf[(cf["horizon"] == h) & (cf["arch_tier"] == "yuksek")]
                    row["publication: mean epochs (high tier)"] = (
                        f"{c['epochs_run'].mean():.0f}" if len(c) else "—")
                    pc = pri[(pri["horizon"] == h) & (pri["arch_tier"] == "yuksek")]
                    lr = (pc.set_index("test_year")["loss_final"] /
                          c.set_index("test_year")["loss_final"])
                    row["publication: loss ratio primary/convergence, median"] = (
                        f"{lr.median():.2f}" if len(c) else "—")
                    row["publication: under-trained (primary→convergence)"] = (
                        f"{int((pri[pri['horizon'] == h]['convergence'] == 'yetersiz_egitilmis').sum())}"
                        f"→{int((cf[cf['horizon'] == h]['convergence'] == 'yetersiz_egitilmis').sum())}")
            rows.append(row)
        w(md_table(pd.DataFrame(rows)))
        w("")
        w("Change = convergence / primary − 1 (fold-mean RMSE, main folds). "
          "At h=66 and h=126 there are no high-tier folds; the results are identical by definition.")
        w("")
        w("**Limit of interpretation (see experiment log 16.4):** in the publication version the stopping criterion "
          "triggered early, the training loss fell only ~1.4-fold (in the timestamp "
          "version 4–6-fold in late folds). On its own this check does not support the claim \"not "
          "under-training but overfitting\"; the claim rests on 7c's fixed "
          "200-epoch check.")
    else:
        w("_`bilstm_*_conv_publication_aligned` not generated yet._")
    w("")
    fx_path = alignment.out("bilstm_fixed200_folds.csv", PUB)
    w("### 7c. BiLSTM fixed 200 epochs (exploratory, post hoc; no early stopping)")
    w("")
    if fx_path.exists():
        fx = rd("bilstm_fixed200_folds.csv", PUB)
        fx = fx[fx["include_in_main"]]
        w("High-tier folds, h=5 and h=22. The cosine schedule is the same as in the convergence run "
          "(`T_max=200`); the k-th epoch is the convergence run itself (loss and test "
          "forecasts bit-identical, assert). k = the epoch at which the convergence rule stopped. "
          "Source: `bilstm_fixed200_folds_publication_aligned.csv`; per-epoch "
          "loss `bilstm_fixed200_loss_history_publication_aligned.csv`. Log 16.5.")
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
                "horizon": f"h={h}", "fold": len(gx), "mean k": f"{gx['k_conv'].mean():.0f}",
                "training loss k→200, median (range)": rng("loss_ratio_k_to_200"),
                "eval-mode training MSE k→200": rng("train_mse_eval_ratio_k_to_200"),
                "loss primary(60)→200, median":
                    f"{gx['loss_ratio_primary_to_200'].median():.2f}×",
                "test RMSE primary / k / 200": f"{gx['rmse_primary'].mean():.6f} / "
                                               f"{gx['rmse_k'].mean():.6f} / "
                                               f"{gx['rmse_200'].mean():.6f}",
                "RMSE 200 vs k": pct(100 * (gx["rmse_200"].mean() / gx["rmse_k"].mean() - 1)),
                "200 better (sign p; exploratory, uncorrected)":
                    f"{k}/{len(gx)} (p={fp(sign_p(k, len(gx)))})",
                "MAE 200 vs k": pct(100 * (gx["mae_200"].mean() / gx["mae_k"].mean() - 1)),
                "sd ratio k→200": f"{gx['pred_std_ratio_k'].mean():.2f} → "
                                  f"{gx['pred_std_ratio_200'].mean():.2f}",
                "horizon-mean RMSE (all folds)":
                    f"{base:.6f} → {full.mean():.6f} ({pct(100 * (full.mean() / base - 1))})"})
        w(md_table(pd.DataFrame(rows)))
        w("")
        w("**Status of the p-values:** the sign-test p-values here (h=5: 0.007) come from an "
          "exploratory diagnostic, **are not part of the primary family of eight and are "
          "not corrected**; their status is the same as HAR+OVX vs HAR-X's uncorrected p = "
          "0.035. The primary family is fixed (formalized after the tests, not a "
          "pre-registration; see Section 6); no test is added afterwards.")
        w("")
        w("**Interpretation:** without the stopping rule the training loss falls sharply and the "
          "test error does not improve but worsens. The \"not under-training but overfitting\" "
          "finding is supported by this check in the publication version. The \"~4×\" figure "
          "of the timestamp version belongs to Appendix A; the publication version's figure is the medians above.")
    else:
        w("_`bilstm_fixed200_*_publication_aligned` not generated yet._")
    w("")

    # ---------------- 8. Two-version comparison ----------------
    w("## 8. Two-version comparison (Appendix A)")
    w("")
    w("Timestamp version: GPR was used every day with a one-day lag (row t saw "
      "the observation dated t−1); but since GPR is published weekly, these observations had "
      "often not yet been published at forecast time ("
      f"{100 * json.load(open(OUT_DIR / 'build_features_publication_aligned_report.json', encoding='utf-8'))['effective_lag']['share_rows_timestamp_uses_unpublished_obs']:.1f}% "
      "of the timestamp rows use an unpublished observation). The two versions are evaluated on "
      "exactly the same sample (same train/test rows); the models that use no GPR are "
      "bit-identical. The difference is a pure alignment effect. Source: "
      "`gpr_alignment_comparison*.csv`.")
    w("")
    w("### 8a. RMSE, models that use GPR (fold mean)")
    w("")
    w("% = 100 × (RMSE_publication / RMSE_timestamp − 1); positive = the accuracy cost of "
      "respecting the publication lag. Fold count: number of folds in which the publication "
      "version is better / total; p uncorrected two-sided sign test.")
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
    w("The models that use no GPR (HAR, HAR-log, HAR+OVX, GARCH, train-mean, "
      "past-volatility) are bit-identical in the two versions; not in the table.")
    w("")
    w("### 8b. Decomposition, two versions")
    w("")
    rows = []
    for h in HORIZONS:
        for al, name in ((TS, "timestamp-aligned"), (PUB, "publication-aligned")):
            r = dec[(dec["horizon"] == h) & (dec["gpr_alignment"] == al)].iloc[0]
            rows.append({"horizon": f"h={h}", "version": name,
                         "(1) exogenous": pct(r["pct_exogenous_HAR_to_HARX"]),
                         "(2) functional form": pct(r["pct_functional_form_HARX_to_XGB6"]),
                         "(3) feature package": pct(r["pct_feature_package_XGB6_to_XGB"]),
                         "total": pct(r["pct_total_HAR_to_XGB"])})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8c. Primary family, two versions")
    w("")
    rows = []
    dt = dm[TS][dm[TS]["aile"] == "birincil"].set_index(["model1", "model2", "horizon"])
    for r in d.itertuples():
        q = dt.loc[(r.model1, r.model2, r.horizon)]
        tw = q["isaret_kazanan"] if r.model1 == "har_x" else q["isaret_fold"] - q["isaret_kazanan"]
        rows.append({"horizon": f"h={r.horizon}",
                     "comparison": "HAR vs HAR-X" if r.model1 == "har" else "HAR-X vs XGBoost",
                     "HLN p (t.s. → pub.)": f"{fp(q['p_HLN'])} → {fp(r.p_HLN)}",
                     "DM BH p": f"{fp(q['p_HLN_bh'])} → {fp(r.p_HLN_bh)}",
                     "HAR-X wins": f"{int(tw)}/{int(q['isaret_fold'])} → "
                                   f"{r.harx_wins}/{r.isaret_fold}",
                     "sign raw p": f"{fp(q['p_isaret'])} → {fp(r.p_isaret)}",
                     "sign Holm p": f"{fp(q['p_isaret_holm'])} → {fp(r.p_isaret_holm)}",
                     "sign BH p": f"{fp(q['p_isaret_bh'])} → {fp(r.p_isaret_bh)}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8d. The weight of GPR, two versions")
    w("")
    sht = json.load(open(alignment.out("shap_summary.json", TS), encoding="utf-8"))
    gt = pd.DataFrame(sht["group_shares_xgb"]).pivot(index="grup", columns="horizon",
                                                     values="pay_pct")
    sbt = rd("ablation_exogenous_std_beta_summary.csv", TS)
    sbt = sbt[sbt["fold_set"] == "main"].set_index(["variant", "horizon", "regressor"])
    rows = []
    for h in HORIZONS:
        rows.append({
            "horizon": f"h={h}",
            "SHAP GPR share": f"{gt.loc['gpr', h]:.1f}% → {g.loc['gpr', h]:.1f}%",
            "SHAP OVX share": f"{gt.loc['ovx', h]:.1f}% → {g.loc['ovx', h]:.1f}%",
            "HAR-X β gprd_lag1": f"{signed(sbt.loc[('har_x', h, 'gprd_lag1'), 'mean'])} → "
                                 f"{signed(sb.loc[('har_x', h, 'gprd_lag1'), 'mean'])}",
            "HAR-X β gprd_threat_lag1":
                f"{signed(sbt.loc[('har_x', h, 'gprd_threat_lag1'), 'mean'])} → "
                f"{signed(sb.loc[('har_x', h, 'gprd_threat_lag1'), 'mean'])}",
            "HAR-X β ovx_lag1": f"{signed(sbt.loc[('har_x', h, 'ovx_lag1'), 'mean'])} → "
                                f"{signed(sb.loc[('har_x', h, 'ovx_lag1'), 'mean'])}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8e. Robustness checks, timestamp version (for comparison)")
    w("")
    rows = []
    for m in ("har", "har_x", "har_x_log"):
        rows.append({"model": LABEL[m],
                     "normal window": " / ".join(pct(eq[(TS, 'normal')].loc[m, h])
                                                 for h in HORIZONS),
                     "equalized window": " / ".join(pct(eq[(TS, 'equalized')].loc[m, h])
                                                    for h in HORIZONS)})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 8f. 2026 partial-year footnote, two versions (h=66, h=126; informational)")
    w("")
    f26c = pd.read_csv(OUT_DIR / "gpr_alignment_comparison_2026_footnote.csv")
    f26c = f26c[f26c["model"].isin(gmodels)]
    t = pd.DataFrame({"model": [LABEL[m] for m in gmodels]})
    for h in (66, 126):
        s = f26c[f26c["horizon"] == h].set_index("model").loc[gmodels]
        t[f"h={h} RMSE t.s. → pub."] = [f"{f6(a)} → {f6(b)} ({pct(c)})" for a, b, c in
                                        zip(s["rmse_ts"], s["rmse_pub"], s["rmse_pct"])]
    w(md_table(t))
    w("")

    # ---------------- 9. Additional numbers used in the root README ----------------
    w("## 9. Additional numbers used in the README (publication-aligned)")
    w("")
    w("Every number in the root `README.md` comes either from Sections 1–8 or from this section.")
    w("")
    R = A.set_index(["model", "horizon"])
    rr = lambda a, b, h: 100 * (R.loc[(a, h), "rmse"] / R.loc[(b, h), "rmse"] - 1)
    w("### 9a. Main RMSE comparisons (fold mean, %)")
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
                      ("H3 (HAR-X+residual) vs HAR-X", "h3_harx_resid", "har_x"),
                      ("HAR-X-log vs HAR-X", "har_x_log", "har_x"),
                      ("HAR+OVX vs HAR-X", "har_ovx", "har_x")):
        rows.append({"comparison": lab, **{f"h={h}": pct(rr(a, b, h)) for h in HORIZONS}})
    har_family = ["har", "har_log", "har_x", "har_x_log", "har_ovx", "har_gpr"]
    best = {h: A[(A["horizon"] == h) & A["model"].isin(har_family)]
            .sort_values("rmse").iloc[0]["model"] for h in HORIZONS}
    for lab, ms in (("best hybrid vs best HAR-family",
                     ["h1_xgb_bilstm", "h2_harx_xgb", "h3_harx_resid"]),
                    ("best primary nonlinear (XGB, XGB-Optuna, BiLSTM) vs HAR",
                     ["xgboost", "xgboost_optuna", "bilstm"]),
                    ("XGBoost-6 (exploratory; HAR-X's inputs, OVX included) vs HAR",
                     ["xgb6"])):
        row = {"comparison": lab}
        for h in HORIZONS:
            s = A[(A["horizon"] == h) & A["model"].isin(ms)].sort_values("rmse").iloc[0]
            ref = best[h] if "HAR-family" in lab else "har"
            row[f"h={h}"] = f"{pct(rr(s['model'], ref, h))} ({s['model']} vs {ref})"
        rows.append(row)
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("Best HAR-family model (RMSE): " + ", ".join(f"h={h}: {LABEL[m]}"
                                                  for h, m in best.items()) + ".")
    w("")
    rows = []
    for h in HORIZONS:
        x = abf.loc[h]
        k = int((x["har_ovx"] < x["har_x"]).sum())
        rows.append({"horizon": f"h={h}", "HAR+OVX beats HAR-X": f"{k}/{len(x)}",
                     "sign p": fp(sign_p(k, len(x)))})
    w("HAR+OVX vs HAR-X, by fold (uncorrected two-sided sign test):")
    w("")
    w(md_table(pd.DataFrame(rows)))
    w("")
    gpr_b = sb.loc[[i for i in sb.index if i[0] in ("har_x", "har_gpr")
                    and i[2] in ("gprd_lag1", "gprd_threat_lag1")], "mean"]
    ovx_b = sb.loc[[i for i in sb.index if i[0] in ("har_x", "har_ovx")
                    and i[2] == "ovx_lag1"], "mean"]
    w(f"Standardized beta range (fold means, HAR-X and ablation, four horizons): "
      f"GPR between {signed(gpr_b.min())} and {signed(gpr_b.max())}; OVX "
      f"between {signed(ovx_b.min())} and {signed(ovx_b.max())}.")
    w("")
    w("### 9b. DM secondary family (24 tests), publication-aligned: survivors")
    w("")
    sec = dm[PUB][dm[PUB]["aile"] == "ikincil"]
    rows = []
    for r in sec.itertuples():
        if min(r.p_HLN_bh, r.p_isaret_bh) < .05:
            rows.append({"horizon": f"h={r.horizon}", "comparison": f"{r.model1} vs {r.model2}",
                         "pooled RMSE difference": pct(r.fark_pct),
                         "DM Holm / BH / BY": f"{fp(r.p_HLN_holm)} / {fp(r.p_HLN_bh)} / "
                                              f"{fp(r.p_HLN_by)}",
                         "sign": f"{r.isaret_kazanan}/{r.isaret_fold}",
                         "sign Holm / BH / BY": f"{fp(r.p_isaret_holm)} / "
                                                f"{fp(r.p_isaret_bh)} / {fp(r.p_isaret_by)}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("Sign: folds won by model1 / total. DM significance against the naive baseline at "
      "long horizons (h=66, h=126): " + (", ".join(
          f"h={r.horizon} {r.model1} vs {r.model2} (BH {fp(r.p_HLN_bh)})"
          for r in sec[(sec["horizon"] >= 66) & (sec["model2"] == "past_vol")
                       & (sec["p_HLN_bh"] < .05)].itertuples()) or "none") + ".")
    w("")
    w("### 9c. Additional SHAP measures")
    w("")
    sa = rd("shap_sign_agreement.csv", PUB)
    unused = sa.groupby("horizon")["kullanilmadi"].mean() * 100
    used = sa[~sa["kullanilmadi"]]
    ovx_ag = used[used["regresor"] == "ovx_lag1"]
    stab = {r["horizon"]: r for r in sh["stability"]}
    t = pd.DataFrame({"measure": [
        "HAR-X: OVX's |std beta| share",
        "XGBoost: SHAP share outside HAR-X's six regressors",
        "share of unused comparisons (5 common regressors × fold; SHAP identically zero)",
        "OVX sign agreement (XGBoost SHAP direction vs HAR-X beta; used comparisons)",
        "attribution ranking stability: first-last fold Spearman ρ"]})
    for h in HORIZONS:
        oh = ovx_ag[ovx_ag["horizon"] == h]
        t[f"h={h}"] = [f"{gh.loc['ovx', h]:.1f}%",
                       f"{sh['outside_harx_share_pct'][str(h)]:.1f}%",
                       f"{unused[h]:.1f}%",
                       f"{100 * oh['uyum'].mean():.0f}% ({int(oh['uyum'].sum())}/{len(oh)})",
                       f"{stab[h]['ilk_son_rho']:.2f}"]
    w(md_table(t))
    w("")
    w(f"OVX sign agreement, four horizons together: {100 * ovx_ag['uyum'].mean():.0f}% "
      f"({int(ovx_ag['uyum'].sum())}/{len(ovx_ag)}). All folds (including 2026), "
      "`shap_sign_agreement_publication_aligned.csv`.")
    w("")
    w("### 9d. Date gaps: direct target-correction test (forecasts fixed)")
    w("")
    gt_ = rd("gap_target_test.csv", PUB).set_index(["horizon", "model"])
    gs = json.load(open(alignment.out("gap_target_test_summary.json", PUB),
                        encoding="utf-8"))["horizons"]
    rows = []
    for h in HORIZONS:
        x = gt_.loc[h]
        s = gs[str(h)]
        rows.append({
            "horizon": f"h={h}", "model": len(x),
            "largest |RMSE change|": f"{x['rmse_change_pct'].abs().max():.2f}%",
            "RMSE rank changes": s["n_rmse_rank_changes"],
            "MAE rank changes": f"{s['n_mae_rank_changes']}"
                                + (f" ({' ↔ '.join(s['mae_rank_swaps'])})"
                                   if s["mae_rank_swaps"] else ""),
            "HAR+OVX < HAR-X (corrected)":
                "yes" if x.loc["har_ovx", "rmse_corrected"] < x.loc["har_x", "rmse_corrected"]
                else "no",
            "HAR < HAR+GPR (corrected)":
                "yes" if x.loc["har", "rmse_corrected"] < x.loc["har_gpr", "rmse_corrected"]
                else "no"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("### 9e. Gap-free subsample (2017–2026 folds)")
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
            "horizon": f"h={h}", "fold (2017+)": int(x["n_folds_2017plus"].iloc[0]),
            "Spearman RMSE rank, full vs 2017+":
                f"{gfs[str(h)]['spearman_rmse_full_vs_2017plus']:.2f}",
            "Spearman, full vs 2012–2016":
                f"{gfs[str(h)]['spearman_rmse_full_vs_2012_2016']:.2f}",
            "best HAR-family < XGBoost and BiLSTM":
                "yes" if fam < min(r17["xgboost"], r17["bilstm"]) else "no",
            "models with RMSE below train-mean": f"{int((r17 < r17['train_mean']).sum())}"
                                                 f"/{len(r17) - 1}",
            "range of the top five RMSEs": f"{100 * (top5.max() / top5.min() - 1):.1f}%"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("\"models with RMSE below train-mean\" is counted on RMSE (unaffected by the R²_oos "
      "reference differences).")
    w("")

    # ---------------- 10. Power analysis ----------------
    w("## 10. Power analysis (primary family; `09_power_analysis.py`)")
    w("")
    w("80% power, 5% two-sided. DM: the sampling unit is the effective block B = n/h. Sign test: "
      "the sampling unit is the fold (year), exact binomial. Source: `power_analysis_publication_"
      "aligned.csv`, `apriori_power_{sign,dm}_publication_aligned.csv`.")
    w("")
    w("**Caveat:** the \"realized power\" computed from the observed effect is a monotone "
      "transformation of the p-value and carries no information beyond the p-value; it cannot be "
      "used as evidence of whether a result is due to chance. The informative parts are the required sample "
      "(10a) and the a priori curves, independent of the observed results (10b).")
    w("")
    pw = {al: rd("power_analysis.csv", al) for al in (PUB, TS)}
    pw_s = json.load(open(alignment.out("power_analysis_summary.json", PUB), encoding="utf-8"))
    w(f"Trading days per year (measured from the test period): {pw_s['days_per_year']:.1f}.")
    w("")
    w("### 10a. Test period required for 80% power if the observed effect is taken as true")
    w("")
    p_ = pw[PUB].copy()
    # HAR-X's fold wins, as in Section 6 (isaret_kazanan counts model1's wins)
    p_["harx_w"] = np.where(p_["karsilastirma"] == "har vs har_x",
                            p_["isaret_fold"] - p_["isaret_kazanan"], p_["isaret_kazanan"])
    p_ = p_.sort_values(["karsilastirma", "horizon"])
    t = pd.DataFrame({
        "horizon": [f"h={h}" for h in p_["horizon"]],
        "comparison": ["HAR vs HAR-X" if c == "har vs har_x" else "HAR-X vs XGBoost"
                       for c in p_["karsilastirma"]],
        "DM: effective blocks": [f"{v:.0f}" for v in p_["dm_etkin_blok"]],
        "DM: realized power": [f"{v:.3f}" for v in p_["dm_gerceklesen_guc"]],
        "DM: required years": [f"{v:,.0f}" for v in p_["dm_gerekli_yil"]],
        "DM: factor": [f"{v:.1f}×" for v in p_["dm_kat_artis"]],
        "sign: HAR-X wins": [f"{k}/{n}" for k, n in zip(p_["harx_w"], p_["isaret_fold"])],
        "sign: realized power": [f"{v:.3f}" for v in p_["isaret_gerceklesen_guc"]],
        "sign: required years": [f"{v:.0f}" for v in p_["isaret_gerekli_fold_yil"]],
        "sign: factor": [f"{v:.1f}×" for v in p_["isaret_kat_artis"]],
    })
    w(md_table(t))
    w("")
    w(f"Ranges: the extension required for DM is "
      f"{p_['dm_kat_artis'].min():.1f}–{p_['dm_kat_artis'].max():.0f} times the current test period; for "
      f"the sign test {p_['isaret_kat_artis'].min():.1f}–{p_['isaret_kat_artis'].max():.1f} times.")
    w("")
    w("### 10b. A priori power curves (do not use the observed results)")
    w("")
    ps = rd("apriori_power_sign.csv", PUB)
    thr = ps.groupby("n_fold")["anlamlilik_icin_gereken_kazanma"].first()
    w("Sign test: minimum wins required for 5% two-sided significance: " + ", ".join(
        f"n={n}: {int(v)}" for n, v in thr.sort_index(ascending=False).items()) + ".")
    w("")
    t = ps.pivot(index="p_gercek", columns="n_fold", values="guc")
    t = pd.DataFrame({"true win probability": t.index,
                      **{f"n={n}": [f"{v:.3f}" for v in t[n]] for n in sorted(t.columns)}})
    w(md_table(t))
    w("")
    pdm = rd("apriori_power_dm.csv", PUB)
    w("DM test: `δ_block = k·|r²−1|`, `ncp = √B·δ_block`; k is calibrated from the noise structure of the data "
      "(the mean of the two pairs in the primary family), not from the observed effect. "
      "In parentheses, the power over the range of k between the two pairs.")
    w("")
    rows = []
    for (h, B), gq in pdm.groupby(["horizon", "etkin_blok"], sort=False):
        row = {"horizon": f"h={h}", "effective blocks": f"{B:.0f}", "k": f"{gq['k'].iloc[0]:.3f}"}
        for r in gq.itertuples():
            row[f"{r.rmse_farki_pct}% RMSE difference"] = (f"{r.guc:.3f} "
                                                           f"({r.guc_k_min:.3f}–{r.guc_k_max:.3f})")
        rows.append(row)
    w(md_table(pd.DataFrame(rows).iloc[::-1]))
    w("")
    w("### 10c. Two versions (Appendix A)")
    w("")
    pt = pw[TS].set_index(["karsilastirma", "horizon"])
    rows = []
    for r in p_.itertuples():
        q = pt.loc[(r.karsilastirma, r.horizon)]
        rows.append({"horizon": f"h={r.horizon}",
                     "comparison": "HAR vs HAR-X" if r.karsilastirma == "har vs har_x"
                     else "HAR-X vs XGBoost",
                     "DM required years (t.s. → pub.)":
                         f"{q['dm_gerekli_yil']:,.0f} → {r.dm_gerekli_yil:,.0f}",
                     "sign required years (t.s. → pub.)":
                         f"{q['isaret_gerekli_fold_yil']:.0f} → {r.isaret_gerekli_fold_yil:.0f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    pst = rd("apriori_power_sign.csv", TS)
    assert ps.equals(pst), "a priori sign-test curves depend only on n and must match"
    pdt = rd("apriori_power_dm.csv", TS).set_index(["horizon", "rmse_farki_pct"])
    kk = pdm.groupby("horizon")["k"].first()
    kt = pdt.groupby(level="horizon")["k"].first()
    w("The a priori sign-test curves are identical in the two versions (they depend only on "
      "the number of folds; assert). k in the a priori DM curves: " + ", ".join(
          f"h={h}: {kt[h]:.3f} → {kk[h]:.3f}" for h in HORIZONS) + ".")
    w("")

    # ---------------- 11. Methodology / Limitations numbers ----------------
    w("## 11. Methodology and limitation numbers (GPR publication alignment, test family)")
    w("")
    rev = json.load(open(OUT_DIR / "gpr_revision_summary.json", encoding="utf-8"))
    rep = json.load(open(OUT_DIR / "build_features_publication_aligned_report.json",
                         encoding="utf-8"))
    vm = pd.read_csv(OUT_DIR / "gpr_vintage_meta.csv")
    gap = rev["vintage_minus_last_obs_days_counts"]
    import re
    m78 = re.search(r"discard (\d+)% of published observations", rep["method"])
    assert m78, "forward-fill discard share not found in the feature report"
    w("### 11a. Publication rule (`16_gpr_vintages.py`, accessed " + rev["access_date"] + ")")
    w("")
    exc = vm[vm["vintage_minus_last_obs_days"] != 0].copy()
    exc["vd"] = pd.to_datetime(exc["vintage_date"])
    exc["lo"] = pd.to_datetime(exc["last_obs_date"])
    stale = exc["vintage_minus_last_obs_days"] > 31
    prev_month_end = (~stale & exc["lo"].dt.is_month_end
                      & (exc["lo"].dt.to_period("M") < exc["vd"].dt.to_period("M")))
    other = exc[~stale & ~prev_month_end]
    w(f"- Archived vintages: **{rev['n_vintages']}** ({rev['first_vintage']} – "
      f"{rev['last_vintage']}).")
    w(f"- Rule \"the file published on day D contains the observations up to and including D\": "
      f"supported by **{gap['0']}/{rev['n_vintages']}** vintages. Exceptions: "
      + ", ".join(f"{v} vintages {k} days behind" for k, v in gap.items() if k != "0")
      + f". Breakdown of the exceptions: **{int(prev_month_end.sum())}** start-of-month files stop at the last "
        f"day of the previous month; **{int(stale.sum())}** stale uploads ("
        + ", ".join(f"{a.date()} file, last observation {b.date()}"
                    for a, b in zip(exc.loc[stale, 'vd'], exc.loc[stale, 'lo']))
        + f"); **{len(other)}** other ("
        + ", ".join(f"{a.date()} file, last observation {b.date()}"
                    for a, b in zip(other["vd"], other["lo"])) + ").")
    w("- Vintage weekdays: " + ", ".join(f"{k} {v}" for k, v in
                                          rev["vintage_weekday_counts"].items()) + ".")
    pl = rev["publication_lag_days"]
    w(f"- Publication lag per observation (calendar days): median {pl['median']:.0f}, "
      f"mean {pl['mean']:.2f}, max {pl['max']}. Median by the observation's "
      "weekday: " + ", ".join(f"{k} {v['median']:.0f}" for k, v in
                              rev["publication_lag_by_obs_weekday"].items()) + ".")
    for s in ("GPRD", "GPRD_THREAT"):
        r_ = rev["revision_rel_first_release_vs_current"][s]
        w(f"- Revision, {s} (first release vs current, relative): mean "
          f"{100 * r_['mean']:+.1f}%, mean absolute {100 * r_['mean_abs']:.1f}%, median "
          f"absolute {100 * r_['median_abs']:.1f}% (n = {r_['n']}). Revisions were not modelled; "
          "the values are from the current vintage.")
    w(f"- Before 2022-02-24: no archive; the rule is applied counterfactually ("
      f"{rep['publication_rule']['before_2022-02-24']}).")
    w(f"- Forward-fill rejected: forward-filling the level series onto the trading calendar would discard "
      f"**{m78.group(1)}%** of the published observations.")
    el = rep["effective_lag"]
    w(f"- Under timestamp alignment, **{100 * el['share_rows_timestamp_uses_unpublished_obs']:.1f}%** of the rows "
      "used an observation not yet published at forecast time.")
    w(f"- Changed features: {rep['n_changed']}/{rep['n_features']}; the first fully populated row is, in both "
      f"versions, {rep['first_fully_valid_row']['publication']}.")
    w("")
    w("### 11b. Effective lag (between trading day t and the date of the GPR observation used)")
    w("")
    cdp, cdt, trp = (el["calendar_days_publication"], el["calendar_days_timestamp"],
                     el["trading_rows_publication"])
    t = pd.DataFrame({
        "measure": ["calendar days, median", "calendar days, mean", "calendar days, max",
                    "trading rows, median / mean / max"],
        "timestamp-aligned": [f"{cdt['median']:.0f}", f"{cdt['mean']:.2f}", f"{cdt['max']:.0f}",
                              "—"],
        "publication-aligned": [f"{cdp['median']:.0f}", f"{cdp['mean']:.2f}", f"{cdp['max']:.0f}",
                                f"{trp['median']:.0f} / {trp['mean']:.2f} / {trp['max']}"]})
    w(md_table(t))
    w("")
    wd = el["calendar_days_by_trading_weekday"]
    order_wd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    w("Publication-aligned, calendar days by trading weekday (median / mean / max): "
      + "; ".join(f"{d} {wd[d]['median']:.0f} / {wd[d]['mean']:.2f} / {wd[d]['max']:.0f}"
                  for d in order_wd) + ".")
    w("")
    w("### 11c. Causality checks")
    w("")
    pi = rep["prefix_invariance"]
    assert all(x["passed"] and x["tolerance"] == 0 for x in pi)
    w("- **Prefix-invariance:** the features were recomputed with the data cut at rows " + " and ".join(
        str(x["cut_row"]) for x in pi) + "; all rows before the cut are identical at **zero "
      "tolerance** to those computed with the full data "
      f"({len(pi)}/{len(pi)} passed).")
    ps_ = pd.DataFrame(rep["publication_sensitivity_test"])
    n_s = len(ps_)
    pub_ok = int(ps_["publication_aligned_unchanged"].sum())
    ctrl = int(ps_["timestamp_aligned_row_changed"].sum())
    w(f"- **Publication sensitivity:** in {n_s} random rows, all GPR observations not yet "
      f"published at that row's date were perturbed ("
      f"{ps_['n_unpublished_perturbed'].min()}–{ps_['n_unpublished_perturbed'].max()} "
      f"observations per row). Publication-aligned arm: **{pub_ok}/{n_s} unchanged**. Control arm (timestamp-"
      f"aligned): **{ctrl}/{n_s} changed**.")
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
    w(f"- **Mechanism of the {n_s - ctrl} unchanged rows in the control arm:** the timestamp "
      "arm uses at row t the observation dated t−1, and the perturbation is applied only to observations "
      "not published up to t−1. The unchanged rows are exactly the rows in which the t−1 observation "
      "had already been published by t−1 ("
      + ", ".join(f"{v} {k}" for k, v in wdc.items()) + "; the t−1 of the Tuesday rows is the "
      "Monday observation published the same day, the Wednesday row is the Labor Day week). The publication "
      f"calendar predicts the control arm's result correctly in **{ok_pred}/{n_s}** rows.")
    w("")
    w("### 11d. Dependence of the two sign tests at h=22")
    w("")
    rel = float(np.corrcoef(d1 / fr["har_x"], d2 / fr["har_x"])[0, 1])
    from scipy import stats as _st
    sp = float(_st.spearmanr(d1, d2)[0])
    hx = float(np.corrcoef(fr["har"], fr["xgboost"])[0, 1])
    agree = int(((d1 > 0) == (d2 > 0)).sum())
    w(f"- Fold-difference vectors (HAR − HAR-X, XGBoost − HAR-X): Pearson "
      f"**{rho22:.2f}**; with relative differences divided by HAR-X's RMSE {rel:.2f}; Spearman "
      f"{sp:.2f}. Years with the same sign: {agree}/{len(d1)}.")
    w(f"- Mechanism: the fold RMSE profiles of HAR and XGBoost are almost identical (correlation "
      f"across folds **{hx:.3f}**). Since both differences contain the same HAR-X RMSE, the two "
      "tests largely test HAR-X against the same yardstick: a year that goes well for HAR-X "
      "shows up as a gain in both comparisons at once, a year that goes badly (2020) as a loss "
      "in both comparisons at once. The correlation between the two difference vectors "
      "is not just a year-level scale difference; it persists in the relative differences and in the ranking.")
    w("")
    w("**0.995 is not a finding on its own.** Fold RMSE scales with the year's volatility "
      "level; hence the fold RMSEs of almost any pair of models are highly "
      "correlated. The table below gives the comparison values for this. The scale-free "
      "measures: correlation via the ratio of fold RMSE to train-mean RMSE "
      "and the daily error correlation. Exploratory, not for inference.")
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
            "horizon": f"h={h}",
            "fold RMSE: HAR~XGB": f"{c.loc['har', 'xgboost']:.3f}",
            "fold RMSE: HAR~past-vol": f"{c.loc['har', 'past_vol']:.3f}",
            "fold RMSE: HAR~train-mean": f"{c.loc['har', 'train_mean']:.3f}",
            "relative: HAR~XGB": f"{rc.loc['har', 'xgboost']:.3f}",
            "relative: HAR~HAR-X": f"{rc.loc['har', 'har_x']:.3f}",
            "relative: HAR~BiLSTM": f"{rc.loc['har', 'bilstm']:.3f}",
            "daily error: HAR~XGB": f"{ec.loc['har', 'xgboost']:.3f}",
            "daily error: HAR-X~XGB": f"{ec.loc['har_x', 'xgboost']:.3f}",
            "daily error: HAR~HAR-X": f"{ec.loc['har', 'har_x']:.3f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("Reading: at h=22 the fold RMSE correlation of HAR with the naive past-volatility baseline is "
      "as high as HAR~XGB. So 0.995 shows not that XGBoost tracks HAR in particular, but "
      "that the years' difficulty level is common to all models; moreover it is "
      "only the h=22 value. The scale-free measures are more informative but "
      "are exploratory and vary by horizon.")
    w("")
    w("### 11e. Sign-test thresholds (exact binomial, 5% two-sided)")
    w("")
    rows = []
    for n in (9, 14, 15):
        kmin = min(k for k in range(n + 1) if k > n / 2 and sign_p(k, n) <= 0.05)
        rows.append({"fold": n, "minimum wins for significance": f"{kmin}/{n}",
                     "p at that threshold": f"{sign_p(kmin, n):.4f}",
                     "p at one fewer": f"{sign_p(kmin - 1, n):.4f}"})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("For n=9 (Section 7c, h=22 high tier) significance is possible, but of the 9 folds at least "
      "8 need the same direction; the observed 6/9 is two folds below this threshold.")
    w("")

    # ---------------- 12. Floor, QLIKE, smearing ----------------
    section_12(w, pr[PUB], A)

    # ---------------- 13. Clark-West, XGB-6 vs HAR, 2026 pointer ----------------
    section_13(w, pr[PUB], F)

    # ---------------- 14. Table 1: descriptive statistics ----------------
    section_14(w)

    # ---------------- 15-16. Appendix A: roll-over, A1, regime analysis, A8 ----------
    section_15(w)
    section_16(w)

    # ---------------- 17. Pooled R2_oos (common reference), H3 residual R2 ----------
    section_17(w, pr[PUB], A)

    # ---------------- 18. XGB-6 vs HAR-X pooled squared-error difference by year ------
    section_18(w, pr[PUB])

    path = OUT_DIR / "paper_numbers_publication_aligned.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Written: {path.name}, primary_family_tests_publication_aligned.csv")


if __name__ == "__main__":
    main()
