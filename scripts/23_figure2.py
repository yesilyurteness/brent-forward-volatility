"""Figure 2 data and figures (no model, no fit()).

(a) Main text: fold-level RMSE ratio XGBoost / HAR-X, four horizons, main folds only (the
    2026 fold at h=66 and h=126 is excluded, as in the main metric; its rows are kept in
    the CSV with include_in_main = False). Ratio > 1 means XGBoost has the larger RMSE,
    i.e. XGBoost is worse in that fold; the axis label says so.
(b) Appendix: SHAP group attribution of the primary XGBoost (share of mean |SHAP|),
    with each group's feature count next to its name, since a group's share grows with
    the number of features in it.

The per-fold RMSEs come from hybrid_metrics_all (the source of the paper's main table);
the count of folds with ratio > 1 is asserted to equal the HAR-X fold wins recorded in
primary_family_tests (18_paper_numbers.py). SHAP shares come from shap_group_importance;
feature counts from shap_feature_importance.

Style: vector output (PDF and SVG, text kept as text), grayscale only so the figure
reads in black-and-white print, Arial (Helvetica metrics).

Outputs: outputs/figure2a_rmse_ratio{sfx}.csv/.pdf/.svg,
         outputs/figure2b_shap_groups{sfx}.csv/.pdf/.svg
Runtime: a few seconds.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import alignment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [5, 22, 66, 126]
GROUP_LABEL = {"brent_vol": "Brent realized volatility", "gpr": "GPR",
               "ovx": "OVX", "brent_fiyat": "Brent price and returns",
               "etkilesim": "Interaction terms", "takvim": "Calendar"}
INK, MUTED, GRID = "#1a1a1a", "#595959", "#d9d9d9"
BAR = "#4d4d4d"
# Ordinal gray ramp for the four horizons (short -> long = light -> dark)
H_GRAY = {5: "#c8c8c8", 22: "#969696", 66: "#636363", 126: "#252525"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "brent-volatility-figure2",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, stem):
    # No creation date and a fixed SVG id salt, so a re-run gives byte-identical files.
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


def figure_a(al):
    hm = pd.read_csv(alignment.out("hybrid_metrics_all.csv", al), float_precision="round_trip")
    x = hm[hm["model"].isin(["xgboost", "har_x"])].pivot_table(
        index=["horizon", "test_year", "include_in_main"], columns="model",
        values="rmse").reset_index()
    x = x.rename(columns={"xgboost": "rmse_xgboost", "har_x": "rmse_har_x"})
    x["ratio_xgboost_over_har_x"] = x["rmse_xgboost"] / x["rmse_har_x"]
    x = x.sort_values(["horizon", "test_year"])
    csv = alignment.out("figure2a_rmse_ratio.csv", al)
    x.to_csv(csv, index=False)

    pf = pd.read_csv(alignment.out("primary_family_tests.csv", al))
    pf = pf[pf["karşılaştırma"].str.contains("xgboost")].set_index("horizon")
    main = x[x["include_in_main"]]
    for h in HORIZONS:
        g = main[main["horizon"] == h]
        assert int((g["ratio_xgboost_over_har_x"] > 1).sum()) == int(pf.loc[h, "harx_fold_wins"])
        assert len(g) == int(pf.loc[h, "n_folds"])

    years = sorted(main["test_year"].unique())
    lo = min(0.8, main["ratio_xgboost_over_har_x"].min() - 0.05)
    hi = max(1.2, main["ratio_xgboost_over_har_x"].max() + 0.3)  # room for the note
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), HORIZONS):
        g = main[main["horizon"] == h].set_index("test_year")["ratio_xgboost_over_har_x"]
        ax.bar(g.index, g.values - 1, bottom=1, width=0.62, color=BAR, linewidth=0,
               zorder=2)
        ax.axhline(1, color=INK, linewidth=0.9, zorder=3)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        k = int((g > 1).sum())
        ax.set_title(f"h = {h}", loc="left", fontweight="bold", color=INK)
        note = f"XGBoost worse in {k} of {len(g)} folds"
        if len(g) < len(years):
            note += "\n2026 fold excluded (partial year)"
        ax.text(0.99, 0.98, note, transform=ax.transAxes, ha="right", va="top",
                color=MUTED, fontsize=7.5, linespacing=1.3)
        ax.set_ylim(lo, hi)
        ax.set_xticks(years)
        ax.set_xticklabels([f"'{str(y)[2:]}" for y in years])
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE ratio, XGBoost / HAR-X\n(above 1: XGBoost worse)")
    for ax in axes[1, :]:
        ax.set_xlabel("Test year (fold)")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, csv.with_suffix(""))
    return csv


def figure_b(al):
    gi = pd.read_csv(alignment.out("shap_group_importance.csv", al))
    fi = pd.read_csv(alignment.out("shap_feature_importance.csv", al))
    n = fi[fi["horizon"] == HORIZONS[0]].groupby("grup")["ozellik"].nunique()
    assert all((fi[fi["horizon"] == h].groupby("grup")["ozellik"].nunique() == n).all()
               for h in HORIZONS)
    assert int(n.sum()) == 65
    w = gi.pivot(index="grup", columns="horizon", values="pay_pct")[HORIZONS]
    w = w.loc[w.mean(axis=1).sort_values(ascending=True).index]  # largest at the top
    out = pd.DataFrame({"group": w.index, "label": [GROUP_LABEL[g] for g in w.index],
                        "n_features": [int(n[g]) for g in w.index]})
    for h in HORIZONS:
        out[f"share_pct_h{h}"] = w[h].values
    csv = alignment.out("figure2b_shap_groups.csv", al)
    out.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(len(w))
    bh = 0.19
    for i, h in enumerate(HORIZONS):
        ax.barh(y + (1.5 - i) * bh, w[h].values, height=bh * 0.86, color=H_GRAY[h],
                linewidth=0, label=f"h = {h}", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{GROUP_LABEL[g]} ({int(n[g])} features)" for g in w.index])
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Share of mean |SHAP| (%), primary XGBoost")
    ax.set_xlim(0, max(50, float(w.values.max()) + 3))
    ax.legend(title="Horizon", frameon=False, loc="lower right", title_fontsize=7.5)
    fig.tight_layout()
    save(fig, csv.with_suffix(""))
    return csv


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment
    for p in (figure_a(al), figure_b(al)):
        print(f"Written: {p.name} + .pdf/.svg")


if __name__ == "__main__":
    main()
