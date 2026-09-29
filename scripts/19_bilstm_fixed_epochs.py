"""BiLSTM, fixed 200 epochs with no stopping rule -- exploratory check designed AFTER seeing
the convergence-mode result (experiment_log.md, Stage 16.4).

WHY
---
In the publication-aligned run the convergence-mode stopping rule ("stop when the loss fell
< 2% over the final 10% of epochs") fired after 62-78 epochs in 14 of 15 high-tier folds
at h=5, so training loss fell only ~1.4x and the check could not test the claim "the
BiLSTM fails through over-fitting, not under-training". This script removes the stopping
rule: every high-tier fold at h=5 and h=22 is trained for exactly MAX_EPOCHS = 200.

STATUS: exploratory and post hoc. It does not change the primary specification, it is
not used for model selection, and nothing in it is chosen by looking at test performance
(the epoch count is the upper bound already declared in 06_attention_bilstm.py).

Either outcome is interpretable:
  * training loss falls substantially and test error does not improve -> the over-fitting
    argument holds without the stopping rule as a confounder;
  * training loss does not fall substantially even in 200 epochs -> the model is
    capacity-limited (at the declared high-tier architecture), not epoch-limited.

SAME TRAJECTORY AS THE CONVERGENCE RUN
--------------------------------------
Everything is imported from 06_attention_bilstm.py (architecture, preprocessing, tiers,
seeds, MAX_EPOCHS). The convergence run already used a cosine schedule with T_max =
MAX_EPOCHS = 200, so with identical seeds and deterministic kernels the first k epochs of
a fixed-200 run ARE the convergence run stopped at k. The script asserts this:
  * fold rows, effective observations and tier equal bilstm_folds_all_conv;
  * the training loss at epoch k equals the convergence run's final loss bit for bit;
  * predictions snapshotted at epoch k equal bilstm_predictions_all_conv bit for bit.
The "k epochs vs 200 epochs" comparison is therefore two points on ONE training path.
(The primary run is a different path: T_max = 60.)

Snapshots are taken in eval mode, which consumes no random numbers (dropout inactive, the
batch permutation uses its own generator), so they do not perturb the trajectory.

Loss measures: `loss_ep<k>` is the epoch-average training MSE in train mode (dropout on;
what the stopping rule saw). `train_mse_eval` is the MSE of the model in eval mode on the
whole training set at that epoch: the cleaner "how well does it fit" number.

Outputs (suffix follows --gpr-alignment):
  outputs/bilstm_fixed200_folds.csv          one row per fold, primary / k / 200 side by side
  outputs/bilstm_fixed200_loss_history.csv   horizon, test_year, epoch, train-mode loss
  outputs/bilstm_fixed200_predictions.csv    test predictions at k and at 200
  outputs/bilstm_fixed200_summary.json

Runtime: ~65-70 minutes for the 24 high-tier folds (publication version), CPU.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
_spec = importlib.util.spec_from_file_location(
    "bilstm06", ROOT / "scripts" / "06_attention_bilstm.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

HORIZONS = [5, 22]
TIER = "yuksek"
N_EPOCHS = B.MAX_EPOCHS          # 200, declared in 06; not chosen here


def train_fixed(X_tr, y_tr, arch, snap_at, on_snapshot):
    """06.train_model(adaptive=True) without the stopping rule, with snapshots.

    The statements and their order mirror train_model exactly, so that the random stream
    and therefore the trajectory are identical."""
    B.set_all_seeds(B.SEED)
    model = B.AttentionBiLSTM(X_tr.shape[2], arch["hidden"], arch["layers"],
                              arch["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=B.LEARNING_RATE,
                           weight_decay=arch["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    lossf = nn.MSELoss()
    n = X_tr.shape[0]
    g = torch.Generator().manual_seed(B.SEED)
    history, snaps = [], {}
    model.train()
    for ep in range(1, N_EPOCHS + 1):
        perm = torch.randperm(n, generator=g)
        total, nb = 0.0, 0
        for i in range(0, n, B.BATCH_SIZE):
            idx = perm[i:i + B.BATCH_SIZE]
            opt.zero_grad()
            loss = lossf(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        history.append(total / max(nb, 1))
        if ep in snap_at:
            snaps[ep] = on_snapshot(model)
            model.train()
    return history, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--test-years", type=int, nargs="+", default=None,
                    help="smoke test on a subset; full run writes the outputs")
    alignment.add_argument(ap)
    args = ap.parse_args()
    al = args.gpr_alignment
    smoke = args.test_years is not None

    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    B.set_all_seeds()

    feat = pd.read_csv(alignment.features_path(al), parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(B.DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in B.LOG_FEATURES if c in feature_cols]
    df = feat.copy()
    df["year"] = df["Date_parsed"].dt.year
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    first_full = int(df[feature_cols].notna().all(axis=1).idxmax())
    first_seq = first_full + B.LOOKBACK - 1

    rd = lambda name: pd.read_csv(alignment.out(name, al), float_precision="round_trip")
    conv_f = rd("bilstm_folds_all_conv.csv").set_index(["horizon", "test_year"])
    conv_p = rd("bilstm_predictions_all_conv.csv")
    conv_m = rd("bilstm_metrics_all_conv.csv")
    prim_f = rd("bilstm_folds_all.csv").set_index(["horizon", "test_year"])
    prim_m = rd("bilstm_metrics_all.csv")
    mkey = lambda m, h, y: m[(m["model"] == "bilstm") & (m["horizon"] == h) &
                             (m["test_year"] == y)].iloc[0]

    print(f"=== BiLSTM fixed {N_EPOCHS} epochs | {al} | horizons {args.horizons} | "
          f"'{TIER}' tier only ===")
    fold_rows, hist_rows, pred_frames = [], [], []
    for h in args.horizons:
        d = df.copy()
        d["y"] = tgt[f"target_vol_{h}"].values
        d["pv"] = daily_ret.rolling(h).std().shift(1).values
        need = feature_cols + ["y", "pv"]
        years = args.test_years or range(B.FIRST_TEST_YEAR, B.LAST_TEST_YEAR + 1)
        for test_year in years:
            cf = conv_f.loc[(h, test_year)]
            if cf["arch_tier"] != TIER:
                continue
            # --- fold construction, identical to 06 -------------------------
            train_idx_all = d.index[d["year"] < test_year]
            test_idx_all = d.index[d["year"] == test_year]
            tr_emb = train_idx_all[:-h]
            assert tr_emb.max() + h < test_idx_all.min()

            def usable(idx):
                idx = idx[idx >= first_seq]
                return idx[d.loc[idx, need].notna().all(axis=1).values]

            tr_rows, te_rows = usable(tr_emb), usable(test_idx_all)
            pp = B.fit_preproc(d.loc[tr_rows, feature_cols], log_cols)   # train only
            Z = B.apply_preproc(d[feature_cols], pp).to_numpy("float32")
            offs = np.arange(-B.LOOKBACK + 1, 1)
            seqs = lambda rows: torch.from_numpy(Z[rows.to_numpy()[:, None] + offs[None, :]])
            X_tr, X_te = seqs(tr_rows), seqs(te_rows)
            y_tr = d.loc[tr_rows, "y"].to_numpy("float64")
            y_te = d.loc[te_rows, "y"].to_numpy("float64")
            pv_tr = d.loc[tr_rows, "pv"].to_numpy("float64")
            pv_te = d.loc[te_rows, "pv"].to_numpy("float64")
            y_fit = np.log(y_tr) - np.log(pv_tr)
            t_fit = torch.from_numpy(y_fit.astype("float32"))
            n_eff = len(tr_rows) / h
            tier, arch = B.select_arch(n_eff)
            # --- same fold as the convergence run -------------------------------
            assert tier == TIER and len(tr_rows) == cf["n_train"] \
                and len(te_rows) == cf["n_test"] \
                and round(n_eff, 2) == cf["n_train_effective"], (h, test_year)
            k = int(cf["epochs_run"])
            train_mean = float(y_tr.mean())

            def snapshot(model):
                fit = B.predict(model, X_tr).astype("float64")
                resid = y_fit - fit
                smearing = float(np.mean(np.exp(resid)))
                pred = pv_te * np.exp(B.predict(model, X_te).astype("float64")) * smearing
                return {"pred": pred, "smearing": smearing,
                        "train_mse_eval": float(np.mean(resid ** 2))}

            ft = time.time()
            history, snaps = train_fixed(X_tr, t_fit, arch, {k, N_EPOCHS}, snapshot)
            fit_secs = time.time() - ft

            # --- replication of the convergence run at epoch k -----------------
            assert history[k - 1] == cf["loss_final"], \
                (h, test_year, history[k - 1], cf["loss_final"])
            cp = conv_p[(conv_p["horizon"] == h) & (conv_p["test_year"] == test_year)]
            assert (cp["Date"].values == d.loc[te_rows, "Date"].values).all()
            assert np.array_equal(cp["pred_bilstm"].to_numpy(), snaps[k]["pred"]), \
                (h, test_year, "snapshot at k != convergence-run predictions")

            include_main = not (test_year == B.PARTIAL_YEAR
                                and h in B.EXCLUDE_2026_HORIZONS)
            m = {e: B.compute_metrics(y_te, snaps[e]["pred"], train_mean)
                 for e in (k, N_EPOCHS)}
            assert m[k]["rmse"] == mkey(conv_m, h, test_year)["rmse"]
            pm = mkey(prim_m, h, test_year)
            sd_true = float(np.std(y_te))
            loss_at = lambda e: history[e - 1]
            row = {
                "horizon": h, "test_year": test_year, "include_in_main": include_main,
                "n_train": len(tr_rows), "n_test": len(te_rows),
                "n_train_effective": round(n_eff, 2), "arch_tier": tier,
                "k_conv": k,
                "loss_primary_ep60_tmax60": float(prim_f.loc[(h, test_year), "loss_final"]),
                "loss_ep1": loss_at(1), "loss_ep60": loss_at(60), "loss_k": loss_at(k),
                "loss_ep100": loss_at(100), "loss_ep150": loss_at(150),
                "loss_ep200": loss_at(N_EPOCHS),
                "loss_ratio_k_to_200": loss_at(k) / loss_at(N_EPOCHS),
                "loss_ratio_primary_to_200": float(prim_f.loc[(h, test_year),
                                                              "loss_final"])
                                             / loss_at(N_EPOCHS),
                "train_mse_eval_k": snaps[k]["train_mse_eval"],
                "train_mse_eval_200": snaps[N_EPOCHS]["train_mse_eval"],
                "train_mse_eval_ratio_k_to_200": snaps[k]["train_mse_eval"]
                                                 / snaps[N_EPOCHS]["train_mse_eval"],
                "smearing_k": snaps[k]["smearing"],
                "smearing_200": snaps[N_EPOCHS]["smearing"],
                "rmse_primary": pm["rmse"], "rmse_k": m[k]["rmse"],
                "rmse_200": m[N_EPOCHS]["rmse"],
                "mae_primary": pm["mae"], "mae_k": m[k]["mae"],
                "mae_200": m[N_EPOCHS]["mae"],
                "r2_oos_k": m[k]["r2_oos"], "r2_oos_200": m[N_EPOCHS]["r2_oos"],
                "pred_std_ratio_primary": float(prim_f.loc[(h, test_year),
                                                           "pred_std_ratio"]),
                "pred_std_ratio_k": float(np.std(snaps[k]["pred"])) / sd_true,
                "pred_std_ratio_200": float(np.std(snaps[N_EPOCHS]["pred"])) / sd_true,
                "fit_seconds": round(fit_secs, 1),
            }
            fold_rows.append(row)
            hist_rows += [{"horizon": h, "test_year": test_year, "epoch": e + 1,
                           "train_loss": v} for e, v in enumerate(history)]
            pred_frames.append(pd.DataFrame({
                "horizon": h, "Date": d.loc[te_rows, "Date"].values,
                "test_year": test_year, "include_in_main": include_main,
                "y_true": y_te, "pred_bilstm_k": snaps[k]["pred"],
                "pred_bilstm_200": snaps[N_EPOCHS]["pred"]}))
            print(f"  h={h:3d} {test_year} | k={k:3d} | loss k {loss_at(k):.4f} -> "
                  f"200 {loss_at(N_EPOCHS):.4f} ({row['loss_ratio_k_to_200']:.2f}x; "
                  f"eval {row['train_mse_eval_ratio_k_to_200']:.2f}x) | RMSE "
                  f"{m[k]['rmse']:.6f} -> {m[N_EPOCHS]['rmse']:.6f} | "
                  f"sd_or {row['pred_std_ratio_k']:.2f} -> {row['pred_std_ratio_200']:.2f}"
                  f" | {fit_secs:.0f}s  [same as the run at k: OK]", flush=True)

    folds = pd.DataFrame(fold_rows)
    agg = []
    for h, g in folds[folds["include_in_main"]].groupby("horizon"):
        agg.append({
            "horizon": h, "n_folds": len(g),
            "k_mean": g["k_conv"].mean(),
            "loss_ratio_k_to_200_median": g["loss_ratio_k_to_200"].median(),
            "loss_ratio_k_to_200_min": g["loss_ratio_k_to_200"].min(),
            "loss_ratio_k_to_200_max": g["loss_ratio_k_to_200"].max(),
            "loss_ratio_primary_to_200_median": g["loss_ratio_primary_to_200"].median(),
            "train_mse_eval_ratio_median": g["train_mse_eval_ratio_k_to_200"].median(),
            "rmse_primary_mean": g["rmse_primary"].mean(),
            "rmse_k_mean": g["rmse_k"].mean(), "rmse_200_mean": g["rmse_200"].mean(),
            "rmse_pct_200_vs_k": 100 * (g["rmse_200"].mean() / g["rmse_k"].mean() - 1),
            "rmse_pct_200_vs_primary": 100 * (g["rmse_200"].mean()
                                              / g["rmse_primary"].mean() - 1),
            "folds_200_better_than_k": int((g["rmse_200"] < g["rmse_k"]).sum()),
            "mae_pct_200_vs_k": 100 * (g["mae_200"].mean() / g["mae_k"].mean() - 1),
            "pred_std_ratio_k_mean": g["pred_std_ratio_k"].mean(),
            "pred_std_ratio_200_mean": g["pred_std_ratio_200"].mean(),
        })
    agg = pd.DataFrame(agg)
    pd.set_option("display.width", 250)
    print("\n=== SUMMARY (main folds; high tier) ===")
    print(agg.round(4).T.to_string())
    print(f"Runtime: {(time.time() - t0) / 60:.1f} minutes")
    if smoke:
        print("Smoke test: no outputs written.")
        return

    sfx = alignment.suffix(al)
    folds.to_csv(OUT_DIR / f"bilstm_fixed200_folds{sfx}.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(OUT_DIR / f"bilstm_fixed200_loss_history{sfx}.csv",
                                   index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        OUT_DIR / f"bilstm_fixed200_predictions{sfx}.csv", index=False)
    summary = {
        "status": "exploratory, post hoc: designed after the convergence-mode result "
                  "(experiment_log.md Stage 16.4); does not change the primary "
                  "specification",
        "gpr_alignment": al, "tier": TIER, "horizons": args.horizons,
        "n_epochs": N_EPOCHS, "lr_schedule": f"cosine, T_max={N_EPOCHS} (same as the "
                                             "convergence run; primary run T_max=60)",
        "replication": "per fold: loss at epoch k and test predictions at epoch k equal "
                       "the convergence run bit for bit (asserted)",
        "loss_definitions": {
            "loss_ep*": "epoch-average training MSE in train mode (dropout on), "
                        "log-ratio target space",
            "train_mse_eval": "MSE on the full training set in eval mode at that epoch"},
        "aggregate": agg.to_dict(orient="records"),
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    with open(OUT_DIR / f"bilstm_fixed200_summary{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Written: bilstm_fixed200_*{sfx}")


if __name__ == "__main__":
    main()
