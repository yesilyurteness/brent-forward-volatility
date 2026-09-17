"""Attention BiLSTM: is XGBoost's loss to the linear specifications specific to the model?

This script is a hypothesis test. HAR-X beats both XGBoost and GARCH at all four
horizons. The question: is that a result specific to XGBoost, or does it show a general
limit of non-linear modelling on this problem? If the BiLSTM also loses, the second
reading is strengthened; if it wins, the blame lies with XGBoost's functional form.
BOTH OUTCOMES ARE REPORTED.

WHAT IS IDENTICAL TO THE PRIMARY SPECIFICATION
-----------------------------------------------
The fold structure, the embargo, the 2026 partial-year rule, the ORDER of within-fold
preprocessing (log1p -> winsorize at training quantiles -> MinMax fit on training), the
ratio target (log(vol_h) - log(past_vol_h)), Duan smearing from the training residuals,
the metrics and the output schema. The only differences are the model family and the
sequence input.

SEQUENCE CONSTRUCTION AND THE FOLD BOUNDARY
--------------------------------------------
The sequence for row t is the L rows in the interval [t-L+1, t]. Because the features are
already shifted with .shift(1), the sequence carries information only up to t-1; causality
is preserved (Critical Rule 6).

Lookback L = 20 (one trading month, which coincides with the brent_vol20 window). It is
PRE-DECLARED and is not tuned.

Interaction with the fold boundary: the loss occurs ONCE at the start of the series, NOT
per fold. Because every fold starts from the same row in an expanding window, the first
sequence row is fixed: 127 (the first fully populated feature row) + 19 = 146. The effect
on effective observations is negligible (5.7 -> 5.6 in the first fold of h=126).

NO TEST ROW IS LOST. The input window of a test sequence may reach back into the training
period; those are PAST observations at forecast time, not leakage. The embargo removes
only the LABELS -- the features of those rows may still be used as sequence input.

THE ARCHITECTURE RULE (the BiLSTM counterpart of the capacity rule)
--------------------------------------------------------------------
Hyperparameters are NOT SEARCHED (CLAUDE.md, "Model Selection Policy"). The architecture
is chosen from a pre-declared fixed table indexed by the fold's effective number of
independent observations (training rows / h). The choice depends only on the training row
count and h; it never looks at test data.

THERE IS NO EARLY STOPPING -- THE JUSTIFICATION
------------------------------------------------
Early stopping means selecting among 30-60 candidate models (epochs) on a validation set.
In this project we measured exactly that kind of selection and saw it fail: selecting via
validation among 30 Optuna trials gave a result 13% worse than the pre-declared fixed rule
at long horizons, and in 32 of 52 folds the best candidate was less than 5% better than
the median. At long horizons the validation slice contains 5-6.6 effective independent
observations; choosing an epoch on a sample that small is choosing noise. Stopping by
looking at the test year would be a direct violation of Critical Rule 5.

The cost: a fixed epoch count cannot adapt to a fold's convergence speed. This is
mitigated in two ways:
  (1) A cosine learning-rate schedule -- because lr approaches zero in the final epochs,
      the end point is not an arbitrary cut but a low-variance, settled state.
  (2) Regularization is done with dropout + weight decay instead of early stopping; both
      depend on the tier and tighten as the effective sample shrinks.

A rejected alternative: early stopping only at the horizons where validation is adequate
(in practice only h=5). That would create a MIXED PROCEDURE across horizons -- exactly the
reporting problem we ran into with Optuna.

BECAUSE THERE IS NO EARLY STOPPING, TWO DIAGNOSTICS ARE ADDED
--------------------------------------------------------------
(1) CONVERGENCE MONITORING: the training loss is recorded at three points (first epoch,
    middle epoch, final epoch) plus the percentage decline over the final 20% of epochs.
    If the loss is still falling markedly over that last stretch, the fold is
    under-trained; if it has flattened, it has converged.
(2) DEGENERATE PREDICTION CHECK: in the low tier (16 units, dropout 0.4, about 5 effective
    observations) the model's predictions can collapse to a constant. The ratio
    std(prediction)/std(actual) is reported for every fold. If the ratio is below
    DEGENERATE_RATIO_THRESHOLD, the model is effectively producing a constant forecast.
    That is NOT AN ERROR, but it changes the interpretation of the result entirely.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
LAST_TEST_YEAR = 2026
EXPECTED_N_FOLDS = 15
PARTIAL_YEAR = 2026
EXCLUDE_2026_HORIZONS = {66, 126}

WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
LOOKBACK = 20            # PRE-DECLARED, not tuned
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Convergence classification: the percentage decline of the loss over the final 20%
# of epochs.
CONV_TAIL_FRAC = 0.2
CONV_FLAT_PCT = 2.0      # below this, "converged"
CONV_DECLINING_PCT = 10.0  # above this, "under-trained"

# Degenerate prediction threshold: std(prediction)/std(actual)
DEGENERATE_RATIO_THRESHOLD = 0.10

# ===========================================================================
# ROBUSTNESS CHECK: TRAINING TO A CONVERGENCE CRITERION (--convergence-mode)
# ===========================================================================
# In the primary run, 8 of 60 folds came out "under-trained"; all of them were in the
# HIGH tier, and after 60 epochs their training loss was still falling by 10-15%. This
# robustness check trains ONLY the high-tier folds up to a PRE-DECLARED convergence
# criterion:
#
#   "train until the decline in training loss over the final 10% of epochs falls below
#    2%, with an upper bound of 200 epochs."
#
# The criterion is NOT an arbitrary epoch count and derives entirely from the TRAINING
# LOSS; it does not look at test performance. It is therefore not a change made by
# looking at the test set and does not violate Critical Rule 5. The lower bound is the
# declared tier epoch count (so the model is trained at least as long as in the primary
# run).
#
# THE RESULT DOES NOT CHANGE THE PRIMARY SPECIFICATION; it is reported separately.
#
# A NOTE ON ONE DIFFERENCE: in this mode the cosine lr schedule's T_max is MAX_EPOCHS,
# whereas in the primary run it was the declared epoch count. The comparison is
# therefore not "the same schedule with more epochs" but "trained until convergence".
# This difference is written into the report.
MAX_EPOCHS = 200
CONV_STOP_TAIL_FRAC = 0.10
CONV_STOP_PCT = 2.0
CONV_MODE_TIERS = {"yuksek"}

# --- The architecture rule: the BiLSTM counterpart of the capacity rule ----
ARCH_TIERS = [
    (100, "yuksek", {"hidden": 64, "layers": 2, "dropout": 0.2,
                     "weight_decay": 1e-4, "epochs": 60}),
    (30, "orta", {"hidden": 32, "layers": 1, "dropout": 0.3,
                  "weight_decay": 1e-3, "epochs": 40}),
    (0, "dusuk", {"hidden": 16, "layers": 1, "dropout": 0.4,
                  "weight_decay": 1e-2, "epochs": 30}),
]

MODELS = ("bilstm", "train_mean", "past_vol")

_SERIES = ("brent", "ovx", "gprd", "gprd_threat")
LOG_FEATURES = (
    [f"{s}_lag{i}" for s in _SERIES for i in range(1, 6)]
    + [f"{s}_ema{n}" for s in _SERIES for n in (5, 10, 20)]
    + [f"brent_vol{w}" for w in (5, 20, 60, 126)]
    + ["vol_ratio", "vol5_vol60", "vol20_vol126",
       "threat_ratio", "ovx_x_gprd", "ovx_x_gprd_threat"]
)


def select_arch(n_effective):
    for threshold, name, params in ARCH_TIERS:
        if n_effective >= threshold:
            return name, params
    raise RuntimeError("ARCH_TIERS son elemani 0 esikli olmali")


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# Preprocessing -- kept IN SYNC with 03_walkforward.py
# ===========================================================================
def _apply_log(X, log_cols):
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols])
    return X


def fit_preproc(X_train, log_cols):
    Xl = _apply_log(X_train, log_cols)
    lo = Xl.quantile(WINSOR_LOWER)
    hi = Xl.quantile(WINSOR_UPPER)
    Xc = Xl.clip(lower=lo, upper=hi, axis=1)
    return {"lo": lo, "hi": hi, "scaler": MinMaxScaler().fit(Xc),
            "log_cols": log_cols}


def apply_preproc(X, p):
    Xl = _apply_log(X, p["log_cols"])
    Xc = Xl.clip(lower=p["lo"], upper=p["hi"], axis=1)
    return pd.DataFrame(p["scaler"].transform(Xc), index=X.index, columns=X.columns)


def compute_metrics(y_true, y_pred, train_mean):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_true - y_pred
    sse = float(np.sum(err ** 2))
    sst_own = float(np.sum((y_true - y_true.mean()) ** 2))
    sst_train = float(np.sum((y_true - train_mean) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2_oos": float(1.0 - sse / sst_train) if sst_train > 0 else float("nan"),
        "r2": float(1.0 - sse / sst_own) if sst_own > 0 else float("nan"),
        "n": int(len(y_true)),
    }


# ===========================================================================
class AttentionBiLSTM(nn.Module):
    """BiLSTM + single-head additive attention -> context vector -> linear output."""

    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(2 * hidden, 1)
        # nn.LSTM applies no dropout with a single layer; it is applied to the context vector.
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)                      # (B, L, 2H)
        w = torch.softmax(self.attn(o), dim=1)   # (B, L, 1)
        ctx = (o * w).sum(dim=1)                 # (B, 2H)
        return self.out(self.drop(ctx)).squeeze(-1)


def train_model(X_tr, y_tr, arch, seed, adaptive=False):
    """Fixed epochs, cosine lr schedule, NO early stopping.

    If adaptive=True (only in robustness mode and only in the high tier), training
    continues until the declared convergence criterion is met or MAX_EPOCHS is reached.
    The criterion derives from the TRAINING LOSS and does not look at test performance.
    """
    set_all_seeds(seed)
    model = AttentionBiLSTM(X_tr.shape[2], arch["hidden"], arch["layers"],
                            arch["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=arch["weight_decay"])
    max_ep = MAX_EPOCHS if adaptive else arch["epochs"]
    min_ep = arch["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    lossf = nn.MSELoss()
    n = X_tr.shape[0]
    g = torch.Generator().manual_seed(seed)
    history = []
    stop_reason = "sabit_epoch"
    model.train()
    for _ in range(max_ep):
        perm = torch.randperm(n, generator=g)
        total, nb = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = lossf(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        history.append(total / max(nb, 1))
        if adaptive and len(history) >= min_ep:
            k = max(1, int(round(len(history) * (1 - CONV_STOP_TAIL_FRAC))) - 1)
            l0 = history[k]
            drop = 100.0 * (l0 - history[-1]) / l0 if l0 > 0 else 0.0
            if drop < CONV_STOP_PCT:
                stop_reason = "yakinsama_kriteri"
                break
    else:
        if adaptive:
            stop_reason = "ust_sinir_200"
    return model, history, stop_reason


@torch.no_grad()
def predict(model, X, batch=512):
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        out.append(model(X[i:i + batch]).numpy())
    return np.concatenate(out) if out else np.array([])


def classify_convergence(history):
    """Classifies by the percentage decline of the loss over the final 20% of epochs."""
    e = len(history)
    k = max(1, int(round(e * (1 - CONV_TAIL_FRAC))) - 1)
    l80, lend = history[k], history[-1]
    drop = 100.0 * (l80 - lend) / l80 if l80 > 0 else float("nan")
    if not np.isfinite(drop):
        label = "belirsiz"
    elif drop < CONV_FLAT_PCT:
        label = "yakinsadi"
    elif drop < CONV_DECLINING_PCT:
        label = "yakinsiyor"
    else:
        label = "yetersiz_egitilmis"
    return {
        "loss_start": float(history[0]),
        "loss_mid": float(history[e // 2]),
        "loss_final": float(lend),
        "loss_p80": float(l80),
        "tail_drop_pct": float(drop),
        "convergence": label,
    }


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--test-years", type=int, nargs="+", default=None)
    ap.add_argument("--convergence-mode", action="store_true",
                    help="SAGLAMLIK KONTROLU: yuksek kademe fold'lari ilan edilmis "
                         "yakinsama kriterine kadar egitir (son %%10 diliminde kayip "
                         "dususu < %%2, ust sinir 200 epoch). Birincil "
                         "spesifikasyonu DEGISTIRMEZ.")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    t0 = time.time()
    set_all_seeds()
    det_mode = "kapali"
    try:
        torch.use_deterministic_algorithms(True)
        det_mode = "acik"
    except Exception as exc:  # some LSTM kernels do not support it
        det_mode = f"kapali ({type(exc).__name__})"

    feat = pd.read_csv(OUT_DIR / "features.csv", parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()
    assert feat["Date_parsed"].is_monotonic_increasing

    feature_cols = [c for c in feat.columns if c not in ("Date", "Date_parsed")]
    log_cols = [c for c in LOG_FEATURES if c in feature_cols]
    assert len(log_cols) == len(LOG_FEATURES)

    df = feat.copy()
    df["year"] = df["Date_parsed"].dt.year
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))

    first_full = int(df[feature_cols].notna().all(axis=1).idxmax())
    first_seq = first_full + LOOKBACK - 1

    test_years = args.test_years or list(range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1))
    if args.test_years is None:
        assert len(test_years) == EXPECTED_N_FOLDS

    print(f"=== Attention BiLSTM | ufuklar {args.horizons} | "
          f"{len(test_years)} fold | {len(feature_cols)} ozellik ===")
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} | "
          f"deterministik algoritmalar: {det_mode}")
    print(f"Lookback L={LOOKBACK} | ozelliklerin ilk tam-dolu satiri {first_full} "
          f"-> ilk sekans satiri {first_seq} ({df['Date'].iloc[first_seq]})")
    print("Early stopping YOK: sabit epoch + kosinus lr. Gerekce modul docstring'inde.\n")

    pred_frames, fold_records = [], []

    for h in args.horizons:
        print(f"--- h={h} ---")
        d = df.copy()
        d["y"] = tgt[f"target_vol_{h}"].values
        d["pv"] = daily_ret.rolling(h).std().shift(1).values

        for fold_id, test_year in enumerate(test_years, start=1):
            train_idx_all = d.index[d["year"] < test_year]
            test_idx_all = d.index[d["year"] == test_year]
            assert train_idx_all.max() < test_idx_all.min()

            # --- Embargo (identical to XGBoost) ----------------------------
            tr_emb = train_idx_all[:-h]
            assert tr_emb.max() + h < test_idx_all.min(), \
                f"h={h} fold {fold_id}: embargo yetersiz"
            assert not (set(tr_emb) & set(test_idx_all))

            # --- Valid rows: the sequence window must fit completely -------
            need = feature_cols + ["y", "pv"]

            def usable(idx):
                idx = idx[idx >= first_seq]
                ok = d.loc[idx, need].notna().all(axis=1)
                return idx[ok.values]

            tr_rows, te_rows = usable(tr_emb), usable(test_idx_all)
            assert len(tr_rows) > 0 and len(te_rows) > 0

            # --- Preprocessing: fit ONLY on the training rows --------------
            pp = fit_preproc(d.loc[tr_rows, feature_cols], log_cols)
            # The whole series is transformed for sequence history; parameters come from train.
            Z = apply_preproc(d[feature_cols], pp).to_numpy("float32")

            def seqs(rows):
                offs = np.arange(-LOOKBACK + 1, 1)
                idx = rows.to_numpy()[:, None] + offs[None, :]
                return torch.from_numpy(Z[idx])

            X_tr, X_te = seqs(tr_rows), seqs(te_rows)
            y_tr = d.loc[tr_rows, "y"].to_numpy("float64")
            y_te = d.loc[te_rows, "y"].to_numpy("float64")
            pv_tr = d.loc[tr_rows, "pv"].to_numpy("float64")
            pv_te = d.loc[te_rows, "pv"].to_numpy("float64")
            assert (y_tr > 0).all() and (pv_tr > 0).all() and (pv_te > 0).all()

            # --- Ratio target (same as the primary specification) ----------
            y_fit = np.log(y_tr) - np.log(pv_tr)
            t_fit = torch.from_numpy(y_fit.astype("float32"))

            n_eff = len(tr_rows) / h
            tier, arch = select_arch(n_eff)

            adaptive = args.convergence_mode and tier in CONV_MODE_TIERS
            ft = time.time()
            model, history, stop_reason = train_model(X_tr, t_fit, arch, SEED,
                                                      adaptive=adaptive)
            fit_secs = time.time() - ft

            # --- Duan smearing: from training residuals (same as primary spec) -
            resid = y_fit - predict(model, X_tr).astype("float64")
            smearing = float(np.mean(np.exp(resid)))
            pred = pv_te * np.exp(predict(model, X_te).astype("float64")) * smearing

            train_mean = float(y_tr.mean())
            preds = {"bilstm": pred,
                     "train_mean": np.full(len(te_rows), train_mean),
                     "past_vol": pv_te}

            include_main = not (test_year == PARTIAL_YEAR
                                and h in EXCLUDE_2026_HORIZONS)
            fold_metrics = {}
            for name, p in preds.items():
                ok = np.isfinite(p)
                fold_metrics[name] = compute_metrics(y_te[ok], p[ok], train_mean)

            # --- Degenerate prediction check -------------------------------
            sd_pred, sd_true = float(np.std(pred)), float(np.std(y_te))
            sd_ratio = sd_pred / sd_true if sd_true > 0 else float("nan")
            degenerate = bool(sd_ratio < DEGENERATE_RATIO_THRESHOLD)

            conv = classify_convergence(history)

            pred_frames.append(pd.DataFrame({
                "horizon": h, "Date": d.loc[te_rows, "Date"].values,
                "Date_parsed": d.loc[te_rows, "Date_parsed"].values,
                "fold": fold_id, "test_year": test_year,
                "include_in_main": include_main, "y_true": y_te,
                "pred_bilstm": pred, "pred_train_mean": preds["train_mean"],
                "pred_past_vol": pv_te,
            }))

            fold_records.append({
                "horizon": h, "fold": fold_id, "test_year": test_year,
                "include_in_main": include_main,
                "n_train": int(len(tr_rows)), "n_test": int(len(te_rows)),
                "n_train_effective": round(n_eff, 2),
                "arch_tier": tier, **{f"arch_{k}": v for k, v in arch.items()},
                "adaptive": bool(adaptive), "epochs_run": int(len(history)),
                "stop_reason": stop_reason,
                **conv,
                "pred_std": sd_pred, "true_std": sd_true,
                "pred_std_ratio": sd_ratio, "degenerate": degenerate,
                "smearing": smearing, "train_mean_target": train_mean,
                "fit_seconds": round(fit_secs, 2),
                "metrics": fold_metrics,
            })

            flag = " [DEJENERE]" if degenerate else ""
            print(f"  h={h:3d} {test_year} | n={len(tr_rows):4d} "
                  f"({n_eff:6.1f}) | {tier:6s} | kayip "
                  f"{conv['loss_start']:.4f}->{conv['loss_mid']:.4f}->"
                  f"{conv['loss_final']:.4f} ({conv['convergence']}) | "
                  f"sd_or {sd_ratio:.2f} | RMSE "
                  f"{fold_metrics['bilstm']['rmse']:.6f} | "
                  f"{len(history)}ep | {fit_secs:.1f}s{flag}")
        print()

    preds_all = pd.concat(pred_frames, ignore_index=True)
    rows = []
    for r in fold_records:
        for name, m in r["metrics"].items():
            rows.append({"horizon": r["horizon"], "fold": r["fold"],
                         "test_year": r["test_year"],
                         "include_in_main": r["include_in_main"],
                         "model": name, "n_test": m["n"], "rmse": m["rmse"],
                         "mae": m["mae"], "r2_oos": m["r2_oos"], "r2": m["r2"]})
    metrics_all = pd.DataFrame(rows)
    fold_all = pd.DataFrame([{k: v for k, v in r.items() if k != "metrics"}
                             for r in fold_records])

    pd.set_option("display.width", 240)

    # --- (1) Convergence monitoring ----------------------------------------
    print("=== YAKINSAMA IZLEME (early stopping olmadigi icin) ===")
    print(f"Siniflandirma: son %{CONV_TAIL_FRAC*100:.0f} epoch diliminde kaybin "
          f"yuzde dususu. <%{CONV_FLAT_PCT:.0f} yakinsadi, "
          f"<%{CONV_DECLINING_PCT:.0f} yakinsiyor, ustu yetersiz egitilmis.")
    print(pd.crosstab(fold_all["horizon"], fold_all["convergence"]).to_string())
    print("\nKademe bazinda:")
    print(pd.crosstab(fold_all["arch_tier"], fold_all["convergence"]).to_string())
    print("\nEgitim kaybi (ufuk ortalamasi):")
    print(fold_all.groupby("horizon").agg(
        baslangic=("loss_start", "mean"), orta=("loss_mid", "mean"),
        son=("loss_final", "mean"), kuyruk_dusus_pct=("tail_drop_pct", "mean"),
    ).to_string(float_format=lambda v: f"{v:.4f}"))
    bad = fold_all[fold_all["convergence"] == "yetersiz_egitilmis"]
    if len(bad):
        print(f"\nYETERSIZ EGITILMIS {len(bad)}/{len(fold_all)} fold -- son dilimde "
              "kayip hala belirgin dusuyor:")
        print(bad[["horizon", "test_year", "arch_tier", "arch_epochs",
                   "tail_drop_pct"]].to_string(index=False,
                                               float_format=lambda v: f"{v:.2f}"))
    print()

    # --- (2) Degenerate prediction check -----------------------------------
    print("=== DEJENERE TAHMIN KONTROLU ===")
    print(f"std(tahmin)/std(gercek) < {DEGENERATE_RATIO_THRESHOLD} ise model pratikte "
          "SABIT tahmin uretiyordur.")
    print("Bu bir hata degildir ama sonucun yorumunu tamamen degistirir.")
    print(fold_all.groupby(["horizon", "arch_tier"]).agg(
        fold=("pred_std_ratio", "size"), oran_ort=("pred_std_ratio", "mean"),
        oran_min=("pred_std_ratio", "min"), oran_maks=("pred_std_ratio", "max"),
        dejenere=("degenerate", "sum"),
    ).to_string(float_format=lambda v: f"{v:.3f}"))
    deg = fold_all[fold_all["degenerate"]]
    print(f"\nDejenere fold: {len(deg)}/{len(fold_all)}")
    if len(deg):
        print(deg[["horizon", "test_year", "arch_tier", "pred_std",
                   "true_std", "pred_std_ratio"]].to_string(
            index=False, float_format=lambda v: f"{v:.6f}"))
    print()

    # --- (3) Metrics + the nine-model table --------------------------------
    agg_rows = []
    main_m = metrics_all[metrics_all["include_in_main"]]
    for h in args.horizons:
        for name in MODELS:
            sub = main_m[(main_m["horizon"] == h) & (main_m["model"] == name)]
            if len(sub):
                agg_rows.append({"horizon": h, "model": name,
                                 "n_folds": int(len(sub)),
                                 "rmse_fold_mean": float(sub["rmse"].mean()),
                                 "mae_fold_mean": float(sub["mae"].mean()),
                                 "r2_oos_fold_mean": float(sub["r2_oos"].mean())})
    agg_all = pd.DataFrame(agg_rows)
    print("=== BiLSTM METRIKLERI (fold ortalamasi) ===")
    print(agg_all.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()

    combined = None
    wf, bench = OUT_DIR / "wf_metrics_all.csv", OUT_DIR / "bench_metrics_all.csv"
    if wf.exists() and bench.exists():
        parts = [main_m[["horizon", "test_year", "model", "rmse", "mae", "r2_oos"]]]
        w = pd.read_csv(wf)
        w = w[(w["model"] == "xgboost") & w["include_in_main"]]
        parts.append(w[["horizon", "test_year", "model", "rmse", "mae", "r2_oos"]])
        b = pd.read_csv(bench)
        b = b[b["include_in_main"] & ~b["model"].isin(["train_mean", "past_vol"])]
        parts.append(b[["horizon", "test_year", "model", "rmse", "mae", "r2_oos"]])
        combined = pd.concat(parts, ignore_index=True).groupby(
            ["horizon", "model"]).agg(
            n_folds=("rmse", "size"), rmse_fold_mean=("rmse", "mean"),
            mae_fold_mean=("mae", "mean"), r2_oos_fold_mean=("r2_oos", "mean"),
        ).reset_index()
        order = ["har_x_log", "har_x", "har", "har_log", "xgboost", "bilstm",
                 "garch", "train_mean", "past_vol"]
        print("=== TUM MODELLER (fold ortalamasi RMSE) ===")
        print(combined.pivot(index="model", columns="horizon",
                             values="rmse_fold_mean").reindex(order).to_string(
            float_format=lambda v: f"{v:.6f}"))
        print("\nR2_oos:")
        print(combined.pivot(index="model", columns="horizon",
                             values="r2_oos_fold_mean").reindex(order).to_string(
            float_format=lambda v: f"{v:+.3f}"))
        print("\nHIPOTEZ TESTI: dogrusal olmayan modeller HAR-X'e gore (RMSE %, "
              "negatif = dogrusal olmayan daha iyi)")
        rows2 = []
        for h in args.horizons:
            s = combined[combined["horizon"] == h].set_index("model")[
                "rmse_fold_mean"]
            if not {"har_x", "xgboost", "bilstm"} <= set(s.index):
                continue
            rows2.append({"horizon": h,
                          "XGBoost_vs_HAR_X": 100 * (s["xgboost"] / s["har_x"] - 1),
                          "BiLSTM_vs_HAR_X": 100 * (s["bilstm"] / s["har_x"] - 1),
                          "BiLSTM_vs_XGBoost": 100 * (s["bilstm"] / s["xgboost"] - 1)})
        print(pd.DataFrame(rows2).to_string(index=False,
                                            float_format=lambda v: f"{v:+.2f}"))
    print()

    sfx = args.suffix
    preds_all.to_csv(OUT_DIR / f"bilstm_predictions_all{sfx}.csv", index=False)
    metrics_all.to_csv(OUT_DIR / f"bilstm_metrics_all{sfx}.csv", index=False)
    agg_all.to_csv(OUT_DIR / f"bilstm_aggregate_all{sfx}.csv", index=False)
    fold_all.to_csv(OUT_DIR / f"bilstm_folds_all{sfx}.csv", index=False)
    if combined is not None:
        combined.to_csv(OUT_DIR / f"all_models_comparison{sfx}.csv", index=False)

    runtime = time.time() - t0
    summary = {
        "horizons": args.horizons, "test_years": test_years, "seed": SEED,
        "torch_version": torch.__version__,
        "deterministic_algorithms": det_mode,
        "lookback": LOOKBACK,
        "first_full_feature_row": first_full, "first_sequence_row": first_seq,
        "sequence_note": (
            "Lookback kaybi fold basina degil, serinin basinda BIR KEZ olusur "
            f"({LOOKBACK - 1} satir). Test satiri kaybedilmez: test sekanslarinin "
            "girdi penceresi train donemine uzanabilir, bunlar gecmis gozlemlerdir."
        ),
        "convergence_mode": bool(args.convergence_mode),
        "convergence_mode_rule": (
            "SAGLAMLIK KONTROLU: yalnizca yuksek kademe. Son %10'luk epoch diliminde "
            f"egitim kaybi dususu < %{CONV_STOP_PCT} olana kadar egit, ust sinir "
            f"{MAX_EPOCHS} epoch, alt sinir ilan edilmis kademe epoch sayisi. Kriter "
            "EGITIM KAYBINDAN turer, test performansina bakmaz. Birincil "
            "spesifikasyonu degistirmez. Not: bu modda kosinus lr T_max=MAX_EPOCHS'tur, "
            "birincil kosuda ilan edilmis epoch sayisiydi."
        ),
        "early_stopping": False,
        "early_stopping_rationale": (
            "Early stopping validation uzerinden 30-60 aday epoch arasindan secim "
            "yapmaktir. Ayni tur secimi Optuna ile olctuk: uzun ufuklarda onceden "
            "ilan edilmis sabit kuraldan %13 daha kotu, 52 fold'un 32'sinde secim "
            "sinyali %5'in altinda. Validation dilimi uzun ufuklarda 5-6.6 etkin "
            "gozlem icerir. Test yiline bakarak durdurmak Kritik Kural 5 ihlalidir."
        ),
        "arch_tiers": [{"min_effective_obs": t, "name": n, "params": p}
                       for t, n, p in ARCH_TIERS],
        "convergence_thresholds": {"tail_frac": CONV_TAIL_FRAC,
                                   "flat_pct": CONV_FLAT_PCT,
                                   "declining_pct": CONV_DECLINING_PCT},
        "degenerate_threshold": DEGENERATE_RATIO_THRESHOLD,
        "aggregate": agg_rows,
        "folds": fold_records,
        "runtime_seconds": round(runtime, 2),
    }
    with open(OUT_DIR / f"bilstm_summary_all{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"Yazildi: bilstm_predictions_all{sfx}.csv, bilstm_metrics_all{sfx}.csv, "
          f"bilstm_aggregate_all{sfx}.csv, bilstm_folds_all{sfx}.csv")
    if combined is not None:
        print(f"Yazildi: all_models_comparison{sfx}.csv")
    print(f"Rapor  : bilstm_summary_all{sfx}.json")
    print(f"Sure   : {runtime / 60:.1f} dakika")


if __name__ == "__main__":
    main()
