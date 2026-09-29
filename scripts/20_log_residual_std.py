"""Log-residual standard deviation of HAR-log and HAR-X-log, per fold (no new model).

WHY
---
05_benchmarks.py records the Duan smearing coefficient of the two log-log HAR
specifications but not the standard deviation of their training log residuals, which the
paper's Methodology section quotes alongside XGBoost's (03 records resid_log_std). This
script re-estimates the same OLS fits to recover that one number.

NOT A NEW MODEL, AND CHECKED TO BE THE SAME FIT
-----------------------------------------------
The data preparation, fold split, h-day embargo, NaN masks, LOG_FLOOR on the regressors
and the OLS routine are taken from 05_benchmarks.py itself (imported, not copied). Before
anything is written, every fold is asserted to reproduce the saved run BIT FOR BIT:
  * the test predictions equal bench_predictions_all (pred_har_log, pred_har_x_log);
  * the smearing coefficient equals bench_folds_all (har_smearing, har_x_smearing).
If either check fails the script stops; nothing is reported from a fit that differs from
the one evaluated in the paper. OLS is deterministic, so no randomness is involved.

The residual std uses ddof=1, as resid_log_std in 03_walkforward.py.

Only the training slice is used (Critical Rule 2); the test set is read only for the
bit-identity assertion.

Outputs: outputs/log_residual_std{_publication_aligned}.csv
Runtime: a few seconds.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import alignment

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

_spec = importlib.util.spec_from_file_location("benchmarks",
                                              ROOT / "scripts" / "05_benchmarks.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    args = ap.parse_args()
    al = args.gpr_alignment

    feat = pd.read_csv(alignment.features_path(al), parse_dates=["Date_parsed"])
    tgt = pd.read_csv(OUT_DIR / "targets.csv", parse_dates=["Date_parsed"])
    raw = pd.read_excel(bm.DATA_PATH)
    assert len(feat) == len(tgt) == len(raw)
    assert (feat["Date"].values == tgt["Date"].values).all()

    # Same frame as 05_benchmarks.main()
    df = feat[["Date", "Date_parsed"] + bm.HAR_COLS[1:] + bm.HARX_EXTRA].copy()
    df["year"] = df["Date_parsed"].dt.year
    for h in bm.HORIZONS:
        df[f"target_vol_{h}"] = tgt[f"target_vol_{h}"].values
    daily_ret = np.log(raw["Brent_Petrol"] / raw["Brent_Petrol"].shift(1))
    df["har_daily"] = daily_ret.abs().shift(1).values

    saved = pd.read_csv(alignment.out("bench_predictions_all.csv", al),
                        float_precision="round_trip")
    sfold = pd.read_csv(alignment.out("bench_folds_all.csv", al),
                        float_precision="round_trip").set_index(["horizon", "fold"])

    test_years = list(range(bm.FIRST_TEST_YEAR, bm.LAST_TEST_YEAR + 1))
    rows = []
    for fold_id, test_year in enumerate(test_years, start=1):
        train_idx_all = df.index[df["year"] < test_year]
        test_idx_all = df.index[df["year"] == test_year]
        for h in bm.HORIZONS:
            y_col = f"target_vol_{h}"

            def slice_for(idx, cols):
                ok = df.loc[idx, list(cols) + [y_col]].notna().all(axis=1)
                return df.loc[idx[ok.values]]

            tr_emb = train_idx_all[:-h]
            assert tr_emb.max() + h < int(test_idx_all.min())
            tr_har = slice_for(tr_emb, bm.HAR_COLS)
            tr_harx = slice_for(tr_emb, bm.HARX_COLS)
            te = slice_for(test_idx_all, bm.HAR_COLS)
            sp = saved[(saved["horizon"] == h) & (saved["fold"] == fold_id)]
            assert (sp["Date"].values == te["Date"].values).all(), (h, test_year)
            include_main = not (test_year == bm.PARTIAL_YEAR
                                and h in bm.EXCLUDE_2026_HORIZONS)

            for model, tr, cols, s_col in (
                    ("har_log", tr_har, bm.HAR_COLS, "har_smearing"),
                    ("har_x_log", tr_harx, bm.HARX_COLS, "har_x_smearing")):
                Xl_tr = np.log(np.maximum(tr[cols].to_numpy("float64"), bm.LOG_FLOOR))
                Xl_te = np.log(np.maximum(te[cols].to_numpy("float64"), bm.LOG_FLOOR))
                ylog = np.log(tr[y_col].to_numpy("float64"))
                beta = bm.ols_fit(Xl_tr, ylog)
                resid = ylog - bm.ols_predict(beta, Xl_tr)
                smear = float(np.mean(np.exp(resid)))
                pred = np.exp(bm.ols_predict(beta, Xl_te)) * smear
                # Bit-identity with the saved run: stop on any difference.
                assert smear == sfold.loc[(h, fold_id), s_col], \
                    f"{model} h={h} {test_year}: smearing differs from bench_folds_all"
                assert (pred == sp[f"pred_{model}"].to_numpy()).all(), \
                    f"{model} h={h} {test_year}: test predictions differ from the saved run"
                rows.append({"horizon": h, "fold": fold_id, "test_year": test_year,
                             "include_in_main": include_main, "model": model,
                             "n_train": len(tr), "smearing": smear,
                             "resid_log_std": float(np.std(resid, ddof=1))})

    out = pd.DataFrame(rows)
    assert len(out) == len(test_years) * len(bm.HORIZONS) * 2
    path = alignment.out("log_residual_std.csv", al)
    out.to_csv(path, index=False)
    print(f"[kontrol] {len(out)} model x ufuk x fold: yeniden tahmin kayitli test "
          "tahminlerini ve smearing katsayisini bit duzeyinde uretti")
    print(out[out["include_in_main"]].groupby(["model", "horizon"])["resid_log_std"]
          .median().unstack().round(4).to_string())
    print(f"Yazildi: {path.name}")


if __name__ == "__main__":
    main()
