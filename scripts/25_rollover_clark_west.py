"""Clark-West (HAR nested in HAR-X) on the roll-over variants (no new model).

DESIGN AND STATUSES were written to the experiment log BEFORE this run (Stage 26,
commit 15ff613); all three variants are reported whatever they show.
  A   primary robustness variant       A2  sensitivity (two rows removed)
  B   sensitivity, h=5 only

This is a replication of the declared Clark-West family (21_clark_west.py, Stage 22) on
variant data. CLAUDE.md: such a replication is not an additional family; it is reported
in Appendix A with its own Holm/BH/BY corrections, within each variant (A and A2: 4
tests; B: 1 test, so its corrections equal the raw p-value), and is never pooled.

SETUP: identical to 21 (the statistic is 21's clark_west function): one-sided, Newey-West
Bartlett L = h-1, HLN factor and t(n-1); corrections on the HLN p-value. Forecasts: the
saved variant predictions of 24_rollover_robustness.py (rollover_predictions), main folds,
pooled. The unmasked (baseline) predictions are asserted to reproduce the saved
clark_west results.

Outputs: outputs/rollover_clark_west{sfx}.csv
Runtime: a few seconds.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import alignment

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cw", ROOT / "scripts" / "21_clark_west.py")
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)

STATUS = {"A": "birincil sağlamlık varyantı", "A2": "duyarlılık (iki satır)",
          "B": "duyarlılık (yalnızca h=5)"}


def main():
    ap = argparse.ArgumentParser()
    alignment.add_argument(ap)
    al = ap.parse_args().gpr_alignment

    P = pd.read_csv(alignment.out("rollover_predictions.csv", al), float_precision="round_trip")
    saved = pd.read_csv(alignment.out("clark_west.csv", al), float_precision="round_trip")
    base = cw.clark_west(P[P["variant"] == "baseline"])
    for c in ("n", "CW", "CW_HLN", "p_normal_one_sided", "p_HLN_one_sided", "p_HLN_holm",
              "p_HLN_bh", "p_HLN_by", "mean_mse_diff", "mean_adjustment"):
        assert np.allclose(base[c], saved[c], rtol=1e-8, atol=1e-15), c
    print("[check] the unmasked forecasts reproduce the stored Clark-West results")

    out = []
    for v in ("A", "A2", "B"):
        g = P[P["variant"] == v]
        hs = sorted(g["horizon"].unique())
        r = cw.clark_west(g, horizons=hs)
        out.append(r.assign(variant=v, status=STATUS[v], n_tests_in_family=len(r)))
    res = pd.concat(out, ignore_index=True)
    path = alignment.out("rollover_clark_west.csv", al)
    res.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(res[["variant", "horizon", "n", "mean_mse_diff", "CW", "CW_HLN",
               "p_normal_one_sided", "p_HLN_one_sided", "p_HLN_holm", "p_HLN_bh",
               "p_HLN_by", "folds_f_positive", "n_folds"]].to_string(index=False))
    print(f"Written: {path.name}")


if __name__ == "__main__":
    main()
