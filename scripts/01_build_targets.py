"""Builds the forward realized volatility targets for the four horizons (h=5, 22, 66, 126).

Formula (CLAUDE.md):
    daily_ret = np.log(brent / brent.shift(1))
    future = pd.concat([daily_ret.shift(-i) for i in range(1, h + 1)], axis=1)
    target_h = future.std(axis=1)

DEVIATION: if `future.std(axis=1)` is called with the default skipna=True, the standard
deviation is computed even when the full h future days are not available (as long as at
least 2 values exist) -- that produces a target from an incomplete/inconsistent window at
the end of the series. For that reason skipna=False is used here: a row's target is
computed only if ALL h future days are present, otherwise it stays NaN. This is NOT a
leakage risk (it makes the target stricter, it does not use information that must never be
used) but it is not a verbatim copy of the code snippet in CLAUDE.md.

This script only computes the target variables; it contains no fit() calls and no
randomness (no SEED needed). Rows that are NaN are NOT dropped -- per-horizon filtering
happens at the modelling stage.

UNIT: the target is the standard deviation of the DAILY log returns of the next h days. It
is not annualized and it is not cumulative volatility over the horizon. All four horizons
are on the same scale (daily volatility); only the length of the forecast window differs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

EXPECTED_ROWS = 4641
HORIZONS = [5, 22, 66, 126]
EXPECTED_VALID_COUNTS = {5: 4636, 22: 4619, 66: 4575, 126: 4515}


def main():
    df = pd.read_excel(DATA_PATH)
    assert len(df) == EXPECTED_ROWS, (
        f"Kaynak veri satir sayisi degismis: {len(df)} (beklenen {EXPECTED_ROWS})"
    )

    df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    daily_ret = np.log(df["Brent_Petrol"] / df["Brent_Petrol"].shift(1))

    out = df[["Date", "Date_parsed"]].copy()
    valid_counts = {}
    for h in HORIZONS:
        future = pd.concat([daily_ret.shift(-i) for i in range(1, h + 1)], axis=1)
        target = future.std(axis=1, skipna=False)
        out[f"target_vol_{h}"] = target
        valid_counts[h] = int(target.notna().sum())

    # Full validation before anything is written to disk
    check_rows = []
    all_passed = True
    for h in HORIZONS:
        n_valid = valid_counts[h]
        expected_generic = EXPECTED_ROWS - h
        expected_pinned = EXPECTED_VALID_COUNTS[h]
        matches = (n_valid == expected_generic == expected_pinned)
        all_passed = all_passed and matches
        check_rows.append({
            "h": h,
            "valid_count": n_valid,
            "expected_generic": expected_generic,
            "expected_pinned": expected_pinned,
            "matches": matches,
        })

    check_df = pd.DataFrame(check_rows)
    print("=== Ufuk basina gecerli hedef sayisi dogrulamasi ===")
    print(check_df.to_string(index=False))
    print()

    for row in check_rows:
        assert row["matches"], (
            f"h={row['h']}: gecerli hedef sayisi {row['valid_count']}, "
            f"beklenen {row['expected_pinned']} (genel kural: {row['expected_generic']})"
        )

    # Distribution check: min, median, max, number of negative values
    dist_rows = []
    for h in HORIZONS:
        col = out[f"target_vol_{h}"]
        n_negative = int((col < 0).sum())
        dist_rows.append({
            "h": h,
            "min": float(col.min()),
            "median": float(col.median()),
            "max": float(col.max()),
            "n_negative": n_negative,
        })

    dist_df = pd.DataFrame(dist_rows)
    print("=== Ufuk basina dagilim kontrolu ===")
    print(dist_df.to_string(index=False))
    print()

    for row in dist_rows:
        assert row["n_negative"] == 0, (
            f"h={row['h']}: {row['n_negative']} negatif standart sapma degeri bulundu -- "
            "standart sapma negatif olamaz, hesaplamada hata var."
        )

    # All checks passed -> write
    out.to_csv(OUT_DIR / "targets.csv", index=False)

    report = {
        "row_count": len(out),
        "date_min": str(out["Date_parsed"].min().date()),
        "date_max": str(out["Date_parsed"].max().date()),
        "horizons": {
            str(row["h"]): {
                "valid_count": row["valid_count"],
                "expected_count": row["expected_pinned"],
                "matches_expected": row["matches"],
            }
            for row in check_rows
        },
        "distribution": {
            str(row["h"]): {
                "min": row["min"],
                "median": row["median"],
                "max": row["max"],
                "n_negative": row["n_negative"],
            }
            for row in dist_rows
        },
        "all_checks_passed": all_passed,
    }
    with open(OUT_DIR / "build_targets_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Yazildi: {OUT_DIR / 'targets.csv'} ({len(out)} satir)")
    print(f"Rapor: {OUT_DIR / 'build_targets_report.json'}")


if __name__ == "__main__":
    main()
