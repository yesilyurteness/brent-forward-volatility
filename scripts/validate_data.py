"""veriseti.xlsx validation: column/row/date integrity, gaps, and valid targets per horizon.

This script only READS and validates; it does not modify the data and contains no fit()
calls.

EXIT STATUS
-----------
The script exits with a NON-ZERO status (1) when a critical condition fails, so that a
pipeline cannot proceed on broken data:
  * an expected column is missing;
  * a required column (all five) contains a missing value;
  * a date cannot be parsed as DD.MM.YYYY;
  * the dates are not in increasing order;
  * a date is duplicated.
The report JSON is still written, with the errors listed, before the script stops.

WARNINGS (exit status 0)
------------------------
  * extra or reordered columns, a row count other than 4641, a changed OVX record value;
  * the SHA-256 digest of the input file differs from the recorded one, or no digest is
    recorded. The raw data is not in the repository (data/README.md), so the recorded
    digest (data/veriseti.xlsx.sha256, `sha256sum` format) is the only way a replicator
    can confirm that they hold exactly the same file.

NOT AN ERROR: date gaps. The merged series has 40 documented gaps (55 skipped trading
days, all 2008-2016; experiment log Stage 11, 14_date_gap_diagnostics.py). They are a
known, reported property of the data; gaps are listed here but never fail validation.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
SHA_PATH = ROOT / "data" / "veriseti.xlsx.sha256"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

EXPECTED_COLUMNS = ["Date", "Brent_Petrol", "OVX", "GPRD", "GPRD_THREAT"]
EXPECTED_ROWS = 4641
HORIZONS = {"1 hafta": 5, "1 ay": 22, "3 ay": 66, "6 ay": 126}

# 2020-04-21: the day after WTI futures went negative. The OVX all-time record (325.15)
# occurred here -- not a data error but a real market event. The winsorization fit
# (train-only, Critical Rule 2) will treat this point like ANY other outlier; beyond that
# it is not deleted or corrected, because it is the reference point for the 2020 COVID
# period analysis.
OVX_RECORD_DATE = "2020-04-21"
OVX_RECORD_VALUE = 325.15


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_sha256():
    """The digest in data/veriseti.xlsx.sha256 (`<hex>  veriseti.xlsx`), or None."""
    if not SHA_PATH.exists():
        return None
    return SHA_PATH.read_text(encoding="ascii").split()[0].lower()


def critical_checks(df):
    """Critical conditions -> list of error messages (empty = pass). Pure; also returns
    the parsed dates (None if parsing failed). Date gaps are deliberately NOT checked."""
    errors = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing expected columns: {missing}")
    present = [c for c in EXPECTED_COLUMNS if c in df.columns]
    nan = df[present].isna().sum()
    for c, k in nan.items():
        if k:
            errors.append(f"missing values in required column {c}: {int(k)}")
    dates = None
    if "Date" in df.columns and not nan.get("Date", 0):
        try:
            dates = pd.to_datetime(df["Date"].astype(str), format="%d.%m.%Y")
        except (ValueError, TypeError) as exc:
            errors.append("Date is not parseable as DD.MM.YYYY: "
                          + str(exc).splitlines()[0])
    if dates is not None:
        dup = int(dates.duplicated().sum())
        if dup:
            errors.append(f"duplicated dates: {dup} "
                          f"(first {dates[dates.duplicated()].iloc[0].date()})")
        back = dates.diff() < pd.Timedelta(0)
        if back.any():
            i = int(np.flatnonzero(back.values)[0])
            errors.append(f"dates not in increasing order: row {i} "
                          f"({dates.iloc[i].date()}) follows {dates.iloc[i - 1].date()}")
    return errors, dates


def main():
    report = {"errors": [], "warnings": []}
    warn = report["warnings"].append

    # 0) Input file digest (warning only)
    actual_sha = sha256_of(DATA_PATH)
    expected_sha = recorded_sha256()
    rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
    report["sha256"] = {"file": rel(DATA_PATH),
                        "actual": actual_sha, "recorded": expected_sha,
                        "record_file": rel(SHA_PATH),
                        "matches": actual_sha == expected_sha}
    if expected_sha is None:
        warn(f"no recorded SHA-256 digest ({SHA_PATH.name} missing)")
    elif actual_sha != expected_sha:
        warn(f"SHA-256 of {DATA_PATH.name} differs from the recorded digest: this is not "
             "the file the published results were produced from")

    # 1) Re-read and validate
    df = pd.read_excel(DATA_PATH)
    report["columns_match"] = list(df.columns) == EXPECTED_COLUMNS
    report["columns_found"] = list(df.columns)
    report["row_count"] = len(df)
    report["row_count_match"] = len(df) == EXPECTED_ROWS
    errors, dates = critical_checks(df)
    report["errors"].extend(errors)
    if not report["columns_match"] and not any("missing expected" in e for e in errors):
        warn(f"columns are not exactly {EXPECTED_COLUMNS} (extra or reordered)")
    if not report["row_count_match"]:
        warn(f"row count {len(df)} != expected {EXPECTED_ROWS}")
    report["nan_counts"] = df.isna().sum().to_dict()
    report["any_new_nan"] = bool(df.isna().sum().sum() > 0)

    if report["errors"]:
        with open(OUT_DIR / "validate_data_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print("=== DATA VALIDATION FAILED ===")
        for e in report["errors"]:
            print(f"ERROR: {e}")
        for w_ in report["warnings"]:
            print(f"WARNING: {w_}")
        print(f"Report saved: {OUT_DIR / 'validate_data_report.json'}")
        sys.exit(1)

    df["Date_parsed"] = dates
    report["date_monotonic_increasing"] = bool(df["Date_parsed"].is_monotonic_increasing)
    report["date_duplicates"] = int(df["Date_parsed"].duplicated().sum())
    report["date_min"] = str(df["Date_parsed"].min().date())
    report["date_max"] = str(df["Date_parsed"].max().date())

    # 2) Confirm the OVX record value (no deletion or correction, validation only)
    record_row = df.loc[df["Date_parsed"] == pd.Timestamp(OVX_RECORD_DATE)]
    if not record_row.empty:
        actual_value = float(record_row["OVX"].iloc[0])
        report["ovx_record_check"] = {
            "date": OVX_RECORD_DATE,
            "expected_value": OVX_RECORD_VALUE,
            "actual_value": actual_value,
            "matches": abs(actual_value - OVX_RECORD_VALUE) < 0.01,
        }
    else:
        report["ovx_record_check"] = {"date": OVX_RECORD_DATE, "found": False}
    if not report["ovx_record_check"].get("matches", False):
        warn(f"OVX record value on {OVX_RECORD_DATE} not found or changed")

    max_ovx_row = df.loc[df["OVX"].idxmax()]
    report["ovx_global_max"] = {
        "date": str(max_ovx_row["Date_parsed"].date()),
        "value": float(max_ovx_row["OVX"]),
    }

    # 3) Gaps between consecutive dates
    gaps = df["Date_parsed"].diff().dt.days
    report["max_gap_days"] = int(gaps.max())
    long_gaps = pd.DataFrame({
        "gap_start": df["Date_parsed"].shift(1)[gaps > 10],
        "gap_end": df["Date_parsed"][gaps > 10],
        "gap_days": gaps[gaps > 10],
    })
    long_gaps["gap_start"] = long_gaps["gap_start"].dt.strftime("%Y-%m-%d")
    long_gaps["gap_end"] = long_gaps["gap_end"].dt.strftime("%Y-%m-%d")
    report["long_gaps_over_10_days"] = long_gaps.to_dict(orient="records")
    long_gaps.to_csv(OUT_DIR / "date_gaps_over_10_days.csv", index=False)

    # 4) Valid targets per horizon and their distribution across years
    n = len(df)
    years = df["Date_parsed"].dt.year
    horizon_table = []
    year_dist_rows = []
    for label, h in HORIZONS.items():
        n_valid = n - h  # no valid target exists for the last h rows
        valid_years = years.iloc[:n_valid]
        year_counts = valid_years.value_counts().sort_index()
        horizon_table.append({
            "ufuk": label,
            "h": h,
            "toplam_satir": n,
            "gecerli_hedef_sayisi": n_valid,
            "hedefsiz_son_satir": h,
        })
        for yr, cnt in year_counts.items():
            year_dist_rows.append({"ufuk": label, "h": h, "yil": int(yr), "gecerli_gozlem": int(cnt)})

    horizon_df = pd.DataFrame(horizon_table)
    year_dist_df = pd.DataFrame(year_dist_rows)
    pivot = year_dist_df.pivot(index="yil", columns="ufuk", values="gecerli_gozlem").fillna(0).astype(int)
    pivot = pivot[list(HORIZONS.keys())]

    horizon_df.to_csv(OUT_DIR / "horizon_valid_counts.csv", index=False)
    pivot.to_csv(OUT_DIR / "horizon_valid_by_year.csv")

    report["horizon_summary"] = horizon_df.to_dict(orient="records")

    with open(OUT_DIR / "validate_data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # Console output
    print("=== 1) Column / row / date validation ===")
    print(f"Columns match: {report['columns_match']} -> {report['columns_found']}")
    print(f"Row count: {report['row_count']} (expected {EXPECTED_ROWS}) -> matches: {report['row_count_match']}")
    print(f"Dates monotone increasing: {report['date_monotonic_increasing']}, duplicates: {report['date_duplicates']}")
    print(f"Date range: {report['date_min']} - {report['date_max']}")
    print(f"NaN counts: {report['nan_counts']}")
    print()
    print("=== 2) OVX record value (2020-04-21) ===")
    print(report["ovx_record_check"])
    print(f"Global OVX max in the dataset: {report['ovx_global_max']}")
    print()
    print("=== 3) Gaps longer than 10 days ===")
    print(f"Largest gap: {report['max_gap_days']} days")
    if long_gaps.empty:
        print("No gap longer than 10 days.")
    else:
        print(long_gaps.to_string(index=False))
    print()
    print("=== 4) Valid targets per horizon ===")
    print(horizon_df.to_string(index=False))
    print()
    print("Distribution of valid observations by year (including 2026):")
    print(pivot.to_string())
    print()
    print(f"Report saved: {OUT_DIR / 'validate_data_report.json'}")
    print()
    sha = report["sha256"]
    print(f"SHA-256: {sha['actual']} -> matches the recorded digest: {sha['matches']}")
    for w_ in report["warnings"]:
        print(f"WARNING: {w_}")
    print("=== DATA VALIDATION PASSED ===" + (" (with warnings)" if report["warnings"] else ""))


if __name__ == "__main__":
    main()
