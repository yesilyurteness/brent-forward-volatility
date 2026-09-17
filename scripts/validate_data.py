"""veriseti.xlsx validation: column/row/date integrity, gaps, and valid targets per horizon.

This script only READS and validates; it does not modify the data and contains no fit()
calls.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
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


def main():
    report = {}

    # 1) Re-read and validate
    df = pd.read_excel(DATA_PATH)
    report["columns_match"] = list(df.columns) == EXPECTED_COLUMNS
    report["columns_found"] = list(df.columns)
    report["row_count"] = len(df)
    report["row_count_match"] = len(df) == EXPECTED_ROWS

    df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    report["date_monotonic_increasing"] = bool(df["Date_parsed"].is_monotonic_increasing)
    report["date_duplicates"] = int(df["Date_parsed"].duplicated().sum())
    report["date_min"] = str(df["Date_parsed"].min().date())
    report["date_max"] = str(df["Date_parsed"].max().date())

    nan_counts = df.drop(columns=["Date_parsed"]).isna().sum()
    report["nan_counts"] = nan_counts.to_dict()
    report["any_new_nan"] = bool(nan_counts.sum() > 0)

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
    print("=== 1) Sutun / satir / tarih dogrulama ===")
    print(f"Sutunlar eslesiyor mu: {report['columns_match']} -> {report['columns_found']}")
    print(f"Satir sayisi: {report['row_count']} (beklenen {EXPECTED_ROWS}) -> eslesiyor: {report['row_count_match']}")
    print(f"Tarih monoton artan: {report['date_monotonic_increasing']}, tekrar: {report['date_duplicates']}")
    print(f"Tarih araligi: {report['date_min']} - {report['date_max']}")
    print(f"NaN sayilari: {report['nan_counts']}")
    print()
    print("=== 2) OVX rekor degeri (2020-04-21) ===")
    print(report["ovx_record_check"])
    print(f"Veri setindeki global OVX max: {report['ovx_global_max']}")
    print()
    print("=== 3) 10 gunden uzun bosluklar ===")
    print(f"En buyuk bosluk: {report['max_gap_days']} gun")
    if long_gaps.empty:
        print("10 gunden uzun bosluk yok.")
    else:
        print(long_gaps.to_string(index=False))
    print()
    print("=== 4) Ufuk basina gecerli hedef sayisi ===")
    print(horizon_df.to_string(index=False))
    print()
    print("Yillara gore gecerli gozlem dagilimi (2026 dahil):")
    print(pivot.to_string())
    print()
    print(f"Rapor kaydedildi: {OUT_DIR / 'validate_data_report.json'}")


if __name__ == "__main__":
    main()
