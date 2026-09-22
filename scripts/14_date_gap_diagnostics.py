"""Date-gap diagnostics: do some "daily" returns span more than one trading day?

The dataset is the inner join of four series on their common dates, so a date missing from
any one series drops the whole row, and the log return computed from consecutive rows then
spans more than one trading day. This script measures how often that happens and how many
forecast targets it touches. Nothing is modelled and nothing is refit.

Calendar used as "normal":
  * Weekends.
  * NYSE holidays (OVX is a CBOE index and follows the NYSE calendar), including the
    special closures in the sample: Hurricane Sandy (2012-10-29/30) and the national days
    of mourning for G. H. W. Bush (2018-12-05) and J. Carter (2025-01-09).
  * England & Wales bank holidays are also computed, but only as a check: every one of
    them that falls on a NYSE trading day is PRESENT in the data, which shows that the
    merged series follows the US calendar. A skipped UK-only holiday would have been
    reported separately; none occurs.

A row is a REAL GAP when at least one weekday between it and the previous row is neither a
NYSE holiday nor a UK bank holiday.

Target exposure: the target at row t uses the returns of rows t+1..t+h. A target is
AFFECTED when one of those h returns is a real-gap return. Two counts are reported and
they must not be confused:
  * full sample   : every valid target, 2008-2026, including the 2008-2011 warm-up that is
                    only ever used for training;
  * test folds    : only the targets that enter the main out-of-sample evaluation, test
                    years 2012-2026, with the 2026 fold excluded at h=66 and h=126
                    (CLAUDE.md partial-year rule). This is the count that bears on the
                    reported metrics, and it is the one 13_gap_target_test.py reproduces.

Outputs (outputs/):
  date_gap_distribution.csv, date_gap_rows.csv, date_gap_by_year.csv,
  date_gap_target_exposure.csv, date_gap_diagnostics_summary.json

Runtime: a few seconds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
                                    USLaborDay, USMartinLutherKingJr, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay, nearest_workday)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "veriseti.xlsx"
OUT_DIR = ROOT / "outputs"

HORIZONS = [5, 22, 66, 126]
FIRST_TEST_YEAR = 2012
PARTIAL_YEAR = 2026
EXCLUDE_PARTIAL_HORIZONS = {66, 126}
NYSE_SPECIAL_CLOSURES = ["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"]


def _new_year_observed(d):
    # NYSE does not observe New Year on the preceding Friday when 1 Jan is a Saturday
    return d + pd.Timedelta(days=1) if d.weekday() == 6 else d


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=_new_year_observed),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_holidays(start, end):
    hol = set(NYSEHolidayCalendar().holidays(start, end))
    return hol | set(pd.to_datetime(NYSE_SPECIAL_CLOSURES))


def _last_monday(year, month, day):
    d = pd.Timestamp(year, month, day)
    return d - pd.Timedelta(days=d.weekday())


def uk_bank_holidays(years):
    """England & Wales bank holidays, including the one-off moves and additions."""
    out = set()
    for y in years:
        ny = pd.Timestamp(y, 1, 1)
        out.add(ny + pd.Timedelta(days={5: 2, 6: 1}.get(ny.weekday(), 0)))
        gf = GoodFriday.dates(f"{y}-01-01", f"{y}-12-31")[0]
        out |= {gf, gf + pd.Timedelta(days=3)}
        may1 = pd.Timestamp(y, 5, 1)
        early_may = may1 + pd.Timedelta(days=(7 - may1.weekday()) % 7)
        out.add(pd.Timestamp(2020, 5, 8) if y == 2020 else early_may)
        if y == 2012:
            out |= {pd.Timestamp(2012, 6, 4), pd.Timestamp(2012, 6, 5)}
        elif y == 2022:
            out |= {pd.Timestamp(2022, 6, 2), pd.Timestamp(2022, 6, 3)}
        else:
            out.add(_last_monday(y, 5, 31))
        out.add(_last_monday(y, 8, 31))
        xmas, boxing = pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)
        if xmas.weekday() == 5:
            xmas, boxing = xmas + pd.Timedelta(days=2), boxing + pd.Timedelta(days=2)
        elif xmas.weekday() == 6:
            xmas = xmas + pd.Timedelta(days=2)
        elif xmas.weekday() == 4:
            boxing = boxing + pd.Timedelta(days=2)
        out |= {xmas, boxing}
    out |= set(pd.to_datetime(["2011-04-29", "2022-09-19", "2023-05-08"]))
    return out


def load_dates():
    raw = pd.read_excel(DATA_PATH)
    raw["Date_parsed"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y")
    assert raw["Date_parsed"].is_monotonic_increasing
    assert not raw["Date_parsed"].duplicated().any()
    return raw


def classify_gaps(dates):
    """One row per return (row i vs row i-1). Row 0 has no return and is not listed."""
    dates = pd.DatetimeIndex(dates)
    nyse = nyse_holidays(dates[0], dates[-1])
    uk = uk_bank_holidays(range(dates[0].year, dates[-1].year + 1))
    rows = []
    for i in range(1, len(dates)):
        a, b = dates[i - 1], dates[i]
        missing = pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))
        us_h = [x for x in missing if x in nyse]
        uk_only = [x for x in missing if x not in nyse and x in uk]
        unexpl = [x for x in missing if x not in nyse and x not in uk]
        rows.append({
            "row": i, "prev_date": a.date(), "date": b.date(), "year": b.year,
            "calendar_gap_days": int((b - a).days),
            "n_missing_weekdays": len(missing),
            "n_nyse_holidays": len(us_h),
            "n_uk_only_holidays": len(uk_only),
            "n_unexplained_weekdays": len(unexpl),
            "unexplained_dates": ";".join(str(x.date()) for x in unexpl),
        })
    g = pd.DataFrame(rows)
    g["real_gap"] = g["n_unexplained_weekdays"] > 0
    return g, nyse, uk


def real_gap_flags(g, n_rows):
    """Boolean array over data rows: True where that row's return spans a real gap."""
    flag = np.zeros(n_rows, dtype=bool)
    flag[g["row"].to_numpy()] = g["real_gap"].to_numpy()
    return flag


def window_hits(flag, h):
    """hit[t] = any(flag[t+1 .. t+h]); False where the target is not defined."""
    n = len(flag)
    cs = np.concatenate([[0], np.cumsum(flag)])
    t = np.arange(n)
    valid = t + h <= n - 1
    hit = np.zeros(n, dtype=bool)
    hit[valid] = (cs[t[valid] + h + 1] - cs[t[valid] + 1]) > 0
    return hit, valid


def test_fold_mask(years, h):
    m = years >= FIRST_TEST_YEAR
    if h in EXCLUDE_PARTIAL_HORIZONS:
        m &= years != PARTIAL_YEAR
    return m


def main():
    raw = load_dates()
    dates = raw["Date_parsed"]
    years = dates.dt.year.to_numpy()
    g, nyse, uk = classify_gaps(dates)
    n_ret = len(g)

    # ---- 1. calendar-day distance between consecutive rows
    bucket = pd.cut(g["calendar_gap_days"], [0, 1, 3, 10_000], labels=["1", "2-3", ">=4"])
    dist = (g.assign(bucket=bucket.astype(str))
             .groupby(["bucket", "calendar_gap_days"]).size().rename("n_rows").reset_index())
    dist["pct_of_returns"] = 100 * dist["n_rows"] / n_ret
    dist.to_csv(OUT_DIR / "date_gap_distribution.csv", index=False)

    # ---- 2. rows with skipped weekdays and their classification
    skipped = g[g["n_missing_weekdays"] > 0].copy()
    skipped["category"] = np.where(
        skipped["real_gap"], "unexplained",
        np.where(skipped["n_uk_only_holidays"] > 0, "uk_only_holiday", "nyse_holiday"))
    skipped.to_csv(OUT_DIR / "date_gap_rows.csv", index=False)

    in_data = set(dates)
    nyse_in_data = sorted(str(x.date()) for x in in_data & nyse)
    uk_trading = [x for x in uk if dates.iloc[0] <= x <= dates.iloc[-1]
                  and x.weekday() < 5 and x not in nyse]
    uk_present = sum(x in in_data for x in uk_trading)
    assert not nyse_in_data, f"NYSE holidays present in data: {nyse_in_data}"
    assert (dates.dt.weekday < 5).all()

    # ---- 3. by year
    by_year = g.groupby("year").agg(
        n_returns=("row", "size"),
        n_real_gap_rows=("real_gap", "sum"),
        n_skipped_trading_days=("n_unexplained_weekdays", "sum"),
        n_calendar_gap_ge4=("calendar_gap_days", lambda s: int((s >= 4).sum())),
    ).reset_index()
    by_year.to_csv(OUT_DIR / "date_gap_by_year.csv", index=False)

    # ---- 4. targets whose h-day window contains a real gap
    flag = real_gap_flags(g, len(raw))
    flag_ge4 = np.zeros(len(raw), dtype=bool)
    flag_ge4[g["row"].to_numpy()] = (g["calendar_gap_days"] >= 4).to_numpy()
    exp_rows = []
    for h in HORIZONS:
        hit, valid = window_hits(flag, h)
        hit4, _ = window_hits(flag_ge4, h)
        tm = valid & test_fold_mask(years, h)
        exp_rows.append({
            "horizon": h,
            "n_valid_targets_full": int(valid.sum()),
            "n_affected_full": int(hit.sum()),
            "pct_affected_full": 100 * hit.sum() / valid.sum(),
            "n_test_targets": int(tm.sum()),
            "n_affected_test": int((hit & tm).sum()),
            "pct_affected_test": 100 * (hit & tm).sum() / tm.sum(),
            "n_affected_2017plus": int((hit & valid & (years >= 2017)).sum()),
            "n_window_with_calendar_gap_ge4_full": int(hit4.sum()),
            "pct_window_with_calendar_gap_ge4_full": 100 * hit4.sum() / valid.sum(),
        })
        assert exp_rows[-1]["n_affected_2017plus"] == 0
    exposure = pd.DataFrame(exp_rows)
    exposure.to_csv(OUT_DIR / "date_gap_target_exposure.csv", index=False)

    summary = {
        "n_returns": n_ret,
        "calendar_gap_buckets": {k: int(v) for k, v in bucket.value_counts().sort_index().items()},
        "n_rows_with_missing_weekdays": int(len(skipped)),
        "n_missing_weekdays": int(skipped["n_missing_weekdays"].sum()),
        "n_rows_explained_by_nyse_holidays": int((skipped["category"] == "nyse_holiday").sum()),
        "n_rows_uk_only_holiday": int((skipped["category"] == "uk_only_holiday").sum()),
        "n_real_gap_rows": int(g["real_gap"].sum()),
        "n_skipped_trading_days": int(g["n_unexplained_weekdays"].sum()),
        "real_gap_years": sorted(int(y) for y in g.loc[g["real_gap"], "year"].unique()),
        "largest_gap": {"prev_date": str(g.loc[g["calendar_gap_days"].idxmax(), "prev_date"]),
                        "date": str(g.loc[g["calendar_gap_days"].idxmax(), "date"]),
                        "calendar_days": int(g["calendar_gap_days"].max())},
        "nyse_holidays_present_in_data": nyse_in_data,
        "uk_only_bank_holidays_on_trading_days": len(uk_trading),
        "uk_only_bank_holidays_present_in_data": int(uk_present),
        "target_exposure": exposure.to_dict(orient="records"),
    }
    with open(OUT_DIR / "date_gap_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Tarih boslugu tanisi | {n_ret} getiri ===")
    print(dist.groupby("bucket")["n_rows"].sum().to_string())
    print(f"\nEksik hafta ici gun iceren satir: {len(skipped)} "
          f"({summary['n_missing_weekdays']} gun)")
    print(f"  NYSE tatiliyle aciklanan: {summary['n_rows_explained_by_nyse_holidays']}")
    print(f"  Yalniz UK tatili: {summary['n_rows_uk_only_holiday']}")
    print(f"  GERCEK BOSLUK: {summary['n_real_gap_rows']} satir, "
          f"{summary['n_skipped_trading_days']} islem gunu")
    print(f"UK-yalniz tatiller veride mevcut: {uk_present}/{len(uk_trading)} "
          f"-> veri ABD takvimini izliyor")
    print("\nYillara gore:")
    print(by_year[by_year["n_real_gap_rows"] > 0].to_string(index=False))
    print("\nHedef penceresinde gercek bosluk:")
    print(exposure[["horizon", "n_affected_full", "n_valid_targets_full", "pct_affected_full",
                    "n_affected_test", "n_test_targets", "pct_affected_test"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nYazildi: date_gap_distribution.csv, date_gap_rows.csv, date_gap_by_year.csv, "
          "date_gap_target_exposure.csv, date_gap_diagnostics_summary.json")


if __name__ == "__main__":
    main()
