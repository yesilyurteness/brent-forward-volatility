"""Publication calendar of the daily GPR index: for every calendar day d, the date p(d)
on which an observation dated d first became publicly available.

Imported by 02_build_features.py; running it directly writes the calendar to
outputs/gpr_publication_calendar.csv for inspection.

HOW p(d) IS DEFINED
-------------------
* d >= 2022-02-24 (EMPIRICAL): the date of the first archived vintage that contains d,
  read from outputs/gpr_first_release.csv (built by 16_gpr_vintages.py from the authors'
  vintage archive). This captures the weekly Monday releases, the month-start releases,
  the extra releases (e.g. near-daily in Feb-Mar 2022) and the late ones (skipped or
  delayed weeks) exactly as they happened.
* d <  2022-02-24 (RULE, COUNTERFACTUAL): the first Monday on or after d, moved to the
  next business day if it is a US federal holiday. This is the rule stated on the
  authors' page ("The daily data are updated every Monday ... If the first day of the
  month or week falls on a federal holiday, data updates will take place the next
  business day") and verified on 279/289 vintages: a file released on day D contains
  observations through D itself. Month-start and extra releases are NOT modelled here,
  which errs on the side of a longer lag.
  COUNTERFACTUAL: the daily GPR index became public with Caldara and Iacoviello (2022);
  before that there is no "value as published on that day". The rule applies today's
  publication regime to the past.
* Monotonicity: every file contains the full history, so an observation is available
  no later than any later observation: p(d) = min_{d' >= d} p(d'). This also makes the
  seam between the rule and the empirical mapping at 2022-02-24 consistent.

p(d) depends only on the calendar date and the fixed vintage metadata, never on the
data values, so it can be applied to any prefix of the data without changing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIRST_RELEASE_PATH = OUT_DIR / "gpr_first_release.csv"

VINTAGE_START = pd.Timestamp("2022-02-24")
CALENDAR_START = pd.Timestamp("2007-01-01")


def _rule_dates(days):
    """First Monday on or after each day, rolled past federal holidays and weekends."""
    days = pd.DatetimeIndex(days)
    hol = set(USFederalHolidayCalendar().holidays(days.min(),
                                                  days.max() + pd.Timedelta(days=30)))
    r = days + pd.to_timedelta((7 - days.dayofweek) % 7, unit="D")
    r = pd.Series(r)
    for _ in range(10):
        bad = r.isin(hol) | (r.dt.dayofweek >= 5)
        if not bad.any():
            break
        r[bad] = r[bad] + pd.Timedelta(days=1)
    return pd.DatetimeIndex(r)


def publication_calendar():
    """DataFrame indexed by calendar day: publication_date and source (rule/empirical)."""
    fr = pd.read_csv(FIRST_RELEASE_PATH, parse_dates=["date", "vintage_date"],
                     index_col="date")["vintage_date"]
    days = pd.date_range(CALENDAR_START, fr.index.max(), freq="D")
    emp_days = days[days >= VINTAGE_START]
    missing = emp_days.difference(fr.index)
    assert missing.empty, f"first-release table has gaps: {missing[:5]}"

    pub = pd.Series(_rule_dates(days), index=days)
    pub[emp_days] = fr.reindex(emp_days).values
    source = pd.Series(np.where(days >= VINTAGE_START, "empirical", "rule"), index=days)
    # Every file contains the full history -> monotone from the end.
    pub = pd.Series(np.minimum.accumulate(pub.values[::-1])[::-1], index=days)
    assert (pub.index <= pub.values).all(), "an observation published before its date"
    return pd.DataFrame({"publication_date": pub, "source": source})


def publication_dates(dates):
    """p(d) for each date in `dates` (a Series/array of Timestamps), same order."""
    cal = publication_calendar()["publication_date"]
    dates = pd.DatetimeIndex(dates)
    outside = dates.difference(cal.index)
    assert outside.empty, f"dates outside the publication calendar: {outside[:5]}"
    return pd.Series(cal.reindex(dates).values, index=dates)


if __name__ == "__main__":
    cal = publication_calendar()
    cal["lag_days"] = (cal["publication_date"] - cal.index.to_series()).dt.days
    cal.to_csv(OUT_DIR / "gpr_publication_calendar.csv", index_label="date",
               date_format="%Y-%m-%d")
    print(cal.groupby("source")["lag_days"].describe())
