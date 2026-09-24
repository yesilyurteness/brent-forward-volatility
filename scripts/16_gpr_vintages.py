"""GPR publication regime: vintage metadata, first-release series and revision check.

PURPOSE
-------
The daily GPR index (Caldara and Iacoviello) is not observed in real time the way OVX or
Brent are: it is published in batches (weekly, plus a monthly update) and later vintages
revise past values. This script establishes, from the authors' own vintage archive, (i)
the publication rule -- which observation dates each published file contains -- (ii) the
effective publication lag per observation date, and (iii) whether and how much past values
are revised. It does NOT build model features; the availability-aligned GPR features are
built in the feature pipeline from the rule established here.

SOURCES (access date 2026-09-24)
--------------------------------
  Page     : https://www.matteoiacoviello.com/gpr.htm
             "The daily data are updated every Monday.* ... *If the first day of the month
             or week falls on a federal holiday, data updates will take place the next
             business day."
  Current  : https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta
  Vintages : https://github.com/iacoviel/iacoviel.github.io/tree/master/gpr_archive_files
             (data_gpr_daily_recent_YYYYMMDD.dta, YYYYMMDD = update date; the archive
             starts on 2022-02-24)
Stata files are used instead of .xls because they carry identical data and need no extra
dependency (pandas.read_stata).

COVERAGE CAVEAT
---------------
Vintages exist only from 2022-02-24. The first-release series is therefore defined only
from that date on. For 2008-2021 there is no "value as published on that day": the daily
GPR index became public with Caldara and Iacoviello (2022), so any availability alignment
for that period is COUNTERFACTUAL -- it assumes today's publication regime had applied in
the past.

Inputs : downloaded vintages (cached in data/gpr_vintages/vintages/, git-ignored)
Outputs: outputs/gpr_vintage_meta.csv, outputs/gpr_first_release.csv,
         outputs/gpr_revision_summary.json

Runtime: dominated by the first download (288 files, ~400 MB, ~5-10 min); a re-run with
the cache in place takes under a minute. Pass --cleanup to delete the cache at the end.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CACHE = ROOT / "data" / "gpr_vintages" / "vintages"

ACCESS_DATE = "2026-09-24"
LISTING_URL = ("https://api.github.com/repos/iacoviel/iacoviel.github.io/contents/"
               "gpr_archive_files")
RAW_URL = ("https://raw.githubusercontent.com/iacoviel/iacoviel.github.io/master/"
           "gpr_archive_files/{name}")
CURRENT_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"

SERIES = ["GPRD", "GPRD_THREAT"]
SAMPLE_START = pd.Timestamp("2008-01-01")


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            return
        except Exception as e:  # network hiccup: back off and retry
            if attempt == 3:
                raise
            print(f"  retry {dest.name}: {e}")
            time.sleep(2 ** attempt)


def read_vintage(path):
    df = pd.read_stata(path)
    df["date"] = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d")
    return df.set_index("date")[SERIES].astype(float).sort_index()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    listing_path = CACHE / f"archive_listing_{ACCESS_DATE}.json"
    fetch(LISTING_URL, listing_path)
    names = sorted(x["name"] for x in json.loads(listing_path.read_text())
                   if x["name"].startswith("data_gpr_daily_recent_")
                   and x["name"].endswith(".dta"))
    print(f"{len(names)} daily vintages in the archive")

    for i, n in enumerate(names, 1):
        fetch(RAW_URL.format(name=n), CACHE / n)
        if i % 25 == 0:
            print(f"  downloaded {i}/{len(names)}")
    current_path = CACHE / f"current_accessed_{ACCESS_DATE}.dta"
    fetch(CURRENT_URL, current_path)

    current = read_vintage(current_path)
    vint = {pd.Timestamp(n[-12:-4]): read_vintage(CACHE / n) for n in names}
    vdates = sorted(vint)

    # ---- (1) vintage metadata --------------------------------------------------------
    meta, prev_last = [], None
    for vd in vdates:
        v = vint[vd]
        last = v.index.max()
        common = v.index[(v.index >= SAMPLE_START)].intersection(current.index)
        rel = (v.loc[common, "GPRD"] - current.loc[common, "GPRD"]).abs() \
            / current.loc[common, "GPRD"]
        meta.append({
            "vintage_date": vd.date(),
            "vintage_weekday": vd.day_name(),
            "last_obs_date": last.date(),
            "vintage_minus_last_obs_days": (vd - last).days,
            "n_new_obs": (v.index > prev_last).sum() if prev_last is not None else np.nan,
            "days_since_prev_vintage": (vd - meta[-1]["_vd"]).days if meta else np.nan,
            "n_rows": len(v),
            "gprd_mean_abs_rel_diff_vs_current_2008on": rel.mean(),
            "_vd": vd,
        })
        prev_last = last
    meta = pd.DataFrame(meta).drop(columns="_vd")
    meta.to_csv(OUT / "gpr_vintage_meta.csv", index=False)

    # ---- (2) first-release series ------------------------------------------------------
    # For each date d >= first vintage date: the value in the earliest vintage containing d.
    rows = []
    first_vd = vdates[0]
    seen = pd.Timestamp.min
    for vd in vdates:
        v = vint[vd]
        new = v.index[(v.index > seen) & (v.index >= first_vd)]
        for d in new:
            rows.append({"date": d, "vintage_date": vd, **{f"{s}_first": v.at[d, s]
                                                           for s in SERIES}})
        seen = max(seen, v.index.max())
    fr = pd.DataFrame(rows).set_index("date")
    fr["obs_weekday"] = fr.index.day_name()
    fr["publication_lag_days"] = (fr["vintage_date"] - fr.index).dt.days
    for s in SERIES:
        fr[f"{s}_current"] = current[s].reindex(fr.index)
        fr[f"{s}_revision"] = fr[f"{s}_current"] - fr[f"{s}_first"]
    fr.to_csv(OUT / "gpr_first_release.csv", date_format="%Y-%m-%d")

    # ---- (3) revision summary ------------------------------------------------------------
    def desc(x):
        x = x.dropna()
        return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
                "mean_abs": float(x.abs().mean()), "median_abs": float(x.abs().median()),
                "max_abs": float(x.abs().max())}

    lag = fr["publication_lag_days"]
    summary = {
        "access_date": ACCESS_DATE,
        "n_vintages": len(vdates),
        "first_vintage": str(vdates[0].date()), "last_vintage": str(vdates[-1].date()),
        "vintage_weekday_counts": meta["vintage_weekday"].value_counts().to_dict(),
        "vintage_minus_last_obs_days_counts":
            meta["vintage_minus_last_obs_days"].value_counts().sort_index()
            .rename(index=str).to_dict(),
        "publication_lag_days": {"min": int(lag.min()), "median": float(lag.median()),
                                 "mean": float(lag.mean()), "max": int(lag.max())},
        "publication_lag_by_obs_weekday":
            fr.groupby("obs_weekday")["publication_lag_days"]
              .agg(["min", "median", "mean", "max"]).round(2).to_dict("index"),
        "publication_lag_by_year":
            fr.groupby(fr.index.year)["publication_lag_days"]
              .agg(["min", "median", "mean", "max"]).round(2).rename(index=str)
              .to_dict("index"),
        "revision_first_release_vs_current": {s: desc(fr[f"{s}_revision"]) for s in SERIES},
        "revision_rel_first_release_vs_current": {
            s: desc(fr[f"{s}_revision"] / fr[f"{s}_current"]) for s in SERIES},
        "revision_by_publication_lag": {
            str(k): desc(g["GPRD_revision"] / g["GPRD_current"])
            for k, g in fr.groupby("publication_lag_days")},
    }
    (OUT / "gpr_revision_summary.json").write_text(json.dumps(summary, indent=2,
                                                                default=str))
    print(json.dumps(summary, indent=2, default=str))

    if "--cleanup" in sys.argv:
        for p in CACHE.iterdir():
            p.unlink()
        CACHE.rmdir()
        print("cache deleted")


if __name__ == "__main__":
    main()
