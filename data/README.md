# Dataset Documentation

The raw data file in this folder (`veriseti.xlsx`) is **not included in the repository.**
The Brent and OVX series were downloaded from Yahoo Finance, whose terms of use do not
permit redistribution of the data. The steps below are sufficient to rebuild the file from
scratch, identically.

Once the file is in place, run `python scripts/validate_data.py`. That script verifies the
row count, column names, date integrity and the number of valid targets per horizon, and
writes `outputs/validate_data_report.json`.

---

## 1. File specification

| Property | Value |
| --- | --- |
| Path | `data/veriseti.xlsx` |
| Format | Excel (`.xlsx`), single sheet, first row is the header |
| Row count | **4641** (excluding the header) |
| Date range | 02.01.2008 to 01.09.2026 |
| Frequency | Daily, trading days only (not calendar days) |
| Missing values | **None** in any column (0 NaN in every column) |
| Date ordering | Strictly increasing, no duplicate dates |

### Columns

The column names and their order must match the table below exactly;
`scripts/validate_data.py` and `scripts/01_build_targets.py` check this directly.

| # | Column name | Type | Unit / description |
| --- | --- | --- | --- |
| 1 | `Date` | text | `DD.MM.YYYY` (for example `02.01.2008`). NOT an Excel date type, but text. |
| 2 | `Brent_Petrol` | float | Brent crude daily closing price, USD/barrel |
| 3 | `OVX` | float | CBOE Crude Oil ETF Volatility Index, daily close (index points) |
| 4 | `GPRD` | float | Daily geopolitical risk index (overall) |
| 5 | `GPRD_THREAT` | float | Daily geopolitical risk index, threat component |

The `Date` column is read as **text** and parsed in code with
`pd.to_datetime(df["Date"], format="%d.%m.%Y")`. If Excel is allowed to auto-format the
dates, this parsing breaks.

---

## 2. Variable sources

### `Brent_Petrol` — Yahoo Finance, ticker `BZ=F`

- Source: <https://finance.yahoo.com/quote/BZ%3DF/history>
- Series: Brent crude oil futures contract (front month, ICE), daily **Close**
- Unit: USD/barrel
- Range downloaded: 2008-01-02 to 2026-09-01
- Terms of use: redistribution is not permitted, which is why the file is excluded via
  `.gitignore`

### `OVX` — Yahoo Finance, ticker `^OVX`

- Source: <https://finance.yahoo.com/quote/%5EOVX/history>
- Series: CBOE Crude Oil ETF Volatility Index, daily **Close**
- Unit: index points (annualized implied volatility in percent)
- **Start constraint:** the OVX index does not exist before **May 2007.** That is the
  reason the dataset begins in 2008; going further back would leave the OVX column empty.
- Validation anchor: on 2020-04-21 the OVX takes its all-time record value of **325.15**
  (the day after WTI futures went negative). This is not a data error and must not be
  deleted or corrected. `validate_data.py` checks this value explicitly; if a rebuilt file
  does not reproduce it, the wrong ticker was downloaded.

### `GPRD` and `GPRD_THREAT` — Caldara & Iacoviello, daily GPR index

- Source: <https://www.matteoiacoviello.com/gpr.htm>
- File: the daily GPR data file (`data_gpr_daily_recent.xls`)
- Columns used: the daily overall index becomes `GPRD`, the threat component becomes
  `GPRD_THREAT`
- Citation: Caldara, Dario and Matteo Iacoviello (2022), "Measuring Geopolitical Risk",
  *American Economic Review*, 112(4), 1194-1225.
- This series is on a calendar-day basis; it is intersected with the Brent/OVX trading days
  (see step 4).

---

## 3. Known data characteristics

These are not errors, and a rebuilt file is expected to show them too.

- **A single long gap:** 17 calendar days between 2009-04-14 and 2009-05-01. This is the
  largest break in the trading-day series and is recorded in
  `outputs/date_gaps_over_10_days.csv`.
- **2026 is a partial year:** the data ends in September 2026 and is not a full year. As
  the horizon lengthens, the number of valid targets in the 2026 fold falls rapidly (162
  observations for h=5, 41 for h=126). The modelling consequence of this is the 2026
  partial-year rule described in `outputs/horizon_valid_by_year.csv` and in the root
  `README.md`.
- **Valid targets per horizon.** Because the target is computed with `skipna=False`, no
  target is formed for the last h rows of the series. A rebuilt file must yield the counts
  below:

| Horizon | h | Total rows | Valid targets | Trailing rows with no target |
| --- | --- | --- | --- | --- |
| 1 week | 5 | 4641 | 4636 | 5 |
| 1 month | 22 | 4641 | 4619 | 22 |
| 3 months | 66 | 4641 | 4575 | 66 |
| 6 months | 126 | 4641 | 4515 | 126 |

---

## 4. Steps to reconstruct the data

1. **Download Brent.** From the `BZ=F` page on Yahoo Finance, download the daily history
   for the range 2008-01-02 to 2026-09-01. Take the `Date` and `Close` columns; `Open`,
   `High`, `Low`, `Adj Close` and `Volume` are not used. Name the column `Brent_Petrol`.

2. **Download OVX.** In the same way, download the daily history from the `^OVX` page for
   the same date range, take the `Close` column and name it `OVX`.

3. **Download the GPR series.** Download the daily GPR file from
   <https://www.matteoiacoviello.com/gpr.htm>. Name the daily overall index `GPRD` and the
   threat component `GPRD_THREAT`.

4. **Merge the four series on date.** Take Brent's trading-day calendar as the reference
   and match the other three series to those dates with an **inner join**. Only days on
   which all four series have a value survive; the result must be 4641 rows. If any row
   still has a NaN in any column, that row is dropped — the final file contains no missing
   values.

5. **Do NOT forward-fill or back-fill.** Filling missing days with `ffill`/`bfill` carries
   a value that was unknown on that day into the series. If a day is missing, the row is
   dropped.

6. **Convert the date to text.** Write the `Date` column as text in `DD.MM.YYYY` format
   (`dt.strftime("%d.%m.%Y")`).

7. **Set the column order and save.** Order the columns as `Date`, `Brent_Petrol`, `OVX`,
   `GPRD`, `GPRD_THREAT` and save as `data/veriseti.xlsx`.

8. **Validate.** Run `python scripts/validate_data.py`. In the resulting report, the
   fields `columns_match`, `row_count_match`, `date_monotonic_increasing`,
   `any_new_nan: false` and `ovx_record_check.matches` must all come out as expected. The
   modelling scripts must not be run before this validation passes: `01_build_targets.py`
   and `02_build_features.py` already raise an error and stop if the row count is not 4641.

---

## 5. Warning about data revisions

The GPR index and Yahoo's historical price series are occasionally revised retroactively.
Data downloaded today may therefore not be byte-for-byte identical to the file used in this
study, and in that case the numerical results will shift slightly as well. The row count
and the OVX record value in the `validate_data.py` report exist precisely so that such a
deviation is not passed over silently.
