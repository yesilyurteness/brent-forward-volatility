# Brent Crude Oil Volatility Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22815169.svg)](https://doi.org/10.5281/zenodo.22815169)

Forecasting the **forward realized volatility** of Brent crude oil using the OVX implied
volatility index and daily geopolitical risk indices. Four forecast horizons (1 week,
1 month, 3 months, 6 months), expanding-window walk-forward validation, and comparison
against econometric benchmarks.

This is **not a price or direction forecasting study.** The target variable is always
volatility, that is, the magnitude of fluctuation.

---

## Purpose and motivation

The question is this: does information in implied volatility (OVX) and geopolitical risk
(GPR) **add anything on top of naive and econometric baselines** when forecasting the
realized volatility of Brent over the next h days, and where it does add something, which
family uses it better, **machine learning** models or the **linear HAR** family?

This work is the corrected version of an earlier release that contained data leakage. In
that version, wavelet denoising computed over the full series leaked test-period
information into the training data. In this version, wavelets and all other non-causal
signal processing steps have been removed entirely; `fit()` calls are applied to training
data only, fold boundaries carry an embargo equal in length to the forecast horizon, and
all features are causal features shifted into the past. The complete list of these rules
is in [CLAUDE.md](CLAUDE.md); the same file also serves as the instruction file given to
an AI-assisted coding assistant during development, so the rules are one and the same text
for both a human reader and the assistant.

---

## Data sources and the OVX constraint

| Variable | Source | Description |
| --- | --- | --- |
| `Brent_Petrol` | Yahoo Finance, `BZ=F` | Brent Last Day Financial futures (NYMEX), continuous front month, daily close, USD/barrel |
| `OVX` | Yahoo Finance, `^OVX` | CBOE Crude Oil ETF Volatility Index, daily close |
| `GPRD` | Caldara & Iacoviello (2022) | Daily geopolitical risk index, overall |
| `GPRD_THREAT` | Caldara & Iacoviello (2022) | Daily geopolitical risk index, threat component |

4641 rows, 02.01.2008 to 01.09.2026, daily trading-day frequency (US trading calendar;
40 returns between 2008 and 2016 skip one or more trading days, see
[Date gaps in the merged series](#date-gaps-in-the-merged-series)).

The daily GPR index is not observed in real time: it is published in batches, so the GPR
features are aligned to the dates on which each observation was actually published; see
[GPR publication-date alignment](#gpr-publication-date-alignment).

**The OVX constraint:** the OVX index does not exist before **May 2007**. That is why the
sample period begins in 2008; going further back would leave the main explanatory variable
empty. This constraint also ensures that the 2008 global financial crisis falls inside the
sample, so the study covers at least one extreme volatility regime.

The raw data file is **not included in this repository**, because of the Yahoo Finance
terms of use. Step-by-step instructions for reconstructing the data from scratch, the
expected row counts and a validation checklist are in [data/README.md](data/README.md).

**Verifying that you hold the same file.** Because the raw data cannot be shared, its
SHA-256 digest is recorded in [data/veriseti.xlsx.sha256](data/veriseti.xlsx.sha256):

```
f13956e7d0eef3dfa49dee1ac83e098f0d1cea872774eb2c72f3fe33661ebd26  veriseti.xlsx
```

Check it with `cd data && sha256sum -c veriseti.xlsx.sha256` (Linux/macOS) or
`Get-FileHash data\veriseti.xlsx -Algorithm SHA256` (Windows PowerShell).
`scripts/validate_data.py` computes the digest as well and warns if it differs. The digest
is of the file's bytes: a rebuilt file with identical values but saved differently (for
example re-saved in Excel) will not match, so a mismatch is a warning, not an error. The
content checks (columns, missing values, date order and duplicates) are errors and stop
the script with a non-zero exit status.

---

## Target variable

For each horizon, the target is the standard deviation of the next h days of daily log
returns:

```python
daily_ret = np.log(brent / brent.shift(1))
future = pd.concat([daily_ret.shift(-i) for i in range(1, h + 1)], axis=1)
target_h = future.std(axis=1, skipna=False)
```

`skipna=False` is mandatory: a row's target is computed only if **all** h future days are
available. Otherwise incomplete windows at the end of the series produce erroneous targets.

| Horizon | h (trading days) | Valid targets |
| --- | --- | --- |
| 1 week | 5 | 4636 |
| 1 month | 22 | 4619 |
| 3 months | 66 | 4575 |
| 6 months | 126 | 4515 |

The target is **not annualized** and is not cumulative over the horizon; all four horizons
are on the same scale (daily volatility) and differ only in the length of the forecast
window. A 1-year horizon is not used, because an annual fold would leave a single
observation per fold and no statistical power.

---

## Walk-forward design

**Expanding window:**

- The 2008-2011 warm-up period is the first training set, and 2012 is the first forecast
  year.
- At each step the model is trained on all history up to that point, the next year is
  forecast, and the window advances by one year. The final fold trains through the end of
  2025 and forecasts 2026.
- 15 folds in total (test years 2012-2026).
- **Embargo:** the last h days of the training set are dropped. Because the target looks h
  days ahead, without this the training labels would spill into the test period. The
  embargo length always equals that model's forecast horizon.
- Window length is **not optimized.** The test set is read only once, at the very end.

**The 2026 partial-year rule.** The 2026 data is available only through September. As the
horizon lengthens, the number of valid targets in this fold falls rapidly (162
observations for h=5, 41 for h=126), and because of overlapping windows the effective
number of independent observations drops to roughly 1 at h=66 and h=126. Therefore:

- **h=5 and h=22:** the 2026 fold is included in the main metric average (n = 15 folds).
- **h=66 and h=126:** the 2026 fold is excluded from the main average (n = 14 folds) and
  is reported as a separate "partial year, low statistical power" footnote.

---

## Models

| Model | Role | Script |
| --- | --- | --- |
| XGBoost | Primary model (tiered capacity rule) | `03_walkforward.py` |
| XGBoost + Optuna | Robustness analysis, shrunk smearing | `04_optuna_walkforward.py` |
| Attention BiLSTM | Primary model | `06_attention_bilstm.py` |
| HAR / HAR-X | Econometric benchmark (HAR-X: with OVX and GPR exogenous regressors) | `05_benchmarks.py` |
| HAR + OVX / HAR + GPR | Ablation of the exogenous block, one source at a time | `11_ablation_exogenous.py` |
| GARCH(1,1) | Econometric benchmark, fit on training data only | `05_benchmarks.py` |
| Hybrid combinations | XGB+BiLSTM, HAR-X+XGB, HAR-X residual modelling | `07_hybrid.py` |
| Train-mean | Naive baseline (R²_oos = 0 by definition) | `05_benchmarks.py` |
| Past-volatility | Naive baseline | `05_benchmarks.py` |

Hyperparameters are **not searched.** The number of trees, depth, `min_child_weight` and
`reg_lambda` are taken from a pre-declared, fixed three-tier table indexed by the fold's
effective number of independent observations. The justification is not test performance
but the weak selection signal measured on the validation side: in 32 of the 52 folds where
Optuna was run, the best trial's validation RMSE was less than 5% better than the median
trial.

---

## Main findings

> **Numbers in this README.** Every number below is taken from
> [`outputs/paper_numbers_publication_aligned.md`](outputs/paper_numbers_publication_aligned.md),
> which `18_paper_numbers.py` generates from the saved outputs of commit
> `0840a54` (the file header records the full hash and the generation time).
> All numbers are from the **publication-aligned** GPR version, the primary results; see
> [GPR publication-date alignment](#gpr-publication-date-alignment).

> **Correction in v1.1.0.** Version 1.0.0 of this README stated that OVX *and* GPR
> information both contribute to the forecast. An ablation of the exogenous block
> (`11_ablation_exogenous.py`) shows that this was wrong: the entire gain comes from OVX,
> and adding GPR makes the forecast slightly worse at all four horizons. The findings
> below replace that claim.

1. **OVX carries the exogenous signal.** Adding OVX (`ovx_lag1`) to HAR lowers the
   fold-average RMSE by 4.9% (h=5), 11.6% (h=22), 7.8% (h=66) and 2.4% (h=126). HAR + OVX
   beats HAR in 13/15, 14/15, 12/14 and 10/14 folds.
2. **GPR does not help; it slightly hurts.** Adding the two GPR regressors (`gprd_lag1`,
   `gprd_threat_lag1`) worsens RMSE by 0.4–2.2%. This holds both when GPR is added to HAR
   alone (+0.5%, +1.3%, +1.8%, +1.1%) and when it is added on top of OVX (+0.4%, +1.4%,
   +2.2%, +1.1%). The direction is the same at every horizon, but the effect is small.
   HAR + OVX beats HAR-X in 9/15, 12/15, 11/14 and 9/14 folds, with exact sign test
   p = 0.61, 0.035, 0.057 and 0.42; these tests are exploratory and not corrected for
   multiplicity. The GPR coefficients are close to zero: fold-average standardized
   coefficients lie between −0.07 and +0.05, against 0.48–0.66 for OVX. The two GPR
   components tend to take opposite signs and partly cancel: in HAR + GPR, GPRD is
   positive in 12/15, 6/15, 11/14 and 9/14 folds and GPRD_THREAT negative in 11/15, 11/15,
   11/14 and 9/14. Inside HAR-X, next to OVX, their signs are unstable across folds (at
   h=126, GPRD is positive in 9 of 14 folds and GPRD_THREAT in 6 of 14).
3. **Non-linear models do not beat the HAR family at any horizon.** XGBoost, under both
   the primary specification and the Optuna robustness specification, and the Attention
   BiLSTM are worse than even plain HAR at every horizon (primary XGBoost: +1.1%, +3.2%,
   +11.4%, +5.8%). The hybrids that contain HAR-X at best draw level with HAR-X (h=5:
   +0.2%, h=126: +0.6%) and never beat the best HAR-family model.

Fold-average RMSE (in units of the standard deviation of daily log returns, lower is
better; the best value at each horizon is in bold):

| Model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR + OVX | **0.010298** | **0.007603** | **0.007469** | 0.008003 |
| HAR-X, log-log | 0.010311 | 0.007610 | 0.007521 | **0.007977** |
| HAR-X (HAR + OVX + GPR) | 0.010343 | 0.007706 | 0.007634 | 0.008088 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR + GPR | 0.010883 | 0.008707 | 0.008247 | 0.008296 |
| XGBoost | 0.010956 | 0.008870 | 0.009024 | 0.008675 |
| GARCH(1,1) | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| Attention BiLSTM | 0.013440 | 0.010386 | 0.012207 | 0.011585 |
| Past-volatility | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| Train-mean | 0.013456 | 0.011238 | 0.009265 | 0.008965 |

All rows are evaluated on the same test rows in every fold; the full table, with MAE,
R²_oos and every model including the hybrids and the robustness variants, is Section 1
of the number package. HAR-X is kept in the main comparison as specified before the
ablation was run, and is not replaced by the better HAR + OVX after the fact.

### GPR publication-date alignment

The daily GPR index (Caldara and Iacoviello) is not observed in real time the way OVX and
Brent are: it is published in batches, and later releases revise past values. An earlier
version of this pipeline gave row t the GPR observation dated t−1, which assumed that
yesterday's value was already public; in fact 80.3% of rows used an observation that had
not yet been published. This was found in an external code audit performed with a
third-party AI assistant, run by the author (see the paper's AI-use statement).

The publication rule was measured from the authors' own vintage archive: 289 archived
releases from 2022-02-24 to 2026-09-21 (`16_gpr_vintages.py`). A file released on day D
contains the observations through D itself (279 of 289 vintages; of the ten exceptions,
six are month-start updates that stop at the previous month's end, one is a stale upload,
and three stop one or two days short). Most
releases fall on Mondays, and the median delay from an observation to its first release
is 3 days (0 for a Monday observation, 6 for a Tuesday one). Before 2022-02-24 no
archive exists, and the same rule is applied counterfactually (first Monday on or after
the observation date, next business day after a federal holiday).

The GPR features are then built **as of** those publication dates: every GPR-derived
feature is computed on the index's own observation sequence and row t uses the latest
observation published by t−1 (`02_build_features.py`, `gpr_publication.py`).
Forward-filling the GPR level onto the trading calendar was rejected because it would
discard 78% of the published observations. The features pass a prefix-invariance test
and a publication-sensitivity test in which every not-yet-published observation is
perturbed: the publication-aligned rows are unchanged in 40 of 40 cases, while the old
alignment changes in 33 of 40 (the other seven are exactly the rows whose t−1 observation
had already been released). The alignment corrects the timing but not the revisions: the
values are from the current vintage.

Every model that uses GPR was re-run on the aligned features, on exactly the same
training and test rows; models without GPR input are bit-identical. The previous,
timestamp-aligned results are kept (unsuffixed output files, paper Appendix A), and the
comparison of the two versions is Section 8 of the number package. For the linear models
the cost of respecting the publication lag is small (HAR-X: +0.0% to +0.8% RMSE), and no
conclusion of the study changes.

Detail and interpretation:

- **The learnable signal is exhausted as the horizon lengthens.** XGBoost beats the
  past-volatility baseline by 17.7% at h=5 and 6.6% at h=22, but at h=66 and h=126 it
  falls behind it (+1.3% and +2.0%) and its R²_oos values fall negative (−0.28 and
  −0.31), meaning it is worse than a constant forecast at the training mean.
- **The Attention BiLSTM is the worst non-naive model at all four horizons**, and at h=5
  it is no better than the train-mean baseline (0.013440 against 0.013456). This supports
  the reading that XGBoost's loss is not specific to XGBoost but reflects a general limit
  of non-linear modelling on this problem.
- **The hybrids do not help.** The XGB+BiLSTM average is 10.4% worse than HAR-X at h=5 and
  18.9% worse at h=22. Blending HAR-X with XGBoost draws level with HAR-X only at h=5
  (+0.2%) and h=126 (+0.6%) and is 3–5% behind at h=22 and h=66, meaning the net
  information added by the ML component is close to zero.
- **Primary hypothesis family** (HAR vs HAR-X and HAR-X vs XGBoost at four horizons,
  8 tests). The Diebold-Mariano test is not significant under any correction (HLN
  p = 0.11–0.88). The fold-level sign test gives HAR-X 13 of 15 folds against both HAR and
  XGBoost at h=22; under Benjamini-Hochberg these two hypotheses are rejected (p = 0.030),
  under Holm (p = 0.059) and under Benjamini-Yekutieli, which is valid under any
  dependence (p = 0.080), nothing survives. The two h=22 tests are not independent
  evidence: their fold-level differences correlate at 0.94, both involve HAR-X, and 2020
  is a losing year in both.
- **Secondary comparisons.** HAR-X's superiority over past-volatility and over the BiLSTM
  is significant at h=5 under every correction; at h=22 it survives Benjamini-Hochberg
  only (DM), or all three corrections against the BiLSTM (sign test). At h=66 and h=126 no
  DM comparison against the naive baseline is significant. The long-horizon results are
  statistically fragile, and that is written up as a finding in exactly those terms.
- **SHAP.** HAR-X assigns 58–73% of its weight to OVX. XGBoost spends 13–29% of its
  attribution on features outside HAR-X's six regressors, more at longer horizons, and at
  long horizons it often does not split on the shared regressors at all (unused in 23% of
  fold comparisons at h=66 and 65% at h=126). On OVX the two models agree on direction in
  74% of fold comparisons (93% at h=5, 80% at h=22, 54–57% at h=66 and h=126). The
  attribution ranking is only moderately stable over time: the rank correlation between
  the first and last fold is 0.55, 0.70, 0.57 and 0.53.

Every configuration tried, together with its result, is recorded chronologically in
[outputs/experiment_log.md](outputs/experiment_log.md).

### Date gaps in the merged series

The dataset is the inner join of four series on their common dates, so a date missing from
one series drops the whole row, and the return computed from consecutive rows then spans
more than one trading day. `14_date_gap_diagnostics.py` measures this. Weekends and NYSE
holidays are treated as normal. All UK-only bank holidays are present in the data, so the
merged series follows the US calendar. After that, **40 returns skip a total of 55 trading
days.** All of them fall between 2008 and 2016; from 2017 onwards there are none. The
largest is a 17-day gap in April 2009.

Targets whose h-day window contains at least one such return, counted over the targets in
the main out-of-sample evaluation (test years 2012-2026; the 2026 fold is excluded at
h=66 and h=126):

| Horizon | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| Affected test targets | 80 / 3662 (2.2%) | 328 / 3645 (9.0%) | 494 / 3500 (14.1%) | 614 / 3500 (17.5%) |

The full-sample counts, which also include the 2008-2011 warm-up period that is used only
for training, are in `outputs/date_gap_target_exposure.csv`.

Two checks show that the gaps do not change the conclusions:

- **Direct test** (`13_gap_target_test.py`). Each gap return is rescaled to its one-day
  equivalent, r / sqrt(1 + k) for k skipped days, and the target is rebuilt from the
  rescaled returns. Every model's published predictions are then re-scored against the
  corrected target, with the predictions held fixed. The fold-average RMSE moves by at
  most 0.5%, and **the RMSE ranking of the 12 models does not change at any horizon.** In
  the MAE ranking, one pair that was already near-tied swaps places: HAR + OVX and
  HAR-X-log at h=22. HAR + OVX beats HAR-X, and HAR beats HAR + GPR, under the corrected
  target as well, at every horizon.
- **Gap-free subsample** (`12_robustness_gapfree.py`). The comparison is repeated on the
  2017-2026 folds only. The ranking is largely preserved at h=5 and h=22 (Spearman 0.95
  and 0.98 against the full sample). At h=66 and h=126 it changes (0.84 and 0.68), mainly
  because train-mean moves up. The 2012-2016 ranking differs from the full-sample ranking
  as well (0.54 and 0.77), so this reflects a difference in volatility regime between the
  two periods, not the gaps; the direct test above is what separates the two. In the
  gap-free subsample, the HAR family stays ahead of XGBoost and the BiLSTM at every
  horizon. At h=126, no model has a lower RMSE than the train-mean baseline over
  2017-2026, and the top five differ by 2.4%.

### Metric reporting

RMSE and MAE are the main metrics; they are in the same unit as the target and do not
depend on any choice of denominator. **R²_oos** is the secondary metric, and its reference
is that fold's **training** target mean, a quantity genuinely known at forecast time. By
definition the train-mean baseline has an R²_oos of exactly 0, which serves as a check that
the calculation is correct. Because the HAR family and XGBoost start their training
windows at different rows, the number package measures every model's R²_oos against the
same reference in each fold (the train-mean baseline's forecast), so that one column
compares every model with one constant forecast. Standard R² (the sklearn definition) is a footnote metric, not
a basis for decisions: because its reference is the test slice's own mean it is an ex-post
quantity, and in calm years the fold SST is so small that it takes large negative values.
MAPE is not used, because volatility can take values near zero.

---

## Version history

- **Unreleased: GPR publication-date alignment.** The GPR features are aligned to the
  dates on which each observation was published, measured from the authors' vintage
  archive (scripts 16 and 02); every GPR-dependent model is re-run on them, and these are
  now the primary results (`_publication_aligned` output files). The earlier
  timestamp-aligned results are kept for paper Appendix A. Adds the two-version comparison
  (script 17), the number package (script 18) and an exploratory BiLSTM fixed-epoch check
  (script 19). The numbers in this README changed accordingly; no conclusion changed.
- **v1.1.0.** Corrects the main finding: the exogenous gain comes from OVX alone, and GPR
  slightly worsens the forecast (exogenous ablation, script 11). Adds the date-gap
  diagnostics (script 14), the direct gap-target test (script 13) and the gap-free 2017+
  robustness check (script 12). No previously published number changes.
- **v1.0.0.** Initial release of the leakage-free pipeline.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

The study was run with Python 3.14.7. CPU is sufficient; no GPU is required.

Because the raw data is not in the repository, the first step is to build
`data/veriseti.xlsx` following the instructions in [data/README.md](data/README.md).

---

## Script execution order

Every script is runnable on its own and writes its outputs under `outputs/`. The order
matters: each step reads the previous step's output.

**GPR alignment is a required argument.** Every model and analysis step that uses GPR
(steps 3-13, 15 and 19) takes `--gpr-alignment {publication,timestamp}` and has **no
default**; a run without it stops with an error. `publication` produces the primary
results (reads `features_publication_aligned.csv`, writes `_publication_aligned` files);
`timestamp` reproduces the earlier alignment (unsuffixed files, paper Appendix A).

```bash
# Data and features (no GPR alignment argument)
python scripts/validate_data.py               # 0. data integrity validation (run this first)
python scripts/16_gpr_vintages.py             # GPR publication rule from the vintage archive (downloads ~400 MB once)
python scripts/01_build_targets.py            # 1. targets for the four horizons
python scripts/02_build_features.py           # 2. causal features, both GPR alignments
python scripts/14_date_gap_diagnostics.py     # 14. date-gap diagnostics of the merged series

# Primary results: publication-aligned GPR
python scripts/03_walkforward.py              --gpr-alignment publication  # 3. XGBoost, primary specification
python scripts/04_optuna_walkforward.py       --gpr-alignment publication  # 4. Optuna robustness analysis (longest step)
python scripts/05_benchmarks.py               --gpr-alignment publication  # 5. HAR, HAR-X, GARCH, naive baselines
python scripts/06_attention_bilstm.py         --gpr-alignment publication  # 6. Attention BiLSTM
python scripts/07_hybrid.py                   --gpr-alignment publication  # 7. hybrid combinations
python scripts/07b_exploratory_vol_regime.py  --gpr-alignment publication  # 7b. exploratory volatility regime analysis
python scripts/08_dm_test.py                  --gpr-alignment publication  # 8. Diebold-Mariano tests
python scripts/09_power_analysis.py           --gpr-alignment publication  # 9. statistical power analysis
python scripts/10_shap_analysis.py            --gpr-alignment publication  # 10. TreeSHAP attribution analysis
python scripts/11_ablation_exogenous.py       --gpr-alignment publication  # 11. OVX / GPR ablation of the HAR family
python scripts/12_robustness_gapfree.py       --gpr-alignment publication  # 12. main comparison on the gap-free 2017+ folds
python scripts/13_gap_target_test.py          --gpr-alignment publication  # 13. direct test: gap-corrected target, fixed predictions
python scripts/15_exploratory_xgb6.py         --gpr-alignment publication  # 15. exploratory: XGBoost on HAR-X's inputs and target
python scripts/20_log_residual_std.py         --gpr-alignment publication  # 20. log-residual std of HAR-log / HAR-X-log (after step 5)
python scripts/21_clark_west.py               --gpr-alignment publication  # 21. Clark-West, HAR nested in HAR-X (supplementary family, after step 7)
python scripts/22_descriptive_stats.py        --gpr-alignment publication  # 22. Table 1, descriptive statistics (after steps 1-2)
python scripts/23_figure2.py                  --gpr-alignment publication  # 23. Figure 2 data and PDF/SVG (after steps 7, 10 and 18)
python scripts/24_rollover_robustness.py      --gpr-alignment publication  # 24. roll-over robustness of the primary family (after steps 3, 5 and 18)
python scripts/25_rollover_clark_west.py      --gpr-alignment publication  # 25. Clark-West on the roll-over variants (after steps 21 and 24)

# Robustness variants (publication-aligned)
python scripts/05_benchmarks.py       --gpr-alignment publication --align-start-row 127 --suffix _aligned  # data equalization
python scripts/06_attention_bilstm.py --gpr-alignment publication --convergence-mode --suffix _conv    # convergence criterion
python scripts/19_bilstm_fixed_epochs.py --gpr-alignment publication  # 19. exploratory: fixed 200 epochs (after the _conv run)

# Appendix A: repeat steps 3-13 and 15 (and the two variants above) with --gpr-alignment timestamp

# Both versions must exist for these two (no alignment argument)
python scripts/17_gpr_alignment_comparison.py  # 17. timestamp vs publication alignment, same samples
python scripts/18_paper_numbers.py             # 18. number package for the paper and this README
```

Steps 11-15 take a few seconds each (step 15 under a minute). Step 12 reads the outputs of steps 3, 5, 6, 7 and 11;
step 13 reads the prediction files of steps 3, 5, 6, 7 and 11, and imports the gap
classification from the step 14 script, so it does not need step 14's outputs.

The later steps must not be run before data validation passes: steps 1 and 2 raise an
error and stop if the row count is not 4641.

Steps 3-7 can be narrowed by horizon and test year, which is useful for quick trials:

```bash
python scripts/03_walkforward.py --gpr-alignment publication --horizons 5 --max-folds 2
python scripts/05_benchmarks.py --gpr-alignment publication --horizons 5 22 --test-years 2012 2013
```

The most expensive step is the Optuna run (30 trials per fold, four horizons). The BiLSTM
step trains one neural network per fold on CPU. The rest take on the order of minutes.

---

## Reproducibility

- `SEED = 42` is fixed in every script; `random`, `numpy` and `torch` are seeded
  separately, and XGBoost is given `random_state=42`.
- On the BiLSTM side, deterministic algorithms are enabled and DataLoader shuffling uses a
  seeded generator. The determinism mode actually used is written to the
  `deterministic_algorithms` field inside `outputs/bilstm_summary_all.json` on every run.
- Package versions are pinned in `requirements.txt`. A run with those same versions on the
  same data reproduces the reported numbers exactly.
- The metric JSON and CSV files under `outputs/`, together with the experiment log, are
  included in the repository, so the results can be audited without running the code. The
  feature matrix and the prediction files are included as well.
- Different package versions, or data that has been revised retroactively, can produce
  small deviations. The row count and the OVX record value (2020-04-21, 325.15) in the data
  validation report exist so that such a deviation is not passed over silently.

---

## Repository structure

```
.
|-- CLAUDE.md              project methodology rules and the data leakage prevention
|                          checklist; the same file is also used as the instruction
|                          file for an AI-assisted coding assistant
|-- README.md
|-- LICENSE                MIT (code only; raw data out of scope)
|-- requirements.txt
|-- data/
|   |-- README.md          data sources and reconstruction instructions
|   |-- veriseti.xlsx.sha256  SHA-256 digest of the data file used for the results
|   +-- veriseti.xlsx      NOT in the repository, built locally
|-- scripts/               27 independently runnable scripts and 2 shared modules
+-- outputs/               metrics, predictions, JSON reports, experiment log
```

A note on language: the documentation, all code comments and the console messages of most
scripts are in English, while the column names of the generated CSV files are in Turkish.
The console logs under `outputs/` and the free-text fields inside some JSON reports were
written by the original runs and so are in Turkish as well; the English narrative of the
analysis is in [outputs/experiment_log_en.md](outputs/experiment_log_en.md). This is
deliberate: translating the column names would require regenerating every
output file, which would break the correspondence between the committed results and the
runs that produced them. Every output file and every column name is translated and
explained in the data dictionary at [outputs/README.md](outputs/README.md).

---

## Citation

For the geopolitical risk indices:

> Caldara, Dario and Matteo Iacoviello (2022). "Measuring Geopolitical Risk."
> *American Economic Review*, 112(4), 1194-1225.

---

## License

The code is released under the [MIT license](LICENSE). The raw market data is not
distributed in this repository and is not covered by that license; each source carries its
own terms of use.
