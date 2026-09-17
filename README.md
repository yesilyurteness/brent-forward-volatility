# Brent Crude Oil Volatility Forecasting

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
| `Brent_Petrol` | Yahoo Finance, `BZ=F` | Brent crude futures (front month), daily close, USD/barrel |
| `OVX` | Yahoo Finance, `^OVX` | CBOE Crude Oil ETF Volatility Index, daily close |
| `GPRD` | Caldara & Iacoviello (2022) | Daily geopolitical risk index, overall |
| `GPRD_THREAT` | Caldara & Iacoviello (2022) | Daily geopolitical risk index, threat component |

4641 rows, 02.01.2008 to 01.09.2026, daily trading-day frequency.

**The OVX constraint:** the OVX index does not exist before **May 2007**. That is why the
sample period begins in 2008; going further back would leave the main explanatory variable
empty. This constraint also ensures that the 2008 global financial crisis falls inside the
sample, so the study covers at least one extreme volatility regime.

The raw data file is **not included in this repository**, because of the Yahoo Finance
terms of use. Step-by-step instructions for reconstructing the data from scratch, the
expected row counts and a validation checklist are in [data/README.md](data/README.md).

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

## Main finding

**The linear HAR-X model beats both machine learning models at all four horizons.** OVX
and GPR information really is useful for forecasting volatility, but what uses that
information best is a linear specification; the tree-based and neural network models
extract nothing further.

Fold-average RMSE (in units of the standard deviation of daily log returns, lower is
better):

| Model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| **HAR-X** | **0.010341** | **0.007669** | **0.007573** | **0.008059** |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| XGBoost | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| GARCH(1,1) | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| Attention BiLSTM | 0.014227 | 0.010512 | 0.011589 | 0.011431 |
| Past-volatility | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| Train-mean | 0.013451 | 0.011243 | 0.009239 | 0.008916 |

Detail and interpretation:

- **The learnable signal is exhausted as the horizon lengthens.** XGBoost beats the
  past-volatility baseline by 18% at h=5, but at h=66 and h=126 it merely draws level with
  it (+0.1% and +1.4%) and its R²_oos values fall negative (−0.28 at h=126), meaning it is
  worse than a constant forecast at the training mean.
- **The Attention BiLSTM is the worst model at all four horizons.** This supports the
  reading that XGBoost's loss is not specific to XGBoost but reflects a general limit of
  non-linear modelling on this problem. Both outcomes are reported.
- **The hybrids do not help.** The XGB+BiLSTM average is 12% worse than HAR-X at h=5 and
  22% worse at h=22. Blending HAR-X with XGBoost only draws level with HAR-X (0.7% behind
  at h=126), meaning the net information added by the ML component is close to zero.
- **Diebold-Mariano test.** HAR-X's superiority over past-volatility and over the BiLSTM
  remains significant at h=5 and h=22 after correction for multiple comparisons. The
  difference between HAR-X and XGBoost is **not significant at any horizon**, and at h=66
  and h=126 no comparison produces significance against the naive baseline. The long-horizon
  results are statistically fragile, and that is written up as a finding in exactly those
  terms.
- **SHAP.** HAR-X assigns two thirds of its weight to OVX. XGBoost, by contrast, looks
  primarily at the Brent volatility window and spends 15-30% of its attention on groups
  HAR-X never uses (price level, calendar, interactions); at long horizons it leaves 44-67%
  of the features entirely unused. On common ground, that is on OVX, the two models agree
  on direction (83% sign agreement). The stability of the attribution ranking is genuine at
  short horizons (first-to-last fold correlation 0.77-0.79) but weak at long horizons (0.53
  at h=126).

Every configuration tried, together with its result, is recorded chronologically in
[outputs/experiment_log.md](outputs/experiment_log.md).

### Metric reporting

RMSE and MAE are the main metrics; they are in the same unit as the target and do not
depend on any choice of denominator. **R²_oos** is the secondary metric, and its reference
is that fold's **training** target mean, a quantity genuinely known at forecast time. By
definition the train-mean baseline has an R²_oos of exactly 0, which serves as a check that
the calculation is correct. Standard R² (the sklearn definition) is a footnote metric, not
a basis for decisions: because its reference is the test slice's own mean it is an ex-post
quantity, and in calm years the fold SST is so small that it takes large negative values.
MAPE is not used, because volatility can take values near zero.

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

```bash
python scripts/validate_data.py               # 0. data integrity validation (run this first)
python scripts/01_build_targets.py            # 1. targets for the four horizons
python scripts/02_build_features.py           # 2. causal features
python scripts/03_walkforward.py              # 3. XGBoost, primary specification
python scripts/04_optuna_walkforward.py       # 4. Optuna robustness analysis (longest step)
python scripts/05_benchmarks.py               # 5. HAR, HAR-X, GARCH, naive baselines
python scripts/06_attention_bilstm.py         # 6. Attention BiLSTM
python scripts/07_hybrid.py                   # 7. hybrid combinations
python scripts/07b_exploratory_vol_regime.py  # 7b. exploratory volatility regime analysis
python scripts/08_dm_test.py                  # 8. Diebold-Mariano tests
python scripts/09_power_analysis.py           # 9. statistical power analysis
python scripts/10_shap_analysis.py            # 10. TreeSHAP attribution analysis
```

The later steps must not be run before data validation passes: steps 1 and 2 raise an
error and stop if the row count is not 4641.

Steps 3-7 can be narrowed by horizon and test year, which is useful for quick trials:

```bash
python scripts/03_walkforward.py --horizons 5 --max-folds 2
python scripts/05_benchmarks.py --horizons 5 22 --test-years 2012 2013
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
|   +-- veriseti.xlsx      NOT in the repository, built locally
|-- scripts/               12 independently runnable scripts
+-- outputs/               metrics, predictions, JSON reports, experiment log
```

A note on language: the documentation and all code comments are in English, while the
console output of the scripts and the column names of the generated CSV files are in
Turkish. The log files under `outputs/` were produced by those runs and so are in Turkish
as well. This is deliberate: translating the column names would require regenerating every
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
