# Brent Crude Oil Volatility Forecasting Project

## Purpose

Forecast the **forward realized volatility** of Brent crude oil prices using OVX and
geopolitical risk indices. This is **NOT a price or direction forecasting project** — the
target variable is always volatility (the magnitude of fluctuation).

This work is the corrected version of an earlier release that contained data leakage. The
rules below exist to prevent that error from recurring and must be applied without
exception.

## CRITICAL RULES (Data Leakage Prevention)

These rules are not negotiable. If any code suggestion violates one of them, do not write
the code; warn me first.

1. **Do NOT use wavelet transforms.** In the earlier version, non-causal denoising via
   `pywt` leaked test-period information into the training data. Do not use any non-causal
   signal processing (wavelets, forward-backward filtering, `filtfilt`, or any smoothing
   computed over the full series).
2. **All `fit()` operations apply to training data only.** This covers everything,
   including MinMaxScaler, winsorization bounds, log-transform parameters and GARCH
   parameter estimation. A line such as `scaler.fit(df_full)` must never be written.
3. **Time series order is preserved; NO shuffling.** Do not use
   `train_test_split(shuffle=True)` or `KFold`. Only chronological splits and
   walk-forward.
4. **Apply purge/embargo at fold boundaries.** Because the target looks h days ahead, the
   last h days of the training set must be dropped (otherwise training labels spill into
   the test period). The embargo length must always equal that model's forecast horizon
   (h).
5. **The test set is read only once, at the very end.** Hyperparameters, window lengths
   and model selection are never decided by looking at test performance — these are chosen
   on the validation set.
6. **All features must be causal.** Every lag/EMA/rolling feature must be shifted into the
   past with `.shift(1)`; a feature at time t cannot contain information from time t or
   later.

## Data

- File: `data/veriseti.xlsx`
- Range: 02.01.2008 – 01.09.2026, 4641 rows, daily frequency
- Columns: `Date`, `Brent_Petrol`, `OVX`, `GPRD`, `GPRD_THREAT`
- Note: the OVX index does not exist before May 2007, which is why the data starts in 2008.

## Target Variable

Forward realized volatility is computed for four separate horizons. Each is the standard
deviation of the next h days of daily log returns:

```python
daily_ret = np.log(brent / brent.shift(1))
future = pd.concat([daily_ret.shift(-i) for i in range(1, h + 1)], axis=1)
target_h = future.std(axis=1, skipna=False)
```

`skipna=False` is mandatory; otherwise incomplete windows at the end of the series produce
erroneous targets.

Horizons: 1 week (h=5), 1 month (h=22), 3 months (h=66), 6 months (h=126).
A 1-year horizon is not used (an annual walk-forward fold would leave a single observation
per fold, giving no statistical power).

## Feature Engineering

All features are `.shift()`-based and causal:

- Lags 1-5 and EMAs (5, 10, 20) for Brent, OVX, GPRD, GPRD_THREAT
- Realized volatility: `brent_vol5`, `brent_vol20`, `vol_ratio`
- OVX derivatives: z-score (60-day window), spike (z > 2), high/low regime, mean reversion
- GPR derivatives: z-score, spike, momentum (EMA5 − EMA20), threat ratio
- Interaction terms: `OVX × GPRD`, `OVX × GPRD_THREAT`, regime × GPRD
- Calendar: day of week, month, start/end-of-month flags

## Validation Scheme

**Expanding window walk-forward:**

- Warm-up period: 2008–2011 (the first training set)
- First forecast year: 2012
- At each step: train on all history up to that point → forecast the next year → advance
  one year
- Final fold: train through the end of 2025 → forecast 2026
- Window length is NOT optimized (see Critical Rule 5)

### 2026 partial-year rule

The 2026 data is available only through September (not a full year). As the horizon
lengthens, the number of valid targets in this fold falls rapidly (162 observations for
h=5, only 41 for h=126 — see `outputs/horizon_valid_by_year.csv`). Because of overlapping
windows, the effective number of independent observations drops to roughly 1 at h=66 and
h=126, which makes the fold metric statistically meaningless / excessively noisy.
Therefore:

- **h=5 and h=22**: the 2026 fold is included in the main metric average exactly like the
  other folds.
- **h=66 and h=126**: the 2026 fold is EXCLUDED from the main metric average (across-folds
  RMSE/MAE/R²). It is reported in the results table as a separate "partial year, low
  statistical power" footnote — it is not used for model comparison or selection, and is
  informational only.

## Models and Benchmarks

Run for every horizon and every fold:

| Model | Role |
| --- | --- |
| XGBoost | Primary model |
| Attention BiLSTM | Primary model |
| Simple average (0.5×XGB + 0.5×BiLSTM) | Hybrid |
| GARCH(1,1) | Econometric benchmark (train-only fit) |
| HAR | Econometric benchmark |
| Train-mean | Naive baseline |
| Past-volatility | Naive baseline |

## Model Selection Policy

### Primary specification: tiered capacity rule

Hyperparameters are **not searched** at any of the four horizons. The number of trees,
depth, `min_child_weight` and `reg_lambda` are taken from a pre-declared, fixed
three-tier table indexed by the fold's effective number of independent observations
(`training rows / h`). The rule is applied per fold; as the fold grows, effective
observations increase and capacity rises automatically. Tier selection depends only on the
training row count and h, and never looks at test data.

| effective obs. | trees | depth | min_child_weight | reg_lambda |
| --- | --- | --- | --- | --- |
| ≥ 100 | 400 | 4 | 1 | 1 |
| 30 – 100 | 150 | 3 | 20 | 10 |
| < 30 | 80 | 2 | 50 | 25 |

**The justification for this choice is NOT test performance.** The justification is the
weak selection signal measured on the validation side: in 32 of the 52 folds where Optuna
was run, the best trial's validation RMSE was less than 5% better than the median trial.
Because of overlapping target windows, the validation slice contains only 5–6.6 effective
independent observations at long horizons; picking the best of 30 trials on a sample that
small selects noise, not signal. The selection signal is a quantity read off the
validation distribution alone, without looking at any test result.

Changing this policy on the basis of test performance is a violation of Critical Rule 5.

### Secondary / robustness analysis: Optuna + shrunk smearing

The Optuna setup is not abandoned; it is retained as a **robustness analysis** and
reported in the paper. Its configuration: within-fold validation (whose length derives
from the "≥ 3 effective independent observations" criterion), an h-day internal embargo
between train-proper and validation, selection from validation RMSE only, and a smearing
coefficient estimated from validation residuals and shrunk via `S = 1 + w(S_raw − 1)` with
`w = n_eff/(n_eff+10)`.

The result varies by horizon and **both directions will be stated explicitly**:

- At h=22 it is **better** than the primary specification (RMSE 3.2% lower, 7.4% ahead of
  past-volatility).
- At h=66 and h=126 it is **worse** (RMSE 13% higher; at h=126 it wins in only 1 of 8
  folds).
- At h=5 there is no difference.

**Choosing the best method per horizon (cherry-picking) is FORBIDDEN.** The primary
specification is the tiered capacity rule at all four horizons; the Optuna results are
reported as a robustness analysis, as they are, at all four horizons.

### Appendix: bias-variance trade-off record

Optuna's **pre-shrinkage** (raw smearing) run is stored under `outputs/opt_rawsmearing_*`
and is used as a methodological example in the paper's appendix. The narrative is this:
in-sample smearing was biased but, because it collapsed to 1.0000, it was a harmless
multiplier; validation-based smearing is unbiased but, because it is estimated from about
5 effective observations, it swings between 0.60 and 1.59 and — being a single scalar that
multiplies every test prediction — feeds straight through to RMSE. The correlation between
degradation and absolute smearing deviation was measured at 0.825 for h=66 and 0.768 for
h=126. After shrinkage these correlations fell to 0.624 and 0.193.

### Experiment log

Every configuration tried and its result is kept in order in
`outputs/experiment_log.md`. This table will be given in the Limitations section.

## Evaluation

- Do not use MAPE — it is mathematically problematic because volatility can take values
  near zero
- Results table broken down by horizon × year (the 2008 crisis, 2020 COVID and 2022 war
  periods will be interpreted separately)
- Diebold-Mariano test: hybrid vs. GARCH, hybrid vs. naive baseline
- SHAP analysis for at least one horizon

### Metric reporting rule

No metric is hidden. All three are always reported; their differences are interpreted
using the hierarchy below.

**Main metrics: RMSE and MAE.** These are the primary basis for model selection, horizon
comparison and the results tables. They carry no reference ambiguity: they are in the same
unit as the target (standard deviation of daily log returns) and do not depend on any
choice of denominator.

**Secondary metric: R²_oos (out-of-sample R² relative to the training mean).**

```
R²_oos = 1 - SSE_model / Σ(y_test - train_mean)²
```

The reference is the mean of that fold's training target. This is a quantity genuinely
known at forecast time and one we already compute as the train-mean baseline. R²_oos is
therefore the honest answer to the question "what does the model gain over a naive constant
forecast?" By definition the train-mean baseline has an R²_oos of exactly 0; this serves as
a check that the calculation is correct. A negative R²_oos means the model is worse than a
constant forecast, and must be written into the report.

**Footnote metric: standard R² (sklearn definition).** Reported, but not a basis for any
decision. Its reference is the test slice's OWN mean; that mean cannot be known at forecast
time, i.e. it is an ex-post quantity. It frequently comes out negative in the results, and
that is not an error. The reason is explained in the report every time, as follows:

- When volatility stays relatively constant within a year (calm years), the fold's SST
  becomes very small; a small denominator pushes R² to large negative values. In the h=5
  study the ratio of SST across folds reached 53.6×, and the most negative R² values
  occurred in the calmest years (2017, 2012).
- When fold SSTs differ this much, per-fold R² values are not on the same scale, so
  **the fold average of standard R² is not taken, or if taken, is presented with this
  caveat**. A pooled R² may also be reported; because the pool also contains
  between-year variance, it comes out systematically higher than the fold average.

For aggregation, both the fold average and the pooled value are given. CLAUDE.md's
"across-folds" main metric is the fold average, read off RMSE/MAE.

## Code Standards

- Python, pandas + numpy + scikit-learn + xgboost + torch + arch (for GARCH)
- `SEED = 42`, all randomness fixed (`random`, `numpy`, `torch`)
- Scripts live under `scripts/`, outputs under `outputs/` (JSON + CSV + PNG)
- Every script must be runnable on its own (`python scripts/xxx.py`)
- Path definitions are built relative to the repo root with
  `Path(__file__).resolve().parents[1]`
- Before any long/expensive computation, give me an estimate of how long it will take

## Working Style

- Present a plan before large changes and wait for approval
- After writing a script, test it on a small slice of data, then run it on the full data
- If you notice anything that carries a leakage risk (even if I asked for it), warn me and
  explain why
