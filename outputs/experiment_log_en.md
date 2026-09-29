> This is a translation. The Turkish original (experiment_log.md) is the authoritative record;
> in case of any discrepancy the original prevails.
>
> It translates `outputs/experiment_log.md` line by line: lines 1–2556 as they stand at commit
> `378d551` (the freeze record), and the dated translation note after them (lines 2557–2576) as
> added in the same commit as this file. Line n of the original is line n + 14 of this file.
> Numbers are carried over exactly as written in the original, including decimal commas and
> space-separated thousands (for example "2 764"); only the words around them are translated.
> The one change of notation is the Turkish prefix percent sign, written here as a suffix
> ("%80" → "80%", "−%13.1" → "−13.1%").
> `scripts/check_translation.py log` checks that every stage of this file contains the same
> numbers, commit hashes and file paths as the original.

<!-- end of translation header -->
# Experiment Log

This file records every configuration tried for the XGBoost main model **in chronological order**,
together with the result of each. This table will be given in the paper's Limitations section.

The aim is transparency: which ideas were tried, which did not work and why the final
specification was chosen are visible here. The final choice **was not made according to test
performance** (see CLAUDE.md "Model Selection Policy"); the justification for the choice is the
weak selection signal measured on the validation side.

All metrics are **fold-mean RMSE**, in units of the standard deviation of daily log returns.
The validation scheme is fixed: expanding window, 15 folds, test years 2012–2026, embargo = h.
At h=66 and h=126 the 2026 fold is excluded from the main average under the partial-year rule (n=14).

## Versions

| # | Configuration | The only thing changed |
| --- | --- | --- |
| 1 | 61 features, log target, fixed capacity (400 trees, depth 4) | baseline |
| 2 | + HAR realized volatility, 252 window included, 67 features | feature set |
| 3 | + HAR realized volatility, 252 window excluded, 65 features | feature set |
| 4 | + ratio target: `log(vol_h) − log(past_vol_h)` | target parametrization |
| 5 | + tiered capacity rule (tied to effective observations) | model capacity |
| 6 | + Optuna, smearing from validation residuals (raw) | hyperparameter selection |
| 7 | + Optuna, smearing shrunk (`w = n_eff/(n_eff+10)`) | smearing variance |

## XGBoost RMSE (fold mean)

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | 0.010772 | 0.008938 | 0.010401 | 0.009923 |
| 2 | 0.011174 | 0.009497 | 0.009595 | 0.009163 |
| 3 | 0.010829 | 0.009008 | 0.010304 | 0.009907 |
| 4 | 0.010999 | 0.009091 | 0.010084 | 0.009886 |
| **5** | **0.010913** | **0.009080** | **0.008924** | **0.008627** |
| 6 | 0.011025 | 0.009178 | 0.011042 | 0.011174 |
| 7 | 0.010927 | 0.008788 | 0.009797 | 0.009715 |

The bold row is the primary specification.

## Difference relative to the past-volatility baseline (%, negative = model better)

The baseline RMSE is constant across all versions (0.013318 / 0.009493 / 0.008913 / 0.008508),
because it does not depend on the feature set or the model. This makes it a valid anchor for
comparing the table across versions.

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | −19.1 | −5.8 | +16.7 | +16.6 |
| 2 | −16.1 | +0.0 | +7.7 | +7.7 |
| 3 | −18.7 | −5.1 | +15.6 | +16.5 |
| 4 | −17.4 | −4.2 | +13.1 | +16.2 |
| **5** | **−18.1** | **−4.3** | **+0.1** | **+1.4** |
| 6 | −17.2 | −3.3 | +23.9 | +31.3 |
| 7 | −17.9 | −7.4 | +9.9 | +14.2 |

## R²_oos (relative to the training mean, fold mean)

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | +0.277 | +0.205 | −0.757 | −0.821 |
| 2 | +0.193 | +0.010 | −0.807 | −0.729 |
| 3 | +0.287 | +0.248 | −0.831 | −1.118 |
| 4 | +0.272 | +0.191 | −0.744 | −1.124 |
| **5** | **+0.283** | **+0.186** | **−0.199** | **−0.283** |
| 6 | +0.271 | +0.098 | −1.529 | −1.673 |
| 7 | +0.281 | +0.219 | −0.770 | −0.744 |

## What was learned at each step

**1 → 2: HAR volatility features added (252 window included).** At long horizons RMSE
fell by 7.7%, at short horizons it rose. But the comparison was contaminated: the 252-day window moved the first
valid row from 60 to 253 and took ~193 rows from every fold. The feature-blind train-mean
baseline also improved by 4.1–4.4%, i.e. more than half of the gain came from the change
in the data window. Its source is the 2008 crisis partly leaving the training set.

**2 → 3: the 252 window was removed, the data cost halved.** The train-mean baseline moved by
less than 1%, i.e. the data-window effect disappeared. In the same version XGBoost too showed no
change exceeding 1% at any horizon. **Result: HAR features with matched windows provide no
measurable contribution once the data cost is removed.** The hypothesis was tested and not supported.
The features were kept anyway, because they are needed for the HAR benchmark and their cost is near zero.

**3 → 4: the target was rewritten as a deviation from past-volatility.** The expectation was that,
in the absence of signal, the model would structurally copy the baseline. The effect stayed small: at long
horizons an improvement of 0.2–2.1%, at short horizons a regression of 1–2%. The worst fold improved markedly
(h=126/2013 ratio from 3.75-fold to 2.49-fold). The structural protection did not fully work, because the model does not
recognize the absence of signal and predict zero; 400 unregularized trees always produce
confident deviations. The memorization measure (in-sample R² calibrated to the target
space) fell at h=126 from 0.994 to 0.992, i.e. practically unchanged.

**4 → 5: capacity was tied to the effective sample size.** The largest single effect. At long horizons
RMSE fell by 11.5% and 12.7%; h=66 and h=126 beat the train-mean baseline for the first time and
drew level with past-volatility. In-sample R² at h=126 fell from 0.992 to 0.746,
i.e. the memorization was really broken. The worst-fold ratios fell at h=66 from 3.21 to 1.86,
at h=126 from 3.24 to 2.35. **Diagnosis: the main driver of the long-horizon failure was the imbalance
between model capacity and the number of effective independent observations.** At h=126 even the largest
fold has only ~33 effective observations, while the model uses 65 features.

**5 → 6: Optuna + validation-based smearing.** Heavy regression: at long horizons RMSE rose by 24% and
30%. The decomposition showed that the culprit is not hyperparameter selection but the smearing coefficient.
The correlation between the degradation and the absolute smearing deviation is 0.825 at h=66,
0.768 at h=126; the correlation with the selection signal is negative (−0.30, −0.17). In-sample
smearing was biased but, because it collapsed to 1.0000, harmless; validation-based smearing is unbiased
but, because it is estimated from ~5 effective observations, it swings between 0.60 and 1.59 and, as a single scalar,
multiplies all test forecasts.

**6 → 7: smearing was shrunk.** `S = 1 + w(S_raw − 1)`, `w = n_eff/(n_eff+10)`. The coefficient range
at h=126 narrowed from 0.599–1.580 to 0.870–1.158; the mean absolute deviation fell from 0.198 to
0.061. RMSE beat the raw version at every horizon (0.9–13.1% improvement) and the smearing correlation
fell to 0.193 at h=126. But it could beat the capacity rule only at h=22. The remaining regression
now comes directly from selection noise.

## Final decision

**Primary specification: version 5, the tiered capacity rule, at all four horizons.**

The justification is not test performance. In 32 of the 52 folds where Optuna was run, the best trial's
validation RMSE was less than 5% better than the median trial; because of overlapping target windows the
validation slice contains only 5–6.6 effective independent observations at long horizons. This criterion
is read from the validation distribution and does not look at the test result.

**Version 7 is reported as a secondary / robustness analysis.** At h=22 it is better than the primary
(RMSE 3.2% lower), at h=66 and h=126 worse (13% higher), at h=5 no difference. Choosing the best one
per horizon (cherry-picking) will not be done.

**Version 6 is used in the paper's appendix** as an example of the bias-variance trade-off.

## Output files

| version | files |
| --- | --- |
| 5 (primary) | `wf_predictions_h*.csv`, `wf_metrics_all.csv`, `wf_aggregate_all.csv`, `wf_summary_all.json` |
| 6 (raw smearing) | `opt_rawsmearing_metrics_all.csv`, `opt_rawsmearing_aggregate_all.csv`, `opt_rawsmearing_folds_all.csv`, `opt_rawsmearing_summary_all.json` |
| 7 (shrunk) | `opt_predictions_all.csv`, `opt_metrics_all.csv`, `opt_aggregate_all.csv`, `opt_folds_all.csv`, `opt_summary_all.json` |

Versions 1–4 are intermediate steps and their outputs were not kept; the values in the tables above
can be reproduced by re-running `scripts/02_build_features.py` and `scripts/03_walkforward.py` with the
corresponding parameters.

---

# Stage 5: Econometric Benchmarks

This section records the outputs of `scripts/05_benchmarks.py` and the specification errors found
during development **in the order they were discovered**.

## Eight-model comparison (fold-mean RMSE)

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| **HAR-X-log** | **0.010335** | **0.007561** | **0.007504** | **0.008012** |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR-log | 0.010857 | 0.008678 | 0.008199 | 0.008275 |
| XGBoost (primary) | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| GARCH(1,1)-t | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| past-volatility | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| train-mean | 0.013440 | 0.011216 | 0.009240 | 0.009028 |

## Main decomposition: the source of the gain

| horizon | exogenous variables (HAR→HAR-X) | nonlinear model (HAR-X→XGBoost) |
| --- | --- | --- |
| 5 | −4.55% | +5.53% |
| 22 | −10.82% | +18.39% |
| 66 | −6.50% | +17.84% |
| 126 | −1.76% | +7.05% |

**The gain comes from the exogenous variables (OVX, GPR); the nonlinear model, with the same information,
does harm at all four horizons.** This is the main finding, which reverses the project's implicit
assumption.

## Data-equalization robustness check

**Aim:** to show that XGBoost's loss does not stem from the HAR family's ~106-day (4%) data
advantage.

**Method note:** it is NOT POSSIBLE to extend XGBoost down to HAR's window (row 21);
`brent_vol126` is NaN at that row, and extending it would require pruning XGBoost's feature set,
which would replace the confounder being removed with a larger one. Therefore
the equalization was done in the opposite direction: the benchmarks were restricted to XGBoost's window (row 127).
No model's feature set changed. After equalization the HAR/HAR-X training
sizes are identical to XGBoost's.

| model | difference relative to XGBoost, normal window | equalized window |
| --- | --- | --- |
| HAR | −0.72 / −5.29 / −9.24 / −4.91 | −0.72 / −5.36 / −9.35 / −6.22 |
| HAR-X | −5.24 / −15.54 / −15.14 / −6.58 | −5.18 / −15.40 / −15.11 / −7.88 |
| HAR-X-log | −5.30 / −16.72 / −15.92 / −7.12 | −5.23 / −16.67 / −16.06 / −8.57 |

(h=5 / h=22 / h=66 / h=126 respectively, in %; negative = benchmark better.)

**Result: the data difference is not the explanation.** After equalization the HAR family's superiority stayed
the same, and at h=126 it INCREASED somewhat. The extra early data (the 2008 crisis) apparently does not help HAR.
This is a robustness check; the primary specification was not changed according to the result.
The outputs are in the `bench_*_aligned.*` files.

## Errors found during development (in the order discovered)

**1. The `fit(last_obs=)` parameter of the `arch` library is positional.** Because the return series'
index starts at 1, `last_obs=first_test` was adding the first day of the test period to the training slice.
Comparing with an explicit slice (`ret[ret.index < first_test]`) showed
that the parameters differed, and explicit slicing was adopted. `forecast(start=)`
is positional too; the position is computed explicitly and the result is verified label-wise with an assert.
In addition, in every fold the equality "first step of origin t−1 = σ²_t" is checked.

**2. `arch` forecast column names are zero-padded according to the horizon** (`h.1` / `h.01` / `h.001`).
h=5 worked and h=126 raised a KeyError; the column is now selected by position, not by name.

**3. HAR-log had been misspecified — the order of discovery matters.** In the plan it was described as "the same
regressors, dependent variable log(target)". **In the two-fold smoke
test HAR-log's RMSE came out at 0.040098 at h=5, more than twice that of HAR in levels
(0.018221).** When this anomaly was examined, it was seen that logging the dependent variable and leaving the regressors
in levels is a misspecification; the canonical log-HAR logs both
sides. The log-log form was adopted. **This change was made not by chasing results but
by correcting to the canonical form; however, the order is recorded as is for transparency:
first the anomalous result was seen, then the specification was corrected.** After the correction HAR-log
came out very close to HAR (0.2–1.2% worse), i.e. the anomaly really
stemmed from the misspecification.

`har_daily = |r_{t−1}|` is exactly zero on 25 days (unchanged close) and the log becomes undefined;
the regressors in the log specification are bounded at the `LOG_FLOOR = 1e-4` floor declared in advance,
and 31 observations are floored.

**4. HAR-X-log was added later.** Initially the 2×2 design (HAR/HAR-X × level/log)
had only three cells. The fourth cell was completed. The log form also structurally solves HAR-X's problem
of predicting negative volatility and hitting the floor: in levels,
at h=66 2.69% of test observations, at h=22 1.62%, were hitting the floor; in the log form
this problem does not exist by definition.

## GARCH notes

No embargo is applied to GARCH (it uses no forward-looking label), so it sees 131–252 days more data
than XGBoost; the difference is reported per fold and broken down into embargo/warm-up
components. Despite the advantage, GARCH is behind HAR at every horizon and at h=66 and
h=126 cannot even beat the past-volatility baseline.

GARCH persistence (α+β) is ~1.000 in every fold, i.e. almost IGARCH. The variance process is very close
to a unit root; this explains why the long-horizon forecasts converge to the unconditional mean and
carry no information.

---

# Stage 6: Attention BiLSTM

`scripts/06_attention_bilstm.py`. The aim is a hypothesis test: is XGBoost's loss to the linear
specifications specific to the model, or is it a general limit of nonlinear
modelling in this problem?

Same as the primary specification: fold structure, embargo, 2026 rule, preprocessing order,
ratio target, Duan smearing from training residuals, metrics. The architecture is chosen from the tiered table
declared in advance that is the BiLSTM counterpart of the capacity rule. Lookback L=20,
declared, not tuned. No early stopping; the justification is the weak selection signal
measured in the Optuna experiment.

## All models (fold-mean RMSE)

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR-X-log | 0.010335 | 0.007561 | 0.007504 | 0.008012 |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR-log | 0.010857 | 0.008678 | 0.008199 | 0.008275 |
| XGBoost | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| GARCH | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| past-vol | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| train-mean | 0.013451 | 0.011243 | 0.009239 | 0.008916 |
| BiLSTM | 0.014227 | 0.010512 | 0.011589 | 0.011431 |

BiLSTM R²_oos is negative at all four horizons: −0.371, −0.249, −2.218, −3.094. It beats past-volatility
at no horizon, and train-mean only at h=22.

## The answer to the hypothesis test

| horizon | XGBoost vs HAR-X | BiLSTM vs HAR-X |
| --- | --- | --- |
| 5 | +5.53% | +37.58% |
| 22 | +18.39% | +37.07% |
| 66 | +17.84% | +53.03% |
| 126 | +7.05% | +41.83% |

**Two independent families of nonlinear models both fail to beat the linear HAR-X
with the same information.** The result is not specific to XGBoost.

## Degenerate-forecast check

0/60 folds degenerate (std(forecast)/std(actual) < 0.1). The form of the failure is not collapsing to a
constant value but **over-dispersion**: at h=126 the mean ratio is 1.486, i.e. the model predicts a series
~1.5 times more volatile than it really is. The model mistakes noise for signal.

## Robustness check: under-training

In the primary run 8/60 folds came out as "under-trained" (in the final 20% epoch slice the training
loss still falls by >10%); all are in the high tier, at h=5 and h=22.

**Criterion declared in advance:** "train until the decrease in training loss over the final 10% epoch slice
falls below 2%, upper bound 200 epochs, lower bound the declared tier's
epoch count." Applied only to the high tier. The criterion derives entirely from the **training loss**
and does not look at test performance; therefore it is not a change made by looking at the test
set.

**Difference note:** in this mode the `T_max` value of the cosine lr schedule is 200; in the primary run it
was the declared epoch count. The comparison is not "the same schedule, more epochs" but
the "trained until convergence" state.

**Training really got longer and the loss really fell.** At h=5 the folds ran on average
124 epochs instead of 60 (range 60–198); in late folds the final training loss fell from 0.107 to 0.025,
i.e. ~4-fold. At h=22 the criterion was met immediately, 62 epochs on average.

**But test performance did not change.**

| horizon | primary | with convergence criterion | change | folds where robustness is better |
| --- | --- | --- | --- | --- |
| 5 | 0.014227 | 0.014374 | +1.03% | 6/15 |
| 22 | 0.010512 | 0.010423 | −0.85% | 5/15 |
| 66 | 0.011589 | 0.011589 | 0.00% | 0/14 |
| 126 | 0.011431 | 0.011431 | 0.00% | 0/14 |

h=66 and h=126 came out exactly identical; those tiers were not adaptive, and the results being identical to six
decimal places **is a confirmation that determinism works**.

**Interpretation: the BiLSTM's failure is not under-training but overfitting.** Lowering the training loss
4-fold did not improve the test RMSE; at h=5 it made it slightly worse. The gap to HAR-X
stayed practically the same (+39.0 / +35.9 / +53.0 / +41.8).

**A limitation of the diagnostic tool.** The stopping criterion (final 10%, <2%) triggered in all 24 high-tier
folds, but the independent reporting classifier (final 20%)
still shows 12 folds as "under-trained" in the robustness run. Among the tail-decrease
values there are negatives too (−12.9, −1.5), i.e. the training loss is noisy between epochs
and both windows measure noise more than trend. The criterion may at times have triggered on a
flat pair by chance. A smoothed loss curve would have been a better diagnostic
tool; this is recorded as a limitation.

**The result does not change the primary specification.** The outputs are in the `bilstm_*_conv.*` files.

---

# Stage 7: Hybrid Models

`scripts/07_hybrid.py`. A direct test of the hybrid econometric-ML structure pointed to by the
Gunnarsson et al. (2024) review: does ML make a complementary contribution to HAR-X?

The weights are fixed at 0.5/0.5, declared in advance, not optimized. H1 and H2 require no new training
(averages of saved forecasts); that the test rows are identical at all four horizons
and the `y_true` values match exactly was verified with an assert. H3's base
HAR-X was refit, with an assert that it equals the standalone HAR-X within a tolerance of
`atol=1e-10`.

## Main question: did the hybrids beat HAR-X standalone?

Fold-mean RMSE, percent difference relative to HAR-X. Negative = hybrid better.

| hybrid | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| H1 XGBoost + BiLSTM | +12.28 | +22.43 | +30.59 | +20.94 |
| H2 HAR-X + XGBoost | −0.01 | +4.35 | +4.33 | +0.67 |
| H3 HAR-X residual modelling | +8.21 | +14.98 | +23.24 | +5.96 |

**No hybrid beats HAR-X on the main metric.** H1 is very bad as expected, the average of two weak
components. H3 loses at every horizon. H2 ties at h=5 and is behind at the other three
horizons.

## RMSE (fold mean)

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR-X-log | 0.010335 | 0.007561 | 0.007504 | 0.008012 |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |
| H2 HAR-X+XGBoost | 0.010340 | 0.008002 | 0.007901 | 0.008113 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| XGBoost | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| H3 HAR-X residual | 0.011190 | 0.008818 | 0.009333 | 0.008539 |
| H1 XGBoost+BiLSTM | 0.011611 | 0.009390 | 0.009890 | 0.009747 |
| BiLSTM | 0.014227 | 0.010512 | 0.011589 | 0.011431 |

## Diagnostic 1: does the residual stage extract information? (H3)

`R²_resid = 1 − SSE(e − ê)/SSE(e)`, its baseline is "not forecasting the residual".

| horizon | in-sample | out-of-sample | out-of-sample median | out-of-sample positive folds |
| --- | --- | --- | --- | --- |
| 5 | +0.863 | −0.240 | −0.166 | 3/15 |
| 22 | +0.873 | −0.573 | −0.365 | 3/15 |
| 66 | +0.698 | −1.460 | −0.334 | 4/14 |
| 126 | +0.579 | −0.270 | +0.038 | 8/14 |

**This table is the most direct answer to the hypothesis.** XGBoost explains 58–87% of HAR-X's in-sample
residuals, but out-of-sample R² is **negative** at all four horizons:
not using the residual model at all (i.e. leaving HAR-X as it is) gives a better result.
Only 3–8 of the folds show a positive contribution. **ML cannot extract any generalizable
information from HAR-X's residuals.**

**The anticipated bias did not materialize.** In the plan we had noted that training the residual model on in-sample
residuals could create a bias. The train/test ratio of the residual standard deviation
came out at 0.96–1.21, i.e. close to 1. Since the base model is an OLS with 6 regressors, this
bias turned out to be negligible. It is not the cause of the failure; the residual signal is simply not
learnable.

## Diagnostic 2: correlation of the component errors

| hybrid | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| H1 XGBoost–BiLSTM | 0.690 | 0.830 | 0.763 | 0.776 |
| H2 HAR-X–XGBoost | 0.875 | 0.800 | 0.814 | 0.871 |

The correlations are far from 1, i.e. there is theoretically potential for diversification. But H2 still
cannot win, because the XGBoost component is systematically worse than HAR-X and the average carries
it.

## An aggregation difference — not hidden

For H2 the pooled RMSE and the fold mean give results **in opposite directions**.

| horizon | HAR-X pooled | H2 pooled | H2 pooled difference | H2 fold-mean difference | folds H2 wins |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.011612 | 0.011381 | −1.99% | −0.01% | 6/15 |
| 22 | 0.009265 | 0.009247 | −0.20% | +4.35% | 3/15 |
| 66 | 0.009894 | 0.009879 | −0.15% | +4.33% | 5/14 |
| 126 | 0.009810 | 0.009534 | −2.81% | +0.67% | 7/14 |

On the pooled measure H2 is slightly better than HAR-X at all four horizons. On the fold mean it is
worse at three horizons. Why: H2 wins markedly in a few high-variance years, and pooling weights these
years; the fold mean gives equal weight to every year, and H2
loses in the majority of folds (fewer than half at all four horizons).

**Our main metric is the fold mean** (CLAUDE.md "Metric reporting rule"), so
the result is "H2 did not beat HAR-X". The pooled value is also reported.

## Conclusion

None of the three hybrid structures beat HAR-X standalone on the main metric. The residual-modelling
diagnostic directly shows that ML cannot extract generalizable information from the residuals HAR-X
leaves. The hybrid direction pointed to by Gunnarsson et al. found no counterpart in this data and at these
horizons.

---

# Exploratory / Secondary Analysis: volatility regime and the aggregation difference

`scripts/07b_exploratory_vol_regime.py`.

**This is an exploratory analysis. It does not change the primary finding and is not used for model
selection.** Its aim is to shed light on a single question: why do the pooled RMSE and the fold mean
give results in opposite directions for H2 (the HAR-X + XGBoost average)?

**The split criterion is mechanical and was not chosen by looking at the results.** For each horizon, the
mean realized volatility of each test year is computed; the years are split in two at the median of this value
into a high and a low regime. The threshold derives from the data, not from
performance. The median is computed separately per horizon because the set of folds included in the main metric
changes with the horizon.

## H2's difference relative to HAR-X, by regime (%, negative = H2 better)

| measure | regime | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| fold mean | low | −0.31 | +5.23 | +8.89 | +3.95 |
| fold mean | high | +0.18 | +3.89 | +2.17 | −1.01 |
| pooled | low | +0.49 | +4.35 | +6.76 | +5.59 |
| pooled | high | −2.75 | −1.16 | −1.24 | −4.56 |

The weight of the high regime in the pooled measure (squared-error share): h=5 76.8%,
h=22 82.9%, h=66 86.9%, h=126 83.5%.

## Finding: the regime difference is only part of the explanation

The pooled measure weights the high-volatility years at 77–87%, and H2 wins in that group
on the pooled measure. That much supports the hypothesis.

**But H2 does not systematically win in high-volatility years.** Within the same group
the fold mean still shows H2 behind at three of the four horizons, and H2 beats HAR-X in only
2/7, 1/7, 3/7 and 4/7 of the high-regime folds — i.e. it loses in the
majority.

## The actual mechanism: concentration of extreme observations

| horizon | pooled share of the worst 1% of observations | H2's difference on those observations |
| --- | --- | --- |
| 5 | 38.8% | −10.8% |
| 22 | 44.8% | −12.3% |
| 66 | 40.6% | −9.3% |
| 126 | 24.3% | −13.6% |

The worst 1% of observations make up between a quarter and a half of the pooled error,
and H2 is 9–14% better than HAR-X exactly on those observations. Everywhere else it is slightly worse.

The same pattern at fold level. At h=22 H2 wins by 9.6% in the 2020 fold (that year's RMSE
0.0252, the highest of all folds) and loses in every one of the other six high-regime years.
At h=126 it wins 2020 (−9.4%), 2019 (−4.1%) and 2025 (−6.0%), and loses by 38.5% in 2022.

**Interpretation:** the value XGBoost adds to the average is limiting the error in rare and severe volatility
spikes. These dominate the pooled measure but count as a single year in the fold
mean. Not regime, but sensitivity to extreme events.

This does not change our choice of main metric (fold mean, CLAUDE.md "Metric reporting
rule") and H2 is counted as not having beaten HAR-X in the primary result. But it is a nuance worth
discussing in the paper: which measure is right depends on whether, in practice, the error in rare shock periods
or the error in a typical year matters.

Outputs: `explore_vol_regime_years.csv`, `explore_vol_regime_groups.csv`,
`explore_vol_regime_folds.csv`, `explore_vol_regime_summary.json`.

## Extreme-event robustness check: "extreme events" or "COVID"?

Two claims of different strength are at issue. The robustness check determined which one is right.

### Calendar distribution of the worst 1% of observations

| year | h=5 | h=22 | h=66 | h=126 | total |
| --- | --- | --- | --- | --- | --- |
| 2020 | 25 | 32 | 35 | 16 | 108 |
| 2019 | 5 | 0 | 0 | 19 | 24 |
| 2022 | 3 | 4 | 0 | 0 | 7 |
| 2026 | 2 | 1 | 0 | 0 | 3 |
| 2015 | 2 | 0 | 0 | 0 | 2 |

2020's share among the extreme observations: h=5 67.6%, h=22 86.5%, **h=66 100%**, h=126 45.7%.

### With the 2020 fold removed entirely (threshold recomputed)

H2's difference on the extreme observations (%, negative = H2 better):

| sample | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| all years | −10.78 | −12.33 | −9.35 | −13.58 |
| 2020 excluded | **+3.01** | **+5.12** | −2.79 | −3.58 |

H2's difference relative to HAR-X on the pooled measure, excluding 2020: +3.16, +8.81, +6.80, +0.68.
**The sign changed at all four horizons.** H2's pooled superiority turned out to come entirely from
2020.

In the 99% outside the extreme observations, H2 is consistently worse in both samples
(between +0.4 and +9.5).

### Critical nuance: at long horizons removing 2020 is not enough

Year distribution of the extreme observations in the sample excluding 2020:

| horizon | distribution |
| --- | --- |
| 5 | 2014:3, 2015:5, 2016:2, 2019:5, 2021:6, 2022:4, 2026:10 |
| 22 | 2015:3, 2022:16, 2026:15 |
| 66 | 2019:21, 2022:7, 2025:5 |
| 126 | **2019:33 (all of them)** |

All of the 2019 extreme observations at h=126 come from the range 27.11.2019 – 31.12.2019.
The target of these rows looks 126 trading days ahead, i.e. **it extends to about May 2020
and measures the COVID crash.** Although the fold label is 2019, the observation is a COVID observation.

Therefore at long horizons removing the 2020 fold does not remove COVID from the sample.
At h=66 21 of the 33 extreme observations, at h=126 all 33, are still in the COVID window. This explains why the
small H2 advantage measured excluding 2020 at h=66 and
h=126 (−2.79%, −3.58%) persists.

### Conclusion: which claim should be written

**The correct claim is "ML added complementary value during the COVID period", not "in extreme events".**

- At h=5 and h=22 COVID can really be removed, and H2's extreme-observation advantage changes
  sign and turns into a disadvantage. Other extreme periods such as the 2015–2016 oil crash, the 2022 war period and 2026
  are present in the sample, but H2 does not win there.
- At h=66 and h=126 the test is inconclusive: removing the 2020 fold does not really remove COVID from the sample,
  because it does not remove the 2019 rows whose target window extends into 2020.

The evidence required for a general "extreme event" claim is not there. A finding resting on a single event must be
reported as a single event.

Outputs: `explore_tail_robustness.csv`, `explore_tail_year_distribution.csv`.

---

# Stage 8: Diebold-Mariano Tests

`scripts/08_dm_test.py`. Squared-error loss, pooled across folds, Newey-West
(Bartlett) HAC correction `L = h−1`, Harvey-Leybourne-Newbold small-sample correction,
Holm-Bonferroni multiple-testing correction (20 tests). In addition a distribution-free fold-level
sign test (binomial).

## The most important result: the main finding is NOT statistically significant

HAR-X vs XGBoost:

| horizon | RMSE difference | DM (HLN) | p | Holm p | sign | sign p | sign Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | −2.52% | −0.604 | 0.546 | 1.000 | 10/15 | 0.302 | 1.000 |
| 22 | −9.46% | −1.384 | 0.167 | 1.000 | 13/15 | 0.007 | 0.089 |
| 66 | −9.15% | −1.468 | 0.142 | 1.000 | 11/14 | 0.057 | 0.631 |
| 126 | −0.98% | −0.142 | 0.887 | 1.000 | 10/14 | 0.180 | 1.000 |

The point estimates consistently favour HAR-X at all four horizons, but **there is no statistical
significance at any horizon.** The sign test passes 5% with the raw p at h=22; after the Holm
correction that one falls too.

**In the paper this finding CANNOT BE WRITTEN as "HAR-X significantly beats XGBoost".** The correct
statement: the point estimates consistently favour HAR-X, but once the autocorrelation created by the overlapping windows
is taken into account the difference cannot be distinguished statistically.

## Survivors after the Holm correction

DM test (3/20):

| horizon | pair | difference | Holm p |
| --- | --- | --- | --- |
| 5 | HAR-X vs past-vol | −19.79% | <0.0001 |
| 5 | XGBoost vs past-vol | −17.71% | <0.0001 |
| 5 | HAR-X vs BiLSTM | −23.17% | <0.0001 |

Sign test (8/20): in addition to the above, h=5 HAR-X vs GARCH, h=22 HAR-X vs past-vol,
h=22 HAR-X vs GARCH, h=22 HAR-X vs BiLSTM, h=66 HAR-X vs GARCH.

So the only thing that can be robustly established: HAR-X and XGBoost beat the naive and weak
alternatives (past-volatility, BiLSTM, GARCH) at short horizons. The difference between the two good models
cannot be distinguished.

## Why the HAC correction was mandatory

Variance inflation factor (HAC variance / variance under an independence assumption):

| horizon | min | mean | max |
| --- | --- | --- | --- |
| 5 | 1.7 | 2.5 | 2.9 |
| 22 | 3.4 | 6.0 | 12.1 |
| 66 | 4.8 | 19.8 | 38.8 |
| 126 | 12.7 | 42.8 | 74.5 |

An uncorrected DM test would have underestimated the standard error at h=126 **by up to 8.6-fold**.
The necessity is numerically confirmed.

## Empirical confirmation of the lag rule

In the plan we had noted that `L = h−1` is a LOWER BOUND and that, since the forecasts are not optimal, the
autocorrelation could extend beyond it. The ACF diagnostic did not confirm this concern
— the autocorrelation of the loss difference drops to zero at `h−1`:

| horizon | ACF(1) | ACF(h−1) | ACF(h) | mean ACF [h, 2h] |
| --- | --- | --- | --- | --- |
| 5 | 0.615 | −0.021 | −0.090 | −0.016 |
| 22 | 0.677 | −0.068 | −0.035 | −0.010 |
| 66 | 0.713 | −0.014 | −0.009 | +0.017 |
| 126 | 0.730 | +0.006 | +0.004 | −0.012 |

The declared rule turned out to be sufficient; a longer lag is not needed.

## Agreement of the two tests

Direction agreement 17/20. All three disagreements are in the past-volatility comparisons and at long horizons:

| horizon | pair | RMSE difference | DM | sign |
| --- | --- | --- | --- | --- |
| 66 | XGBoost vs past-vol | −8.56% | −0.798 | 7/14 |
| 126 | HAR-X vs past-vol | −11.23% | −0.824 | 7/14 |
| 126 | XGBoost vs past-vol | −10.35% | −0.837 | 5/14 |

The pooled RMSE shows a marked difference but wins in half of the folds or fewer.
This is the same extreme-observation concentration we found in Stage 7's exploratory analysis:
the pooled measure is under the influence of a few severe episodes.

## Assessment

In this study the sign test produced more significant results than DM (8 versus 3) and does not depend on
any of the HAC bandwidth debate. In the 17 tests where the two tests point the same way, the
finding is immune to the bandwidth choice. On the main finding (HAR-X vs XGBoost), however,
neither produces significance, i.e. the weakness of the result does not stem from the choice of method.

## Extended test family (32 tests)

Three pairs were added: HAR vs HAR-X (the main decomposition claim), HAR vs past-volatility,
HAR-X vs HAR-X-log. The Holm correction was **recomputed** over 8 pairs × 4 horizons = 32 tests,
so the threshold tightened relative to the earlier 20-test family.

### Main decomposition claim: HAR vs HAR-X

| horizon | pooled difference | DM (HLN) | p | Holm p | folds HAR-X wins | sign p | sign Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | +1.49% | +0.357 | 0.722 | 1.000 | 12/15 | 0.035 | 0.738 |
| 22 | +4.81% | +0.655 | 0.512 | 1.000 | 13/15 | 0.007 | 0.170 |
| 66 | −1.28% | −0.202 | 0.840 | 1.000 | 10/14 | 0.180 | 1.000 |
| 126 | −2.52% | −0.531 | 0.596 | 1.000 | 9/14 | 0.424 | 1.000 |

Positive difference = HAR worse, i.e. HAR-X better.

**The contribution of the exogenous variables is not statistically significant either.** The sign test passes the
threshold with the raw p at short horizons (at h=22 13/15 folds, p=0.007) but after the 32-test Holm
correction none survives.

**At long horizons the measures change direction:**

| horizon | pooled | fold mean |
| --- | --- | --- |
| 5 | +1.49% | +4.77% |
| 22 | +4.81% | +12.13% |
| 66 | −1.28% | +6.95% |
| 126 | −2.52% | +1.79% |

At h=66 and h=126 the pooled RMSE points to HAR, the fold mean and the sign test to HAR-X.
Again extreme-observation concentration: HAR is better than HAR-X in a few severe episodes,
worse in a typical year.

### The other two pairs

**HAR vs past-volatility:** at h=5 both DM (−18.59%, Holm p<0.0001) and the sign test
(15/15, Holm p=0.002) are significant. No significance at long horizons.

**HAR-X vs HAR-X-log:** at all four horizons no test shows significance (DM p = 0.21, 0.14,
0.32, 0.38; sign p = 0.61, 0.61, 0.79, 0.79). The levels and the log-log specification
cannot be distinguished statistically. **This is a useful null result:** our having chosen levels as the primary
specification does not change the results.

### Survivors after Holm (32-test family)

DM test, 4/32 — all at h=5 and all against weak alternatives:

| pair | difference | Holm p |
| --- | --- | --- |
| HAR-X vs BiLSTM | −23.17% | <0.0001 |
| HAR-X vs past-vol | −19.79% | <0.0001 |
| HAR vs past-vol | −18.59% | <0.0001 |
| XGBoost vs past-vol | −17.71% | <0.0001 |

Sign test, 9/32: the sign counterparts of the four above, plus at h=5 and h=22 HAR-X vs
GARCH, at h=22 HAR-X vs past-vol and HAR-X vs BiLSTM, at h=66 HAR-X vs GARCH.

### Direction agreement

24/32. Seven of the eight disagreements are at h=66 and h=126, six are in past-volatility or HAR
comparisons. All from the same mechanism: the pooled measure is under the influence of a few severe
episodes, the fold mean and the sign test reflect the typical year.

### Overall assessment

After the multiple-testing correction, the only class of claim that can be statistically defended is
that the good models (HAR, HAR-X, XGBoost) beat the naive and weak alternatives (past-volatility,
BiLSTM, GARCH) **at short horizons**. The study's two main claims —
"HAR-X beats XGBoost" and "the gain comes from the exogenous variables" — while showing a consistent direction in the point
estimates, cannot be distinguished statistically.
The paper should build these claims in the language of effect size and directional consistency instead of the language of
significance.

## Separation of the test families and FWER / FDR

The tests were split into two families and the corrections were computed **separately within each family**. For each family
both Holm (FWER, confirmatory claims) and Benjamini-Hochberg (FDR, exploratory
findings) are given side by side; the reader can choose their own threshold.

**Transparency record:** the primary family corresponds to the study's two main claims declared in
Stages 5 and 6, and was not chosen by looking at p-values. However, the family definition
was formalized **after** the tests; this is not a pre-registration and must be stated as such in the
paper.

### Primary (confirmatory) family: 2 comparisons × 4 horizons = 8 tests

| horizon | comparison | difference | DM p | DM Holm | DM BH | sign | sign p | sign Holm | sign BH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | +1.49% | 0.722 | 1.000 | 0.887 | 3/15 | 0.035 | 0.211 | 0.094 |
| 22 | HAR vs HAR-X | +4.81% | 0.512 | 1.000 | 0.887 | 2/15 | 0.007 | 0.059 | **0.030** |
| 66 | HAR vs HAR-X | −1.28% | 0.840 | 1.000 | 0.887 | 4/14 | 0.180 | 0.718 | 0.239 |
| 126 | HAR vs HAR-X | −2.52% | 0.596 | 1.000 | 0.887 | 5/14 | 0.424 | 0.718 | 0.424 |
| 5 | HAR-X vs XGBoost | −2.52% | 0.546 | 1.000 | 0.887 | 10/15 | 0.302 | 0.718 | 0.345 |
| 22 | HAR-X vs XGBoost | −9.46% | 0.167 | 1.000 | 0.666 | 13/15 | 0.007 | 0.059 | **0.030** |
| 66 | HAR-X vs XGBoost | −9.15% | 0.142 | 1.000 | 0.666 | 11/14 | 0.057 | 0.287 | 0.115 |
| 126 | HAR-X vs XGBoost | −0.98% | 0.887 | 1.000 | 0.887 | 10/14 | 0.180 | 0.718 | 0.239 |

Number of surviving tests (5% threshold):

| test | Holm (FWER) | BH (FDR) |
| --- | --- | --- |
| DM | 0/8 | 0/8 |
| sign | 0/8 | **2/8** |

**The strongest defensible statement:** under FDR control, according to the sign test, both
main claims are supported at h=22. HAR-X beats XGBoost in 13 of the 15 folds
(BH p = 0.030) and HAR-X beats HAR in 13 of the 15 folds (BH p = 0.030). At the other three
horizons there is no support.

The DM test produces significance under no condition in the primary family; the raw p-values
are in the range 0.142–0.887, i.e. shrinking the family does not help — the test has no power on this data.

### Secondary / exploratory family: 6 comparisons × 4 horizons = 24 tests

| test | Holm (FWER) | BH (FDR) |
| --- | --- | --- |
| DM | 4/24 | 6/24 |
| sign | 9/24 | 9/24 |

The two tests BH adds to Holm in DM: h=22 HAR-X vs BiLSTM (BH p = 0.023) and h=22
HAR-X vs past-volatility (BH p = 0.035). Both are against weak alternatives.

In the sign test the two corrections keep the same 9 tests standing; all are at h=5, h=22 and h=66, and
all against weak alternatives (past-volatility, BiLSTM, GARCH).

**HAR-X vs HAR-X-log** is not significant under any correction at all four horizons (BH p = 0.27–0.43,
sign BH p = 0.77–0.90). The choice between levels and log-log does not affect the
results — a useful null result.

### Conclusion

In the confirmatory framework (primary family, FWER) neither of the study's two main claims is statistically
supported. In the FDR framework they are supported only at h=22 and only with the sign test.
All the strong results in the secondary family point to the good models beating the naive and weak
alternatives at short horizons — this is an expected finding of low information
value.

The paper should build its main claims in the language of **effect size and directional consistency** instead of the
language of significance, and report the FDR support at h=22 in a qualified way.

## Power analysis: could these differences have been detected with this data?

`scripts/09_power_analysis.py` → `outputs/power_analysis.csv`. The 8
comparisons in the primary family, 80% power and a 5% two-sided threshold.

**Method.** For the DM test the sampling unit is the effective independent block: `B = n/h` (at h=126
3500/126 ≈ 28). Standardized effect `δ = |DM| / √B`; the required non-centrality
`z₀.₉₇₅ + z₀.₈₀ = 2.802`, hence `B_required = (2.802/δ)²`. For the sign test the
sampling unit is directly the fold (year); the number of folds is increased until the exact binomial power reaches 80%.
Trading days per year were measured from the test period: 244.1.

**Assumptions.** The observed effect is taken to be the true effect; the error structure does not change as the sample
grows; `B = n/h` is a rough approximation. "Year" is a test-period year.

**Caveat.** The "realized power" computed from the observed effect is a monotone
transformation of the p-value and carries no information beyond the p-value. The actual answer is in the required-sample
columns.

### DM test

| horizon | comparison | effective blocks | realized power | required years | fold increase |
| --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 732 | 0.065 | 927 | 62× |
| 22 | HAR vs HAR-X | 166 | 0.101 | 273 | 18× |
| 66 | HAR vs HAR-X | 53 | 0.055 | 2 764 | 193× |
| 126 | HAR vs HAR-X | 28 | 0.083 | 400 | 28× |
| 5 | HAR-X vs XGBoost | 732 | 0.093 | 323 | 22× |
| 22 | HAR-X vs XGBoost | 166 | 0.283 | 61 | 4× |
| 66 | HAR-X vs XGBoost | 53 | 0.312 | 52 | 4× |
| 126 | HAR-X vs XGBoost | 28 | 0.052 | 5 582 | 389× |

**The realized power of the DM test does not exceed 0.32 in any test.** That is, in this design the DM
test never had a chance of catching these differences. In the most optimistic case (h=66, HAR-X vs
XGBoost) a 52-year test period would have been needed; we have 14 years.

### Sign test

| horizon | comparison | folds | ratio | realized power | required years | fold increase |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 15 | 12/15 | 0.648 | 20 | 1.3× |
| 22 | HAR vs HAR-X | 15 | 13/15 | **0.871** | 15 | 1.0× |
| 66 | HAR vs HAR-X | 14 | 10/14 | 0.190 | 42 | 3.0× |
| 126 | HAR vs HAR-X | 14 | 9/14 | 0.076 | 94 | 6.7× |
| 5 | HAR-X vs XGBoost | 15 | 10/15 | 0.210 | 72 | 4.8× |
| 22 | HAR-X vs XGBoost | 15 | 13/15 | **0.871** | 15 | 1.0× |
| 66 | HAR-X vs XGBoost | 14 | 11/14 | 0.396 | 25 | 1.8× |
| 126 | HAR-X vs XGBoost | 14 | 10/14 | 0.190 | 42 | 3.0× |

**At h=22 both comparisons already have sufficient power (0.871 > 0.80).** This is important:
the significant results at h=22 are not a low-powered fluke; they mean that the study was sufficiently
powerful at that horizon and detected the effect.

At the other horizons the power is insufficient: at h=5 for HAR-X vs XGBoost 0.21 (72 years needed),
at h=126 for HAR vs HAR-X 0.076 (94 years needed).

### Conclusion

In this design the sign test is **one to two orders of magnitude more efficient** than DM. The required extension for DM
is 4–389 times the current sample, for the sign test 1–6.7 times. This quantifies why in Stage 8
the sign test produced more significant results: fold-level
consistency carries much more information than overlapping daily observations.

**Implication for the paper.** The statement "no significant difference was found" does not mean "no difference"
in this study; the design did not have the power to detect the difference in most comparisons.
h=22 is the exception, and there both main claims are supported with sufficient power. The Limitations
section should contain this table.

## Prior (a priori) power curves

This section **does not use the observed results at all**; it answers the question "what could this design
have detected" in terms of hypothetical effect sizes. It is not circular.
Outputs: `apriori_power_sign.csv`, `apriori_power_dm.csv`.

### Sign test: number of folds × true win probability

The minimum number of wins required for 5% two-sided significance is **12 at n=15, 12 at n=14**.
That is, one needs to win at least 12 (80%) of 15 years, at least 12 (86%) of 14 years.

| true p | n=14 | n=15 |
| --- | --- | --- |
| 0.60 | 0.040 | 0.092 |
| 0.65 | 0.084 | 0.173 |
| 0.70 | 0.161 | 0.297 |
| 0.75 | 0.281 | 0.461 |
| 0.80 | 0.448 | 0.648 |
| 0.85 | 0.648 | **0.823** |
| 0.90 | **0.842** | 0.944 |

**This design can reliably detect only a model that wins 85% (n=15) or 90% (n=14) of the
years.** A model that is truly superior but wins 70% of the years is detected with 15 folds
with a probability of only 30%.

### DM test: number of effective blocks × hypothetical RMSE difference

Transformation: `δ_block = k·|r²−1|`, `ncp = √B·δ_block`. The coefficient `k` is calibrated from the **noise**
structure of the data (the error correlation of the two models and the loss distribution), not from the observed
**effect** size. Mean of the two pairs in the primary family, per horizon:
h=5 → 0.444, h=22 → 0.557, h=66 → 1.124, h=126 → 1.700.

| horizon | effective blocks | 5% difference | 10% difference | 20% difference |
| --- | --- | --- | --- | --- |
| 5 | 732 | 0.216 | 0.626 | **0.991** |
| 22 | 166 | 0.108 | 0.276 | 0.733 |
| 66 | 53 | 0.126 | 0.343 | **0.838** |
| 126 | 28 | 0.142 | 0.401 | **0.899** |

Power is not monotone in the number of blocks because `k` changes with the horizon: h=5 has the most blocks
(732) but the smallest `k` (0.444); h=126 the opposite.

The sensitivity to the uncertainty in `k` is narrow: ±0.02–0.06 at h=66 and h=22, marked only at h=126
(0.284–0.528 at a 10% difference).

### Conclusion: the detection limit of the design

| test | reliable detection limit |
| --- | --- |
| DM | about **20%** RMSE difference (except h=22, where even 20% gives 0.73) |
| sign | winning **85–90%** of the years |

The observed values stayed below these limits: RMSE differences of 1–9%, win rates of
64–87%. **The study was structurally underpowered for effects of the size it was looking for.**

The two significant results at h=22 (13/15 = 86.7% win rate) fall right above the detection
limit — this explains why they are the only significant results and
supports that they are not flukes.

**For the paper:** this table shows a priori, without recourse to the observed results, why the statement "no significant difference was found"
does not mean "no difference". These two tables should be given in the Limitations
section.

---

# Stage 9: SHAP Analysis

`scripts/10_shap_analysis.py`. Question: do XGBoost and HAR-X use the same variables?

TreeSHAP was computed with XGBoost's built-in `pred_contribs` feature (the `shap` package
was not needed); the additivity property was verified with an assert in every fold. The models were retrained exactly as in the
primary specification, SHAP on each fold's test slice.
**Unit caveat:** since the model is trained in log-ratio space, the SHAP values are in that
unit too.

## (1) Group shares — the actual comparison

XGBoost (share of total |SHAP|, %):

| group | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| brent_vol | 47.3 | 44.1 | 34.2 | 30.8 |
| gpr | 19.5 | 16.5 | 11.3 | 9.9 |
| ovx | 18.0 | 18.3 | 30.2 | 29.8 |
| brent_fiyat | 11.3 | 15.6 | 18.5 | 26.1 |
| takvim | 1.7 | 4.1 | 5.0 | 1.8 |
| etkilesim | 2.1 | 1.4 | 0.8 | 1.6 |

HAR-X (share of total |std beta|, %):

| group | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| ovx | 72.2 | 72.8 | 66.1 | 59.1 |
| brent_vol | 19.9 | 18.8 | 14.4 | 19.6 |
| gpr | 8.0 | 8.5 | 19.5 | 21.2 |

**The two models weight very differently.** HAR-X leans overwhelmingly on OVX
(59–73%). XGBoost's primary source is Brent volatility (31–47%); it gives OVX only
18–30%.

## (2) Weight outside HAR-X

The total share XGBoost gives to groups HAR-X does not use at all (Brent price/return, interaction, calendar):
**h=5 15.1%, h=22 21.1%, h=66 24.3%, h=126 29.5%.**

It increases as the horizon lengthens. The calendar features take a 4–5% share at h=22 and h=66; there is
no economic justification for this and it is most likely fitting to noise.

The most important features pick the right window for the horizon: at h=5 `vol5_vol60` (14.5%),
at h=22 `brent_vol20` (18.7%), at h=66 `brent_vol60` (13.9%), at h=126 `brent_vol126`
(21.3%). The model finds the volatility window matching the horizon by itself.

## (3) Sign agreement — two design flaws were found and corrected

**Flaw 1 (code error, corrected).** If the model made no split on a feature at all, its
SHAP values are identically zero and the direction is undefined. In the first implementation these were counted as
"opposite direction". This is wrong: it means "the model did not use that feature". Corrected and now
reported separately.

Share of unused comparisons: h=5 0%, h=22 2.7%, **h=66 44.3%, h=126 67.1%.**
At long horizons the low capacity tier (16 units, depth 2) does not use most features
at all.

**Flaw 2 (design limit, documented).** XGBoost is trained on the log-ratio target;
`brent_vol5` and `brent_vol20` are mechanically related to the `past_vol_h` in the denominator.
At h=5 `brent_vol5` is the denominator itself (correlation 1.000), at h=22 the correlation with `brent_vol20`
is 0.991. For these features a negative SHAP direction is a mechanical consequence of mean reversion,
it does not mean "an opposite relation was learned". HAR-X's beta, on the other hand, is on the level target.
**For these two features the comparison is invalid.**

The valid comparison is with the exogenous regressors only:

| horizon | comparisons | agreeing | agreement |
| --- | --- | --- | --- |
| 5 | 45 | 24 | 53.3% |
| 22 | 43 | 31 | 72.1% |
| 66 | 15 | 9 | 60.0% |
| 126 | 7 | 5 | 71.4% |
| **total** | **110** | **69** | **62.7%** |

By regressor:

| regressor | folds used | agreement | XGB direction corr. | HAR-X beta |
| --- | --- | --- | --- | --- |
| ovx_lag1 | 46 | **82.6%** | +0.373 | +0.581 |
| gprd_threat_lag1 | 30 | 50.0% | +0.166 | −0.026 |
| gprd_lag1 | 34 | 47.1% | −0.001 | +0.021 |

**On OVX the two models agree strongly:** the direction matches in 38 of 46 folds, positive in
both. OVX is also HAR-X's dominant regressor. On the GPR variables the agreement is at
chance level (47–50%), but HAR-X's GPR betas are very small anyway
(|β| ≈ 0.02–0.03), i.e. there is no strong direction there to compare.

## (4) Stability across folds

| horizon | mean consecutive ρ | min | max | ρ to pooled | distinct features entering the top 5 | most frequent first | stays first |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.858 | 0.772 | 0.943 | 0.870 | 11 | vol5_vol60 | 10/15 |
| 22 | 0.866 | 0.788 | 0.955 | 0.895 | 12 | brent_vol20 | 11/15 |
| 66 | 0.863 | 0.670 | 0.926 | 0.861 | 12 | brent_vol60 | 5/14 |
| 126 | 0.845 | 0.657 | 0.993 | 0.802 | 12 | brent_vol126 | 6/14 |

**The ranking does not drift.** The mean Spearman between consecutive folds is 0.85–0.87, the
lowest value 0.66. The SHAP interpretation is not fragile. The top-ranked feature is stable in two thirds of the folds
at short horizons, more volatile at long horizons (5–6/14). The share of the top five features
in total attribution rises from 42% to 62%, i.e. at long horizons importance is more concentrated.

## (5) Spearman — uninformative, as warned

Group level (3 items): ρ = −0.5, +0.5, −0.5, −0.5; p = 0.667 in all.
Common-regressor level (5 items): ρ = −0.2, +0.5, −0.6, +0.5; p = 0.28–0.75.

None of them is interpretable. As stated in the plan, this measure cannot produce significance at these
sizes; the group-share table is the actual evidence.

## Conclusion

**No, they do not use the same variables.** HAR-X gives two thirds of its weight to OVX;
XGBoost looks primarily at the Brent volatility window and spends
15–30% of its attention on groups HAR-X does not use at all (price level, calendar, interaction).
At long horizons this share increases, and at the same time it does not use 44–67% of the features
at all.

On the common ground, i.e. OVX, the two models find the same direction (83% agreement). So XGBoost does not learn the basic
relation wrongly; it spreads its attention to different places, and part of that
(like the calendar variables) is probably noise.

This finding also explains Stage 7: the residual modelling could extract nothing
out of sample, because the information XGBoost could add to HAR-X consists largely of
noise.

## Correction to the stability measure: overlap caveat and distant fold pairs

The consecutive-fold correlations above **cannot on their own be counted as evidence of stability.**
In an expanding window the training set of fold k is a subset of that of fold k+1; consecutive
folds share on average **88–89% of their training data.** Under this condition a high Spearman
correlation is partly mechanical.

The most honest measure is the pair with the least overlap, **the first fold and the last fold**.

| horizon | consecutive ρ | consecutive overlap | first-last ρ | first-last overlap |
| --- | --- | --- | --- | --- |
| 5 | 0.858 | 89.1% | **0.773** | 19.4% |
| 22 | 0.866 | 89.0% | **0.785** | 19.1% |
| 66 | 0.863 | 88.3% | **0.686** | 19.4% |
| 126 | 0.845 | 87.9% | **0.529** | 18.2% |

By fold distance, the correlation falls steadily together with the overlap:

| distance | overlap | ρ (h=5) | ρ (h=22) | ρ (h=66) | ρ (h=126) |
| --- | --- | --- | --- | --- | --- |
| 1 | 88–89% | 0.858 | 0.866 | 0.863 | 0.845 |
| 4 | 62–65% | 0.802 | 0.821 | 0.802 | 0.775 |
| 8 | 39–43% | 0.724 | 0.766 | 0.714 | 0.740 |
| 13 | 18–23% | 0.697 | 0.752 | 0.686 | 0.529 |

**Corrected interpretation.** At short horizons the stability is real: even when the overlap falls to 19%
the correlation stays at 0.77–0.79. At long horizons, however, the stability is markedly weak.
At h=126 the first-last correlation is 0.529, i.e. far below what the consecutive measure shows.

This is consistent with the other findings: at h=126 the number of effective independent observations is ~28, the model
does not use 67% of the features at all and the attribution ranking really moves
between years. **At long horizons the SHAP interpretation is fragile and must be qualified as such in the paper.**

Output: `shap_stability_by_distance.csv`.


---

# Stage 10: Exogenous-Variable Ablation (OVX and GPR separately)

`scripts/11_ablation_exogenous.py`. **What was looked for:** an external review claimed that the GPR
block in HAR-X makes no contribution, and even does harm. The claim was reproduced in our own pipeline,
with the same fold structure, embargo, NaN mask, train-only floor and metrics.
Four nested OLS (level) specifications: HAR, HAR + OVX, HAR + GPR, HAR-X (HAR + OVX +
GPR). The script verifies with an assert that `har` and `har_x` come out identical to `bench_aggregate_all.csv` and
that all variants see the same train/test rows.

## Fold-mean RMSE

| variant | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR + OVX | **0.010298** | **0.007603** | **0.007469** | **0.008003** |
| HAR + GPR | 0.010882 | 0.008668 | 0.008198 | 0.008262 |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |

(The best within these four variants is in bold. At h=22 HAR-X log-log, with 0.007561, is ahead of HAR + OVX
as well.)

## What was found

- **OVX's contribution:** relative to HAR, RMSE falls by 4.9%, 11.6%, 7.8%, 2.4%. HAR + OVX beats HAR in
  13/15, 14/15, 12/14, 10/14 folds.
- **GPR's contribution is negative:** added to HAR on its own it worsens by 0.4%, 0.8%, 1.2%, 0.7%;
  added on top of OVX by 0.4%, 0.9%, 1.4%, 0.7%. The direction is the same at all four
  horizons.
- **Not significant at fold level:** HAR + OVX beats HAR-X in 9/15, 11/15, 9/14, 9/14 folds;
  exact sign test p = 0.61, 0.12, 0.42, 0.42. When the worst fold (at h=5, 22, 66
  2024; at h=126 2017) is removed, the mean difference is still positive at every horizon, i.e. the result
  does not come from a single fold.
- **Coefficients:** the fold mean of the standardized GPR coefficients is between −0.06 and +0.08;
  OVX's is 0.49–0.65. The two GPR components take opposite signs (GPRD mostly
  positive, GPRD_THREAT mostly negative) and partly cancel each other. At the long horizon
  the sign is unstable: at h=126 GPRD is positive in 9/14 folds, GPRD_THREAT in 5/14. At h=5, by contrast, the sign
  is largely stable (13/15 and 4/15 positive); saying "it drifts at every horizon" would not be correct.

## Effect on the interpretation

The statement "OVX and GPR contribute" in the v1.0.0 README **was wrong** and was
corrected in v1.1.0. The correct statement: all of the exogenous gain comes from OVX; GPR does small but
directionally consistent harm, and this harm is not significant at fold level. The GPR daily index
is published with weekly updates; `gprd_lag1` assumes that yesterday's value is known at forecast
time, which is an assumption in GPR's favour. Despite this assumption there is no contribution, so
the finding is conservative.

HAR-X stays in the main comparison as it was specified before the ablation; it **was not replaced** with
HAR + OVX, which later turned out better (CLAUDE.md Rule 5, cherry-picking ban). HAR +
OVX and HAR + GPR are reported as separate rows.

Outputs: `ablation_exogenous*.csv`, `ablation_exogenous_summary.json`. The forecasts
were added later as `ablation_exogenous_predictions.csv` (for Stage 13); the existing
outputs stayed byte-identical on the re-run.

---

# Stage 11: Date-Gap Diagnostics

`scripts/14_date_gap_diagnostics.py`. **What was looked for:** since the data were merged on the common dates of
the four series, a day missing in one series drops the whole row. In that case the "daily" return
computed from consecutive rows may span more than one trading day.

## What was found

- In 3600 (77.6%) of the 4640 returns the difference between consecutive rows is 1 calendar day,
  in 898 (19.4%) 2-3 days, in 142 (3.1%) 4 days or more.
- In 209 rows at least one weekday was skipped. 169 of these are fully explained by NYSE
  holidays. All 86 UK holidays falling on trading days are present in the data,
  i.e. the merged series follows the US calendar.
- **Real gaps: 40 rows, 55 skipped trading days.** All between 2008–2016; none after
  2017. The largest is the 17-day gap in April 2009. The single-day gaps in 2008–2013
  mostly fall on the 13th–19th of the month; this is consistent with contract roll-over days but
  was not verified. Which series is missing cannot be told from the merged file.

## Two counts: not an inconsistency, a different denominator

The number of observations with a real gap in the target window was counted in two different sets:

| horizon | full sample (2008–2026) | test folds (2012–2026) |
| --- | --- | --- |
| 5 | 198 / 4636 (4.3%) | 80 / 3662 (2.2%) |
| 22 | 734 / 4619 (15.9%) | 328 / 3645 (9.0%) |
| 66 | 1182 / 4575 (25.8%) | 494 / 3500 (14.1%) |
| 126 | 1482 / 4515 (32.8%) | 614 / 3500 (17.5%) |

In the earlier ad hoc diagnostic the figure "198" appeared for h=5, in the direct test "80". Both are
correct, but they count different things. 198 is the full-sample count that also includes the 2008–2011 warm-up
period used only in training. 80 is the count of the test targets that enter the main out-of-sample
evaluation. **The number that affects the results is the test count; that is the one used
in the README.** The test count is computed independently in two scripts and matched with an
assert in Stage 13.

## Effect on the interpretation

The gaps pose no leakage risk; all computations rely on row order and do not look into the
future. The problem concerns measurement consistency. Its effect was measured in Stages 12 and 13.

Outputs: `date_gap_distribution.csv`, `date_gap_rows.csv`, `date_gap_by_year.csv`,
`date_gap_target_exposure.csv`, `date_gap_diagnostics_summary.json`.

---

# Stage 12: Gap-Free Subsample Robustness Check (2017–2026)

`scripts/12_robustness_gapfree.py`. **What was looked for:** the main comparison was repeated with the 2017–2026
folds, which contain no date gaps at all. The model was not retrained; the existing fold
metrics were re-averaged. The full sample reproduces the main table exactly (assert). The sub-
sample was determined a priori by the Stage 11 diagnostic, not by looking at the test result.

## What was found

| horizon | folds (full → 2017+) | Spearman RMSE rank | Spearman MAE rank |
| --- | --- | --- | --- |
| 5 | 15 → 10 | 0.955 | 0.964 |
| 22 | 15 → 10 | 0.982 | 0.991 |
| 66 | 14 → 9 | 0.836 | 0.873 |
| 126 | 14 → 9 | 0.673 | 0.700 |

At h=5 and h=22 the ranking is preserved. At h=66 the top three models stay the same, but train-mean rises from 9th
to 4th place. At h=126 train-mean rises from 8th to 1st place, and in this period
the R²_oos of all models is negative; the difference among the top five models is about 3%.

**But this change stems from the period, not from the gaps.** The 2012–2016 ranking also differs from the full
sample; there train-mean is 11th at h=66 and h=126. A gap-induced
distortion would shift the target of all models in the same way. Here, instead, the relative success of a single model
(a constant forecast) changes greatly between the periods, i.e. the two periods have
different volatility regimes. This check alone cannot separate the gap effect from the period effect;
this is why Stage 13 was done.

## Effect on the interpretation

The argument "if the ranking does not change, the gaps have no effect" cannot be built with this check; at the long horizon
the ranking changes. The finding that does not change: in the 2017+ period too the HAR family is ahead of XGBoost
and the BiLSTM at every horizon. The h=126, 2017+ finding (no model beats train-mean)
must be written in the Limitations section.

Outputs: `robustness_gapfree_2017plus.csv`, `robustness_gapfree_2017plus_summary.json`.

---

# Stage 13: Direct Target-Correction Test

`scripts/13_gap_target_test.py`. **What was looked for:** separating the gap effect from the period effect.
If a real gap return r spans 1 + k trading days, it was scaled to its single-day
equivalent with r / sqrt(1 + k) (random-walk assumption: variance grows linearly with time).
The target was rebuilt from the corrected returns with the same formula. Keeping all models' published
forecasts **fixed**, fold RMSE and MAE were recomputed. Only
the evaluation target changes; the features and training targets were not corrected, the model was not
retrained. 12 models, including HAR + OVX and HAR + GPR.

## What was found

| horizon | changed test targets | mean target change | RMSE change | RMSE rank changes | MAE rank changes |
| --- | --- | --- | --- | --- | --- |
| 5 | 80 / 3662 | −13.1% | −0.15% … +0.16% | 0 | 2 (train-mean ↔ BiLSTM) |
| 22 | 328 / 3645 | −3.3% | −0.10% … +0.26% | 0 | 0 |
| 66 | 494 / 3500 | −2.3% | +0.20% … +0.42% | 0 | 2 (XGBoost ↔ past-vol) |
| 126 | 614 / 3500 | −2.0% | +0.20% … +0.49% | 0 | 0 |

The number of changed targets is identical to the Stage 11 test count (assert). The two pairs that
swap places in MAE were already nearly equal (difference 0.2% and 0.06%). With the corrected target too,
HAR + OVX beats HAR-X and HAR beats HAR + GPR; i.e. the Stage 10 finding is not sensitive
to the gaps either.

## Effect on the interpretation

**The date gaps do not change the reported results.** They shift the metrics by at most 0.5%
and the RMSE ranking changes at no horizon. In the paper this test should be given as the primary evidence,
and Stage 12 as an additional robustness analysis. Limitation: the gap returns on the training side, including the
training data of the 2017+ folds, were not corrected.

Outputs: `gap_target_test.csv`, `gap_target_test_folds.csv`, `gap_target_test_summary.json`.


---

# Stage 14: Standardization of the Ablation Coefficients

`scripts/11_ablation_exogenous.py` (extended). **What was looked for:** the raw coefficients
cannot be compared; the regressors are at unscaled levels (OVX in the tens, GPRD in the
hundreds), and a difference in units can be read as a difference in effect. For each fold and horizon
beta_std = beta × sd(X) / sd(y) was computed; the sds from that fold's training slice. The `har_x`
rows come out identical to `harx_standardized_betas.csv` (assert).

**Number of folds:** each variant is estimated in 15 folds at each horizon (240 rows = 4 variants
× 4 horizons × 15 folds). The primary metric aggregation has 14 folds at h=66 and h=126 (2026 partial-
year rule). That is, 15 estimation folds, at long horizons 14 metric folds. The summary file gives the two
sets separately (`fold_set` = `main` / `all_estimated`).

## What was found (fold_set = main; mean, positive/negative folds)

| horizon | HAR-X: ovx_lag1 | HAR-X: gprd_lag1 | HAR-X: gprd_threat_lag1 | HAR+GPR: gprd_lag1 | HAR+GPR: gprd_threat_lag1 |
| --- | --- | --- | --- | --- | --- |
| 5 | +0.584 (15/0) | +0.028 (13/2) | −0.024 (4/11) | +0.040 (15/0) | −0.056 (0/15) |
| 22 | +0.650 (15/0) | +0.019 (11/4) | −0.028 (6/9) | +0.032 (11/4) | −0.063 (4/11) |
| 66 | +0.586 (14/0) | +0.077 (13/1) | −0.057 (3/11) | +0.091 (11/3) | −0.093 (3/11) |
| 126 | +0.487 (14/0) | +0.040 (9/5) | −0.007 (5/9) | +0.052 (9/5) | −0.035 (5/9) |

- The GPR coefficients are roughly a tenth of OVX's or smaller.
- The two GPR components systematically have opposite signs. Their correlation in the training slices is
  0.82–0.89; the opposite sign is consistent with this high collinearity, and their net effects partly
  cancel each other.
- The sign is stable at the short horizon: in HAR+GPR at h=5 GPRD is 15/15 positive, THREAT 15/15 negative.
  At the long horizon it is unstable: at h=126 9/5 and 5/9.
- When OVX is added, the standardized coefficient of brent_vol20 falls from 0.44–0.62 to 0.06–0.13.
  OVX takes over most of the persistence information.

## Effect on the interpretation

Consistent with the statement in the README: the GPR coefficients are near zero relative to OVX, the two components have opposite
signs, the sign is unstable at the long horizon. Saying "the sign drifts from fold to fold at every horizon"
would be wrong; at h=5 the sign is fully stable.

Outputs: `ablation_exogenous_coefficients_standardized.csv`,
`ablation_exogenous_std_beta_summary.csv`. The existing ablation outputs did not change at the
byte level.

---

# Stage 15: XGBoost-6 — an exploratory run isolating the functional form

`scripts/15_exploratory_xgb6.py`. **Status: exploratory and post hoc**, the same as the ablation
ladder. Not part of the primary hypothesis family, not used in model selection, does not change the primary
specification. Designed after the main results were known.

**What was looked for:** the XGBoost vs HAR-X comparison changes three things at once: the model
family, the input set (65 vs 6) and the target/preprocessing (log-ratio + smearing + on the features
log1p/winsorize/MinMax vs level, no transformation). XGBoost-6 changes only the model family:
HAR-X's six raw regressors, level target, no smearing, no preprocessing, the same
train/test rows as HAR-X (assert), the same train-min floor. The capacity rule and the shared
parameters were imported from `03_walkforward.py`. Note: in the primary XGBoost winsorization is applied
only to the features, not to the target.

## What was found (fold-mean RMSE, folds entering the main metric)

| horizon | XGBoost-6 | HAR-X | XGBoost primary | XGB-6 vs HAR-X | primary vs XGB-6 | XGB-6 / HAR-X winning folds | sign p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.010645 | 0.010341 | 0.010913 | +2.94% | +2.51% | 3 / 12 | 0.035 |
| 22 | 0.008184 | 0.007669 | 0.009080 | +6.72% | +10.94% | 4 / 11 | 0.118 |
| 66 | 0.008015 | 0.007573 | 0.008924 | +5.84% | +11.34% | 2 / 12 | 0.013 |
| 126 | 0.008050 | 0.008059 | 0.008627 | −0.12% | +7.17% | 7 / 7 | 1.000 |

In MAE XGBoost-6 is 4.3%, 10.3%, 10.9%, 2.6% worse than HAR-X (sign p = 0.035, 0.035,
0.057, 0.42). Capacity tiers: at h=5 15 folds high; at h=22 9 high and 6 medium;
at h=66 9 medium and 5 low; at h=126 2 medium and 12 low. No forecast hit the floor.

- **Functional form, with inputs and target fixed:** at h=5, 22, 66 the flexible form is 3–7% worse than the linear
  form. At h=126 no difference (−0.12%, 7 to 7). The sign tests are uncorrected; when Holm is applied to the 4 RMSE
  tests none stays below 0.05 (the smallest 0.013 × 4 = 0.052).
- **The additional loss of the primary XGBoost:** 65 features + log-ratio target + smearing + preprocessing
  together are 2.5–11.3% worse than XGBoost-6. Part of the difference between the primary XGBoost and HAR-X
  (5.5%, 18.4%, 17.8%, 7.0%) comes from the functional form, the rest from this
  package.
- XGBoost-6 beats plain HAR at all four horizons, because it sees OVX. It does not beat the best HAR-family
  model (HAR + OVX or HAR-X-log) at any horizon.

## Effect on the interpretation

The claim "nonlinear modelling does not contribute" holds also with inputs and target fixed,
but more measuredly: at h=5–66 the flexible form gives a small but consistent loss,
at h=126 a tie. The primary XGBoost's large loss comes not only from the functional form, but
also from the rich feature set and the target reparametrization. This decomposition should be reported in the paper
as exploratory; it does not replace the primary comparison.

Outputs: `exploratory_xgb6.csv`, `exploratory_xgb6_folds.csv`,
`exploratory_xgb6_predictions.csv`, `exploratory_xgb6_summary.json`.

---

# Stage 16: GPR publication-date alignment (accessibility correction)

**Trigger — transparency record.** This work was **triggered by an independent external
code review** that examined the project's code. The review pointed out that the GPR features use information
not accessible at forecast time. It will appear in this form in the paper's transparency
statement.

**Problem.** The daily GPR index (Caldara and Iacoviello) is not observed in real time like OVX and Brent.
It is published in batches (weekly, plus a monthly update), and later vintages
revise past values. The model, however, used GPR every day with a one-day lag:
row t saw the observation dated t−1. This formally complies with the `.shift(1)` rule
but is an **accessibility violation**. On most days the t−1 observation had not yet been published at time t.
Measurement (Stage 16.2): under timestamp alignment **80.3%** of the rows use an unpublished
observation. This is a leak contrary to the spirit of Critical Rule 6 ("information not known at time t cannot be used").
Because the code passed the `.shift()` check, it was not caught in earlier
audits.

The correction was made in three steps. Steps 1 and 2 in commits `20cbfab` and `a1482d6`, Step 3
in commit `4ef8c39`.

## 16.1 Establishing the publication rule (`scripts/16_gpr_vintages.py`)

**Source (accessed 2026-09-24):** the authors' vintage archive on GitHub
(`iacoviel/iacoviel.github.io/gpr_archive_files`). There are **289 daily vintages**, between 2022-02-24
and 2026-09-21. The page's statement: "The daily data are updated every Monday … If the
first day of the month or week falls on a federal holiday, data updates will take place
the next business day."

**Rule: the file published on day D contains the observations up to and including D.** **279** of the 289
vintages support this rule (file date = last observation date). Of the 10 exceptions
7 are 1 day, 1 is 2 days, 1 is 3 days behind. All of these are start-of-month updates; the file contains data
up to the last day of the previous month. One exception is 124 days behind
(the 2023-01-02 file, last observation 2022-08-31). This looks like a stale upload.

Vintage weekdays: Monday 207, Tuesday 42, Wednesday 14, Thursday 12, Friday 14.

**Publication lag per observation** (calendar days from the observation date to the first vintage that contains it):
mean 2.84, median 3, max 10. Median lag by the observation's weekday:
Monday 0, Tuesday 6, Wednesday 5, Thursday 4, Friday 3, Saturday 2, Sunday 1. Stable across
years (2022–2026 mean 2.76–2.88).

**Revisions.** The relative difference between the first release and the current vintage is on average −5.2% for GPRD
(mean absolute 12.7%, median absolute 10.0%), on average −3.7% for GPRD_THREAT (mean
absolute 14.6%, median absolute 11.3%). The revisions are not small and have a systematic sign.

**Scope caveat: before 2022 it is counterfactual.** The vintage archive starts on 2022-02-24.
The daily GPR was made public with Caldara and Iacoviello (2022). For 2008–2021 there is no such thing as "the value
published on that day". For this period the rule assumes that today's publication regime was also valid in the
past: observation d is published on the first Monday after d (d included);
if that day is a federal holiday, the next business day.

**Revisions were not modelled.** The values come from `data/veriseti.xlsx` (identical to the 2026-09-01
vintage), i.e. they are current-vintage values. The publication alignment corrects the **timing**,
not the **revision**. A fully real-time design would use the first-release values;
this is possible only after 2022. Recorded as a limitation.

## 16.2 Publication-aligned features (`scripts/02_build_features.py`, `scripts/gpr_publication.py`)

**Method.** The GPR features (lag, EMA, z-score, spike, momentum, threat ratio, interactions)
are computed on the index's **own observation sequence**, then mapped to trading
days by publication date. Row t sees the latest observation published up to t−1: p(d*) ≤ t−1.
Forward-filling the level series onto the trading calendar and deriving from it would discard 78% of the published
observations. **One day conservative:** the vintages are published at ~13:30 UTC, before the Brent close,
i.e. same-day use would have been possible. Still, p(d*) ≤ t−1 is required for consistency with the
`.shift(1)` rule of the other predictors.

**27** of the 65 features changed (all derivatives of GPRD and GPRD_THREAT and the four OVX × GPR
interactions). The first fully populated row is 127 in both versions. The timestamp-aligned `features.csv` is
reproduced bit for bit. A correction in the same commit: first-release tracking for dates
added in a later vintage was corrected (2024-02-29).

**Effective lag** (between trading day t and the date of the GPR observation used):

| measure | timestamp-aligned | publication-aligned |
| --- | --- | --- |
| calendar days, median | 1 | 3 |
| calendar days, mean | 1.47 | 3.69 |
| calendar days, max | 17 | 20 |
| trading rows, median / mean / max | — | 3 / 2.97 / 8 |

In the publication-aligned version, median calendar days by trading weekday: Monday 7, Tuesday 1,
Wednesday 2, Thursday 3, Friday 4. Monday sees the oldest information, because under the t−1 rule
that day's release can be used only on Tuesday.

**Tests (all passed):**

- **Prefix-invariance:** the features were recomputed with the data cut at rows 3000 and
  4000. All rows before the cut are identical to those computed with the full data (tolerance 0).
  No feature looks at future rows.
- **Publication sensitivity, with a control arm:** in 40 random rows, all GPR observations not yet
  published at that row's date were perturbed (136–4065 observations per row).
  **Publication-aligned arm: 40/40 rows unchanged.** **Control arm (timestamp-aligned): 33/40
  rows changed**, i.e. the test can catch the leak.

  **Explanation of the 7 unchanged rows in the control arm.** The timestamp arm always uses at row t
  the observation dated t−1. The perturbation is applied only to observations not published up to
  t−1. Therefore a control row changes only if the t−1 observation had not been published by
  t−1.
  - **6 Tuesday rows.** The t−1 observation is dated Monday. Monday is the observation day with the
    smallest publication lag (median 0): the Monday observation is published the same day in the Monday
    file.
  - **1 Wednesday row (2023-09-06).** Labor Day week. The update shifted to Tuesday (09-05)
    and included the Tuesday observation, i.e. the t−1 observation had again been published the same day.

  Verification: from the release calendar, the question "was the t−1 observation published by t−1?" predicts
  the control arm's result correctly in **40** of the 40 rows (7 of the 7 published rows did not
  change, 33 of the 33 unpublished rows changed). Therefore not detecting these 7 rows
  is not a weakness of the test but the expected consequence of the measured publication-lag structure:
  in these rows the timestamp alignment already uses an accessible observation and there is
  nothing to perturb.

  _Sentence suggested for the paper:_ "The seven control rows that did not change are exactly
  those whose t−1 observation had already been released by t−1 (six Tuesdays, whose t−1
  is a Monday released the same day, and one Wednesday in Labor Day week, when the update
  moved to Tuesday); the release calendar predicts the control outcome in 40 of 40 rows,
  so non-detection reflects the measured release schedule, not a weakness of the test."

## 16.3 Re-running all models that use GPR

`scripts/alignment.py` adds a shared switch: `--gpr-alignment publication`
reads `features_publication_aligned.csv` and appends the `_publication_aligned` suffix to every output
it writes and every output it reads (the two versions never mix in any chain). Re-
run: 03 (XGBoost), 04 (Optuna, shrunk and raw smearing), 05 (benchmarks),
06 (BiLSTM), 07 (hybrids), 07b, 08 (DM), 10 (SHAP), 11 (ablation), 12, 13, 15 (XGB-6).
75 `_publication_aligned` files were produced. The comparison was done with
`scripts/17_gpr_alignment_comparison.py`.

**Same sample, pure alignment effect.** In every model × horizon × fold the train/test rows are the same in the two
versions (assert). The models that use no GPR (HAR, HAR-log, HAR+OVX, GARCH, train-mean,
past-volatility) are **bit-identical** in 360 fold rows. The difference comes only from the
alignment.

**RMSE change, fold mean** (`100 × (publication / timestamp − 1)`, positive =
the cost of respecting the publication lag; in parentheses the folds in which the publication version is better and the
uncorrected sign p):

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| XGBoost | +0.40% (8/15) | −2.31% (12/15, p=0.035) | +1.12% (6/14) | +0.56% (7/14) |
| BiLSTM | −5.53% (9/15) | −1.20% (6/15) | +5.33% (9/14) | +1.35% (4/14) |
| H1 | −1.67% | −2.41% | +3.59% | +1.21% |
| H2 | +0.26% | −0.49% | +1.08% | +0.28% |
| H3 | −0.45% | −1.88% | +0.12% | +1.21% |
| HAR-X | +0.02% | +0.49% | +0.80% | +0.36% |
| HAR-X-log | −0.23% | +0.64% | +0.23% | −0.44% |
| HAR + GPR | +0.01% | +0.45% | +0.60% | +0.41% |
| XGB-6 | +1.81% | +2.73% (2/15, p=0.007) | +0.15% | +1.11% |
| Optuna (shrunk) | +1.67% | +4.03% | +7.98% | −0.30% |
| Optuna (raw) | +1.82% | +3.08% | +0.50% | +1.69% |

- **In the linear models the cost is very small:** HAR-X between +0.02% and +0.80%; no sign
  test is significant. This is consistent with GPR already carrying almost zero weight in
  HAR-X.
- **The flexible models are more volatile:** the BiLSTM changes by −5.5% at h=5, +5.3% at h=66, in both directions.
  This is not a "loss of leak gains" but the sensitivity of a high-variance model to a change in its
  input. XGBoost is **better** in the publication version at h=22 (−2.3%, 12/15). Uncorrected
  p=0.035; within 11 models × 4 horizons it must not be counted as significant under multiple comparisons.
- **The main findings did not change:**
  - The exogenous contribution comes from OVX.
  - GPR adds nothing on top of OVX at all four horizons: HAR-X is worse than HAR+OVX by
    +0.43% / +1.36% / +2.21% / +1.06%.
  - The nonlinear models beat HAR-X at no horizon.
- **Primary hypothesis family (8 tests):** the survivors under BH are **the same two tests**: at h=22
  HAR-X > HAR and HAR-X > XGBoost, sign test 13/15, BH p = 0.030. DM 0/8 (HLN p
  0.113–0.876). What changed turns no result around:
  - h=5 HAR vs HAR-X: HAR-X wins 12/15 → 11/15 (raw p 0.035 → 0.118).
  - h=66 HAR-X vs XGBoost: 11/14 → 10/14 (0.057 → 0.180).
  - h=126 HAR-X vs XGBoost: 10/14 → 9/14.
- **Secondary family (24 tests):** in the sign test the survivors 9/24 (Holm and BH) → 6/24 Holm,
  8/24 BH. DM 4/24 Holm, 6/24 BH, unchanged.
- **Decomposition** (HAR → HAR-X → XGB-6 → XGB), publication-aligned:
  - exogenous: −4.53 / −10.39 / −5.75 / −1.41%
  - functional form: +4.78 / +9.11 / +5.16 / +0.63%
  - feature package: +1.09 / +5.50 / +12.42 / +6.59%

  The functional-form cost increased at h=5 and h=22, and at h=126 went from −0.12 to +0.63. The direction
  did not change.
- **The weight of GPR moves in two different directions:**
  - In the XGBoost SHAP the GPR group share **increased**: 19.5% → 25.5, 16.5 → 29.8, 11.3 → 20.0,
    9.9 → 15.3.
  - In HAR-X the GPR betas **moved towards** zero: gprd_lag1 at h=22 +0.019 → +0.001,
    at h=126 +0.040 → +0.006.
  - Interpretation: a SHAP share measures use, not contribution. The weekly step-like GPR series
    offer the trees more split points, but XGBoost's accuracy does not increase. The GPR group,
    with 23 features, is the most crowded group; the share is inflated for this reason too.
- **2026 partial year (h=66/126, informational):** most of the models that use GPR are 1–16% worse in the publication
  version, XGB-6 is −6.7% / −6.2% better. It does not enter the main metric; 41–101
  overlapping observations, effective observations ~1.

**Decision.** The primary results are now the publication-aligned version. The timestamp-aligned version is moved to the paper's
Appendix A, and the two-version comparison is reported there. The model specification,
the hyperparameter rule and the test families were not changed. The only thing that changed is the timing of the GPR
features.

## 16.4 Re-running the robustness checks in publication mode

**Data equalization (Stage 5), `bench_*_aligned_publication_aligned`.** The benchmarks were restricted to
XGBoost's window (row 127). Difference relative to XGBoost (h=5/22/66/126):

| model | normal window | equalized window |
| --- | --- | --- |
| HAR | −1.11 / −3.05 / −10.25 / −5.44 | −1.12 / −3.13 / −10.36 / −6.74 |
| HAR-X | −5.60 / −13.12 / −15.41 / −6.77 | −5.58 / −13.01 / −15.37 / −8.29 |
| HAR-X-log | −5.89 / −14.21 / −16.66 / −8.04 | −5.81 / −14.19 / −16.81 / −9.48 |

**The result did not change:** after equalization the HAR family's superiority is preserved, and at h=126 it again
increases somewhat. The timestamp reference was reproduced bit for bit with the same command
(Stage 5 table). Runtime 32 seconds.

**BiLSTM convergence check (Stage 6), `bilstm_*_conv_publication_aligned`.** The same command
(`--convergence-mode --suffix _conv`), the same criterion declared in advance. Runtime 29 minutes.

| horizon | primary | with convergence criterion | change | folds where convergence is better | (timestamp: change, folds) |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.013440 | 0.013518 | +0.58% | 7/15 | +1.03%, 6/15 |
| 22 | 0.010386 | 0.010653 | +2.58% | 3/15 | −0.85%, 5/15 |
| 66 | 0.012207 | 0.012207 | 0.00% | 0/14 | 0.00% |
| 126 | 0.011585 | 0.011585 | 0.00% | 0/14 | 0.00% |

At h=66 and h=126 there are no high-tier folds; the results are identical by definition (the determinism
confirmation holds again). **The test error did not improve**: worse by +0.6% at h=5, +2.6% at h=22.

**However, the premise the check rests on was not repeated in this version.**
- **Training did not get longer.** In the timestamp version the folds at h=5 had run 124 epochs on average;
  the late folds (2021–2026) ran 182–198 epochs and the training loss fell 4.1–6.1-fold per fold.
  In the publication version the stopping criterion triggered at 62–78 epochs in 14 of the 15 folds.
  Only the 2025 fold ran 119 epochs. Mean 72 epochs.
- **The loss decrease is small.** The final training-loss ratio (primary / convergence) has a median of 1.36;
  in the timestamp version the median was 2.40, ≥2 in 8 folds. In the publication version only 1 fold is ≥2.
- **The independent classifier objects.** By the final 20% window, 11/15 folds at h=5 are
  still "under-trained" (4/15 in the primary run).

This is a direct consequence of the diagnostic limitation recorded in Stage 6. The stopping criterion (final 10%,
<2% decrease) can trigger by chance on a flat pair of epochs in a noisy loss curve.
The change in the GPR input changed the loss trajectory and this time the criterion triggered
early.

**Interpretation.** The finding "not under-training but overfitting" is **not supported by this check in the
publication version, but not refuted either**. The additional training did not improve the test error.
But the check did not reach the regime on which the evidence in the timestamp version rests (a large
decrease in training loss); i.e. in this version it does not measure the effect of really lowering the training loss
on the test error. The over-dispersion indicator also weakened in this version:
the mean std(forecast)/std(actual) is ~1.0 at h=5 and h=22 (timestamp: 1.16 and 0.96),
1.31 and 1.68 at h=66 and h=126 (timestamp: 1.23 and 1.49). The over-dispersion argument
holds only at long horizons.

**Open decision (for the user):** should this claim in the paper (a) be written in a weakened form, or
(b) should a stronger check be run? (b) could be done, for example, with a fixed 200
epochs in the high-tier folds; estimated runtime ~70 minutes. That would be a check designed after the result was seen
and must be recorded as such; it does not change the primary specification. The criterion must again
derive only from the training loss and must not look at test performance.

**Decision: (b) was chosen.** The result is in 16.5.

## 16.5 Exploratory check: BiLSTM fixed 200 epochs (`scripts/19_bilstm_fixed_epochs.py`)

**Status: exploratory and post hoc.** Designed after the convergence result in 16.4 was seen.
It does not change the primary specification and is not used in model selection. The number of epochs
(200) is the upper bound declared in advance in 06; it was not chosen by looking at test performance.
Publication-aligned version, runtime 69 minutes.

**Design:**
- Early stopping entirely off; a fixed 200 epochs.
- Scope: the high-tier folds, 15 at h=5, 9 folds at h=22. At h=66 and h=126 there is no high
  tier.
- The cosine lr schedule is the same as in the convergence run (`T_max=200`). Hence this run's
  k-th epoch is the convergence run that stopped at k itself. Asserted in every fold:
  the training loss at k and the test forecasts at k are bit-identical (24/24).
  The "k epochs vs 200 epochs" comparison is two points on a single training path; the stopping
  rule does not interfere. The primary run (60 epochs, `T_max=60`) is a separate path, given side by
  side.
- The per-epoch training loss was recorded (`bilstm_fixed200_loss_history`). In addition, at k and at
  200, the MSE on the whole training set was computed in eval mode (dropout off); this is a cleaner
  measure of fit.

**Interpretation rule written in advance (before the result was seen):**
- If the loss falls sharply and the test error does not improve, the overfitting argument is built without the stopping rule
  interfering.
- If the loss again does not fall, the model is capacity-constrained, not epoch-constrained.

**Result (main folds, high tier):**

| | h=5 (15 folds) | h=22 (9 folds) |
| --- | --- | --- |
| k (stopping epoch), mean | 72 | 62 |
| training loss k → 200, median factor | **2.48×** (1.51–4.65) | **3.09×** (2.58–3.50) |
| eval-mode training MSE k → 200, median factor | 2.40× (1.41–6.29) | 3.08× (2.52–4.94) |
| training loss primary (60 ep) → 200, median factor | 3.63× | 2.21× |
| folds whose loss fell ≥2-fold | 14/15 | 9/9 |
| test RMSE, primary / k / 200 | 0.013440 / 0.013518 / **0.014493** | 0.012137 / 0.012583 / **0.013170** |
| test RMSE change, 200 vs k | **+7.21%** | **+4.67%** |
| folds where 200 is better than k (sign p, exploratory, uncorrected) | 2/15 (p = 0.007) | 3/9 (p = 0.51) |
| test MAE change, 200 vs k | +5.22% | +3.32% |
| std(forecast)/std(actual), k → 200 | 1.00 → 1.10 | 1.02 → 1.05 |

At h=22, when the 200-epoch results of the 9 high-tier folds are combined with the other 6 folds,
the horizon-mean RMSE rises from 0.010653 to 0.011006 (+3.31%). At h=5 all folds are
in the high tier; horizon mean +7.21%.

**Interpretation.** The first branch of the interpretation rule was realized:
- Without the stopping rule the training loss fell sharply: median 2.5× (h=5) and 3.1× (h=22).
  The loss fell at least 2-fold in 23 of the 24 folds. The eval-mode MSE fell to the same extent; i.e.
  the decrease is not dropout noise but a real increase in fit.
- The test error did not improve, it worsened: +7.2% at h=5 (worse in 13 of the 15 folds,
  p = 0.007), +4.7% at h=22 (worse in 6 of 9, p = 0.51).

  **Status of these p-values:** they come from an exploratory diagnostic, **are NOT PART of the primary
  hypothesis family of eight** and are **not corrected** for multiple comparisons.
  Their status is the same as HAR+OVX vs HAR-X's uncorrected p = 0.035. The primary family
  (HAR vs HAR-X, HAR-X vs XGBoost × 4 horizons) was fixed in advance; no test is ever
  added afterwards. In the paper these p-values are given not as inferential evidence but to
  describe the direction of the diagnostic.
- Forecast dispersion increased at h=5 (1.00 → 1.10). This is consistent with the reading that the model
  learns training noise.

**The finding "not under-training but overfitting" is supported by this check in the publication-aligned
version**, and without interference from early stopping. The "~4×" decrease of the timestamp version
is here a median of 2.5–3.1× (fold range 1.5–4.7×). In the paper the figure should be given from this
file; the 4× figure of the timestamp version belongs to Appendix A.

This check does not change the primary specification; the primary BiLSTM stays with the declared tier
rule of 60/40/30 epochs.

Outputs: `bilstm_fixed200_folds_publication_aligned.csv`,
`bilstm_fixed200_loss_history_publication_aligned.csv`,
`bilstm_fixed200_predictions_publication_aligned.csv`,
`bilstm_fixed200_summary_publication_aligned.json`.

---

# Stage 17: Number package (`scripts/18_paper_numbers.py`)

`outputs/paper_numbers_publication_aligned.md` is the single source of all numbers used in the paper and the
root README. It is generated only from saved outputs; no model is trained. Its header
records the commit it was generated from, the generation time and whether the working tree was clean.

**Consistency checks (assert):**
- Each fold's RMSE is recomputed from the prediction files and compared with the stored metric
  (960 and 1020 rows in the two versions).
- The fold means are the same as in `gpr_alignment_comparison.csv`.
- Train-mean R²_oos is exactly 0 in every fold.
- 08's stored BH values are reproduced in the package.

**Finding 1: the R²_oos reference was mixed across sources.** `hybrid_metrics_all`
computes R²_oos against the target mean of the XGBoost training window. `bench`,
`ablation`, `exploratory_xgb6` and `opt_*`, on the other hand, compute it against each model's own training window.
Since the HAR family starts at row 21 and XGBoost at 127, these references differ.
When the two were mixed in one table, the models were being measured against different constant forecasts.

- **Correction:** in the package the R²_oos of all models is recomputed from the forecasts with a single common reference per fold
  (the forecast of the train-mean baseline). For the models sourced from the hybrid file
  it equals the stored value (assert).
- **Effect:** the only sign change is in XGBoost-6, at h=126, +0.010 → −0.018. The other
  changes are at most ~0.09 and do not change sign.
- RMSE/MAE are not affected by this. The source files were not changed; this is given in the package with a separate
  note.

**Finding 2: the two sign tests at h=22 are not independent; Benjamini-Yekutieli was added.**
The two tests that survive under BH in the primary family (h=22: HAR-X > HAR and HAR-X > XGBoost,
both 13/15, raw p 0.007) were examined.

- **No column reuse.** The two fold-difference vectors differ (assert). The years HAR-X
  loses are 2020 and 2024 against HAR, 2014 and 2020 against XGBoost. No
  ties. The same p-value comes from the binomial test depending only on the win count.
- **But the two vectors are correlated at 0.94.** Both contain HAR-X, and 2020 is a common losing
  year.
- **BY was added.** BH's FDR guarantee rests on the positive dependence (PRDS) assumption.
  BY is valid under any dependence structure: BH × c(m), c(8) = 2.718. In the package Holm, BH and BY
  are given side by side, for both families.

**Primary family result (publication-aligned):**
- **Holm:** no test survives.
- **BH:** at h=22 two hypotheses are rejected (p = 0.030). These are two separate hypotheses, but
  not two independent pieces of evidence.
- **BY:** no test survives (smallest p = 0.080).
- **DM:** not significant under any correction.

Surviving under BY in the secondary family: DM 4/24, sign 6/24. Under BH DM 6/24, sign
8/24.

**The root README was updated with the publication-aligned numbers.** Every number in the README comes from the package
(the additional numbers the README needs are in Section 9 of the package). Two statements changed:
- The BiLSTM is no longer "the worst model at all horizons" but "the worst non-naive model"; at h=5
  it is on par with train-mean.
- The HAR+OVX vs HAR-X sign test gives an uncorrected p = 0.035 at h=22. It was written as exploratory and
  uncorrected.

The old paragraph on GPR ("the one-day lag assumption is in GPR's favour") was removed;
the measured publication rule and the as-of alignment are described in its place.

**Scope note:** `09_power_analysis.py` has no `--gpr-alignment` option; it still reads the timestamp-aligned
DM results. It was not re-run in publication mode.

---

# Stage 18 (2026-09-27): `--gpr-alignment` required; power analysis in publication mode

## The flag is now required — provenance note

**The record of the commands before this date was left as it is; no flag was added
retroactively.** The log is not documentation, it is a provenance record.
- **Stages 1–15:** the commands were run without the flag. The flag did not exist then; the behaviour
  was the same as today's `--gpr-alignment timestamp`.
- **From Stage 16.3 until 2026-09-27:** the flag existed but was optional, its default
  was `timestamp`. Every output with the `_publication_aligned` suffix was produced with the flag explicitly
  given as `publication`; the suffix arises only this way. In this period
  a command without the flag would silently have produced the timestamp version.
- **From 2026-09-27 on** (this stage's commit): the flag is **required**, with no
  default (`scripts/alignment.py`, `required=True`). A call without the flag errors out and stops, and in every
  command the flag is written explicitly. The run instructions in the root README were
  updated to this form.

**Justification.** The primary specification is `publication`. With the default `timestamp`, someone who did not give
the flag would produce the secondary version without noticing. Since the repository will be cited via
Zenodo, this silent path was closed.

**Risk check.**
- The arguments are parsed only inside `main()`.
- 13, 15 and 19 import other scripts as modules and do not call their `main()`.
- 17 and 18 take no flag; they read both versions themselves.
- The repository has no shell scripts or CI.

**Systematic scan.** In every script that reads model, metric or DM outputs, it was checked one by one that
the inputs are read via `alignment.out()` / `features_path()`. The only one missing
was `09_power_analysis.py`. Those that need no flag:
- 01 (targets) and 14 (date gaps): independent of GPR.
- 02: produces both feature files at once.
- 16: the vintage archive.
- `validate_data`.

## Power analysis in publication mode (`09_power_analysis.py --gpr-alignment publication`)

The flag was added to 09. The inputs it reads in publication mode:
- the DM results, i.e. 08's publication-aligned output including the HAC inflation factors;
- the prediction file.

The calibration coefficient k of the a priori DM curves was recomputed from these DM results.

**Non-breaking check:** the `--gpr-alignment timestamp` run reproduced the three stored CSVs exactly
at the git-blob level. In the JSON the only differences are `gpr_alignment` (a new
field) and `runtime_seconds`.

**Two-version comparison (Appendix A):**

| horizon | comparison | DM required years (t.s. → pub.) | sign required years (t.s. → pub.) |
| --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 927 → 760 | 20 → 37 |
| 22 | HAR vs HAR-X | 273 → 320 | 15 → 15 |
| 66 | HAR vs HAR-X | 2 764 → 1 680 | 42 → 42 |
| 126 | HAR vs HAR-X | 400 → 307 | 94 → 94 |
| 5 | HAR-X vs XGBoost | 323 → 233 | 72 → 72 |
| 22 | HAR-X vs XGBoost | 61 → 144 | 15 → 15 |
| 66 | HAR-X vs XGBoost | 52 → 45 | 25 → 42 |
| 126 | HAR-X vs XGBoost | 5 582 → 4 639 | 42 → 94 |

- **The a priori sign-test curves are exactly the same.** They depend only on the number of folds. Significance
  requires at least 12 wins in 15 folds, at least 12 in 14 folds.
- **The a priori DM curves shifted slightly via k.** k: h=5 0.444 → 0.437, h=22 0.557 →
  0.542, h=66 1.124 → 1.137, h=126 1.700 → 1.688. At a 20% RMSE difference the power for h=5, 22, 66 and
  126 is 0.989, 0.710, 0.846 and 0.895 (timestamp: 0.991, 0.733, 0.838, 0.899).
- **The detection limit did not change.** For DM about a 20% RMSE difference (at h=22 even 20% gives 0.71),
  for the sign test winning 85–90% of the years.
- **The requirements based on the observed effect changed to the extent the effects changed.** The extension required for DM
  is 3.1–324 times the current period (timestamp: 3.6–389). For the sign test
  1.0–6.7 times (the same range). At h=22 HAR-X vs XGBoost the DM requirement rose from 61 to 144 years,
  because in that test the DM statistic fell from 1.38 to 0.90.
- **The HAC inflation factors** (08, primary family) are almost the same in the two versions: h=5 ~2.4,
  h=22 ~3.8–4.0, h=66 ~4.9–5.5, h=126 ~6.8–12.7.

**Interpretation note.** The power section in Stage 8 interprets the 0.871
"realized power" of the sign tests at h=22 as if it were evidence supporting that those results "are not a fluke".
This inference is not valid. Realized power is a monotone
transformation of the p-value, and the section itself says so. The old text was left as a provenance record.
This inference is not used in the paper or in the number package (Section 10). The informative parts
are the required sample and the a priori curves.

Outputs: `power_analysis_publication_aligned.csv`,
`apriori_power_sign_publication_aligned.csv`, `apriori_power_dm_publication_aligned.csv`,
`power_analysis_summary_publication_aligned.json`. Number package Section 10.

## Correction note (2026-09-27, after Stage 18)

While adding the methodology section (Section 11) to the number package, it was noticed that two statements in this log were
wrong. The original texts were left in place as a provenance record; the correct version is here.

1. **16.1, exceptions to the publication rule.** The statement "All of these are start-of-month updates" is
   wrong. Breakdown of the 10 exceptions:
   - 6 are files published at the start of a month that stop at the last day of the previous month.
   - 1 is a stale upload (the 2023-01-02 file, last observation 2022-08-31).
   - 3 are other: the 2023-11-01 file (last observation 2023-10-30), the 2024-03-12 file (last
     observation 2024-03-11), the 2025-12-02 file (last observation 2025-12-01).

   The support for the rule (279/289) does not change. The same statement in the root README was corrected.
2. **16.5 and old versions of the package: "the primary family was fixed in advance".** This statement
   contradicts the record of Stage 8. The family **was formalized after the tests and is not a
   pre-registration**. But it was chosen not by looking at p-values but according to the two claims declared in Stages 5–6.
   It has been fixed since then and no test is added afterwards. The number package was corrected with this
   statement.

## Clarification note (2026-09-27): the review statement, 7b and the fold RMSE correlation

1. **"An independent external code review" (introduction of Stage 16).** "Independent" implies a third-party
   human review. In reality this was an external code audit carried out with a third-party AI
   assistant (a GPT-based tool) run by the author. The original statement was left
   as a provenance record. The root README was corrected to "an external code audit performed with a
   third-party AI assistant, run by the author"; the paper's AI-use
   statement should say the same.
2. **The 7b volatility regime analysis is post hoc.**
   The docstring of `scripts/07b_exploratory_vol_regime.py` quotes the contradiction measured in Stage 7
   (H2 better than HAR-X in the pool, worse in the fold mean) and defines the aim of the analysis as
   explaining it. That is, it was designed after the hybrid results were seen.
   The split criterion (the median of the year's mean realized volatility) is mechanical; it was not chosen
   by looking at performance. The git history does not show the timing, because both scripts arrive together in the first
   commit; the evidence is the content of the text.
3. **The fold RMSE correlation of HAR and XGBoost (0.995, h=22) is, on its own, not a
   finding.** Fold RMSE scales with the year's volatility level. At h=22 HAR and the naive
   past-volatility baseline are also correlated at 0.996. The value holds only for h=22
   (h=5: 0.981, h=66: 0.932, h=126: 0.931).

   It is valid as the mechanism explaining why the two h=22 sign tests are dependent,
   but it is not evidence for the claim "XGBoost finds nothing that HAR does not find". The scale-free
   comparisons are in the number package (Section 11d); all are exploratory.

---

# Stage 19 (2026-09-29): Floor frequency, QLIKE and per-fold smearing

**Aim.** To close the two [PENDING] markers in the paper's Methodology section. No
retraining; everything from the saved prediction and fold files, publication-aligned version. Code:
`scripts/18_paper_numbers.py`, Section 12 (commit `e7f308d`). The numbers are in Section 12 of the package.
Only a summary is here.

## 19.1 How often the HAR prediction floor binds (package 12a)

**Detection method.** The prediction files carry no floor flag. A floored row was detected as a row where the forecast
is exactly equal to the fold's recorded floor (`pred_floor`, the minimum of the training target).
Three checks were made (assert):
- The per-fold number of equalities is identical to the counter recorded at fit time
  (`n_clipped_har`, `n_clipped_har_x`, ablation `n_clipped`, H3 `n_floored`).
- In the models without a floor (HAR-log, HAR-X-log, GARCH, past-vol) the number of forecasts exactly equal to the floor is
  0. HAR-log in 18, HAR-X-log in 100 rows gives forecasts below the floor.
- The closest non-floored forecast is 3.8e−06 (relative 0.04%) above the floor.

The equality does not arise by chance.

Floored test observations in the main folds (h=5 / 22 / 66 / 126):

| model | count | share | densest fold |
| --- | --- | --- | --- |
| HAR | 0 / 0 / 0 / 0 | 0% | — |
| HAR + OVX | 0 / 22 / 46 / 2 | 0% / 0.60 / 1.31 / 0.06 | 2013, at h=66 10.3% |
| HAR + GPR | 0 / 10 / 17 / 4 | 0% / 0.27 / 0.49 / 0.11 | 2014, at h=22 2.8% |
| HAR-X | 0 / 70 / 106 / 21 | 0% / 1.92 / 3.03 / 0.60 | 2014, at h=66 20.8% |
| HAR-log, HAR-X-log | no floor | — | — |
| Hybrid H3 (additional) | 8 / 9 / 58 / 205 | 0.22% / 0.25 / 1.66 / 5.86 | 2014, at h=126 44.8% |

## 19.2 QLIKE (package 12b–12c)

`QLIKE = σ²/σ̂² − log(σ²/σ̂²) − 1`, on the variance scale (Patton 2011). All models, four
horizons, fold mean (primary) and pooled (secondary).

**Recorded choice:** for floored observations QLIKE was computed on the published (floored) forecast.
What is evaluated is the forecast given. There is no non-positive forecast or target,
the stop condition was not triggered. **Status: descriptive only.** No DM or
sign test was run with the QLIKE loss; the primary family is fixed at 8 tests.

**The ranking differs from RMSE.** Kendall τ (17 models, fold mean) is 0.632 at h=5, 0.765 at h=22,
0.662 at h=66, 0.603 at h=126. The number of models whose rank changes is 13, 12, 11 and 14.

Best model:
- h=5: HAR+OVX in RMSE, HAR-X-log in QLIKE.
- h=22: HAR+OVX in RMSE, HAR-X-log in QLIKE.
- h=66: HAR+OVX in both.
- h=126: HAR-X-log in RMSE, GARCH in QLIKE.

The largest shifts:
- **GARCH** rises in QLIKE at every horizon: 12→2, 11→7, 11→4, 13→1.
- **HAR-X** falls 3→9 at h=5.
- **Past-volatility** falls 9→16 at h=66.

**Direction in the primary family's two comparisons** (`100 × (a/b − 1)`, positive = a worse;
fold mean):

| a vs b | RMSE | QLIKE |
| --- | --- | --- |
| HAR vs HAR-X | +4.75 / +11.59 / +6.10 / +1.43% | **−12.59** / +25.62 / +6.90 / +0.69% |
| XGBoost vs HAR-X | +5.93 / +15.10 / +18.21 / +7.26% | +10.81 / +25.28 / +28.52 / +8.30% |

At h=5 the direction between HAR and HAR-X depends on the loss: HAR-X is better in RMSE, HAR
in QLIKE. In the other seven cells the direction is the same.

**Concentration.** At h=5, 30.0% of HAR-X's pooled QLIKE comes from the largest
1% of rows. The largest single row is 27.11.2013, σ/σ̂ = 11.2. For HAR the same share is 21.6%,
for HAR-X-log 19.0%. Since QLIKE penalizes under-prediction harshly, the h=5 HAR-X result rests on a few
under-prediction observations. The QLIKE share of the floored rows:
- small for HAR-X: at most 4.4%, 1.9% of the rows.
- large for H3 at h=5: 0.22% of the rows, 27.4% of QLIKE.

## 19.3 Per-fold smearing and log-residual std (package 12d)

Not in the package before. Sources:
- XGBoost: `wf_summary_all` (`smearing`, `resid_log_std`)
- BiLSTM: `bilstm_folds_all`
- HAR-log and HAR-X-log: `bench_folds_all` (`har_smearing`, `har_x_smearing`)

Median smearing in the main folds (h=5 / 22 / 66 / 126):

| model | median smearing |
| --- | --- |
| XGBoost | 1.018 / 1.004 / 1.017 / 1.024 |
| BiLSTM | 1.035 / 1.005 / 1.011 / 1.027 |
| HAR-log | 1.130 / 1.055 / 1.049 / 1.058 |
| HAR-X-log | 1.111 / 1.039 / 1.039 / 1.052 |

XGBoost log-residual std median 0.190 / 0.097 / 0.181 / 0.204.

**Coverage gap.** The log-residual std is recorded only for XGBoost. 05 and 06 record only the
coefficient. For HAR-log and HAR-X-log a re-estimation of the OLS, for the BiLSTM
retraining would be needed. Not done at this stage; shown as "—" in the package.

---

# Stage 20 (2026-09-29): Log-residual std and h=5 HAR-X's largest QLIKE row

The two loose ends of Stage 19. Code: `scripts/20_log_residual_std.py` (new) and
`scripts/18_paper_numbers.py`, Section 12c–12d. Commit `7e4084d`.

## 20.1 Log-residual std of HAR-log and HAR-X-log

05 records only the smearing coefficient for these two models. `20_log_residual_std.py
--gpr-alignment publication` re-estimates the OLS. Data preparation, folds, embargo,
NaN masks, `LOG_FLOOR` and `ols_fit` are imported from 05, not copied.

**First, an equality check (assert):** in every fold the re-estimation reproduced the saved test forecasts
(`pred_har_log`, `pred_har_x_log`) and the smearing coefficient (`har_smearing`,
`har_x_smearing`) **bit for bit**: 120/120 model × horizon × fold. Had there been a mismatch,
the script would have stopped.

The std was computed with ddof=1 (like `resid_log_std` in 03). Median in the main folds (h=5 / 22 /
66 / 126):

| model | median log-residual std |
| --- | --- |
| HAR-log | 0.500 / 0.323 / 0.303 / 0.327 |
| HAR-X-log | 0.466 / 0.277 / 0.274 / 0.309 |

These values cannot be compared directly with XGBoost's 0.190 / 0.097 / 0.181 / 0.204:
XGBoost's target is `log(σ_h / past_vol_h)`, the HAR-log family's is `log(σ_h)`.

## 20.2 BiLSTM log-residual std: not recorded

06 does not store the trained weights. There is no `torch.save` in the scripts, and no
`.pt/.pth/.ckpt/.pkl` file in the repository. Computing it would require retraining; not done. In the package it
stands as "not recorded".

## 20.3 h=5 HAR-X's largest single QLIKE row (27.11.2013)

27.11.2013 is the row's own date t, i.e. the **forecast origin**; it is not the start of the target
window.
- **Features:** via `.shift(1)` they use information up to t−1 (26.11.2013).
- **Target:** `std(r_{t+1}, …, r_{t+5})`. The window is the five trading days after t: 29.11.2013,
  02.12.2013, 03.12.2013, 04.12.2013, 05.12.2013. The first return is from the 27.11.2013 close
  to the 29.11.2013 close; there is no 28.11.2013 row in the dataset. The return of day t enters neither the
  features nor the target.

Numbers:
- Realized σ = 0.013080, HAR-X forecast σ̂ = 0.001172. σ/σ̂ = 11.2, QLIKE = 118.7.
- Fold floor 0.001120; the forecast is 4.67% above the floor, not floored.
- The target was recomputed from the raw prices and compared with the stored value (assert).

---

# Stage 21 (2026-09-29): h=5 QLIKE direction reversal — exploratory checks

**Status: exploratory and post hoc.** Both checks were designed after the QLIKE result of Stage 19 (the direction of HAR vs
HAR-X at h=5 reversing relative to RMSE) was seen. The thresholds (1.25 ×
floor, 0.5 × σ̂_HAR) were chosen by looking at the results. No test. Code `scripts/18_paper_numbers.py`,
Section 12e (commit `f544558`). Computations over the main folds, pooled.

## 21.1 Closeness to the floor (h=5)

At h=5 the fold floor is 0.00111966 in every fold.

| set | n | HAR-X σ̂/floor, median (min–max) | HAR-X ≤ 1.25 × floor | HAR σ̂/floor, median (min–max) | HAR ≤ 1.25 × floor |
| --- | --- | --- | --- | --- | --- |
| HAR-X's largest 1% QLIKE rows | 36 | 3.77 (1.05–25.94) | 2 | 11.02 (5.35–20.89) | 0 |
| all h=5 test observations | 3662 | 15.64 (1.05–182.79) | 2 | 14.93 (2.79–77.25) | 0 |

- The largest 1% of rows make up 30.0% of HAR-X's h=5 QLIKE total.
- The 2 rows with σ̂ ≤ 1.25 × floor (27.11.2013 and 25.07.2014) make up 5.5% of the total.
- Mean QLIKE excluding these 2 rows: HAR-X 0.6292, HAR 0.5751. With all rows 0.6654 and
  0.5749.

## 21.2 Mechanical rule: σ̂_HAR-X < 0.5 × σ̂_HAR ("HAR-X markedly low")

"Share" is the rule rows' share of Σ(QLIKE_HAR-X − QLIKE_HAR). If the total difference
is negative, the share is read with its sign.

| horizon | rule rows | years | Σ difference, all | Σ difference, rule rows | share | mean QLIKE excl., pooled (HAR-X / HAR) | excl., fold mean (HAR-X / HAR) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 82 / 3662 | 2013 (46), 2014 (34), 2017 (2) | +331.56 | +580.01 | 174.9% | 0.5072 / 0.5766 | 0.5094 / 0.5777 |
| h=22 | 9 / 3645 | 2014 (9) | −238.44 | +4.63 | −1.9% | 0.2825 / 0.3494 | 0.2837 / 0.3576 |
| h=66 | 0 / 3500 | — | −86.00 | 0 | — | 0.3602 / 0.3847 | 0.3581 / 0.3828 |
| h=126 | 0 / 3500 | — | −9.10 | 0 | — | 0.4005 / 0.4031 | 0.3992 / 0.4020 |

At h=5, when the rule rows are removed, HAR-X is lower in mean QLIKE (pooled and in the fold
mean). The rule triggers in no row at h=66 and h=126.

---

# Stage 22 (2026-09-29): Clark–West supplementary family; XGB-6 vs HAR fold counts; 2026 footnote

Code: `scripts/21_clark_west.py` (new) and `scripts/18_paper_numbers.py`, Section 13. Commit
`3bb6ead`. All with `--gpr-alignment publication`.

## 22.1 Clark–West, HAR ⊂ HAR-X (supplementary family, 4 tests)

**Family rule.** The primary family stays fixed at 8 tests. CW was not added to it and does not replace
the DM tests. CW is a separate supplementary family of 4 tests; Holm, BH and BY within itself.

**Label:** "statistic suited to the nested structure; added after the primary DM tests were seen, before the CW
results were seen".

**Date record.** The family was declared in CLAUDE.md with commit `72106e7` (2026-09-29
16:15:34 +0300). There is no CW computation in the repository before that: in the scripts, the outputs and the
commit messages a search for "clark" finds only this declaration.

**Setup.**
- `f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`, one-sided, H1: HAR-X better.
- The saved published (floored) forecasts were used: `hybrid_predictions_all`, main
  folds, pooled series, sorted by date.
- HAC: Newey-West, Bartlett, L = h−1. The functions were imported from 08.
- Inference as in the primary DM family: HLN factor and t(n−1). Holm, BH and BY applied to the HLN p
  value.

| horizon | n | mean (e²_HAR − e²_HARX) | mean adjustment | CW (HLN) | p HLN (raw) | Holm | BH | BY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 3662 | 4.49e−06 | 3.02e−05 | 5.336 | <0.001 | <0.001 | <0.001 | <0.001 |
| h=22 | 3645 | 7.89e−06 | 2.60e−05 | 3.587 | <0.001 | <0.001 | <0.001 | <0.001 |
| h=66 | 3500 | −3.24e−06 | 1.80e−05 | 2.975 | 0.001 | 0.003 | 0.002 | 0.004 |
| h=126 | 3500 | −5.56e−06 | 1.10e−05 | 1.194 | 0.116 | 0.116 | 0.116 | 0.242 |

The exact values are in `clark_west_publication_aligned.csv`.
- CW rejects H0 under all three corrections at h=5, h=22 and h=66; at h=126 it does not
  reject.
- At h=66 the pooled raw MSE difference is negative (HAR's MSE is lower); CW rejects
  nevertheless.
- Number of folds with a positive mean f at fold level (descriptive): 15/15, 15/15,
  12/14, 10/14.

## 22.2 XGBoost-6 vs HAR, by fold (exploratory; no p-value)

| horizon | years XGB-6 wins / folds | fold-mean difference (XGB-6 / HAR − 1) | direction of count and mean |
| --- | --- | --- | --- |
| h=5 | 5/15 | +0.03% | agree (both HAR) |
| h=22 | 8/15 | −2.23% | agree (XGB-6) |
| h=66 | 8/14 | −0.89% | agree (XGB-6) |
| h=126 | 8/14 | −0.79% | agree (XGB-6) |

**Relation to Stage 15.** The statement in Stage 15, "XGBoost-6 beats plain HAR at all four horizons"
belongs to the timestamp-aligned version (h=5: 0.010645 vs 0.010834). In the publication-aligned version at h=5
XGB-6 is 0.03% worse than HAR in the fold mean (0.010838 vs 0.010834). Years won 5/15.
The statement does not hold at h=5 for the publication-aligned version. The text of Stage 15 was left as it is
as a provenance record.

## 22.3 2026 partial year, h=66 and h=126

Already in the package: Section 1e. All models' RMSE, MAE and R²_oos values in that fold;
n = 101 (h=66) and 41 (h=126); excluded from the primary aggregation. No new computation was made;
Section 13c only points there.

---

# Stage 23 (2026-09-29): Table 1 and hardening of the data validation (external review, finding 7)

Commits: `664a8fb` (validation and SHA-256), `62a65a4` (Table 1). The numbers are in
Section 14 of the package.

## 23.1 Table 1: Descriptive statistics (`scripts/22_descriptive_stats.py`)

**Variables:** Brent daily log return, the four targets, OVX, GPRD and GPRD_THREAT.

**Sample:** the model's sample, 4641 rows, trading calendar.
- The GPR series **as the model sees them**, i.e. the publication-aligned `gprd_lag1` and
  `gprd_threat_lag1`. N = 4637; the 4 rows before the first release are undefined.
- Return N = 4640; targets N = 4641 − h.

**Statistics:** N, mean, std, min, max, skewness, excess kurtosis, ADF (with constant,
AIC lag) and Ljung–Box Q(20). For all series ADF rejects the unit root and Ljung–Box rejects
no autocorrelation (p < 0.001).

**The Ljung–Box rejection for the targets is mechanical.** Consecutive targets are computed from overlapping windows that share h−1 of
their h returns. This is written explicitly in the table note; the rejection is not presented as evidence of
persistence. Since the publication-aligned GPR stays constant between two releases, its
autocorrelation also partly comes from the construction. The Q(20) value is 23 236; in the observation-dated series
22 379.

**GPR's own calendar is markedly different; moved to a footnote.** The source is all calendar days
(6818 days, weekends included). Comparison with the publication-aligned model input:

| measure | own calendar | publication-aligned model input |
| --- | --- | --- |
| GPRD mean | 103.4 | 114.0 |
| min | 0 | 24.8 |
| excess kurtosis | 7.6 | 11.1 |

The other rows in the footnote are observation-dated trading-day series: GPRD mean 115.1,
excess kurtosis 8.1.

The source of the own-calendar series is the local vintage file
`data_gpr_daily_recent_accessed_2026-09-24.dta`. The file is outside git; its SHA-256 is recorded
in the JSON. This vintage is newer than the 2026-09-01 vintage the dataset matches: 44 trading-day values
differ (2025-06-02 – 2026-09-01), largest absolute difference 33.8. Written in the package note.

`statsmodels` is now imported directly; the `requirements.txt` note was updated.

## 23.2 `validate_data.py` stops on errors (external review, finding 7)

Previously the script gave a non-zero exit under no condition. Now, under the following critical conditions, it
writes the report and stops with **exit code 1**:
- an expected column is missing;
- a missing value in a required column (all five);
- a date that cannot be parsed as `DD.MM.YYYY`;
- dates out of increasing order;
- a duplicated date.

**The 40 documented date gaps are not an error**; they are reported and do not fail the validation.

What remains a warning:
- an extra column or a different column order;
- a row count different from 4641;
- a changed OVX record value;
- a SHA-256 mismatch or a missing recorded digest.

**SHA-256.** The digest of `data/veriseti.xlsx` is
`f13956e7d0eef3dfa49dee1ac83e098f0d1cea872774eb2c72f3fe33661ebd26`. The digest
is in the file `data/veriseti.xlsx.sha256`, in `sha256sum -c` format; it was not kept outside git.
It is also in the data section of the README and in `data/README.md`. Since the digest is computed
over the file bytes, a file with the same values saved by another program does not match. This
is why a mismatch is a warning, not an error.

**Result with the current data:** PASSED, exit code 0, no warnings, SHA-256 matches. The date
gap and horizon CSVs did not change; the fields `errors`, `warnings` and `sha256`
were added to the report JSON.

**Test of the error paths.** Tried with corrupted copies in the scratchpad:

| scenario | exit code |
| --- | --- |
| column missing | 1 |
| missing value in GPRD | 1 |
| two rows swapped | 1 |
| duplicated date | 1 |
| ISO-formatted date | 1 |
| gap added by deleting 10 rows | 0 (row count and SHA warnings) |
| identical copy saved again | 0 (SHA warning) |

## Correction note (2026-09-29, after Stage 23): the GPR vintage of the Table 1 footnote

In Stage 23.1 the "own calendar" rows had been computed from the local file dated 2026-09-24.
That vintage differed from the dataset on 44 trading days. The original text was left in place as a provenance
record; the correct version is here.

**The matching vintage.** The vintage cache (`data/gpr_vintages/vintages/`) had been deleted with 16's `--cleanup`
option; the 289 vintages were not available locally. Three files were
downloaded from the archive 16 uses and compared with the dataset:

| vintage | missing trading days | largest absolute difference | result |
| --- | --- | --- | --- |
| 2026-08-31 | 1 | 45.8 | does not match |
| **2026-09-01** | 0 | **5.7e−14** | matches (on all 4641 trading days; floating-point rounding) |
| 2026-09-08 | 0 | 33.8 | does not match |

This is consistent with the record "identical to the 2026-09-01 vintage" in 02's report.

The own-calendar rows were recomputed from this vintage (script 22, commit `cb2166e`).
The script asserts the match; the mismatch warning was removed. Changed values:

| series | mean | excess kurtosis |
| --- | --- | --- |
| GPRD | 103.4 → 103.5 | 7.64 → 7.60 |
| GPRD_THREAT | 112.1 → 112.2 | 10.46 → 10.42 |

**Zero-valued days** (this vintage, 2008-01-02 – 2026-09-01):
- GPRD is 0 only on 2025-02-09 (a Sunday); on the same day GPRD_THREAT is also 0.
- GPRD_THREAT is 0 on 8 days: 2009-04-19, 2016-08-21, 2019-05-12, 2020-08-30, 2023-07-30,
  2023-10-22, 2024-09-22, 2025-02-09. All Sundays.

Only the dates are written in the package.

---

# Stage 24 (2026-09-29): Figure 2 and the Appendix A source map

Commit `f6f2173`. No new computation; from saved outputs.

**Figure 2 (`scripts/23_figure2.py`).**
- **(a) Main text:** fold-level RMSE ratio, four horizons, main folds.
  - The ratio was plotted as **XGBoost / HAR-X**. The axis reading in the request ("above 1 XGBoost
    worse") is correct only in this order; with HAR-X / XGBoost, above 1 would mean HAR-X is
    worse. The order is also consistent with the %-difference convention (a = XGBoost).
  - There is a reference line at 1; the axis label is "(above 1: XGBoost worse)".
  - The number of folds with the ratio above 1 was asserted against the HAR-X win count in the primary family:
    10/15, 13/15, 10/14, 9/14.
  - At h=66 and h=126 the 2026 fold is not in the plot; it is in the CSV with
    `include_in_main = False`.
- **(b) Appendix:** SHAP group shares; each group's number of features next to its label. 65 features
  in total (assert).
- **Format:** grayscale, Arial, vector PDF and SVG (text stays as text). The files are
  byte-identical from run to run; no creation date is embedded, the SVG id salt is fixed.

**Appendix A source map** (`outputs/appendix_a_source_map.md`). The items A1, A3, A6 and A8 cited by
Methodology v3 were mapped to the source file, the script and the package section.

| status | items |
| --- | --- |
| missing | A6 roll-over (not run) |
| source exists but no table in the package | A1 feature list; A6 Optuna selection signal; A6 raw smearing correlation; A6 07b volatility regime; A8 training-length asymmetry |

The package's §7b (BiLSTM convergence check) and the 07b script (volatility regime) are different
things; a note was made in the map.

---

# Stage 25 (2026-09-29): Roll-over robustness analysis — design and statuses written BEFORE THE RUN

**This section was written before the run and committed before the run.** The results will be in
Stage 25.2. The statuses will not change, whatever the result. All three of the three variants will be
reported whatever the result. Luo et al. (2024) was not used as a basis for the design.

## 25.1 Design

**Expiry rule.** The source is the ICE Brent contract specification (circular 13165 Attach 6) and
Circular 15/235.

| contracts | day trading ceases |
| --- | --- |
| up to February 2016 | The business day before the day 15 calendar days before the first day of the contract month. If that day is not a business day, one first goes to the preceding business day. |
| from March 2016 on | The last business day of the second month preceding the contract month. If it falls on the business day before Christmas or New Year, the preceding business day. |

- The rule changed within the sample. The March 2016 contract expired on 29.01.2016, the February 2016 contract
  on 14.01.2016.
- Business day: a trading day that is not a public holiday in England and Wales.
- The coded rule reproduces exactly the 88 expiries in ICE's official table (December 2015 –
  March 2023).
- No official table was found for 2008–2015. The dates of this period were derived from the rule and the holiday
  calendar.

**Data-source assumption.** Yahoo `BZ=F` is the Brent Last Day Financial (BZ) contract on NYMEX.
That this contract follows the ICE expiry calendar was not verified from a primary CME document;
it is used as an assumption. On which day Yahoo rolls in the continuous series is not
documented.

**Roll-over row.** The first data row after the expiry day. That row's return extends from the old
contract's close to the new contract's close.

**Variants and their statuses:**

| variant | status | what is done |
| --- | --- | --- |
| **A** | primary robustness variant | The roll-over row's return is removed (NaN). The target is the std of the remaining returns in the same h-day window; the requirement that the window be complete is kept. The return features are recomputed with clean returns: `brent_ret_lag1-5`, `brent_vol5/20/60/126`, `vol_ratio`, `vol5_vol60`, `vol20_vol126`, HAR's `har_daily` term and the past-vol baseline (the denominator of XGBoost's ratio). The return lags and `har_daily` use the latest clean returns. |
| **A′** | sensitivity check | The same as A, but the two rows after expiry are removed. Against the roll-over day being uncertain by one day. |
| **B** | sensitivity check, h=5 only | No return is removed. The rows with a roll-over row in their target window are dropped from the sample; about 24%. |

**A and the date gaps.** A also removes the return of the date-gap rows that coincide with expiry
days. How many of the 40 documented gaps overlap with the rows removed in A (and A′)
will be reported.

*Hypothesis (not proven):* some of the gaps may stem from roll-overs. Yahoo may be skipping the row of the expiry
day or of the day after. This analysis only counts the overlap,
it does not show the cause.

**Limitation.** The check removes the roll-over effect in the target and in the return features.
It does not remove the effect in XGBoost's price-level features (`brent_lag1-5`, `brent_ema5/10/20`).
These features were deliberately not changed; removing that effect would require a back-adjusted
continuous series, and the data for it does not exist.

**Tests and reporting:**
- The 8 tests of the primary family will be run in each variant with their own Holm/BH/BY corrections
  (CLAUDE.md: replications on variant data are not new families and are not pooled with the primary
  family). Setup as in 08: DM (HLN) and the sign test.
- Within-variant model differences will be reported.
- Since the target changes, absolute RMSE will not be compared with the primary result.

**Models:** HAR, HAR-X, XGBoost; `--gpr-alignment publication`.

**Implementation check.** The script does not run the variants unless, in the unmasked setup, it reproduces bit for bit
the following:
- the targets, the feature file and the HAR, HAR-X, XGBoost forecasts;
- the stored test values of the primary family.

The only code change is an optional `past_vol` parameter added to `run_horizon` in 03.
Its default keeps the existing behaviour; it will be checked that the primary outputs are reproduced
unchanged.

## 25.2 Results (2026-09-29, after the run)

Commit `a4d990f`; `scripts/24_rollover_robustness.py --gpr-alignment publication`, runtime
93 s. The numbers are in §15 of the package. The statuses are as in 25.1, not changed.

**Verification (assert):**
- The expiry calendar reproduces 88 of the 88 expiries in ICE's official table.
- With an empty mask the targets, the return features and the HAR, HAR-X, XGBoost forecasts are reproduced bit
  for bit; the values of the 8 tests are the same as the stored primary values.
- The default path of the `past_vol` parameter in 03: 03 was re-run in publication mode;
  the prediction and metric files are identical, only `runtime_seconds` changed and the stored
  value was restored.

**Sample:**
- 225 expiries; 14 expiry days are absent from the data.
- A removes 225 returns, A′ 450.
- B drops 1122 rows at h=5 (the rows with a roll-over in their window).

**Within-variant differences**, fold-mean RMSE, `100 × (a/b − 1)`, h=5 / 22 / 66 / 126.
The primary result is given only as a reference for the percentages; absolute RMSE
is not compared.

| variant | HAR vs HAR-X | XGBoost vs HAR-X |
| --- | --- | --- |
| primary (reference) | +4.75 / +11.59 / +6.10 / +1.43% | +5.93 / +15.10 / +18.21 / +7.26% |
| A | +4.88 / +12.10 / +5.86 / +1.13% | +5.83 / +18.80 / +14.48 / +8.56% |
| A′ | +5.19 / +12.43 / +6.09 / +0.96% | +8.06 / +19.24 / +15.50 / +8.53% |
| B (h=5 only) | +4.90% | +3.99% |

In all three variants HAR-X has a lower RMSE than both HAR and XGBoost at every horizon. This is also the case
in the primary result.

**8 tests** (DM and sign test; Holm/BH/BY within the variant's own tests):

| variant | surviving at 5% | raw values |
| --- | --- | --- |
| A | no test, under no correction | DM HLN p 0.122–0.937. In the sign test at h=22 two tests 12/15, raw p 0.035; BH 0.141. |
| A′ | no test | h=22 HAR vs HAR-X 13/15, raw p 0.007; Holm and BH 0.059, BY 0.161. |
| B (2 tests) | no test | — |

The two h=22 sign tests rejected under BH in the primary family (13/15, BH 0.030) drop to 12/15 in A
and do not survive under BH. In A′ one stays at 13/15, but with BH 0.059 it is above the 5%
threshold.

**Overlap with date gaps:** of the 40 documented gap rows, 18 in A and 27 in A′ are rows whose
return is removed.
- In A: 18 rows between 2008–2013.
- In A′ an additional 9 rows: between 2013-03-19 – 2014-02-19.

*Hypothesis, not proven:* some of the gaps may stem from roll-overs. The overlap does not show the
cause; since A′ removes two rows, chance overlap also increases.

**Limitation:** the roll-over effect in XGBoost's price-level features was not removed (25.1).

---

# Stage 26 (2026-09-29): Clark–West on the roll-over variants — design written BEFORE THE RUN

**This section was written before the run and committed before the run.** The results will be
in 26.2. All three variants will be reported whatever the result.

**What:** a replication of the Clark–West supplementary family (HAR ⊂ HAR-X, 4 tests; Stage 22) on the roll-over
variants of Stage 25. CLAUDE.md (commit `0a33dd4`): a replication of any declared family
on variant data is not a new family. It is reported in Appendix A with its own Holm/BH/BY
corrections and is not pooled with the primary family.

**Statuses (as in the roll-over analysis):**

| variant | status |
| --- | --- |
| A | primary robustness variant |
| A′ | sensitivity check |
| B | sensitivity check, h=5 only |

**Setup the same as in Stage 22:**
- `f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`, one-sided, H1: HAR-X better.
- Newey-West Bartlett, L = h−1; HLN factor and t(n−1). The corrections are applied to the HLN p
  value; the normal p in a side column.
- The forecasts are the saved variant forecasts of Stage 25: `rollover_predictions_publication_aligned.csv`,
  main folds, pooled.

**Corrections within the variant:** 4 tests in A and A′; a single test in B (h=5). With a single test Holm,
BH and BY equal the raw p.

**Verification:** the computation of the CW statistic will be moved from 21 into a function. It will be checked that
21's output does not change. The unmasked roll-over forecasts must reproduce the stored
`clark_west_publication_aligned.csv` values (assert).

## 26.2 Results (2026-09-29, after the run)

Commit `ad27136`; `scripts/25_rollover_clark_west.py --gpr-alignment publication`. The numbers are
in §15c of the package. The statuses are as in Stage 26, not changed.

**Verification:**
- 21's computation was moved into the `clark_west()` function; 21's outputs are byte-
  identical.
- The unmasked roll-over forecasts reproduce the stored CW results (assert).

**CW (HLN), raw p and corrected p** (HLN p; within the variant Holm / BH / BY):

| variant | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| A (4 tests) | 5.253; all < 0.001 | 3.761; all < 0.001 | 2.992; 0.001 / 0.003 / 0.002 / 0.004 | 1.416; 0.078 / 0.078 / 0.078 / 0.163 |
| A′ (4 tests) | 5.985; all < 0.001 | 4.499; all < 0.001 | 4.047; all < 0.001 | 1.643; 0.0502 / 0.0502 / 0.0502 / 0.105 |
| B (1 test) | 4.621; < 0.001 | — | — | — |

- In A and A′, CW rejects under all three corrections at h=5, h=22 and h=66, and does not reject
  at h=126. In B it rejects at h=5. The same pattern as in Stage 22 (primary data).
- At h=66 the pooled raw MSE difference is negative in all three datasets (A −9.28e−07, A′ −1.51e−06).

**Effect of the HLN choice.** The note in Stage 22 ("the result does not change") was for the primary data.
On the variants, in one cell the HLN choice decides the 5% decision:

| cell | normal p | HLN p | decision with normal p | decision with HLN |
| --- | --- | --- | --- | --- |
| A′, h=126 | 0.0442 | 0.0502 | rejection under raw, Holm and BH | no rejection |

Under BY there is no rejection in either. Since the primary inference is HLN, the result is reported
as "no rejection". In §15c of the package this cell is listed explicitly.

---

# Stage 27 (2026-09-29): Pooled R²_oos (common reference) and the H3 residual R², publication mode

Commit `c062182`; §17 of the package. No new model; from saved outputs.

## 27.1 Pooled R²_oos, common reference (§17a)

`1 − Σ SSE / Σ (y − train_mean_fold)²`, over all test observations of the main folds.
The reference in each fold is the forecast of the train-mean baseline; the same as the common reference in §1c.
In the heading it is also stated separately so that it is not confused with the standard R² of §1d. The train-mean row is 0
(assert).

There are 9 cells in which the sign differs between the fold mean and the pool; in all of them the fold mean is
negative and the pool positive:

| model | horizon | fold mean → pooled |
| --- | --- | --- |
| BiLSTM | h=5 | −0.302 → +0.070 |
| Past-volatility | h=5 | −0.075 → +0.024 |
| BiLSTM | h=22 | −0.144 → +0.212 |
| HAR-log | h=66 | −0.028 → +0.206 |
| H2 | h=126 | −0.033 → +0.131 |
| H3 | h=126 | −0.204 → +0.010 |
| HAR-log | h=126 | −0.050 → +0.140 |
| XGB-6 | h=126 | −0.018 → +0.148 |
| XGBoost | h=126 | −0.310 → +0.056 |

## 27.2 H3 mechanism, R² of the residual stage (§17b; descriptive, no test)

The values are `resid_r2_in_sample` and `resid_r2_oos`, recorded per fold by 07 in publication mode.
Because the in-sample R² is recorded, XGBoost was not refit.

Check: in the 54 folds with no floored row, the out-of-sample R² was recomputed from the saved H3 and HAR-X
forecasts and came out identical to the stored value (assert). In floored
folds the residual forecast cannot be recovered from the saved forecasts.

| horizon | in-sample, fold mean | out-of-sample, fold mean | out-of-sample, median | folds with out-of-sample > 0 |
| --- | --- | --- | --- | --- |
| h=5 | +0.885 | −0.203 | −0.155 | 1/15 |
| h=22 | +0.900 | −0.457 | −0.222 | 2/15 |
| h=66 | +0.708 | −1.104 | −0.406 | 3/14 |
| h=126 | +0.587 | −0.295 | +0.019 | 8/14 |

---

# Stage 28 (2026-09-29): XGB-6 and HAR-X, decomposition by year of the pooled squared-error difference

Commit `29b20dd`; §18 of the package. Descriptive, no test; from saved forecasts.

The contribution is the fold's Σ(e²_HAR-X − e²_XGB-6) value. A positive contribution means that in that year XGB-6's
squared error is lower. The total is positive at both horizons: at h=66
+2.74e−02, at h=126 +2.63e−02.

| horizon | share of 2020 | years with a positive contribution | years with a negative contribution | largest negative share |
| --- | --- | --- | --- | --- |
| h=66 | +193.6% | 2 (2020 and 2019, +7.6%) | 12 | 2013, −33.5% |
| h=126 | +154.9% | 6 | 8 | 2013, −34.8% |

---

## Freeze note (2026-09-29)

Analysis frozen as of 59158c5, 2026-09-29. After this point no new tests, models or
analyses are run. Requests arising during writing are answered from existing outputs or
by changes in presentation only. Any exception requires an explicit decision by the
author and is labeled post-freeze in the log and the paper.

`59158c5` (2026-09-29 22:11:19 +0300), the last analysis commit, with §18 of the number package and
Stage 28. At that commit the number package was generated from a clean tree with the header `29b20dd`. The same note
is in CLAUDE.md in the "Analysis Freeze" section.

---

## Translation note (2026-09-30, post-freeze; presentation only)

Presentation change only; consistent with the freeze rule (CLAUDE.md, "Analysis Freeze").
No new test, model or analysis.

- **Number package.** The texts in `scripts/18_paper_numbers.py` were translated into English
  (commit `229a7e6`). The package was regenerated from that commit on a clean tree (commit `c06c0e8`).
  `scripts/check_translation.py package --gpr-alignment publication`: 6050 tokens (6008
  numbers, 4 commit hashes, 38 file paths) identical in value and in order to the package at
  `59158c5`; both packages have 1254 lines. The only lines left out of the comparison are the two
  header lines that record the generating commit and time. `primary_family_tests_publication_aligned.csv`
  is byte-identical.
- **Log translation.** `outputs/experiment_log_en.md` was created: a line-by-line English translation
  of this file. Lines 1–2556 were translated as they stand at `378d551` (the freeze record), this note
  as added in the same commit. The authoritative record is this Turkish file; in case of discrepancy
  this file prevails. `scripts/check_translation.py log`: in every stage the numbers, commit
  hashes and file paths are identical in the translation.
