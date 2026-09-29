# Appendix A source map (the items cited by Methodology v3)

For each item: the output file the numbers are taken from, the script that produces it and, where
there is one, the section of the number package (`paper_numbers_publication_aligned.md`). The numbering
is that of Methodology v3; it was not changed here. All files are under `outputs/` and in the
publication-aligned (`_publication_aligned`) version, except the "full timestamp-aligned re-run" row in A6.

**Status column:**
- **complete:** the source file exists and the numbers are in the package.
- **partial:** the source file exists but the package has no corresponding table; for it to be taken
  from the package, it would have to be added to script 18.
- **MISSING:** no source.

| item | content | source output | script | package section | status |
| --- | --- | --- | --- | --- | --- |
| A1 | list of the 65 features | `features_publication_aligned.csv` (header: 65 features + `Date`, `Date_parsed`); feature → group mapping `shap_feature_importance_publication_aligned.csv` (`ozellik`, `grup`) | `02_build_features.py`; group mapping `10_shap_analysis.py` | §16a (full list, grouped) | complete |
| A3 | the excluded 2026 fold (h=66, h=126) | the `include_in_main = False` rows in the model metric files (`hybrid_metrics_all`, `bench_metrics_all`, `ablation_exogenous_folds`, `exploratory_xgb6_folds`, `opt_*metrics_all`); valid observation counts `horizon_valid_by_year.csv` | 03, 04, 05, 07, 11, 15; counts `validate_data.py` | §1e (all models, n = 101 / 41); §8f (two versions); §13c | complete |
| A6 | data equalization | `bench_*_aligned_publication_aligned.*` | `05_benchmarks.py --align-start-row 127 --suffix _aligned` | §7a | complete |
| A6 | Optuna | `opt_{folds,metrics,aggregate,predictions,summary}_all_publication_aligned.*` | `04_optuna_walkforward.py` | §1 (RMSE/MAE/R²_oos rows); the selection signal (best trial vs median) and the folds that fall back to the capacity rule are in `opt_folds_all` (`selection_signal_pct`, `selection_procedure`) | partial |
| A6 | convergence check 1 (convergence criterion) | `bilstm_*_conv_publication_aligned.*` | `06_attention_bilstm.py --convergence-mode --suffix _conv` | §7b | complete |
| A6 | convergence check 2 (fixed 200 epochs) | `bilstm_fixed200_*_publication_aligned.*` | `19_bilstm_fixed_epochs.py` | §7c | complete |
| A6 | alternative smearing (unshrunk, raw) | `opt_rawsmearing_*_publication_aligned.*` | `04_optuna_walkforward.py --raw-smearing` | §1 ("Optuna + raw smearing" rows); the smearing deviation–degradation correlation is not in the package | partial |
| A6 | direct gap-correction test | `gap_target_test{,_folds}_publication_aligned.csv`, `gap_target_test_summary_publication_aligned.json` | `13_gap_target_test.py` (the gap classification is imported from `14_date_gap_diagnostics.py`) | §9d | complete |
| A6 | full timestamp-aligned re-run | all model outputs without a suffix (03–13, 15, the `_aligned` and `_conv` variants); comparison `gpr_alignment_comparison*.csv`, `gpr_alignment_decomposition.csv` | 03–13 and 15, `--gpr-alignment timestamp`; comparison `17_gpr_alignment_comparison.py` | §8 (8a–8f) | complete |
| A6 | 7b volatility regime | `explore_vol_regime_{folds,groups,years}_publication_aligned.csv`, `explore_vol_regime_summary_publication_aligned.json` | `07b_exploratory_vol_regime.py` | §16b, "volatility regime analysis" | complete |
| A6 | roll-over | `rollover_{calendar,predictions,metrics,family_tests}*.csv`, `rollover_summary_publication_aligned.json` | `24_rollover_robustness.py` (design before the run, log Stage 25.1); Clark–West replication `rollover_clark_west_publication_aligned.csv`, `25_rollover_clark_west.py` (Stage 26) | §15 (§15c: Clark–West) | complete |
| A8 | training-length asymmetry | training rows per fold: `bench_folds_all_publication_aligned.csv` (`n_train_har`, `n_train_harx`, `n_train_garch`, `n_train_xgb_equiv`, `garch_extra_vs_xgb`, `garch_extra_vs_har`, `extra_from_embargo`, `extra_from_warmup`); for XGBoost `wf_summary_all_publication_aligned.json` (`folds[].n_train_final`) | `05_benchmarks.py`, `03_walkforward.py`, `06_attention_bilstm.py` | §16c (table); its effect §7a | complete |

**Note, naming clash.** The package's "§7b" section is the BiLSTM convergence check.
`07b_exploratory_vol_regime.py` appears in the package in §16b under the name "volatility regime
analysis". The two must not be confused in Appendix A.

**Note, the exploratory statuses within A6** (from the log): the 7b volatility regime is post hoc
(clarification note, 2026-09-27); the fixed 200 epochs is post hoc (Stage 16.5); the alternative smearing
is a methodological example belonging to the Appendix (CLAUDE.md, "Appendix: bias-variance trade-off record").
