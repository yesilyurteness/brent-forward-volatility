# Ek A kaynak haritası (Methodology v3'ün atıf yaptığı maddeler)

Her madde için sayıların alınacağı çıktı dosyası, onu üreten script ve varsa sayı
paketindeki (`paper_numbers_publication_aligned.md`) bölüm. Numaralandırma Methodology
v3'ünkü; burada değiştirilmedi. Tüm dosyalar `outputs/` altında ve yayım-hizalı
(`_publication_aligned`) sürüm; A6'daki "tam zaman damgalı tekrar koşu" satırı hariç.

**Durum sütunu:**
- **tam:** kaynak dosya var ve sayılar pakette.
- **kısmi:** kaynak dosya var ama pakette ilgili tablo yok; paket üzerinden alınabilmesi
  için script 18'e eklenmesi gerekir.
- **EKSİK:** kaynak yok.

| madde | içerik | kaynak çıktı | script | paket bölümü | durum |
| --- | --- | --- | --- | --- | --- |
| A1 | 65 özellik listesi | `features_publication_aligned.csv` (başlık: 65 özellik + `Date`, `Date_parsed`); özellik → grup eşlemesi `shap_feature_importance_publication_aligned.csv` (`ozellik`, `grup`) | `02_build_features.py`; grup eşlemesi `10_shap_analysis.py` | §5'te yalnızca grup başına sayı (7/23/13/13/4/5) | kısmi |
| A3 | dışlanan 2026 fold'u (h=66, h=126) | model metrik dosyalarında `include_in_main = False` satırları (`hybrid_metrics_all`, `bench_metrics_all`, `ablation_exogenous_folds`, `exploratory_xgb6_folds`, `opt_*metrics_all`); geçerli gözlem sayıları `horizon_valid_by_year.csv` | 03, 04, 05, 07, 11, 15; sayımlar `validate_data.py` | §1e (tüm modeller, n = 101 / 41); §8f (iki sürüm); §13c | tam |
| A6 | veri eşitleme | `bench_*_aligned_publication_aligned.*` | `05_benchmarks.py --align-start-row 127 --suffix _aligned` | §7a | tam |
| A6 | Optuna | `opt_{folds,metrics,aggregate,predictions,summary}_all_publication_aligned.*` | `04_optuna_walkforward.py` | §1 (RMSE/MAE/R²_oos satırları); seçim sinyali (en iyi deneme vs medyan) ve kapasite kuralına düşen fold'lar `opt_folds_all` içinde (`selection_signal_pct`, `selection_procedure`) | kısmi |
| A6 | yakınsama kontrolü 1 (yakınsama kriteri) | `bilstm_*_conv_publication_aligned.*` | `06_attention_bilstm.py --convergence-mode --suffix _conv` | §7b | tam |
| A6 | yakınsama kontrolü 2 (sabit 200 epoch) | `bilstm_fixed200_*_publication_aligned.*` | `19_bilstm_fixed_epochs.py` | §7c | tam |
| A6 | alternatif smearing (büzülmesiz, ham) | `opt_rawsmearing_*_publication_aligned.*` | `04_optuna_walkforward.py --raw-smearing` | §1 ("Optuna + ham smearing" satırları); smearing sapması–bozulma korelasyonu pakette yok | kısmi |
| A6 | doğrudan boşluk düzeltme testi | `gap_target_test{,_folds}_publication_aligned.csv`, `gap_target_test_summary_publication_aligned.json` | `13_gap_target_test.py` (boşluk sınıflaması `14_date_gap_diagnostics.py`'den import edilir) | §9d | tam |
| A6 | tam zaman damgalı tekrar koşu | ekleri olmayan tüm model çıktıları (03–13, 15, `_aligned` ve `_conv` varyantları); karşılaştırma `gpr_alignment_comparison*.csv`, `gpr_alignment_decomposition.csv` | 03–13 ve 15, `--gpr-alignment timestamp`; karşılaştırma `17_gpr_alignment_comparison.py` | §8 (8a–8f) | tam |
| A6 | 7b volatilite rejimi | `explore_vol_regime_{folds,groups,years}_publication_aligned.csv`, `explore_vol_regime_summary_publication_aligned.json` | `07b_exploratory_vol_regime.py` | yok | kısmi |
| A6 | roll-over | — | — | — | **EKSİK** (plan aşamasında, koşulmadı) |
| A8 | eğitim uzunluğu asimetrisi | fold başına eğitim satırları: `bench_folds_all_publication_aligned.csv` (`n_train_har`, `n_train_harx`, `n_train_garch`, `n_train_xgb_equiv`, `garch_extra_vs_xgb`, `garch_extra_vs_har`, `extra_from_embargo`, `extra_from_warmup`); XGBoost için `wf_summary_all_publication_aligned.json` (`folds[].n_train_final`) | `05_benchmarks.py`, `03_walkforward.py` | §1 notunda tek cümle ("HAR ailesi satır 21'den, XGBoost 127'den"); §7a eşitlenmiş sonuç | kısmi |

**Not, adlandırma çakışması.** Paketin "§7b" bölümü BiLSTM yakınsama kontrolüdür.
"7b volatilite rejimi" ise `07b_exploratory_vol_regime.py` script'idir ve pakette yer
almaz. Ek A'da ikisi karıştırılmamalı.

**Not, A6 içindeki keşifsel statüler** (günlükten): 7b volatilite rejimi post hoc
(açıklama notu, 2026-09-27); sabit 200 epoch post hoc (Aşama 16.5); alternatif smearing
Ek'e ait yöntem örneği (CLAUDE.md, "Appendix: bias-variance trade-off record").
