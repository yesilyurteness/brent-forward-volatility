# Makale sayıları — yayım-hizalı GPR sürümü (BİRİNCİL)

- **Üretildiği commit:** `fbf20724ffa94db0ec5d9708fa1d815721b0f683` (Make --gpr-alignment required; power analysis in publication mode)
- **Üretim tarihi:** 2026-09-27T17:37:08+03:00
- **Çalışma ağacı:** temiz — girdiler bu commit'teki dosyalarla birebir aynı.
- Doğrulama: `git checkout <commit> && python scripts/18_paper_numbers.py` aynı sayıları üretmelidir (yalnızca bu başlık değişir).

> Bu dosya `scripts/18_paper_numbers.py` tarafından kayıtlı çıktılardan üretilir; elle düzenlenmez. Makale yazımında sayılar **yalnızca bu dosyadan** alınır. Zaman damgalı (eski) sürümün sayıları yalnızca Bölüm 8'de, Ek A için yer alır.

**Genel kurallar.** Ana metrik fold ortalaması RMSE ve MAE; R²_oos ikincil (referans: o fold'un train hedef ortalaması); standart R² dipnot metriği. h=5 ve h=22'de 15 fold (2012–2026), h=66 ve h=126'da 14 fold (2026 kısmi yıl ana metrikten çıkarılır, ayrıca dipnotta verilir). Birim: günlük log getirilerin standart sapması. Yüzdeler `100 × (RMSE_a / RMSE_b − 1)`; negatif = a daha iyi.

**p değerleri.** Çıkarım için kullanılan tek test ailesi, önceden sabitlenmiş **birincil sekizlik ailedir** (Bölüm 6: HAR vs HAR-X ve HAR-X vs XGBoost, dört ufuk); ona sonradan test eklenmez. Bu dosyadaki diğer tüm p değerleri (ablasyon, XGBoost-6, iki sürüm karşılaştırması, BiLSTM kontrolleri, 9. bölüm) **keşifsel ve çoklu karşılaştırma için düzeltilmemiştir**; betimleyici olarak verilir. İkincil DM ailesi (24 test) kendi içinde Holm/BH/BY ile düzeltilir ama doğrulayıcı değildir.

**Tutarlılık kontrolleri (assert):** her fold'un RMSE'si tahmin dosyalarından yeniden hesaplanıp kayıtlı metrikle karşılaştırıldı; fold ortalamaları `gpr_alignment_comparison.csv` ile aynı; train-mean R²_oos her fold'da tam 0; ana metrikten yalnızca 2026 fold'u h=66/126'da dışlanıyor.

## 1. Ana sonuç tablosu (fold ortalaması)

Kaynak: `hybrid_metrics_all` (XGBoost, BiLSTM, hibritler, HAR, HAR-X, HAR-X-log, naif), `bench_metrics_all` (GARCH, HAR-log), `ablation_exogenous_folds` (HAR+OVX, HAR+GPR), `exploratory_xgb6_folds`, `opt_*metrics_all` — hepsi `_publication_aligned`. Ortak örneklem: her fold'da tüm modellerin test satırı sayısı aynı (assert); train pencereleri modele göre farklı olabilir (HAR ailesi satır 21'den, XGBoost 127'den başlar; Aşama 5 veri eşitleme kontrolü, Bölüm 7a). Kalın = sütundaki en düşük değer.

### 1a. RMSE

| model | rol | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | birincil | 0.010956 | 0.008870 | 0.009024 | 0.008675 |
| Attention BiLSTM | birincil | 0.013440 | 0.010386 | 0.012207 | 0.011585 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | hibrit | 0.011416 | 0.009164 | 0.010244 | 0.009865 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | hibrit | 0.010367 | 0.007963 | 0.007986 | 0.008136 |
| Hibrit H3: HAR-X + XGB artığı | hibrit | 0.011140 | 0.008652 | 0.009345 | 0.008642 |
| HAR-X | ekonometrik | 0.010343 | 0.007706 | 0.007634 | 0.008088 |
| HAR-X-log | ekonometrik | 0.010311 | 0.007610 | 0.007521 | **0.007977** |
| HAR | ekonometrik | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR-log | ekonometrik | 0.010857 | 0.008678 | 0.008199 | 0.008275 |
| GARCH(1,1) | ekonometrik | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| Train-mean | naif | 0.013456 | 0.011238 | 0.009265 | 0.008965 |
| Past-volatility | naif | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| HAR + OVX | ablasyon | **0.010298** | **0.007603** | **0.007469** | 0.008003 |
| HAR + GPR | ablasyon | 0.010883 | 0.008707 | 0.008247 | 0.008296 |
| XGBoost-6 | keşifsel | 0.010838 | 0.008408 | 0.008028 | 0.008139 |
| XGBoost, Optuna + büzülmüş smearing | sağlamlık | 0.011110 | 0.009143 | 0.010579 | 0.009686 |
| XGBoost, Optuna + ham smearing | ek (appendix) | 0.011226 | 0.009461 | 0.011097 | 0.011364 |

### 1b. MAE

| model | rol | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | birincil | 0.007911 | 0.006712 | 0.007069 | 0.006968 |
| Attention BiLSTM | birincil | 0.009730 | 0.008132 | 0.010128 | 0.009213 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | hibrit | 0.008319 | 0.007134 | 0.008393 | 0.007928 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | hibrit | 0.007451 | 0.005975 | 0.006127 | 0.006426 |
| Hibrit H3: HAR-X + XGB artığı | hibrit | 0.008248 | 0.006550 | 0.007233 | 0.006712 |
| HAR-X | ekonometrik | 0.007418 | 0.005676 | 0.005655 | 0.006166 |
| HAR-X-log | ekonometrik | 0.007368 | **0.005593** | 0.005635 | 0.006181 |
| HAR | ekonometrik | 0.007807 | 0.006451 | 0.006086 | 0.006233 |
| HAR-log | ekonometrik | 0.007830 | 0.006501 | 0.006203 | 0.006321 |
| GARCH(1,1) | ekonometrik | 0.008597 | 0.007019 | 0.007098 | 0.007588 |
| Train-mean | naif | 0.010036 | 0.009017 | 0.007444 | 0.007254 |
| Past-volatility | naif | 0.009538 | 0.007228 | 0.007012 | 0.006895 |
| HAR + OVX | ablasyon | **0.007341** | 0.005594 | **0.005522** | **0.006071** |
| HAR + GPR | ablasyon | 0.007850 | 0.006515 | 0.006188 | 0.006317 |
| XGBoost-6 | keşifsel | 0.007854 | 0.006376 | 0.006215 | 0.006381 |
| XGBoost, Optuna + büzülmüş smearing | sağlamlık | 0.008152 | 0.006927 | 0.008189 | 0.007920 |
| XGBoost, Optuna + ham smearing | ek (appendix) | 0.008301 | 0.007312 | 0.008819 | 0.009585 |

### 1c. R²_oos (ikincil metrik)

`R²_oos = 1 − SSE_model / Σ(y_test − train_mean)²`. **Referans tüm modellerde aynıdır:** her fold'da train-mean baseline'ının tahmini (XGBoost train penceresinin hedef ortalaması, `hybrid_metrics_all`). Bu yüzden train-mean satırı tam 0'dır ve sütun içindeki değerler aynı sabit tahmine göre ölçülür. Tahmin dosyalarından yeniden hesaplanmıştır; hibrit kaynaklı modellerde kayıtlı değerle birebir aynıdır (assert).

| model | rol | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | birincil | +0.279 | +0.232 | −0.281 | −0.310 |
| Attention BiLSTM | birincil | −0.302 | −0.144 | −3.359 | −3.155 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | hibrit | +0.161 | +0.183 | −1.145 | −1.181 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | hibrit | +0.359 | +0.383 | +0.074 | −0.033 |
| Hibrit H3: HAR-X + XGB artığı | hibrit | +0.262 | +0.250 | −0.377 | −0.204 |
| HAR-X | ekonometrik | +0.369 | +0.430 | +0.198 | +0.087 |
| HAR-X-log | ekonometrik | +0.382 | +0.440 | +0.202 | +0.054 |
| HAR | ekonometrik | +0.295 | +0.277 | +0.061 | +0.052 |
| HAR-log | ekonometrik | +0.291 | +0.260 | −0.028 | −0.050 |
| GARCH(1,1) | ekonometrik | +0.241 | +0.192 | −0.231 | −0.519 |
| Train-mean | naif | +0.000 | +0.000 | +0.000 | +0.000 |
| Past-volatility | naif | −0.075 | +0.072 | −0.105 | −0.190 |
| HAR + OVX | ablasyon | +0.383 | +0.448 | +0.244 | +0.098 |
| HAR + GPR | ablasyon | +0.283 | +0.260 | +0.014 | +0.031 |
| XGBoost-6 | keşifsel | +0.279 | +0.286 | +0.032 | −0.018 |
| XGBoost, Optuna + büzülmüş smearing | sağlamlık | +0.249 | +0.153 | −1.469 | −0.651 |
| XGBoost, Optuna + ham smearing | ek (appendix) | +0.227 | +0.084 | −1.599 | −1.612 |

Not — kaynak dosyalardaki değerler farklı bir referans kullanır: `bench`, `ablation`, `exploratory_xgb6` ve `opt_*` her modelin **kendi** train penceresinin ortalamasını referans alır (HAR ailesi satır 21'den, XGBoost 127'den başlar). Aşağıdaki değerler o dosyalardadır; makalede kullanılmaz, yalnızca kaynak dosyalarla karşılaştırma yapılırsa farkın nedenini göstermek için verilir:

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR-log | +0.291 | +0.259 | −0.030 | −0.035 |
| GARCH(1,1) | +0.240 | +0.191 | −0.235 | −0.511 |
| HAR + OVX | +0.382 | +0.448 | +0.243 | +0.116 |
| HAR + GPR | +0.283 | +0.259 | +0.012 | +0.041 |
| XGBoost-6 | +0.278 | +0.285 | +0.031 | +0.010 |
| XGBoost, Optuna + büzülmüş smearing | +0.249 | +0.154 | −1.451 | −0.741 |
| XGBoost, Optuna + ham smearing | +0.227 | +0.085 | −1.594 | −1.763 |

Negatif R²_oos (model sabit train-mean tahmininden kötü): h=5: Attention BiLSTM, Past-volatility; h=22: Attention BiLSTM; h=66: Attention BiLSTM, GARCH(1,1), Hibrit H1: 0.5 XGB + 0.5 BiLSTM, Hibrit H3: HAR-X + XGB artığı, HAR-log, Past-volatility, XGBoost (birincil, 65 özellik), XGBoost, Optuna + büzülmüş smearing, XGBoost, Optuna + ham smearing; h=126: Attention BiLSTM, GARCH(1,1), Hibrit H1: 0.5 XGB + 0.5 BiLSTM, Hibrit H2: 0.5 HAR-X + 0.5 XGB, Hibrit H3: HAR-X + XGB artığı, HAR-log, Past-volatility, XGBoost-6, XGBoost (birincil, 65 özellik), XGBoost, Optuna + büzülmüş smearing, XGBoost, Optuna + ham smearing.

### 1d. Standart R² (dipnot metriği, karar için kullanılmaz)

Fold ortalaması ve havuzlanmış değer. Referansı test diliminin kendi ortalamasıdır (ex-post). Sakin yıllarda fold SST'si çok küçüldüğü için fold R²'leri aynı ölçekte değildir; fold ortalaması bu uyarıyla verilir. Havuzlanmış R² yıllar arası varyansı da içerdiği için sistematik olarak daha yüksektir.

| model | h=5 fold ort. | h=5 havuz | h=22 fold ort. | h=22 havuz | h=66 fold ort. | h=66 havuz | h=126 fold ort. | h=126 havuz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | −0.058 | +0.305 | −0.692 | +0.350 | −3.067 | −0.130 | −4.681 | −0.078 |
| Attention BiLSTM | −1.027 | +0.029 | −1.799 | +0.158 | −9.150 | −2.393 | −12.851 | −1.193 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | −0.257 | +0.278 | −0.946 | +0.326 | −4.655 | −0.803 | −7.443 | −0.421 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | +0.053 | +0.370 | −0.218 | +0.444 | −1.803 | +0.071 | −4.335 | +0.007 |
| Hibrit H3: HAR-X + XGB artığı | −0.147 | +0.252 | −0.559 | +0.334 | −4.024 | −0.162 | −4.975 | −0.131 |
| HAR-X | +0.057 | +0.347 | −0.074 | +0.430 | −1.300 | +0.076 | −4.986 | −0.054 |
| HAR-X-log | +0.099 | +0.323 | −0.047 | +0.440 | −1.211 | +0.121 | −4.354 | +0.014 |
| HAR | −0.022 | +0.325 | −0.466 | +0.378 | −2.605 | +0.107 | −6.827 | +0.006 |
| HAR-log | −0.038 | +0.326 | −0.548 | +0.370 | −2.716 | +0.114 | −6.279 | +0.018 |
| GARCH(1,1) | −0.112 | +0.271 | −0.623 | +0.306 | −3.447 | −0.156 | −8.349 | −0.351 |
| Train-mean | −0.926 | −0.044 | −3.269 | −0.068 | −12.268 | −0.116 | −16.252 | −0.142 |
| Past-volatility | −0.572 | −0.019 | −0.663 | +0.224 | −1.557 | −0.329 | −2.171 | −0.328 |
| HAR + OVX | +0.085 | +0.348 | −0.015 | +0.438 | −1.001 | +0.087 | −4.500 | −0.041 |
| HAR + GPR | −0.046 | +0.323 | −0.532 | +0.367 | −2.892 | +0.090 | −7.262 | −0.013 |
| XGBoost-6 | −0.134 | +0.336 | −0.777 | +0.400 | −4.386 | +0.149 | −8.333 | +0.027 |
| XGBoost, Optuna + büzülmüş smearing | −0.095 | +0.292 | −0.695 | +0.302 | −4.706 | −1.059 | −5.888 | −0.369 |
| XGBoost, Optuna + ham smearing | −0.123 | +0.279 | −0.736 | +0.243 | −4.389 | −0.973 | −12.667 | −0.717 |

Havuzlanmış gözlem sayısı: h=5: 3662, h=22: 3645, h=66: 3500, h=126: 3500.

### 1e. Dipnot: 2026 kısmi yıl, h=66 ve h=126 (düşük istatistiksel güç)

Model karşılaştırması veya seçimi için kullanılmaz; yalnızca bilgi amaçlı.

| model | h=66 RMSE | h=66 MAE | h=66 R²_oos | h=126 RMSE | h=126 MAE | h=126 R²_oos |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | 0.018138 | 0.014845 | +0.406 | 0.023586 | 0.023565 | −0.128 |
| Attention BiLSTM | 0.025648 | 0.021275 | −0.187 | 0.024842 | 0.024498 | −0.252 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 0.021638 | 0.017233 | +0.155 | 0.024105 | 0.024031 | −0.178 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | 0.017384 | 0.013272 | +0.454 | 0.021943 | 0.021927 | +0.023 |
| Hibrit H3: HAR-X + XGB artığı | 0.014303 | 0.010681 | +0.631 | 0.020874 | 0.020819 | +0.116 |
| HAR-X | 0.017037 | 0.012877 | +0.476 | 0.020331 | 0.020290 | +0.162 |
| HAR-X-log | 0.015987 | 0.011378 | +0.539 | 0.018759 | 0.018598 | +0.286 |
| HAR | 0.019621 | 0.015981 | +0.305 | 0.021676 | 0.021652 | +0.047 |
| HAR-log | 0.019179 | 0.015356 | +0.336 | 0.021018 | 0.020994 | +0.104 |
| GARCH(1,1) | 0.020323 | 0.017698 | +0.254 | 0.019727 | 0.019674 | +0.211 |
| Train-mean | 0.023536 | 0.022558 | +0.000 | 0.022205 | 0.022172 | +0.000 |
| Past-volatility | 0.023763 | 0.021442 | −0.019 | 0.027144 | 0.027138 | −0.494 |
| HAR + OVX | 0.017250 | 0.013052 | +0.463 | 0.020412 | 0.020376 | +0.155 |
| HAR + GPR | 0.019560 | 0.016069 | +0.309 | 0.021598 | 0.021573 | +0.054 |
| XGBoost-6 | 0.015405 | 0.012053 | +0.572 | 0.018758 | 0.018674 | +0.286 |
| XGBoost, Optuna + büzülmüş smearing | 0.018447 | 0.014357 | +0.386 | 0.026546 | 0.026540 | −0.429 |
| XGBoost, Optuna + ham smearing | 0.020986 | 0.017403 | +0.205 | 0.027262 | 0.027254 | −0.507 |

n: h=66: 101, h=126: 41 test gözlemi.

Not (Optuna satırları): validation dilimi kurulamayan erken fold'larda Optuna koşusu kapasite kuralına düşer: h=66: 2 fold, h=126: 6 fold (tüm fold'lar, 2026 dahil).

## 2. Dört basamaklı ayrıştırma (HAR → HAR-X → XGBoost-6 → XGBoost)

Her basamak tek bir şeyi değiştirir: (1) dışsal değişkenler (OVX, GPR) eklenir; (2) aynı altı regresör, düzey hedef, ön işleme yok — yalnızca fonksiyonel form doğrusaldan ağaca; (3) 65 özellik + log-oran hedef + smearing + ön işleme paketi. Kaynak: `gpr_alignment_decomposition.csv`. XGBoost-6 keşifsel ve post hoc'tur.

| ufuk | RMSE HAR | RMSE HAR-X | RMSE XGB-6 | RMSE XGB | (1) dışsal HAR→HAR-X | (2) fonksiyonel form HAR-X→XGB-6 | (3) özellik paketi XGB-6→XGB | toplam HAR→XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 0.010834 | 0.010343 | 0.010838 | 0.010956 | −4.53% | +4.78% | +1.09% | +1.13% |
| h=22 | 0.008600 | 0.007706 | 0.008408 | 0.008870 | −10.39% | +9.11% | +5.50% | +3.15% |
| h=66 | 0.008100 | 0.007634 | 0.008028 | 0.009024 | −5.75% | +5.16% | +12.42% | +11.42% |
| h=126 | 0.008203 | 0.008088 | 0.008139 | 0.008675 | −1.41% | +0.63% | +6.59% | +5.75% |

Fold bazında (RMSE, düzeltmesiz iki yönlü işaret testi):

| ufuk | XGB-6 < HAR-X (fold) | işaret p | XGB < XGB-6 (fold) | işaret p  |
| --- | --- | --- | --- | --- |
| h=5 | 2/15 | 0.007 | 9/15 | 0.607 |
| h=22 | 3/15 | 0.035 | 8/15 | 1.000 |
| h=66 | 2/14 | 0.013 | 5/14 | 0.424 |
| h=126 | 6/14 | 0.791 | 4/14 | 0.180 |

## 3. Ablasyon merdiveni (HAR, HAR+OVX, HAR+GPR, HAR-X)

Kaynak: `ablation_exogenous_folds_publication_aligned.csv`. Keşifsel/post hoc; birincil hipotez ailesine dahil değil. HAR ve HAR+OVX GPR kullanmaz, iki sürümde bit düzeyinde aynıdır. RMSE/MAE ablasyon dosyasıyla aynı (assert); R²_oos Bölüm 1c'deki ortak referansla.

| ufuk | varyant | RMSE | MAE | R²_oos | RMSE vs HAR |
| --- | --- | --- | --- | --- | --- |
| h=5 | HAR | 0.010834 | 0.007807 | +0.295 | — |
| h=5 | HAR + OVX | 0.010298 | 0.007341 | +0.383 | −4.95% |
| h=5 | HAR + GPR | 0.010883 | 0.007850 | +0.283 | +0.45% |
| h=5 | HAR-X (HAR + OVX + GPR) | 0.010343 | 0.007418 | +0.369 | −4.53% |
| h=22 | HAR | 0.008600 | 0.006451 | +0.277 | — |
| h=22 | HAR + OVX | 0.007603 | 0.005594 | +0.448 | −11.59% |
| h=22 | HAR + GPR | 0.008707 | 0.006515 | +0.260 | +1.25% |
| h=22 | HAR-X (HAR + OVX + GPR) | 0.007706 | 0.005676 | +0.430 | −10.39% |
| h=66 | HAR | 0.008100 | 0.006086 | +0.061 | — |
| h=66 | HAR + OVX | 0.007469 | 0.005522 | +0.244 | −7.79% |
| h=66 | HAR + GPR | 0.008247 | 0.006188 | +0.014 | +1.82% |
| h=66 | HAR-X (HAR + OVX + GPR) | 0.007634 | 0.005655 | +0.198 | −5.75% |
| h=126 | HAR | 0.008203 | 0.006233 | +0.052 | — |
| h=126 | HAR + OVX | 0.008003 | 0.006071 | +0.098 | −2.44% |
| h=126 | HAR + GPR | 0.008296 | 0.006317 | +0.031 | +1.13% |
| h=126 | HAR-X (HAR + OVX + GPR) | 0.008088 | 0.006166 | +0.087 | −1.41% |

Fold bazında kazanma sayıları (düzeltmesiz iki yönlü işaret testi) ve fold ortalaması RMSE farkı:

| ufuk | fold | OVX katkısı: HAR+OVX < HAR | OVX katkısı RMSE % | GPR katkısı: HAR+GPR < HAR | GPR katkısı RMSE % | OVX üstüne GPR: HAR-X < HAR+OVX | OVX üstüne GPR RMSE % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 15 | 13/15 (p=0.007) | −4.95% | 5/15 (p=0.302) | +0.45% | 6/15 (p=0.607) | +0.43% |
| h=22 | 15 | 14/15 (p<0.001) | −11.59% | 6/15 (p=0.607) | +1.25% | 3/15 (p=0.035) | +1.36% |
| h=66 | 14 | 12/14 (p=0.013) | −7.79% | 1/14 (p=0.002) | +1.82% | 3/14 (p=0.057) | +2.21% |
| h=126 | 14 | 10/14 (p=0.180) | −2.44% | 4/14 (p=0.180) | +1.13% | 5/14 (p=0.424) | +1.06% |

## 4. Standartlaştırılmış betalar

beta_std = beta × sd(X) / sd(y), sd'ler o fold'un train diliminden. Fold ortalaması (pozitif / negatif fold sayısı), ana fold kümesi. Kaynak: `ablation_exogenous_std_beta_summary_publication_aligned.csv`.

### 4a. HAR-X, altı regresör

| regresör | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| har_daily | +0.007 (9/6) | +0.015 (15/0) | +0.019 (14/0) | +0.018 (14/0) |
| brent_vol5 | −0.015 (5/10) | +0.014 (6/9) | +0.036 (11/3) | +0.032 (11/3) |
| brent_vol20 | +0.131 (15/0) | +0.128 (15/0) | +0.072 (13/1) | +0.058 (13/1) |
| ovx_lag1 | +0.586 (15/0) | +0.649 (15/0) | +0.586 (14/0) | +0.484 (14/0) |
| gprd_lag1 | +0.025 (10/5) | +0.001 (7/8) | +0.046 (13/1) | +0.006 (9/5) |
| gprd_threat_lag1 | −0.010 (7/8) | −0.019 (6/9) | −0.039 (4/10) | +0.003 (6/8) |

### 4b. Ablasyon varyantlarında dışsal katsayılar

| ufuk | HAR + OVX: ovx_lag1 | HAR + OVX: brent_vol20 | HAR: brent_vol20 | HAR + GPR: gprd_lag1 | HAR + GPR: gprd_threat_lag1 |
| --- | --- | --- | --- | --- | --- |
| h=5 | +0.584 (15/0) | +0.132 (15/0) | +0.578 (15/0) | +0.032 (12/3) | −0.037 (4/11) |
| h=22 | +0.655 (15/0) | +0.123 (15/0) | +0.622 (15/0) | +0.007 (6/9) | −0.049 (4/11) |
| h=66 | +0.591 (14/0) | +0.070 (12/2) | +0.527 (14/0) | +0.054 (11/3) | −0.069 (3/11) |
| h=126 | +0.487 (14/0) | +0.058 (13/1) | +0.440 (14/0) | +0.012 (9/5) | −0.020 (5/9) |

## 5. SHAP grup payları (birincil XGBoost)

Yöntem: TreeSHAP via xgboost pred_contribs (shap paketi kullanilmadi). Birim: log-oran uzayi (log(vol)-log(past_vol)); ham volatilite birimi DEGIL. Pay = grup mean|SHAP| / toplam. **Uyarı:** grup payı özellik sayısıyla birlikte büyür; grup başına özellik sayısı ikinci sütunda. Kaynak: `shap_summary_publication_aligned.json`.

| grup | özellik sayısı | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| brent_vol | 7 | 45.4% | 38.9% | 29.6% | 28.0% |
| gpr | 23 | 25.5% | 29.8% | 20.0% | 15.3% |
| ovx | 13 | 15.7% | 15.1% | 27.9% | 27.3% |
| brent_fiyat | 13 | 8.3% | 11.2% | 16.7% | 24.8% |
| etkilesim | 4 | 3.8% | 1.9% | 1.6% | 3.0% |
| takvim | 5 | 1.3% | 3.1% | 4.3% | 1.6% |

Karşılaştırma: HAR-X'te |standartlaştırılmış beta| grup payları. mean|SHAP| ile standartlaştırılmış beta aynı büyüklük değildir; yalnızca sıralama ve pay düzeyinde karşılaştırılabilir.

| grup | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| brent_vol | 19.9% | 19.2% | 15.4% | 20.2% |
| ovx | 72.5% | 73.4% | 69.8% | 58.5% |
| gpr | 7.6% | 7.4% | 14.8% | 21.3% |

XGBoost SHAP'ında HAR-X'in altı regresörü dışındaki özelliklerin payı: h=5: 13.4%, h=22: 16.2%, h=66: 22.6%, h=126: 29.4%.

## 6. Birincil hipotez ailesi (8 test) — merkez çıkarım sonucu

İki iddia × dört ufuk: **HAR vs HAR-X** (dışsal değişkenler katkı sağlar mı) ve **HAR-X vs XGBoost** (doğrusal olmayan model katkı sağlar mı). Holm (FWER), Benjamini-Hochberg (FDR, pozitif bağımlılık/PRDS altında geçerli) ve Benjamini-Yekutieli (FDR, her bağımlılık yapısında geçerli; BH × c(m), c(8) = 2.718) aile içinde, 8 test üzerinden. Kayıp: karesel hata. HAC: Newey-West, Bartlett, L = h-1 (onceden ilan edilmis). HLN: `DM* = DM * sqrt((n+1-2h+h(h-1)/n)/n), t(n-1)`. DM işareti: negatif = ilk model daha iyi. **Holm, BH ve BY düzeltmeleri HLN p değerine uygulanır** (ham DM p'sine değil). BY bu dosyada hesaplanır; BH, `08_dm_test.py`'nin kayıtlı değeriyle aynı fonksiyonla yeniden üretilip doğrulanır (assert). İşaret testi fold düzeyinde binom (H0: p=0.5, iki yönlü); HAC/normallik varsayımı kullanmaz. DM havuzlanmış seri üzerindedir (her yıl yeniden eğitilmiş modellerin tahminleri), fold ortalaması değil; bu yüzden havuzlanmış RMSE farkı Bölüm 1'deki fold ortalaması farkından farklıdır ve h=66/126'da HAR vs HAR-X'te işaret değiştirir.

Şeffaflık: aile tanımı testlerden sonra resmileştirilmiştir (ön-kayıt değildir); karşılaştırmalar p değerine göre değil, Aşama 5–6'da ilan edilmiş iddialara göre seçilmiştir.

| ufuk | karşılaştırma | havuz RMSE farkı | DM | DM (HLN) | ham p | HLN p | Holm p | BH p | BY p | işaret: HAR-X kazanır | işaret ham p | işaret Holm p | işaret BH p | işaret BY p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | HAR vs HAR-X | +1.66% | +0.394 | +0.394 | 0.693 | 0.694 | 1.000 | 0.876 | 1.000 | 11/15 | 0.118 | 0.711 | 0.287 | 0.781 |
| h=22 | HAR vs HAR-X | +4.47% | +0.609 | +0.605 | 0.543 | 0.545 | 1.000 | 0.872 | 1.000 | 13/15 | 0.007 | 0.059 | **0.030** | 0.080 |
| h=66 | HAR vs HAR-X | −1.66% | −0.264 | −0.259 | 0.792 | 0.796 | 1.000 | 0.876 | 1.000 | 10/14 | 0.180 | 0.898 | 0.287 | 0.781 |
| h=126 | HAR vs HAR-X | −2.91% | −0.628 | −0.606 | 0.530 | 0.545 | 1.000 | 0.872 | 1.000 | 9/14 | 0.424 | 0.905 | 0.424 | 1.000 |
| h=5 | HAR-X vs XGBoost | −3.04% | −0.711 | −0.710 | 0.477 | 0.477 | 1.000 | 0.872 | 1.000 | 10/15 | 0.302 | 0.905 | 0.402 | 1.000 |
| h=22 | HAR-X vs XGBoost | −6.37% | −0.908 | −0.903 | 0.364 | 0.367 | 1.000 | 0.872 | 1.000 | 13/15 | 0.007 | 0.059 | **0.030** | 0.080 |
| h=66 | HAR-X vs XGBoost | −9.60% | −1.617 | −1.586 | 0.106 | 0.113 | 0.902 | 0.872 | 1.000 | 10/14 | 0.180 | 0.898 | 0.287 | 0.781 |
| h=126 | HAR-X vs XGBoost | −1.08% | −0.162 | −0.156 | 0.872 | 0.876 | 1.000 | 0.876 | 1.000 | 9/14 | 0.424 | 0.905 | 0.424 | 1.000 |

Makine okunur kopya: `primary_family_tests_publication_aligned.csv`.

Ayakta kalan test sayısı (%5 eşiği):

| sürüm | aile | DM Holm | DM BH | DM BY | işaret Holm | işaret BH | işaret BY |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yayım-hizalı (birincil) | birincil (8 test) | 0/8 | 0/8 | 0/8 | 0/8 | 2/8 | 0/8 |
| yayım-hizalı (birincil) | ikincil (24 test) | 4/24 | 6/24 | 4/24 | 6/24 | 8/24 | 6/24 |
| zaman damgalı (Ek A) | birincil (8 test) | 0/8 | 0/8 | 0/8 | 0/8 | 2/8 | 0/8 |
| zaman damgalı (Ek A) | ikincil (24 test) | 4/24 | 6/24 | 4/24 | 9/24 | 9/24 | 9/24 |

**Birincil aile sonucu (yayım-hizalı):**

- **Holm (FWER):** hiçbir test ayakta kalmıyor.
- **BH (FDR, PRDS varsayımıyla):** h=22'de iki hipotez reddediliyor: HAR-X, HAR'ı geçer (işaret 13/15, BH p = 0.030); HAR-X, XGBoost'u geçer (işaret 13/15, BH p = 0.030). **Bunlar iki ayrı hipotez, ama birbirinden bağımsız iki kanıt değil:** iki fold farkı vektörü (HAR − HAR-X ve XGBoost − HAR-X) 0.94 korelasyonlu. İkisi de HAR-X'i içeriyor ve 2020 ortak kayıp yılı (HAR-X'in kaybettiği yıllar: HAR'a karşı 2020, 2024; XGBoost'a karşı 2014, 2020). Aynı 13/15 ve aynı ham p, binom testinin yalnızca kazanma sayısına bağlı olmasından geliyor; vektörler farklı (assert).
- **BY (FDR, bağımlılık yapısından bağımsız geçerli):** hiçbir test ayakta kalmıyor. En küçük BY p = 0.080 (h=22 işaret testleri; BH p 0.0295 × c(8) = 2.718).
- **DM:** hiçbir düzeltmede anlamlılık yok (HLN p aralığı 0.113–0.876).

Yön uyumu: DM (havuz) ve işaret testi 6/8 testte aynı modeli işaret ediyor; uyuşmayanlar: h=66 har vs har_x, h=126 har vs har_x.

## 7. Sağlamlık kontrolleri (yayım modunda yeniden koşuldu)

### 7a. Veri eşitleme: benchmark'lar XGBoost'un penceresine (satır 127) indirildi

XGBoost'a göre RMSE farkı (%, negatif = benchmark daha iyi). Kaynak: `bench_model_comparison_all{,_aligned}_publication_aligned.csv`.

| model | normal pencere | eşitlenmiş pencere |
| --- | --- | --- |
| HAR | −1.11% / −3.05% / −10.25% / −5.44% | −1.12% / −3.13% / −10.36% / −6.74% |
| HAR-X | −5.60% / −13.12% / −15.41% / −6.77% | −5.58% / −13.01% / −15.37% / −8.29% |
| HAR-X-log | −5.89% / −14.21% / −16.66% / −8.04% | −5.81% / −14.19% / −16.81% / −9.48% |

(h=5 / h=22 / h=66 / h=126.)

### 7b. BiLSTM yakınsama kontrolü (yalnızca yüksek kademe; son %10 epoch diliminde kayıp düşüşü < %2, üst sınır 200 epoch)

| ufuk | yayım: birincil | yayım: yakınsama | yayım: değişim | yayım: yakınsama daha iyi | yayım: ort. epoch (yüksek kademe) | yayım: kayıp oranı birincil/yakınsama, medyan | yayım: yetersiz eğitilmiş (birincil→yakınsama) | zaman d.: birincil | zaman d.: yakınsama | zaman d.: değişim | zaman d.: yakınsama daha iyi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 0.013440 | 0.013518 | +0.58% | 7/15 | 72 | 1.36 | 4→11 | 0.014227 | 0.014374 | +1.03% | 6/15 |
| h=22 | 0.010386 | 0.010653 | +2.58% | 3/15 | 62 | 0.72 | 3→5 | 0.010512 | 0.010423 | −0.85% | 5/15 |
| h=66 | 0.012207 | 0.012207 | +0.00% | 0/14 | — | — | 1→1 | 0.011589 | 0.011589 | +0.00% | 0/14 |
| h=126 | 0.011585 | 0.011585 | +0.00% | 0/14 | — | — | 0→0 | 0.011431 | 0.011431 | +0.00% | 0/14 |

Değişim = yakınsama / birincil − 1 (fold ortalaması RMSE, ana fold'lar). h=66 ve h=126'da yüksek kademe fold yok, sonuçlar tanım gereği aynı.

**Yorum sınırı (bkz. deney günlüğü 16.4):** yayım sürümünde durdurma kriteri erken tetiklendi, eğitim kaybı yalnızca ~1.4 kat düştü (zaman damgalı sürümde geç fold'larda 4–6 kat). Bu kontrol tek başına "yetersiz eğitim değil aşırı uyum" iddiasını desteklemez; iddianın dayanağı 7c'deki sabit 200 epoch kontrolüdür.

### 7c. BiLSTM sabit 200 epoch (keşifsel, post hoc; erken durdurma yok)

Yüksek kademe fold'lar, h=5 ve h=22. Kosinüs programı yakınsama koşusuyla aynı (`T_max=200`); k'ıncı epoch yakınsama koşusunun kendisidir (kayıp ve test tahminleri bit düzeyinde aynı, assert). k = yakınsama kuralının durduğu epoch. Kaynak: `bilstm_fixed200_folds_publication_aligned.csv`; epoch bazında kayıp `bilstm_fixed200_loss_history_publication_aligned.csv`. Günlük 16.5.

| ufuk | fold | ort. k | eğitim kaybı k→200, medyan (aralık) | eval-modu eğitim MSE k→200 | kayıp birincil(60)→200, medyan | test RMSE birincil / k / 200 | RMSE 200 vs k | 200 daha iyi (işaret p; keşifsel, düzeltmesiz) | MAE 200 vs k | sd oranı k→200 | ufuk ortalaması RMSE (tüm fold'lar) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 15 | 72 | 2.48× (1.51–4.65) | 2.40× (1.41–6.29) | 3.63× | 0.013440 / 0.013518 / 0.014493 | +7.21% | 2/15 (p=0.007) | +5.22% | 1.00 → 1.10 | 0.013518 → 0.014493 (+7.21%) |
| h=22 | 9 | 62 | 3.09× (2.58–3.50) | 3.08× (2.52–4.94) | 2.21× | 0.012137 / 0.012583 / 0.013170 | +4.67% | 3/9 (p=0.508) | +3.32% | 1.02 → 1.05 | 0.010653 → 0.011006 (+3.31%) |

**p değerlerinin statüsü:** buradaki işaret testi p'leri (h=5: 0.007) keşifsel bir teşhisten gelir, **birincil sekizlik aileye dahil değildir ve düzeltilmemiştir**; statüsü HAR+OVX vs HAR-X'in düzeltmesiz p = 0.035'iyle aynıdır. Birincil aile önceden sabitlendi; sonradan test eklenmez.

**Yorum:** eğitim kaybı durdurma kuralı olmadan ciddi düşüyor ve test hatası iyileşmiyor, kötüleşiyor. "Yetersiz eğitim değil aşırı uyum" bulgusu yayım sürümünde bu kontrolle destekleniyor. Zaman damgalı sürümdeki "~4×" rakamı Ek A'ya aittir; yayım sürümünün rakamı yukarıdaki medyanlardır.

## 8. İki sürüm karşılaştırması (Ek A)

Zaman damgalı sürüm: GPR her gün bir gün gecikmeyle kullanılıyordu (satır t, t−1 tarihli gözlemi görüyordu); GPR ise haftalık yayımlandığı için bu gözlemler tahmin anında çoğu zaman yayımlanmamıştı (zaman damgalı satırların %80.3'i yayımlanmamış gözlem kullanıyor). İki sürüm tamamen aynı örneklemde değerlendirilir (aynı train/test satırları); GPR kullanmayan modeller bit düzeyinde aynıdır. Fark saf hizalama etkisidir. Kaynak: `gpr_alignment_comparison*.csv`.

### 8a. RMSE, GPR kullanan modeller (fold ortalaması)

% = 100 × (RMSE_yayım / RMSE_zaman damgalı − 1); pozitif = yayım gecikmesine uymanın doğruluk maliyeti. Fold sayımı: yayım sürümünün daha iyi olduğu fold sayısı / toplam; p düzeltmesiz iki yönlü işaret testi.

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | 0.010913 → 0.010956 (+0.40%; 8/15, p=1.000) | 0.009080 → 0.008870 (−2.31%; 12/15, p=0.035) | 0.008924 → 0.009024 (+1.12%; 6/14, p=0.791) | 0.008627 → 0.008675 (+0.56%; 7/14, p=1.000) |
| Attention BiLSTM | 0.014227 → 0.013440 (−5.53%; 9/15, p=0.607) | 0.010512 → 0.010386 (−1.20%; 6/15, p=0.607) | 0.011589 → 0.012207 (+5.33%; 9/14, p=0.424) | 0.011431 → 0.011585 (+1.35%; 4/14, p=0.180) |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 0.011611 → 0.011416 (−1.67%; 10/15, p=0.302) | 0.009390 → 0.009164 (−2.41%; 7/15, p=1.000) | 0.009890 → 0.010244 (+3.59%; 10/14, p=0.180) | 0.009747 → 0.009865 (+1.21%; 6/14, p=0.791) |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | 0.010340 → 0.010367 (+0.26%; 7/15, p=1.000) | 0.008002 → 0.007963 (−0.49%; 10/15, p=0.302) | 0.007901 → 0.007986 (+1.08%; 6/14, p=0.791) | 0.008113 → 0.008136 (+0.28%; 8/14, p=0.791) |
| Hibrit H3: HAR-X + XGB artığı | 0.011190 → 0.011140 (−0.45%; 8/15, p=1.000) | 0.008818 → 0.008652 (−1.88%; 11/15, p=0.118) | 0.009333 → 0.009345 (+0.12%; 5/14, p=0.424) | 0.008539 → 0.008642 (+1.21%; 7/14, p=1.000) |
| HAR-X | 0.010341 → 0.010343 (+0.02%; 8/15, p=1.000) | 0.007669 → 0.007706 (+0.49%; 5/15, p=0.302) | 0.007573 → 0.007634 (+0.80%; 4/14, p=0.180) | 0.008059 → 0.008088 (+0.36%; 6/14, p=0.791) |
| HAR-X-log | 0.010335 → 0.010311 (−0.23%; 5/15, p=0.302) | 0.007561 → 0.007610 (+0.64%; 4/15, p=0.118) | 0.007504 → 0.007521 (+0.23%; 4/14, p=0.180) | 0.008012 → 0.007977 (−0.44%; 8/14, p=0.791) |
| HAR + GPR | 0.010882 → 0.010883 (+0.01%; 5/15, p=0.302) | 0.008668 → 0.008707 (+0.45%; 6/15, p=0.607) | 0.008198 → 0.008247 (+0.60%; 5/14, p=0.424) | 0.008262 → 0.008296 (+0.41%; 7/14, p=1.000) |
| XGBoost-6 | 0.010645 → 0.010838 (+1.81%; 4/15, p=0.118) | 0.008184 → 0.008408 (+2.73%; 2/15, p=0.007) | 0.008015 → 0.008028 (+0.15%; 4/14, p=0.180) | 0.008050 → 0.008139 (+1.11%; 4/14, p=0.180) |
| XGBoost, Optuna + büzülmüş smearing | 0.010927 → 0.011110 (+1.67%; 4/15, p=0.118) | 0.008788 → 0.009143 (+4.03%; 6/15, p=0.607) | 0.009797 → 0.010579 (+7.98%; 9/14, p=0.424) | 0.009715 → 0.009686 (−0.30%; 7/14, p=1.000) |
| XGBoost, Optuna + ham smearing | 0.011025 → 0.011226 (+1.82%; 5/15, p=0.302) | 0.009178 → 0.009461 (+3.08%; 8/15, p=1.000) | 0.011042 → 0.011097 (+0.50%; 7/14, p=1.000) | 0.011174 → 0.011364 (+1.69%; 6/14, p=0.791) |

GPR kullanmayan modeller (HAR, HAR-log, HAR+OVX, GARCH, train-mean, past-volatility) iki sürümde bit düzeyinde aynı; tablo dışı.

### 8b. Ayrıştırma, iki sürüm

| ufuk | sürüm | (1) dışsal | (2) fonksiyonel form | (3) özellik paketi | toplam |
| --- | --- | --- | --- | --- | --- |
| h=5 | zaman damgalı | −4.55% | +2.94% | +2.51% | +0.73% |
| h=5 | yayım-hizalı | −4.53% | +4.78% | +1.09% | +1.13% |
| h=22 | zaman damgalı | −10.82% | +6.72% | +10.94% | +5.58% |
| h=22 | yayım-hizalı | −10.39% | +9.11% | +5.50% | +3.15% |
| h=66 | zaman damgalı | −6.50% | +5.84% | +11.34% | +10.18% |
| h=66 | yayım-hizalı | −5.75% | +5.16% | +12.42% | +11.42% |
| h=126 | zaman damgalı | −1.76% | −0.12% | +7.17% | +5.16% |
| h=126 | yayım-hizalı | −1.41% | +0.63% | +6.59% | +5.75% |

### 8c. Birincil aile, iki sürüm

| ufuk | karşılaştırma | HLN p (z.d. → yayım) | DM BH p | HAR-X kazanır | işaret ham p | işaret Holm p | işaret BH p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | HAR vs HAR-X | 0.722 → 0.694 | 0.887 → 0.876 | 12/15 → 11/15 | 0.035 → 0.118 | 0.211 → 0.711 | 0.094 → 0.287 |
| h=22 | HAR vs HAR-X | 0.512 → 0.545 | 0.887 → 0.872 | 13/15 → 13/15 | 0.007 → 0.007 | 0.059 → 0.059 | 0.030 → 0.030 |
| h=66 | HAR vs HAR-X | 0.840 → 0.796 | 0.887 → 0.876 | 10/14 → 10/14 | 0.180 → 0.180 | 0.718 → 0.898 | 0.239 → 0.287 |
| h=126 | HAR vs HAR-X | 0.596 → 0.545 | 0.887 → 0.872 | 9/14 → 9/14 | 0.424 → 0.424 | 0.718 → 0.905 | 0.424 → 0.424 |
| h=5 | HAR-X vs XGBoost | 0.546 → 0.477 | 0.887 → 0.872 | 10/15 → 10/15 | 0.302 → 0.302 | 0.718 → 0.905 | 0.345 → 0.402 |
| h=22 | HAR-X vs XGBoost | 0.166 → 0.367 | 0.666 → 0.872 | 13/15 → 13/15 | 0.007 → 0.007 | 0.059 → 0.059 | 0.030 → 0.030 |
| h=66 | HAR-X vs XGBoost | 0.142 → 0.113 | 0.666 → 0.872 | 11/14 → 10/14 | 0.057 → 0.180 | 0.287 → 0.898 | 0.115 → 0.287 |
| h=126 | HAR-X vs XGBoost | 0.887 → 0.876 | 0.887 → 0.876 | 10/14 → 9/14 | 0.180 → 0.424 | 0.718 → 0.905 | 0.239 → 0.424 |

### 8d. GPR'ın ağırlığı, iki sürüm

| ufuk | SHAP GPR payı | SHAP OVX payı | HAR-X β gprd_lag1 | HAR-X β gprd_threat_lag1 | HAR-X β ovx_lag1 |
| --- | --- | --- | --- | --- | --- |
| h=5 | 19.5% → 25.5% | 18.0% → 15.7% | +0.028 → +0.025 | −0.024 → −0.010 | +0.584 → +0.586 |
| h=22 | 16.5% → 29.8% | 18.3% → 15.1% | +0.019 → +0.001 | −0.028 → −0.019 | +0.650 → +0.649 |
| h=66 | 11.3% → 20.0% | 30.2% → 27.9% | +0.077 → +0.046 | −0.057 → −0.039 | +0.586 → +0.586 |
| h=126 | 9.9% → 15.3% | 29.8% → 27.3% | +0.040 → +0.006 | −0.007 → +0.003 | +0.487 → +0.484 |

### 8e. Sağlamlık kontrolleri, zaman damgalı sürüm (karşılaştırma için)

| model | normal pencere | eşitlenmiş pencere |
| --- | --- | --- |
| HAR | −0.72% / −5.29% / −9.24% / −4.91% | −0.72% / −5.36% / −9.35% / −6.22% |
| HAR-X | −5.24% / −15.54% / −15.14% / −6.58% | −5.18% / −15.40% / −15.11% / −7.88% |
| HAR-X-log | −5.30% / −16.72% / −15.92% / −7.12% | −5.23% / −16.67% / −16.06% / −8.57% |

### 8f. 2026 kısmi yıl dipnotu, iki sürüm (h=66, h=126; bilgi amaçlı)

| model | h=66 RMSE z.d. → yayım | h=126 RMSE z.d. → yayım |
| --- | --- | --- |
| XGBoost (birincil, 65 özellik) | 0.016988 → 0.018138 (+6.77%) | 0.021091 → 0.023586 (+11.83%) |
| Attention BiLSTM | 0.022960 → 0.025648 (+11.71%) | 0.023669 → 0.024842 (+4.95%) |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 0.019669 → 0.021638 (+10.01%) | 0.022314 → 0.024105 (+8.02%) |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | 0.016701 → 0.017384 (+4.09%) | 0.020384 → 0.021943 (+7.64%) |
| Hibrit H3: HAR-X + XGB artığı | 0.014196 → 0.014303 (+0.75%) | 0.019395 → 0.020874 (+7.62%) |
| HAR-X | 0.016810 → 0.017037 (+1.35%) | 0.019741 → 0.020331 (+2.99%) |
| HAR-X-log | 0.015849 → 0.015987 (+0.87%) | 0.018295 → 0.018759 (+2.54%) |
| HAR + GPR | 0.019297 → 0.019560 (+1.36%) | 0.020989 → 0.021598 (+2.90%) |
| XGBoost-6 | 0.016509 → 0.015405 (−6.69%) | 0.019999 → 0.018758 (−6.21%) |
| XGBoost, Optuna + büzülmüş smearing | 0.017787 → 0.018447 (+3.71%) | 0.026580 → 0.026546 (−0.13%) |
| XGBoost, Optuna + ham smearing | 0.018129 → 0.020986 (+15.76%) | 0.027269 → 0.027262 (−0.02%) |

## 9. README'de kullanılan ek sayılar (yayım-hizalı)

Kök `README.md`'deki her sayı ya Bölüm 1–8'den ya da bu bölümden gelir.

### 9a. Başlıca RMSE karşılaştırmaları (fold ortalaması, %)

| karşılaştırma | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| XGBoost vs past-volatility | −17.73% | −6.56% | +1.25% | +1.97% |
| XGBoost vs HAR | +1.13% | +3.15% | +11.42% | +5.75% |
| XGBoost (Optuna) vs HAR | +2.55% | +6.32% | +30.60% | +18.07% |
| BiLSTM vs HAR | +24.05% | +20.77% | +50.71% | +41.22% |
| XGBoost vs HAR-X | +5.93% | +15.10% | +18.21% | +7.26% |
| BiLSTM vs HAR-X | +29.94% | +34.77% | +59.90% | +43.24% |
| H1 (XGB+BiLSTM) vs HAR-X | +10.38% | +18.91% | +34.19% | +21.98% |
| H2 (HAR-X+XGB) vs HAR-X | +0.24% | +3.33% | +4.61% | +0.60% |
| H3 (HAR-X+artık) vs HAR-X | +7.71% | +12.27% | +22.41% | +6.86% |
| HAR-X-log vs HAR-X | −0.31% | −1.25% | −1.49% | −1.37% |
| HAR+OVX vs HAR-X | −0.43% | −1.34% | −2.16% | −1.05% |
| en iyi hibrit vs en iyi HAR-ailesi | +0.67% (h2_harx_xgb vs har_ovx) | +4.73% (h2_harx_xgb vs har_ovx) | +6.92% (h2_harx_xgb vs har_ovx) | +1.99% (h2_harx_xgb vs har_x_log) |
| en iyi birincil doğrusal olmayan (XGB, XGB-Optuna, BiLSTM) vs HAR | +1.13% (xgboost vs har) | +3.15% (xgboost vs har) | +11.42% (xgboost vs har) | +5.75% (xgboost vs har) |
| XGBoost-6 (keşifsel; HAR-X'in girdileri, OVX dahil) vs HAR | +0.03% (xgb6 vs har) | −2.23% (xgb6 vs har) | −0.89% (xgb6 vs har) | −0.79% (xgb6 vs har) |

En iyi HAR-ailesi modeli (RMSE): h=5: HAR + OVX, h=22: HAR + OVX, h=66: HAR + OVX, h=126: HAR-X-log.

HAR+OVX vs HAR-X, fold bazında (düzeltmesiz iki yönlü işaret testi):

| ufuk | HAR+OVX, HAR-X'i geçer | işaret p |
| --- | --- | --- |
| h=5 | 9/15 | 0.607 |
| h=22 | 12/15 | 0.035 |
| h=66 | 11/14 | 0.057 |
| h=126 | 9/14 | 0.424 |

Standartlaştırılmış beta aralığı (fold ortalamaları, HAR-X ve ablasyon, dört ufuk): GPR −0.069 ile +0.054 arası; OVX +0.484 ile +0.655 arası.

### 9b. DM ikincil aile (24 test), yayım-hizalı: ayakta kalanlar

| ufuk | karşılaştırma | havuz RMSE farkı | DM Holm / BH / BY | işaret | işaret Holm / BH / BY |
| --- | --- | --- | --- | --- | --- |
| h=5 | har_x vs past_vol | −19.92% | <0.001 / <0.001 / <0.001 | 15/15 | 0.001 / <0.001 / 0.003 |
| h=5 | xgboost vs past_vol | −17.41% | <0.001 / <0.001 / <0.001 | 14/15 | 0.021 / 0.005 / 0.018 |
| h=5 | har_x vs bilstm | −17.98% | <0.001 / <0.001 / <0.001 | 14/15 | 0.021 / 0.005 / 0.018 |
| h=5 | har vs past_vol | −18.59% | <0.001 / <0.001 / <0.001 | 15/15 | 0.001 / <0.001 / 0.003 |
| h=22 | har_x vs past_vol | −14.35% | 0.198 / 0.042 / 0.158 | 13/15 | 0.133 / 0.022 / 0.084 |
| h=22 | har_x vs garch | −9.40% | 1.000 / 0.211 / 0.795 | 13/15 | 0.133 / 0.022 / 0.084 |
| h=22 | har_x vs bilstm | −17.74% | 0.156 / 0.037 / 0.141 | 14/15 | 0.021 / 0.005 / 0.018 |
| h=66 | har_x vs garch | −10.59% | 0.427 / 0.075 / 0.284 | 13/14 | 0.035 / 0.007 / 0.028 |

İşaret: model1'in kazandığı fold / toplam. Uzun ufuklarda (h=66, h=126) naif baseline'a karşı DM anlamlılığı: yok.

### 9c. SHAP ek ölçüler

| ölçü | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR-X: OVX'in |std beta| payı | 72.5% | 73.4% | 69.8% | 58.5% |
| XGBoost: HAR-X'in altı regresörü dışındaki SHAP payı | 13.4% | 16.2% | 22.6% | 29.4% |
| kullanılmayan karşılaştırma oranı (5 ortak regresör × fold; SHAP özdeş sıfır) | 0.0% | 0.0% | 22.7% | 65.3% |
| OVX işaret uyumu (XGBoost SHAP yönü vs HAR-X beta; kullanılan karşılaştırmalar) | 93% (14/15) | 80% (12/15) | 54% (7/13) | 57% (4/9) |
| atıf sıralaması kararlılığı: ilk-son fold Spearman ρ | 0.55 | 0.70 | 0.57 | 0.53 |

OVX işaret uyumu, dört ufuk birlikte: 74% (37/52). Tüm fold'lar (2026 dahil), `shap_sign_agreement_publication_aligned.csv`.

### 9d. Tarih boşluğu: doğrudan hedef düzeltme testi (tahminler sabit)

| ufuk | model | en büyük |RMSE değişimi| | RMSE sıra değişimi | MAE sıra değişimi | HAR+OVX < HAR-X (düzeltilmiş) | HAR < HAR+GPR (düzeltilmiş) |
| --- | --- | --- | --- | --- | --- | --- |
| h=5 | 12 | 0.16% | 0 | 0 | evet | evet |
| h=22 | 12 | 0.26% | 0 | 2 (har_ovx ↔ har_x_log) | evet | evet |
| h=66 | 12 | 0.43% | 0 | 0 | evet | evet |
| h=126 | 12 | 0.49% | 0 | 0 | evet | evet |

### 9e. Boşluksuz alt örneklem (2017–2026 fold'ları)

| ufuk | fold (2017+) | Spearman RMSE sırası, tüm vs 2017+ | Spearman, tüm vs 2012–2016 | en iyi HAR-ailesi < XGBoost ve BiLSTM | train-mean'den düşük RMSE'li model | ilk beş RMSE aralığı |
| --- | --- | --- | --- | --- | --- | --- |
| h=5 | 10 | 0.95 | 0.96 | evet | 8/10 | 3.4% |
| h=22 | 10 | 0.98 | 0.95 | evet | 9/10 | 10.6% |
| h=66 | 9 | 0.84 | 0.54 | evet | 3/10 | 3.0% |
| h=126 | 9 | 0.68 | 0.77 | evet | 0/10 | 2.4% |

"train-mean'den düşük RMSE'li model" RMSE üzerinden sayılır (R²_oos referans farklarından etkilenmez).

## 10. Güç analizi (birincil aile; `09_power_analysis.py`)

%80 güç, %5 iki yönlü. DM: örneklem birimi etkin blok B = n/h. İşaret testi: örneklem birimi fold (yıl), tam binom. Kaynak: `power_analysis_publication_aligned.csv`, `apriori_power_{sign,dm}_publication_aligned.csv`.

**Uyarı:** gözlenen etkiden hesaplanan "gerçekleşen güç" p değerinin monoton bir dönüşümüdür ve p değerinin ötesinde bilgi taşımaz; bir sonucun tesadüf olup olmadığına kanıt olarak kullanılamaz. Bilgi taşıyan kısımlar gerekli örneklem (10a) ve gözlenen sonuçlardan bağımsız önsel eğrilerdir (10b).

Yıl başına işlem günü (test döneminden ölçüldü): 244.1.

### 10a. Gözlenen etki gerçek kabul edilirse %80 güç için gereken test dönemi

| ufuk | karşılaştırma | DM: etkin blok | DM: gerçekleşen güç | DM: gerekli yıl | DM: kat | işaret: HAR-X kazanır | işaret: gerçekleşen güç | işaret: gerekli yıl | işaret: kat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | HAR vs HAR-X | 732 | 0.068 | 760 | 50.7× | 11/15 | 0.403 | 37 | 2.5× |
| h=22 | HAR vs HAR-X | 166 | 0.093 | 320 | 21.4× | 13/15 | 0.871 | 15 | 1.0× |
| h=66 | HAR vs HAR-X | 53 | 0.058 | 1,680 | 117.2× | 10/14 | 0.190 | 42 | 3.0× |
| h=126 | HAR vs HAR-X | 28 | 0.093 | 307 | 21.4× | 9/14 | 0.076 | 94 | 6.7× |
| h=5 | HAR-X vs XGBoost | 732 | 0.110 | 233 | 15.5× | 10/15 | 0.209 | 72 | 4.8× |
| h=22 | HAR-X vs XGBoost | 166 | 0.147 | 144 | 9.6× | 13/15 | 0.871 | 15 | 1.0× |
| h=66 | HAR-X vs XGBoost | 53 | 0.355 | 45 | 3.1× | 10/14 | 0.190 | 42 | 3.0× |
| h=126 | HAR-X vs XGBoost | 28 | 0.053 | 4,639 | 323.6× | 9/14 | 0.076 | 94 | 6.7× |

Aralıklar: DM için gereken uzatma mevcut test döneminin 3.1–324 katı; işaret testi için 1.0–6.7 katı.

### 10b. Önsel güç eğrileri (gözlenen sonuçları kullanmaz)

İşaret testi: %5 iki yönlü anlamlılık için gereken en az kazanma: n=15: 12, n=14: 12.

| gerçek kazanma olasılığı | n=14 | n=15 |
| --- | --- | --- |
| 0.6 | 0.040 | 0.092 |
| 0.65 | 0.084 | 0.173 |
| 0.7 | 0.161 | 0.297 |
| 0.75 | 0.281 | 0.461 |
| 0.8 | 0.448 | 0.648 |
| 0.85 | 0.648 | 0.823 |
| 0.9 | 0.842 | 0.944 |

DM testi: `δ_blok = k·|r²−1|`, `ncp = √B·δ_blok`; k verinin gürültü yapısından (birincil ailedeki iki çiftin ortalaması) kalibre edilir, gözlenen etkiden değil. Parantezde k'nın iki çift arasındaki aralığıyla güç.

| ufuk | etkin blok | k | %5 RMSE farkı | %10 RMSE farkı | %20 RMSE farkı |
| --- | --- | --- | --- | --- | --- |
| h=5 | 732 | 0.437 | 0.210 (0.209–0.211) | 0.612 (0.609–0.615) | 0.989 (0.989–0.989) |
| h=22 | 166 | 0.542 | 0.105 (0.099–0.110) | 0.264 (0.243–0.286) | 0.710 (0.665–0.752) |
| h=66 | 53 | 1.137 | 0.127 (0.120–0.135) | 0.349 (0.322–0.378) | 0.846 (0.809–0.878) |
| h=126 | 28 | 1.688 | 0.140 (0.109–0.179) | 0.396 (0.281–0.522) | 0.895 (0.743–0.968) |

### 10c. İki sürüm (Ek A)

| ufuk | karşılaştırma | DM gerekli yıl (z.d. → yayım) | işaret gerekli yıl (z.d. → yayım) |
| --- | --- | --- | --- |
| h=5 | HAR vs HAR-X | 927 → 760 | 20 → 37 |
| h=22 | HAR vs HAR-X | 273 → 320 | 15 → 15 |
| h=66 | HAR vs HAR-X | 2,764 → 1,680 | 42 → 42 |
| h=126 | HAR vs HAR-X | 400 → 307 | 94 → 94 |
| h=5 | HAR-X vs XGBoost | 323 → 233 | 72 → 72 |
| h=22 | HAR-X vs XGBoost | 61 → 144 | 15 → 15 |
| h=66 | HAR-X vs XGBoost | 52 → 45 | 25 → 42 |
| h=126 | HAR-X vs XGBoost | 5,582 → 4,639 | 42 → 94 |

Önsel işaret testi eğrileri iki sürümde birebir aynıdır (yalnızca fold sayısına bağlı; assert). Önsel DM eğrilerinde k: h=5: 0.444 → 0.437, h=22: 0.557 → 0.542, h=66: 1.124 → 1.137, h=126: 1.700 → 1.688.

