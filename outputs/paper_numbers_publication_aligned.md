# Makale sayıları — yayım-hizalı GPR sürümü (BİRİNCİL)

- **Üretildiği commit:** `3bb6ead8b80ab9c1a58b3e3f3db7c9a0bfa907f7` (Add the Clark-West supplementary family and XGB-6 vs HAR fold counts)
- **Üretim tarihi:** 2026-09-29T17:11:13+03:00
- **Çalışma ağacı:** temiz — girdiler bu commit'teki dosyalarla birebir aynı.
- Doğrulama: `git checkout <commit> && python scripts/18_paper_numbers.py` aynı sayıları üretmelidir (yalnızca bu başlık değişir).

> Bu dosya `scripts/18_paper_numbers.py` tarafından kayıtlı çıktılardan üretilir; elle düzenlenmez. Makale yazımında sayılar **yalnızca bu dosyadan** alınır. Zaman damgalı (eski) sürümün sayıları yalnızca Bölüm 8'de, Ek A için yer alır.

**Genel kurallar.** Ana metrik fold ortalaması RMSE ve MAE; R²_oos ikincil (referans: o fold'un train hedef ortalaması); standart R² dipnot metriği. h=5 ve h=22'de 15 fold (2012–2026), h=66 ve h=126'da 14 fold (2026 kısmi yıl ana metrikten çıkarılır, ayrıca dipnotta verilir). Birim: günlük log getirilerin standart sapması. Yüzdeler `100 × (RMSE_a / RMSE_b − 1)`; negatif = a daha iyi.

**p değerleri.** Çıkarım için kullanılan tek test ailesi **birincil sekizlik ailedir** (Bölüm 6: HAR vs HAR-X ve HAR-X vs XGBoost, dört ufuk). Aile **testlerden sonra resmileştirildi, ön-kayıt değildir**; ancak p değerlerine bakılarak değil, Aşama 5–6'da ilan edilmiş iki iddiaya göre seçildi ve o tarihten beri sabittir: sonradan test eklenmez. Bu dosyadaki diğer tüm p değerleri (ablasyon, XGBoost-6, iki sürüm karşılaştırması, BiLSTM kontrolleri, 9. bölüm) **keşifsel ve çoklu karşılaştırma için düzeltilmemiştir**; betimleyici olarak verilir. İkincil DM ailesi (24 test) kendi içinde Holm/BH/BY ile düzeltilir ama doğrulayıcı değildir.

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

**p değerlerinin statüsü:** buradaki işaret testi p'leri (h=5: 0.007) keşifsel bir teşhisten gelir, **birincil sekizlik aileye dahil değildir ve düzeltilmemiştir**; statüsü HAR+OVX vs HAR-X'in düzeltmesiz p = 0.035'iyle aynıdır. Birincil aile sabittir (testlerden sonra resmileştirildi, ön-kayıt değil; bkz. Bölüm 6); sonradan test eklenmez.

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

## 11. Yöntem ve sınırlılık sayıları (GPR yayım hizalaması, test ailesi)

### 11a. Yayım kuralı (`16_gpr_vintages.py`, erişim 2026-09-24)

- Arşivlenmiş sürüm: **289** (2022-02-24 – 2026-09-21).
- Kural "D günü yayımlanan dosya D dahil D'ye kadarki gözlemleri içerir": **279/289** sürüm destekliyor. İstisnalar: 7 sürüm 1 gün geride, 1 sürüm 2 gün geride, 1 sürüm 3 gün geride, 1 sürüm 124 gün geride. İstisnaların dökümü: **6** ay başı dosyası bir önceki ayın son gününde duruyor; **1** bayat yükleme (2023-01-02 dosyası, son gözlem 2022-08-31); **3** diğer (2023-11-01 dosyası, son gözlem 2023-10-30, 2024-03-12 dosyası, son gözlem 2024-03-11, 2025-12-02 dosyası, son gözlem 2025-12-01).
- Sürüm günleri: Monday 207, Tuesday 42, Friday 14, Wednesday 14, Thursday 12.
- Gözlem başına yayım gecikmesi (takvim günü): medyan 3, ortalama 2.84, en fazla 10. Gözlemin haftanın gününe göre medyan: Friday 3, Monday 0, Saturday 2, Sunday 1, Thursday 4, Tuesday 6, Wednesday 5.
- Revizyon, GPRD (ilk yayım vs güncel, göreli): ortalama -5.2%, ortalama mutlak 12.7%, medyan mutlak 10.0% (n = 1664). Revizyonlar modellenmedi; değerler güncel sürümden.
- Revizyon, GPRD_THREAT (ilk yayım vs güncel, göreli): ortalama -3.7%, ortalama mutlak 14.6%, medyan mutlak 11.3% (n = 1661). Revizyonlar modellenmedi; değerler güncel sürümden.
- 2022-02-24 öncesi: arşiv yok, kural karşı-olgusal uygulanır (first Monday on/after d, next business day if federal holiday (COUNTERFACTUAL)).
- Forward-fill reddi: düzey seriyi işlem takvimine ileri doldurmak yayımlanan gözlemlerin **%78**'ini atardı.
- Zaman damgalı hizalamada satırların **%80.3**'i tahmin anında henüz yayımlanmamış bir gözlem kullanıyordu.
- Değişen özellik: 27/65; ilk tam dolu satır iki sürümde de 127.

### 11b. Etkin gecikme (işlem günü t ile kullanılan GPR gözleminin tarihi arası)

| ölçü | zaman damgalı | yayım-hizalı |
| --- | --- | --- |
| takvim günü, medyan | 1 | 3 |
| takvim günü, ortalama | 1.47 | 3.69 |
| takvim günü, en fazla | 17 | 20 |
| işlem satırı, medyan / ortalama / en fazla | — | 3 / 2.97 / 8 |

Yayım-hizalı, işlem gününe göre takvim günü (medyan / ortalama / en fazla): Monday 7 / 7.15 / 20; Tuesday 1 / 1.92 / 11; Wednesday 2 / 2.27 / 9; Thursday 3 / 3.20 / 7; Friday 4 / 4.20 / 18.

### 11c. Nedensellik doğrulamaları

- **Prefix-invariance:** özellikler veri 3000 ve 4000. satırda kesilerek yeniden hesaplandı; kesim öncesi tüm satırlar tam veriyle hesaplananla **sıfır toleransta** aynı (2/2 geçti).
- **Yayım duyarlılığı:** rastgele 40 satırda, o satırın tarihinde henüz yayımlanmamış tüm GPR gözlemleri bozuldu (satır başına 136–4065 gözlem). Yayım-hizalı kol: **40/40 değişmedi**. Kontrol kolu (zaman damgalı): **33/40 değişti**.
- **Kontrol kolunda değişmeyen 7 satırın mekanizması:** zaman damgalı kol satır t'de t−1 tarihli gözlemi kullanır ve bozulma yalnızca t−1'e kadar yayımlanmamış gözlemlere uygulanır. Değişmeyen satırlar tam olarak t−1 gözleminin t−1'e kadar zaten yayımlanmış olduğu satırlardır (6 Tuesday, 1 Wednesday; Salı satırlarının t−1'i aynı gün yayımlanan Pazartesi gözlemi, Çarşamba satırı İşçi Bayramı haftası). Yayım takvimi kontrol kolunun sonucunu **40/40** satırda doğru öngörüyor.

### 11d. h=22'deki iki işaret testinin bağımlılığı

- Fold farkı vektörleri (HAR − HAR-X, XGBoost − HAR-X): Pearson **0.94**; HAR-X RMSE'sine bölünmüş göreli farklarla 0.90; Spearman 0.90. İşaret aynı olan yıl: 13/15.
- Mekanizma: HAR ve XGBoost'un fold RMSE profilleri neredeyse aynı (fold'lar arası korelasyon **0.995**). İki fark da aynı HAR-X RMSE'sini içerdiğinden, iki test büyük ölçüde HAR-X'i aynı ölçüte karşı sınıyor: HAR-X'in iyi geçirdiği yıl iki karşılaştırmada birden kazanç, kötü geçirdiği yıl (2020) iki karşılaştırmada birden kayıp olarak görünüyor. İki fark vektörü arasındaki korelasyon yıl bazlı ölçek farkından ibaret değil; göreli farklarda ve sıralamada da sürüyor.

**0.995 kendi başına bir bulgu değildir.** Fold RMSE, yılın volatilite düzeyiyle birlikte ölçeklenir; bu yüzden hemen her model çiftinin fold RMSE'leri yüksek korelasyonludur. Aşağıdaki tablo bunun karşılaştırma değerlerini veriyor. Ölçekten arındırılmış ölçüler: fold RMSE'nin train-mean RMSE'sine oranı üzerinden korelasyon ve günlük hata korelasyonu. Keşifsel, çıkarım için değil.

| ufuk | fold RMSE: HAR~XGB | fold RMSE: HAR~past-vol | fold RMSE: HAR~train-mean | göreli: HAR~XGB | göreli: HAR~HAR-X | göreli: HAR~BiLSTM | günlük hata: HAR~XGB | günlük hata: HAR-X~XGB | günlük hata: HAR~HAR-X |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 0.981 | 0.997 | 0.928 | 0.919 | 0.925 | 0.501 | 0.914 | 0.873 | 0.890 |
| h=22 | 0.995 | 0.996 | 0.893 | 0.973 | 0.937 | 0.608 | 0.914 | 0.827 | 0.857 |
| h=66 | 0.932 | 0.981 | 0.894 | 0.722 | 0.917 | 0.261 | 0.836 | 0.819 | 0.907 |
| h=126 | 0.931 | 0.868 | 0.934 | 0.836 | 0.866 | 0.332 | 0.867 | 0.863 | 0.942 |

Okuma: h=22'de HAR ile naif past-volatility baseline'ının fold RMSE korelasyonu da HAR~XGB kadar yüksek. Yani 0.995, XGBoost'un HAR'ı özel olarak izlediğini değil, yılların zorluk düzeyinin bütün modellere ortak olduğunu gösteriyor; üstelik yalnızca h=22 değeri. Ölçekten arındırılmış ölçüler daha bilgilendirici ama keşifseldir ve ufka göre değişir.

### 11e. İşaret testi eşikleri (tam binom, %5 iki yönlü)

| fold | anlamlılık için en az kazanma | o eşikte p | bir eksiğinde p |
| --- | --- | --- | --- |
| 9 | 8/9 | 0.0391 | 0.1797 |
| 14 | 12/14 | 0.0129 | 0.0574 |
| 15 | 12/15 | 0.0352 | 0.1185 |

n=9 (Bölüm 7c, h=22 yüksek kademe) için anlamlılık mümkündür ama 9 fold'un en az 8'inde aynı yön gerekir; gözlenen 6/9 bu eşiğin iki fold altındadır.

## 12. Taban sıklığı, QLIKE ve fold başına smearing (yayım-hizalı)

Yeniden eğitim yok; her şey kayıtlı tahmin ve fold dosyalarından. QLIKE yalnızca betimleyicidir: QLIKE kaybıyla DM veya işaret testi koşulmadı, birincil aile 8 testle sabittir.

### 12a. Tahmin tabanının devreye girme sıklığı

Düzey ölçekli OLS tahminleri (HAR, HAR-X, ablasyon basamakları) ve Hibrit H3, fold'un **eğitim hedefinin minimumunda** tabanlanır: `max(tahmin, min(y_train))` (train-only). Log ölçekli HAR-log ve HAR-X-log tahminleri `exp(·) × smearing` olduğu için yapısal olarak pozitiftir; tahmine taban uygulanmaz (0 tanım gereği). (Log spesifikasyonların *regresörlerine* uygulanan `LOG_FLOOR = 1e-4` ayrı bir şeydir ve burada sayılmaz.)

**Tespit yöntemi.** Tahmin dosyalarında taban işareti yok. Tabanlanmış satır, tahminin o fold'un kayıtlı tabanına (`bench_folds_all.pred_floor`, H3 için `hybrid_folds_all.pred_floor`) **tam eşit** olduğu satır olarak tespit edildi. Kontroller (assert):
- Her fold'da eşitlik sayısı, uyum anında kaydedilen sayaçla birebir aynı (`n_clipped_har`, `n_clipped_har_x`, ablasyon `n_clipped`, H3 `n_floored`).
- Tesadüfi eşitlik yok: tabansız modellerde (HAR-log, HAR-X-log, GARCH, past-volatility) fold tabanına tam eşit tahmin sayısı 0. Oysa HAR-log 18, HAR-X-log 100 satırda tabanın **altında** tahmin veriyor; yani eşitlik ancak `max()` işleminden doğuyor.
- Tabanlı modellerde tabanlanmamış en yakın tahmin, tabanın 3.83e-06 (göreli %0.043) üstünde; sürekli bir OLS tahmininin tabana bit düzeyinde tesadüfen eşit çıkması pratikte olanaksız.
- Pakette kullanılan HAR-X (hibrit dosyası) aynı satırlarda tabanlanıyor.

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR | 0 / 3662 (%0.00) | 0 / 3645 (%0.00) | 0 / 3500 (%0.00) | 0 / 3500 (%0.00) |
| HAR + OVX | 0 / 3662 (%0.00) | 22 / 3645 (%0.60; en yoğun 2013 %4.9) | 46 / 3500 (%1.31; en yoğun 2013 %10.3) | 2 / 3500 (%0.06; en yoğun 2014 %0.8) |
| HAR + GPR | 0 / 3662 (%0.00) | 10 / 3645 (%0.27; en yoğun 2014 %2.8) | 17 / 3500 (%0.49; en yoğun 2012 %2.9) | 4 / 3500 (%0.11; en yoğun 2013 %0.8) |
| HAR-X | 0 / 3662 (%0.00) | 70 / 3645 (%1.92; en yoğun 2014 %17.6) | 106 / 3500 (%3.03; en yoğun 2014 %20.8) | 21 / 3500 (%0.60; en yoğun 2014 %5.6) |
| HAR-log | 0 / 3662 (taban yok) | 0 / 3645 (taban yok) | 0 / 3500 (taban yok) | 0 / 3500 (taban yok) |
| HAR-X-log | 0 / 3662 (taban yok) | 0 / 3645 (taban yok) | 0 / 3500 (taban yok) | 0 / 3500 (taban yok) |
| Hibrit H3 (ek) | 8 / 3662 (%0.22; en yoğun 2013 %3.3) | 9 / 3645 (%0.25; en yoğun 2013 %3.7) | 58 / 3500 (%1.66; en yoğun 2013 %14.4) | 205 / 3500 (%5.86; en yoğun 2014 %44.8) |

Ana metriğe giren fold'lar (h=66/126'da 2026 hariç). 2026 dahil tüm fold'lar: HAR 0/14449, HAR + OVX 70/14449, HAR + GPR 31/14449, HAR-X 197/14449, Hibrit H3 (ek) 280/14449. XGBoost-6 aynı train-min tabanını kullanır; kayıtlı sayaç 0 (tabana takılan tahmin yok).

### 12b. QLIKE (Patton 2011), varyans ölçeğinde

`QLIKE = σ²/σ̂² − log(σ²/σ̂²) − 1`, σ = gerçekleşen hedef, σ̂ = tahmin. Hedef bir standart sapma olduğu için ikisi de karelenir. Düşük = iyi; mükemmel tahminde 0. **Taban bağlanan gözlemlerde QLIKE yayımlanan (tabanlanmış) tahmin üzerinden hesaplanır** — değerlendirilen şey modelin verdiği tahmindir. Tüm tahminler ve hedefler pozitif (en küçük tahmin 0.001120); durdurma koşulu tetiklenmedi. Kalın = sütundaki en düşük.

**Fold ortalaması (birincil):**

| model | rol | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | birincil | 0.7336 | 0.3565 | 0.4602 | 0.4324 |
| Attention BiLSTM | birincil | 0.8885 | 0.4868 | 0.4924 | 0.4920 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | hibrit | 0.6790 | 0.3703 | 0.4570 | 0.4370 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | hibrit | 0.5604 | 0.2964 | 0.3886 | 0.3909 |
| Hibrit H3: HAR-X + XGB artığı | hibrit | 0.8265 | 0.3265 | 0.4339 | 0.4431 |
| HAR-X | ekonometrik | 0.6620 | 0.2845 | 0.3581 | 0.3992 |
| HAR-X-log | ekonometrik | **0.4815** | **0.2650** | 0.3501 | 0.3981 |
| HAR | ekonometrik | 0.5787 | 0.3574 | 0.3828 | 0.4020 |
| HAR-log | ekonometrik | 0.5787 | 0.3605 | 0.3851 | 0.4093 |
| GARCH(1,1) | ekonometrik | 0.5219 | 0.3304 | 0.3706 | **0.3594** |
| Train-mean | naif | 1.0557 | 0.7156 | 0.4994 | 0.4480 |
| Past-volatility | naif | 1.5800 | 0.4537 | 0.5325 | 0.4622 |
| HAR + OVX | ablasyon | 0.5389 | 0.2706 | **0.3471** | 0.3900 |
| HAR + GPR | ablasyon | 0.5917 | 0.3779 | 0.4018 | 0.4189 |
| XGBoost-6 | keşifsel | 0.5901 | 0.3273 | 0.3818 | 0.4176 |
| XGBoost, Optuna + büzülmüş smearing | sağlamlık | 0.6767 | 0.3449 | 0.4803 | 0.5894 |
| XGBoost, Optuna + ham smearing | ek (appendix) | 0.6793 | 0.3725 | 0.5702 | 0.9973 |

**Havuzlanmış (ikincil):** ana metriğe giren tüm test satırları üzerinden ortalama.

| model | rol | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | birincil | 0.7354 | 0.3499 | 0.4629 | 0.4335 |
| Attention BiLSTM | birincil | 0.8921 | 0.4641 | 0.4950 | 0.4931 |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | hibrit | 0.6811 | 0.3598 | 0.4595 | 0.4381 |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | hibrit | 0.5616 | 0.2928 | 0.3909 | 0.3920 |
| Hibrit H3: HAR-X + XGB artığı | hibrit | 0.8282 | 0.3249 | 0.4363 | 0.4448 |
| HAR-X | ekonometrik | 0.6654 | 0.2835 | 0.3602 | 0.4005 |
| HAR-X-log | ekonometrik | **0.4842** | **0.2642** | 0.3522 | 0.3995 |
| HAR | ekonometrik | 0.5749 | 0.3489 | 0.3847 | 0.4031 |
| HAR-log | ekonometrik | 0.5745 | 0.3524 | 0.3870 | 0.4106 |
| GARCH(1,1) | ekonometrik | 0.5217 | 0.3262 | 0.3722 | **0.3601** |
| Train-mean | naif | 1.0194 | 0.6801 | 0.5002 | 0.4482 |
| Past-volatility | naif | 1.5792 | 0.4501 | 0.5359 | 0.4639 |
| HAR + OVX | ablasyon | 0.5401 | 0.2685 | **0.3492** | 0.3913 |
| HAR + GPR | ablasyon | 0.5911 | 0.3710 | 0.4038 | 0.4201 |
| XGBoost-6 | keşifsel | 0.5939 | 0.3255 | 0.3833 | 0.4187 |
| XGBoost, Optuna + büzülmüş smearing | sağlamlık | 0.6779 | 0.3397 | 0.4832 | 0.5912 |
| XGBoost, Optuna + ham smearing | ek (appendix) | 0.6807 | 0.3684 | 0.5728 | 0.9999 |

**Tabanlanmış satırların QLIKE payı** (havuzlanmış, ana fold'lar): tabanlanmış satırların toplam QLIKE içindeki payı / satır payı. Taban, eğitim hedefinin minimumu olduğundan bu satırlarda σ̂ küçüktür ve QLIKE büyür.

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR | — | — | — | — |
| HAR + OVX | — | %0.8 / %0.60 | %0.2 / %1.31 | %0.1 / %0.06 |
| HAR + GPR | — | %0.3 / %0.27 | %0.2 / %0.49 | %0.1 / %0.11 |
| HAR-X | — | %4.4 / %1.92 | %1.5 / %3.03 | %1.2 / %0.60 |
| Hibrit H3 (ek) | %27.4 / %0.22 | %0.4 / %0.25 | %0.1 / %1.66 | %3.8 / %5.86 |

### 12c. QLIKE ve RMSE sıralamaları (fold ortalaması)

Sıra 1 = en iyi. Yalnızca iki kayıpta sırası farklı olan modeller listelenir. İki tutarlı kayıp farklı sıralayabilir (Patton 2011); farklılık bir bulgu olarak raporlanır.

| ufuk | Kendall τ (17 model) | sırası değişen model | RMSE'de en iyi | QLIKE'ta en iyi |
| --- | --- | --- | --- | --- |
| h=5 | 0.632 | 13/17 | HAR + OVX | HAR-X-log |
| h=22 | 0.765 | 12/17 | HAR + OVX | HAR-X-log |
| h=66 | 0.662 | 11/17 | HAR + OVX | HAR + OVX |
| h=126 | 0.603 | 14/17 | HAR-X-log | GARCH(1,1) |

| ufuk | model | RMSE sırası | QLIKE sırası | fark |
| --- | --- | --- | --- | --- |
| h=5 | HAR + OVX | 1 | 3 | +2 |
| h=5 | HAR-X-log | 2 | 1 | −1 |
| h=5 | HAR-X | 3 | 9 | +6 |
| h=5 | XGBoost-6 | 6 | 7 | +1 |
| h=5 | HAR-log | 7 | 6 | −1 |
| h=5 | XGBoost (birincil, 65 özellik) | 9 | 13 | +4 |
| h=5 | Hibrit H3: HAR-X + XGB artığı | 11 | 14 | +3 |
| h=5 | GARCH(1,1) | 12 | 2 | −10 |
| h=5 | XGBoost, Optuna + ham smearing | 13 | 12 | −1 |
| h=5 | Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 14 | 11 | −3 |
| h=5 | Past-volatility | 15 | 17 | +2 |
| h=5 | Attention BiLSTM | 16 | 15 | −1 |
| h=5 | Train-mean | 17 | 16 | −1 |
| h=22 | HAR + OVX | 1 | 2 | +1 |
| h=22 | HAR-X-log | 2 | 1 | −1 |
| h=22 | XGBoost-6 | 5 | 6 | +1 |
| h=22 | HAR | 6 | 10 | +4 |
| h=22 | Hibrit H3: HAR-X + XGB artığı | 7 | 5 | −2 |
| h=22 | HAR-log | 8 | 11 | +3 |
| h=22 | HAR + GPR | 9 | 14 | +5 |
| h=22 | XGBoost (birincil, 65 özellik) | 10 | 9 | −1 |
| h=22 | GARCH(1,1) | 11 | 7 | −4 |
| h=22 | XGBoost, Optuna + büzülmüş smearing | 12 | 8 | −4 |
| h=22 | Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 13 | 12 | −1 |
| h=22 | XGBoost, Optuna + ham smearing | 14 | 13 | −1 |
| h=66 | Hibrit H2: 0.5 HAR-X + 0.5 XGB | 4 | 8 | +4 |
| h=66 | HAR + GPR | 8 | 9 | +1 |
| h=66 | Past-volatility | 9 | 16 | +7 |
| h=66 | XGBoost (birincil, 65 özellik) | 10 | 12 | +2 |
| h=66 | GARCH(1,1) | 11 | 4 | −7 |
| h=66 | Train-mean | 12 | 15 | +3 |
| h=66 | Hibrit H3: HAR-X + XGB artığı | 13 | 10 | −3 |
| h=66 | Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 14 | 11 | −3 |
| h=66 | XGBoost, Optuna + büzülmüş smearing | 15 | 13 | −2 |
| h=66 | XGBoost, Optuna + ham smearing | 16 | 17 | +1 |
| h=66 | Attention BiLSTM | 17 | 14 | −3 |
| h=126 | HAR-X-log | 1 | 4 | +3 |
| h=126 | HAR-X | 3 | 5 | +2 |
| h=126 | Hibrit H2: 0.5 HAR-X + 0.5 XGB | 4 | 3 | −1 |
| h=126 | XGBoost-6 | 5 | 8 | +3 |
| h=126 | HAR + GPR | 8 | 9 | +1 |
| h=126 | Past-volatility | 9 | 14 | +5 |
| h=126 | Hibrit H3: HAR-X + XGB artığı | 10 | 12 | +2 |
| h=126 | XGBoost (birincil, 65 özellik) | 11 | 10 | −1 |
| h=126 | Train-mean | 12 | 13 | +1 |
| h=126 | GARCH(1,1) | 13 | 1 | −12 |
| h=126 | XGBoost, Optuna + büzülmüş smearing | 14 | 16 | +2 |
| h=126 | Hibrit H1: 0.5 XGB + 0.5 BiLSTM | 15 | 11 | −4 |
| h=126 | XGBoost, Optuna + ham smearing | 16 | 17 | +1 |
| h=126 | Attention BiLSTM | 17 | 15 | −2 |

**QLIKE'ın yoğunlaşması** (havuzlanmış, ana fold'lar): QLIKE eksik tahmini (σ̂ ≪ σ) sert cezalandırır, bu yüzden ortalama birkaç gözleme dayanabilir. Hücre: en büyük %1 satırın QLIKE toplamındaki payı; parantezde en büyük tek satırın σ/σ̂ oranı ve tarihi. Betimleyicidir.

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| XGBoost (birincil, 65 özellik) | %23.9 (8.0×, 04.03.2022) | %24.4 (4.8×, 03.03.2020) | %32.8 (5.3×, 31.12.2019) | %13.7 (3.2×, 31.12.2019) |
| Attention BiLSTM | %20.3 (7.4×, 04.03.2020) | %24.2 (5.0×, 02.03.2026) | %25.5 (5.2×, 31.12.2019) | %15.1 (3.5×, 03.01.2020) |
| Hibrit H1: 0.5 XGB + 0.5 BiLSTM | %21.3 (5.9×, 04.03.2022) | %24.7 (4.3×, 20.02.2020) | %28.6 (5.2×, 31.12.2019) | %14.3 (3.3×, 31.12.2019) |
| Hibrit H2: 0.5 HAR-X + 0.5 XGB | %18.3 (4.8×, 03.03.2020) | %25.9 (4.5×, 20.02.2020) | %34.4 (4.8×, 22.01.2020) | %18.3 (3.5×, 27.12.2019) |
| Hibrit H3: HAR-X + XGB artığı | %40.4 (11.7×, 27.11.2013) | %26.9 (5.0×, 20.02.2020) | %28.4 (4.7×, 30.12.2019) | %20.8 (4.4×, 27.12.2019) |
| HAR-X | %30.0 (11.2×, 27.11.2013) | %24.6 (4.6×, 20.02.2020) | %33.7 (4.5×, 22.01.2020) | %22.5 (4.1×, 27.12.2019) |
| HAR-X-log | %19.0 (5.1×, 03.03.2020) | %26.1 (4.5×, 20.02.2020) | %33.8 (4.5×, 30.12.2019) | %22.6 (4.2×, 27.12.2019) |
| HAR | %21.6 (6.7×, 06.03.2020) | %24.5 (5.0×, 04.03.2020) | %30.1 (4.3×, 03.02.2020) | %21.4 (3.8×, 31.12.2019) |
| HAR-log | %20.9 (6.6×, 04.03.2020) | %23.3 (4.9×, 04.03.2020) | %28.5 (4.3×, 31.12.2019) | %21.5 (3.9×, 31.12.2019) |
| GARCH(1,1) | %17.4 (5.9×, 06.03.2020) | %22.3 (4.5×, 05.03.2020) | %27.2 (4.2×, 30.01.2020) | %18.4 (3.6×, 31.12.2019) |
| Train-mean | %26.0 (7.6×, 16.04.2020) | %24.3 (5.3×, 05.03.2020) | %22.8 (4.0×, 04.03.2020) | %13.2 (3.0×, 30.12.2019) |
| Past-volatility | %36.2 (24.0×, 13.09.2012) | %19.5 (4.8×, 06.03.2020) | %32.3 (5.2×, 30.01.2020) | %14.2 (3.3×, 08.10.2014) |
| HAR + OVX | %18.8 (5.1×, 03.03.2020) | %25.8 (4.6×, 20.02.2020) | %34.4 (4.6×, 22.01.2020) | %22.8 (4.0×, 27.12.2019) |
| HAR + GPR | %21.8 (6.7×, 06.03.2020) | %25.0 (5.0×, 04.03.2020) | %29.9 (4.4×, 03.02.2020) | %21.4 (3.8×, 15.11.2019) |
| XGBoost-6 | %18.2 (5.9×, 05.03.2020) | %23.1 (4.7×, 25.02.2020) | %29.7 (4.4×, 11.02.2020) | %23.4 (4.7×, 27.12.2019) |
| XGBoost, Optuna + büzülmüş smearing | %20.9 (8.3×, 13.09.2012) | %23.1 (4.2×, 18.02.2020) | %28.5 (5.0×, 31.12.2019) | %13.5 (3.7×, 02.09.2014) |
| XGBoost, Optuna + ham smearing | %20.2 (8.2×, 13.09.2012) | %23.1 (4.7×, 18.02.2020) | %22.3 (5.3×, 31.12.2019) | %16.1 (5.0×, 02.09.2014) |

**Birincil ailenin iki karşılaştırmasında yön** (betimleyici; test değil). `100 × (kayıp_a / kayıp_b − 1)`, pozitif = a daha kötü.

| a vs b | ufuk | RMSE (fold ort.) | QLIKE (fold ort.) | QLIKE (havuz) |
| --- | --- | --- | --- | --- |
| HAR vs HAR-X | h=5 | +4.75% | −12.59% | −13.61% |
| HAR vs HAR-X | h=22 | +11.59% | +25.62% | +23.08% |
| HAR vs HAR-X | h=66 | +6.10% | +6.90% | +6.82% |
| HAR vs HAR-X | h=126 | +1.43% | +0.69% | +0.65% |
| XGBoost (birincil, 65 özellik) vs HAR-X | h=5 | +5.93% | +10.81% | +10.52% |
| XGBoost (birincil, 65 özellik) vs HAR-X | h=22 | +15.10% | +25.28% | +23.44% |
| XGBoost (birincil, 65 özellik) vs HAR-X | h=66 | +18.21% | +28.52% | +28.53% |
| XGBoost (birincil, 65 özellik) vs HAR-X | h=126 | +7.26% | +8.30% | +8.25% |

**h=5 HAR-X'in en büyük tek QLIKE satırı (27.11.2013).** Tarih, satırın kendi tarihi t'dir, yani **tahmin kökeni**; hedef penceresinin başı değildir. Özellikler `.shift(1)` ile t−1'e (26.11.2013) kadarki bilgiyi kullanır. Hedef `std(r_{t+1}, …, r_{t+5})` olduğundan pencere, t'den sonraki beş işlem gününün getirileridir: 29.11.2013, 02.12.2013, 03.12.2013, 04.12.2013, 05.12.2013 (her getiri bir önceki işlem gününün kapanışından; ilki 27.11.2013 kapanışından 29.11.2013 kapanışına). t günü getirisi ne özelliklerde ne hedefte yer alır. Gerçekleşen σ = 0.013080, HAR-X tahmini σ̂ = 0.001172 (σ/σ̂ = 11.2, QLIKE = 118.7). Fold tabanı 0.001120; tahmin tabanın %4.67 üstünde, tabanlanmamış.

### 12d. Fold başına Duan smearing katsayısı ve log-artık std'si

Smearing `S = mean(exp(e))`, e = eğitim setindeki log ölçekli artıklar (örneklem-içi, train-only). XGBoost ve BiLSTM'de hedef `log(σ_h / past_vol_h)`, HAR-log ve HAR-X-log'da `log(σ_h)`. Kaynak: `wf_summary_all` (XGBoost: `smearing`, `resid_log_std`), `bilstm_folds_all`, `bench_folds_all` (`har_smearing`, `har_x_smearing`). Log-artık std'si (ddof=1): XGBoost için 03'ün kaydı (`resid_log_std`); HAR-log ve HAR-X-log için 05 yalnızca katsayıyı kaydettiğinden OLS `20_log_residual_std.py` ile yeniden tahmin edildi. Yeniden tahminin **her fold'da kayıtlı test tahminlerini ve smearing katsayısını bit düzeyinde ürettiği** assert edildi (120/120). **BiLSTM için log-artık std'si kaydedilmedi:** 06 eğitilmiş ağırlıkları saklamıyor, hesaplamak yeniden eğitim gerektirir. XGBoost/BiLSTM ile HAR-log ailesinin std'leri farklı hedeflerde (log-oran vs log-düzey) olduğundan doğrudan karşılaştırılamaz.

Özet, ana metriğe giren fold'lar: medyan (en küçük–en büyük). BiLSTM log-artık std'si: kaydedilmedi.

| ölçü | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| XGBoost, S | 1.0176 (1.0026–1.0292) | 1.0044 (1.0017–1.0139) | 1.0174 (1.0067–1.0245) | 1.0237 (1.0052–1.0410) |
| XGBoost, log-artık std | 0.1898 (0.0721–0.2439) | 0.0970 (0.0601–0.1644) | 0.1805 (0.1103–0.2080) | 0.2038 (0.0782–0.2662) |
| BiLSTM, S | 1.0346 (1.0242–1.0541) | 1.0045 (1.0020–1.0183) | 1.0108 (1.0058–1.0338) | 1.0274 (1.0057–1.0480) |
| HAR-log, S | 1.1298 (1.1179–1.1403) | 1.0546 (1.0441–1.0714) | 1.0485 (1.0371–1.0759) | 1.0575 (1.0482–1.0814) |
| HAR-log, log-artık std | 0.5000 (0.4795–0.5147) | 0.3226 (0.2934–0.3601) | 0.3034 (0.2610–0.3610) | 0.3274 (0.2944–0.3726) |
| HAR-X-log, S | 1.1106 (1.0992–1.1194) | 1.0393 (1.0356–1.0551) | 1.0392 (1.0285–1.0672) | 1.0517 (1.0382–1.0758) |
| HAR-X-log, log-artık std | 0.4658 (0.4429–0.4799) | 0.2772 (0.2636–0.3182) | 0.2736 (0.2290–0.3352) | 0.3087 (0.2640–0.3592) |

**h=5**

| yıl | XGBoost S | XGBoost log-artık std | BiLSTM S | BiLSTM log-artık std | HAR-log S | HAR-log log-artık std | HAR-X-log S | HAR-X-log log-artık std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 1.0026 | 0.0721 | 1.0432 | kaydedilmedi | 1.1179 | 0.4795 | 1.0992 | 0.4429 |
| 2013 | 1.0059 | 0.1045 | 1.0541 | kaydedilmedi | 1.1256 | 0.4970 | 1.1061 | 0.4610 |
| 2014 | 1.0081 | 0.1241 | 1.0370 | kaydedilmedi | 1.1235 | 0.4910 | 1.1075 | 0.4631 |
| 2015 | 1.0106 | 0.1419 | 1.0393 | kaydedilmedi | 1.1263 | 0.4950 | 1.1093 | 0.4659 |
| 2016 | 1.0129 | 0.1583 | 1.0321 | kaydedilmedi | 1.1283 | 0.4986 | 1.1106 | 0.4680 |
| 2017 | 1.0147 | 0.1717 | 1.0291 | kaydedilmedi | 1.1298 | 0.5014 | 1.1079 | 0.4633 |
| 2018 | 1.0162 | 0.1802 | 1.0380 | kaydedilmedi | 1.1287 | 0.5000 | 1.1068 | 0.4619 |
| 2019 | 1.0176 | 0.1898 | 1.0444 | kaydedilmedi | 1.1269 | 0.4978 | 1.1065 | 0.4613 |
| 2020 | 1.0201 | 0.2000 | 1.0242 | kaydedilmedi | 1.1309 | 0.5034 | 1.1110 | 0.4670 |
| 2021 | 1.0227 | 0.2122 | 1.0271 | kaydedilmedi | 1.1364 | 0.5080 | 1.1162 | 0.4740 |
| 2022 | 1.0255 | 0.2266 | 1.0262 | kaydedilmedi | 1.1403 | 0.5147 | 1.1194 | 0.4799 |
| 2023 | 1.0272 | 0.2337 | 1.0324 | kaydedilmedi | 1.1370 | 0.5096 | 1.1160 | 0.4736 |
| 2024 | 1.0269 | 0.2338 | 1.0346 | kaydedilmedi | 1.1343 | 0.5043 | 1.1138 | 0.4691 |
| 2025 | 1.0277 | 0.2371 | 1.0363 | kaydedilmedi | 1.1310 | 0.4982 | 1.1114 | 0.4641 |
| 2026 | 1.0292 | 0.2439 | 1.0346 | kaydedilmedi | 1.1322 | 0.5001 | 1.1122 | 0.4658 |

**h=22**

| yıl | XGBoost S | XGBoost log-artık std | BiLSTM S | BiLSTM log-artık std | HAR-log S | HAR-log log-artık std | HAR-X-log S | HAR-X-log log-artık std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 1.0054 | 0.1049 | 1.0092 | kaydedilmedi | 1.0441 | 0.2934 | 1.0356 | 0.2636 |
| 2013 | 1.0076 | 0.1244 | 1.0092 | kaydedilmedi | 1.0484 | 0.3091 | 1.0376 | 0.2715 |
| 2014 | 1.0092 | 0.1378 | 1.0155 | kaydedilmedi | 1.0492 | 0.3109 | 1.0391 | 0.2772 |
| 2015 | 1.0106 | 0.1470 | 1.0143 | kaydedilmedi | 1.0495 | 0.3100 | 1.0384 | 0.2748 |
| 2016 | 1.0124 | 0.1560 | 1.0183 | kaydedilmedi | 1.0520 | 0.3158 | 1.0393 | 0.2769 |
| 2017 | 1.0139 | 0.1644 | 1.0139 | kaydedilmedi | 1.0546 | 0.3226 | 1.0392 | 0.2772 |
| 2018 | 1.0017 | 0.0601 | 1.0030 | kaydedilmedi | 1.0527 | 0.3163 | 1.0365 | 0.2678 |
| 2019 | 1.0019 | 0.0652 | 1.0020 | kaydedilmedi | 1.0525 | 0.3167 | 1.0369 | 0.2691 |
| 2020 | 1.0024 | 0.0718 | 1.0025 | kaydedilmedi | 1.0552 | 0.3265 | 1.0409 | 0.2824 |
| 2021 | 1.0030 | 0.0784 | 1.0045 | kaydedilmedi | 1.0667 | 0.3494 | 1.0534 | 0.3128 |
| 2022 | 1.0034 | 0.0845 | 1.0044 | kaydedilmedi | 1.0705 | 0.3587 | 1.0551 | 0.3182 |
| 2023 | 1.0038 | 0.0898 | 1.0065 | kaydedilmedi | 1.0714 | 0.3601 | 1.0549 | 0.3176 |
| 2024 | 1.0040 | 0.0924 | 1.0033 | kaydedilmedi | 1.0701 | 0.3569 | 1.0535 | 0.3140 |
| 2025 | 1.0044 | 0.0970 | 1.0039 | kaydedilmedi | 1.0683 | 0.3522 | 1.0523 | 0.3107 |
| 2026 | 1.0049 | 0.1010 | 1.0042 | kaydedilmedi | 1.0679 | 0.3510 | 1.0521 | 0.3100 |

**h=66**

| yıl | XGBoost S | XGBoost log-artık std | BiLSTM S | BiLSTM log-artık std | HAR-log S | HAR-log log-artık std | HAR-X-log S | HAR-X-log log-artık std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 1.0067 | 0.1119 | 1.0212 | kaydedilmedi | 1.0371 | 0.2610 | 1.0285 | 0.2290 |
| 2013 | 1.0144 | 0.1624 | 1.0262 | kaydedilmedi | 1.0385 | 0.2673 | 1.0335 | 0.2507 |
| 2014 | 1.0163 | 0.1766 | 1.0267 | kaydedilmedi | 1.0424 | 0.2835 | 1.0350 | 0.2583 |
| 2015 | 1.0196 | 0.1845 | 1.0293 | kaydedilmedi | 1.0436 | 0.2874 | 1.0356 | 0.2592 |
| 2016 | 1.0245 | 0.2080 | 1.0338 | kaydedilmedi | 1.0484 | 0.3028 | 1.0405 | 0.2762 |
| 2017 | 1.0067 | 0.1103 | 1.0058 | kaydedilmedi | 1.0486 | 0.3039 | 1.0390 | 0.2723 |
| 2018 | 1.0077 | 0.1222 | 1.0100 | kaydedilmedi | 1.0474 | 0.3001 | 1.0366 | 0.2634 |
| 2019 | 1.0086 | 0.1286 | 1.0091 | kaydedilmedi | 1.0462 | 0.2960 | 1.0356 | 0.2601 |
| 2020 | 1.0102 | 0.1405 | 1.0141 | kaydedilmedi | 1.0498 | 0.3096 | 1.0394 | 0.2750 |
| 2021 | 1.0185 | 0.1857 | 1.0077 | kaydedilmedi | 1.0759 | 0.3610 | 1.0672 | 0.3352 |
| 2022 | 1.0194 | 0.1905 | 1.0059 | kaydedilmedi | 1.0737 | 0.3571 | 1.0646 | 0.3294 |
| 2023 | 1.0197 | 0.1914 | 1.0115 | kaydedilmedi | 1.0736 | 0.3583 | 1.0622 | 0.3247 |
| 2024 | 1.0208 | 0.1954 | 1.0092 | kaydedilmedi | 1.0706 | 0.3515 | 1.0595 | 0.3178 |
| 2025 | 1.0201 | 0.1934 | 1.0100 | kaydedilmedi | 1.0686 | 0.3464 | 1.0579 | 0.3142 |
| 2026 (ana metrik dışı) | 1.0201 | 0.1932 | 1.0067 | kaydedilmedi | 1.0668 | 0.3421 | 1.0565 | 0.3109 |

**h=126**

| yıl | XGBoost S | XGBoost log-artık std | BiLSTM S | BiLSTM log-artık std | HAR-log S | HAR-log log-artık std | HAR-X-log S | HAR-X-log log-artık std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 1.0052 | 0.0782 | 1.0075 | kaydedilmedi | 1.0498 | 0.2996 | 1.0382 | 0.2640 |
| 2013 | 1.0143 | 0.1531 | 1.0246 | kaydedilmedi | 1.0482 | 0.2944 | 1.0447 | 0.2861 |
| 2014 | 1.0159 | 0.1654 | 1.0220 | kaydedilmedi | 1.0522 | 0.3076 | 1.0456 | 0.2896 |
| 2015 | 1.0199 | 0.1828 | 1.0248 | kaydedilmedi | 1.0546 | 0.3150 | 1.0440 | 0.2836 |
| 2016 | 1.0243 | 0.2031 | 1.0309 | kaydedilmedi | 1.0617 | 0.3370 | 1.0571 | 0.3223 |
| 2017 | 1.0243 | 0.2054 | 1.0273 | kaydedilmedi | 1.0593 | 0.3329 | 1.0538 | 0.3155 |
| 2018 | 1.0239 | 0.2045 | 1.0307 | kaydedilmedi | 1.0556 | 0.3218 | 1.0497 | 0.3020 |
| 2019 | 1.0227 | 0.1993 | 1.0265 | kaydedilmedi | 1.0526 | 0.3123 | 1.0465 | 0.2926 |
| 2020 | 1.0235 | 0.2060 | 1.0275 | kaydedilmedi | 1.0529 | 0.3153 | 1.0469 | 0.2960 |
| 2021 | 1.0410 | 0.2662 | 1.0480 | kaydedilmedi | 1.0814 | 0.3726 | 1.0758 | 0.3592 |
| 2022 | 1.0403 | 0.2619 | 1.0448 | kaydedilmedi | 1.0766 | 0.3624 | 1.0715 | 0.3483 |
| 2023 | 1.0404 | 0.2662 | 1.0408 | kaydedilmedi | 1.0758 | 0.3633 | 1.0684 | 0.3432 |
| 2024 | 1.0408 | 0.2661 | 1.0451 | kaydedilmedi | 1.0717 | 0.3526 | 1.0650 | 0.3339 |
| 2025 | 1.0196 | 0.1870 | 1.0057 | kaydedilmedi | 1.0694 | 0.3472 | 1.0627 | 0.3283 |
| 2026 (ana metrik dışı) | 1.0184 | 0.1804 | 1.0074 | kaydedilmedi | 1.0665 | 0.3404 | 1.0602 | 0.3222 |

### 12e. Keşifsel: h=5 QLIKE yön dönmesi ve HAR-X'in düşük tahminleri

**Statü: keşifsel ve post hoc.** Her iki kontrol de QLIKE sonuçları görüldükten sonra tasarlandı; eşikler (1.25 × taban, 0.5 × σ̂_HAR) sonuçlara bakılarak seçildi. Test yok. Havuzlanmış, ana fold'lar.

**(1) Tabana yakınlık, h=5.** σ̂ / fold tabanı; eşik σ̂ ≤ 1.25 × taban.

| küme | n | HAR-X σ̂/taban, medyan (min–maks) | HAR-X ≤ 1.25 × taban | HAR σ̂/taban, medyan (min–maks) | HAR ≤ 1.25 × taban |
| --- | --- | --- | --- | --- | --- |
| HAR-X'in en büyük %1 QLIKE satırı | 36 | 3.77 (1.05–25.94) | 2 (%5.6) | 11.02 (5.35–20.89) | 0 (%0.0) |
| bütün h=5 test gözlemleri | 3662 | 15.64 (1.05–182.79) | 2 (%0.1) | 14.93 (2.79–77.25) | 0 (%0.0) |

En büyük %1 satır HAR-X'in h=5 QLIKE toplamının %30.0'ini, σ̂ ≤ 1.25 × taban olan 2 satır %5.5'ini oluşturuyor. Bu satırlar hariç ortalama QLIKE: HAR-X 0.6292, HAR 0.5751 (tümü: 0.6654 / 0.5749).

| tarih | σ | σ̂ HAR-X | HAR-X/taban | QLIKE HAR-X | σ̂ HAR | HAR/taban | QLIKE HAR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27.11.2013 | 0.01308 | 0.001172 | 1.05 | 118.7 | 0.01232 | 11.01 | 0.007 |
| 26.11.2013 | 0.01245 | 0.001491 | 1.33 | 64.4 | 0.01332 | 11.90 | 0.009 |
| 29.11.2013 | 0.01052 | 0.001777 | 1.59 | 30.5 | 0.01235 | 11.03 | 0.046 |
| 22.07.2014 | 0.00907 | 0.001596 | 1.43 | 27.8 | 0.00784 | 7.01 | 0.046 |
| 23.07.2014 | 0.00971 | 0.001715 | 1.53 | 27.6 | 0.00776 | 6.93 | 0.118 |
| 24.07.2014 | 0.00921 | 0.001644 | 1.47 | 26.9 | 0.00801 | 7.16 | 0.043 |
| 02.12.2013 | 0.01208 | 0.002210 | 1.97 | 25.5 | 0.01299 | 11.60 | 0.010 |
| 03.03.2020 | 0.13320 | 0.026369 | 23.55 | 21.3 | 0.02339 | 20.89 | 27.941 |
| 28.08.2014 | 0.01962 | 0.003955 | 3.53 | 20.4 | 0.00864 | 7.72 | 2.510 |
| 12.08.2014 | 0.01229 | 0.002539 | 2.27 | 19.3 | 0.00810 | 7.24 | 0.466 |
| 10.09.2019 | 0.07674 | 0.016087 | 14.37 | 18.6 | 0.02011 | 17.96 | 10.887 |
| 27.08.2014 | 0.01926 | 0.004054 | 3.62 | 18.4 | 0.00882 | 7.87 | 2.208 |
| 04.03.2020 | 0.13121 | 0.027760 | 24.79 | 18.2 | 0.01972 | 17.62 | 39.463 |
| 29.08.2014 | 0.01875 | 0.003986 | 3.56 | 18.0 | 0.00863 | 7.71 | 2.164 |
| 11.09.2019 | 0.07576 | 0.016262 | 14.52 | 17.6 | 0.01877 | 16.77 | 12.500 |
| 06.03.2020 | 0.13506 | 0.029040 | 25.94 | 17.6 | 0.02013 | 17.97 | 40.233 |
| 05.03.2020 | 0.12841 | 0.027718 | 24.76 | 17.4 | 0.02017 | 18.01 | 35.838 |
| 26.08.2014 | 0.01882 | 0.004231 | 3.78 | 15.8 | 0.00891 | 7.96 | 1.967 |
| 25.07.2014 | 0.00529 | 0.001206 | 1.08 | 15.3 | 0.00824 | 7.36 | 0.299 |
| 12.09.2019 | 0.07523 | 0.017309 | 15.46 | 15.0 | 0.01842 | 16.45 | 12.865 |
| 13.09.2019 | 0.07519 | 0.017511 | 15.64 | 14.5 | 0.01689 | 15.08 | 15.838 |
| 02.04.2013 | 0.01929 | 0.004540 | 4.05 | 14.2 | 0.00962 | 8.59 | 1.629 |
| 28.07.2014 | 0.00747 | 0.001803 | 1.61 | 13.3 | 0.00892 | 7.96 | 0.056 |
| 09.09.2019 | 0.06597 | 0.016021 | 14.31 | 13.1 | 0.01993 | 17.80 | 7.559 |
| 02.03.2020 | 0.11464 | 0.028169 | 25.16 | 12.8 | 0.02129 | 19.02 | 24.620 |
| 28.03.2013 | 0.01639 | 0.004133 | 3.69 | 12.0 | 0.00940 | 8.39 | 0.931 |
| 13.08.2014 | 0.01086 | 0.002774 | 2.48 | 11.6 | 0.00899 | 8.03 | 0.081 |
| 27.03.2013 | 0.01623 | 0.004215 | 3.76 | 11.1 | 0.01004 | 8.97 | 0.652 |
| 18.11.2021 | 0.06065 | 0.016303 | 14.56 | 10.2 | 0.01582 | 14.13 | 11.006 |
| 21.11.2014 | 0.05007 | 0.013549 | 12.10 | 10.0 | 0.01276 | 11.40 | 11.650 |
| 02.09.2014 | 0.01479 | 0.004082 | 3.65 | 9.6 | 0.00872 | 7.79 | 0.821 |
| 26.03.2013 | 0.01671 | 0.004620 | 4.13 | 9.5 | 0.00957 | 8.55 | 0.933 |
| 24.11.2014 | 0.04966 | 0.014293 | 12.77 | 8.6 | 0.01313 | 11.73 | 10.645 |
| 09.06.2014 | 0.01264 | 0.003655 | 3.26 | 8.5 | 0.00599 | 5.35 | 1.961 |
| 19.11.2021 | 0.06201 | 0.017931 | 16.01 | 8.5 | 0.01580 | 14.11 | 11.667 |
| 22.11.2021 | 0.06109 | 0.017706 | 15.81 | 8.4 | 0.01687 | 15.07 | 9.536 |

**(2) Mekanik kural: "HAR-X belirgin düşük" = σ̂_HAR-X < 0.5 × σ̂_HAR.** Pay: kural satırlarının Σ(QLIKE_HAR-X − QLIKE_HAR) içindeki payı (toplam fark negatifse pay işaretiyle okunmalı). Çıkarma sonrası ortalamalar havuzlanmış ve fold ortalaması olarak verilir.

| ufuk | kural satırı | yıllar | Σ fark (tümü) | Σ fark (kural satırları) | pay | hariç ort. QLIKE, havuz (HAR-X / HAR) | hariç ort. QLIKE, fold ort. (HAR-X / HAR) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 82 / 3662 | 2013 (46), 2014 (34), 2017 (2) | +331.56 | +580.01 | %174.9 | 0.5072 / 0.5766 | 0.5094 / 0.5777 |
| h=22 | 9 / 3645 | 2014 (9) | −238.44 | +4.63 | %−1.9 | 0.2825 / 0.3494 | 0.2837 / 0.3576 |
| h=66 | 0 / 3500 | — | −86.00 | +0.00 | — | 0.3602 / 0.3847 | 0.3581 / 0.3828 |
| h=126 | 0 / 3500 | — | −9.10 | +0.00 | — | 0.4005 / 0.4031 | 0.3992 / 0.4020 |

## 13. Ek aile: Clark–West; XGBoost-6 vs HAR; 2026 dipnotu

### 13a. Clark–West testi, HAR ⊂ HAR-X (ek aile, 4 test)

**Etiket:** iç içe yapıya uygun istatistik; birincil DM testleri görüldükten sonra, CW sonuçları görülmeden eklendi. Aile 2026-09-29 16:15:34 +0300 tarihinde CLAUDE.md'de ilan edildi (commit `72106e7`); CW istatistiği depoda bundan önce hesaplanmamıştı. **Birincil aile 8 testle sabittir; CW oraya eklenmez ve DM testlerinin yerine geçmez.** Holm/BH/BY bu 4 test içinde.

`f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`; H1: E[f] > 0 (HAR-X daha iyi), **tek yanlı**. Kayıtlı (yayımlanan, tabanlanmış) tahminler, ana fold'lar, havuzlanmış seri (DM gibi). HAC: Newey-West, Bartlett, L = h−1. Çıkarım DM birincil ailesiyle aynı: HLN çarpanı ve t(n−1); düzeltmeler HLN p değerine uygulanır. HLN'siz normal p yan sütunda. Kaynak: `21_clark_west.py`, `clark_west_publication_aligned.csv`.

| ufuk | n | ort. (e²_HAR − e²_HARX) | ort. düzeltme (ŷ_HAR − ŷ_HARX)² | ort. f | CW | CW (HLN) | p normal (ham) | p HLN (ham) | Holm | BH | BY | HAC şişme |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 3662 | 4.49e−06 | 3.02e−05 | 3.47e−05 | 5.343 | 5.336 | <0.001 | <0.001 | <0.001 | <0.001 | <0.001 | 2.62 |
| h=22 | 3645 | 7.89e−06 | 2.60e−05 | 3.39e−05 | 3.608 | 3.587 | <0.001 | <0.001 | <0.001 | <0.001 | <0.001 | 6.73 |
| h=66 | 3500 | −3.24e−06 | 1.80e−05 | 1.47e−05 | 3.032 | 2.975 | 0.001 | 0.001 | 0.003 | 0.002 | 0.004 | 3.08 |
| h=126 | 3500 | −5.56e−06 | 1.10e−05 | 5.48e−06 | 1.238 | 1.194 | 0.108 | 0.116 | 0.116 | 0.116 | 0.242 | 5.55 |

**Fold düzeyinde (betimleyici, test değil):** f'nin fold ortalamalarının ortalaması ve f ortalaması pozitif olan fold sayısı: h=5: 3.74e−05, 15/15; h=22: 3.71e−05, 15/15; h=66: 1.48e−05, 12/14; h=126: 5.55e−06, 10/14.

Okuma notu: ilk sütun DM'nin kullandığı ham MSE farkıdır; h=66 ve h=126'da negatiftir (havuzlanmış seride HAR'ın MSE'si daha düşük). CW istatistiği buna tahmin farkının karesini ekler.

### 13b. XGBoost-6 vs HAR, fold bazında (keşifsel; test yok)

XGBoost-6 keşifseldir; p değeri verilmez. Kazanma: fold RMSE'si HAR'ınkinden düşük. Uyuşma: kazanma çoğunluğunun yönü ile fold ortalaması RMSE farkının yönü aynı mı. `100 × (RMSE_XGB-6 / RMSE_HAR − 1)`, pozitif = XGBoost-6 daha kötü.

| ufuk | XGB-6 kazandığı yıl / fold | fold ort. farkı | sayım yönü | ortalama yönü | uyuşuyor mu | ortalamayı taşıyan yıllar |
| --- | --- | --- | --- | --- | --- | --- |
| h=5 | 5/15 | +0.03% | HAR | HAR | evet | — |
| h=22 | 8/15 | −2.23% | XGB-6 | XGB-6 | evet | — |
| h=66 | 8/14 | −0.89% | XGB-6 | XGB-6 | evet | — |
| h=126 | 8/14 | −0.79% | XGB-6 | XGB-6 | evet | — |

### 13c. 2026 kısmi yıl, h=66 ve h=126

Bölüm 1e'de: tüm modellerin o fold'daki RMSE/MAE/R²_oos'u, n = 101 (h=66) ve 41 (h=126). Birincil toplulaştırmadan dışlanmıştır; model karşılaştırması veya seçimi için kullanılmaz.

