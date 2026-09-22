# Deney Günlüğü

Bu dosya, XGBoost ana modeli için denenen tüm konfigürasyonları **kronolojik sırayla** ve
her birinin sonucunu kaydeder. Makalenin Sınırlılıklar bölümünde bu tablo verilecektir.

Amaç şeffaflıktır: hangi fikirlerin denendiği, hangilerinin işe yaramadığı ve nihai
spesifikasyonun neden seçildiği burada görünür. Nihai seçim **test performansına göre
yapılmamıştır** (bkz. CLAUDE.md "Model Seçim Politikası"); seçim gerekçesi validation
tarafında ölçülen zayıf seçim sinyalidir.

Tüm metrikler **fold ortalaması RMSE**, günlük log-getiri standart sapması biriminde.
Doğrulama şeması sabit: genişleyen pencere, 15 fold, test yılları 2012–2026, embargo = h.
h=66 ve h=126'da 2026 fold'u kısmi yıl kuralı gereği ana ortalamadan hariç (n=14).

## Sürümler

| # | Konfigürasyon | Değişen tek şey |
| --- | --- | --- |
| 1 | 61 özellik, log hedef, sabit kapasite (400 ağaç, derinlik 4) | temel |
| 2 | + HAR gerçekleşen volatilite, 252 penceresi dahil, 67 özellik | özellik seti |
| 3 | + HAR gerçekleşen volatilite, 252 penceresi hariç, 65 özellik | özellik seti |
| 4 | + ratio hedef: `log(vol_h) − log(past_vol_h)` | hedef parametrelendirmesi |
| 5 | + kademeli kapasite kuralı (etkin gözleme bağlı) | model kapasitesi |
| 6 | + Optuna, smearing validation artıklarından (ham) | hiperparametre seçimi |
| 7 | + Optuna, smearing büzülmüş (`w = n_eff/(n_eff+10)`) | smearing varyansı |

## XGBoost RMSE (fold ortalaması)

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | 0.010772 | 0.008938 | 0.010401 | 0.009923 |
| 2 | 0.011174 | 0.009497 | 0.009595 | 0.009163 |
| 3 | 0.010829 | 0.009008 | 0.010304 | 0.009907 |
| 4 | 0.010999 | 0.009091 | 0.010084 | 0.009886 |
| **5** | **0.010913** | **0.009080** | **0.008924** | **0.008627** |
| 6 | 0.011025 | 0.009178 | 0.011042 | 0.011174 |
| 7 | 0.010927 | 0.008788 | 0.009797 | 0.009715 |

Kalın satır birincil spesifikasyondur.

## Past-volatility baseline'ına göre fark (%, negatif = model daha iyi)

Baseline RMSE'si tüm sürümlerde sabittir (0.013318 / 0.009493 / 0.008913 / 0.008508),
çünkü özellik setine ve modele bağlı değildir. Bu, tabloyu sürümler arası karşılaştırma
için geçerli bir çapa yapar.

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | −19.1 | −5.8 | +16.7 | +16.6 |
| 2 | −16.1 | +0.0 | +7.7 | +7.7 |
| 3 | −18.7 | −5.1 | +15.6 | +16.5 |
| 4 | −17.4 | −4.2 | +13.1 | +16.2 |
| **5** | **−18.1** | **−4.3** | **+0.1** | **+1.4** |
| 6 | −17.2 | −3.3 | +23.9 | +31.3 |
| 7 | −17.9 | −7.4 | +9.9 | +14.2 |

## R²_oos (train ortalaması referanslı, fold ortalaması)

| # | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| 1 | +0.277 | +0.205 | −0.757 | −0.821 |
| 2 | +0.193 | +0.010 | −0.807 | −0.729 |
| 3 | +0.287 | +0.248 | −0.831 | −1.118 |
| 4 | +0.272 | +0.191 | −0.744 | −1.124 |
| **5** | **+0.283** | **+0.186** | **−0.199** | **−0.283** |
| 6 | +0.271 | +0.098 | −1.529 | −1.673 |
| 7 | +0.281 | +0.219 | −0.770 | −0.744 |

## Her adımda ne öğrenildi

**1 → 2: HAR volatilite özellikleri eklendi (252 penceresi dahil).** Uzun ufuklarda RMSE
%7.7 düştü, kısa ufuklarda yükseldi. Ancak karşılaştırma kirliydi: 252 günlük pencere ilk
geçerli satırı 60'tan 253'e taşıyıp her fold'dan ~193 satır götürdü. Özellik körü train-mean
baseline'ı da %4.1–4.4 iyileşti, yani kazancın yarısından fazlası veri penceresi
değişikliğindendi. Kaynağı 2008 krizinin eğitim setinden kısmen çıkması.

**2 → 3: 252 penceresi çıkarıldı, veri maliyeti yarıya indi.** Train-mean baseline'ı %1'in
altında oynadı, yani veri penceresi etkisi ortadan kalktı. Aynı sürümde XGBoost da hiçbir
ufukta %1'i geçen değişim göstermedi. **Sonuç: eşleşen pencereli HAR özellikleri, veri
maliyeti kalktığında ölçülebilir katkı sağlamıyor.** Hipotez test edildi ve desteklenmedi.
Özellikler yine de korundu, çünkü HAR benchmark'ı için gerekli ve maliyeti sıfıra yakın.

**3 → 4: hedef, past-volatility'den sapma olarak yeniden yazıldı.** Beklenti, sinyal
yokluğunda modelin baseline'ı yapısal olarak kopyalamasıydı. Etki küçük kaldı: uzun
ufuklarda %0.2–2.1 iyileşme, kısa ufuklarda %1–2 gerileme. En kötü fold belirgin düzeldi
(h=126/2013 oranı 3.75 kattan 2.49 kata). Yapısal koruma tam işlemedi, çünkü model sinyal
yokluğunu tanıyıp sıfır tahmin etmiyor; düzenlileştirilmemiş 400 ağaç her zaman kendinden
emin sapmalar üretiyor. Ezberleme ölçüsü (hedef uzayına göre kalibre edilmiş örneklem-içi
R²) h=126'da 0.994'ten 0.992'ye indi, yani pratikte değişmedi.

**4 → 5: kapasite etkin örneklem büyüklüğüne bağlandı.** En büyük tek etki. Uzun ufuklarda
RMSE %11.5 ve %12.7 düştü; h=66 ve h=126 ilk kez train-mean baseline'ını geçti ve
past-volatility ile berabere hale geldi. Örneklem-içi R² h=126'da 0.992'den 0.746'ya indi,
yani ezberleme gerçekten kırıldı. En kötü fold oranları h=66'da 3.21'den 1.86'ya,
h=126'da 3.24'ten 2.35'e indi. **Teşhis: uzun ufuk başarısızlığının ana sürücüsü, model
kapasitesi ile etkin bağımsız gözlem sayısı arasındaki dengesizlikti.** h=126'da en büyük
fold'da bile yalnızca ~33 etkin gözlem var, model ise 65 özellik kullanıyor.

**5 → 6: Optuna + validation tabanlı smearing.** Ağır gerileme: uzun ufuklarda RMSE %24 ve
%30 arttı. Ayrıştırma, suçlunun hiperparametre seçimi değil smearing katsayısı olduğunu
gösterdi. Bozulma ile mutlak smearing sapması arasındaki korelasyon h=66'da 0.825,
h=126'da 0.768; seçim sinyaliyle korelasyon ise negatif (−0.30, −0.17). Örneklem-içi
smearing yanlıydı ama 1.0000'a çöktüğü için zararsızdı; validation tabanlı smearing yansız
ama ~5 etkin gözlemden tahmin edildiği için 0.60–1.59 arasında salınıyor ve tek skalar
olarak tüm test tahminlerini çarpıyor.

**6 → 7: smearing büzüldü.** `S = 1 + w(S_ham − 1)`, `w = n_eff/(n_eff+10)`. Katsayı aralığı
h=126'da 0.599–1.580'den 0.870–1.158'e daraldı; ortalama mutlak sapma 0.198'den 0.061'e
indi. RMSE her ufukta ham sürümü geçti (%0.9–13.1 iyileşme) ve smearing korelasyonu
h=126'da 0.193'e düştü. Ama kapasite kuralını yalnızca h=22'de geçebildi. Kalan gerileme
artık doğrudan seçim gürültüsünden geliyor.

## Nihai karar

**Birincil spesifikasyon: sürüm 5, kademeli kapasite kuralı, dört ufukta da.**

Gerekçe test performansı değildir. Optuna çalıştırılan 52 fold'un 32'sinde en iyi denemenin
validation RMSE'si medyan denemeden %5'ten az iyiydi; örtüşen hedef pencereleri nedeniyle
validation dilimi uzun ufuklarda yalnızca 5–6.6 etkin bağımsız gözlem içeriyor. Bu ölçüt
validation dağılımından okunur, test sonucuna bakmaz.

**Sürüm 7 ikincil / sağlamlık analizi olarak raporlanır.** h=22'de birincilden daha iyi
(RMSE %3.2 düşük), h=66 ve h=126'da daha kötü (%13 yüksek), h=5'te fark yok. Ufuk bazında
en iyisini seçmek (cherry-picking) yapılmayacaktır.

**Sürüm 6 makalenin ekinde** yanlılık-varyans takası örneği olarak kullanılır.

## Çıktı dosyaları

| sürüm | dosyalar |
| --- | --- |
| 5 (birincil) | `wf_predictions_h*.csv`, `wf_metrics_all.csv`, `wf_aggregate_all.csv`, `wf_summary_all.json` |
| 6 (ham smearing) | `opt_rawsmearing_metrics_all.csv`, `opt_rawsmearing_aggregate_all.csv`, `opt_rawsmearing_folds_all.csv`, `opt_rawsmearing_summary_all.json` |
| 7 (büzülmüş) | `opt_predictions_all.csv`, `opt_metrics_all.csv`, `opt_aggregate_all.csv`, `opt_folds_all.csv`, `opt_summary_all.json` |

Sürüm 1–4 ara adımlardır ve çıktıları saklanmamıştır; yukarıdaki tablolardaki değerleri
`scripts/02_build_features.py` ve `scripts/03_walkforward.py` ilgili parametrelerle yeniden
çalıştırılarak üretilebilir.

---

# Aşama 5: Ekonometrik Benchmark'lar

Bu bölüm `scripts/05_benchmarks.py` çıktılarını ve geliştirme sırasında bulunan
spesifikasyon hatalarını **keşif sırasıyla** kaydeder.

## Sekiz modelli karşılaştırma (fold ortalaması RMSE)

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| **HAR-X-log** | **0.010335** | **0.007561** | **0.007504** | **0.008012** |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR-log | 0.010857 | 0.008678 | 0.008199 | 0.008275 |
| XGBoost (birincil) | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| GARCH(1,1)-t | 0.011211 | 0.008970 | 0.009036 | 0.009436 |
| past-volatility | 0.013318 | 0.009493 | 0.008913 | 0.008508 |
| train-mean | 0.013440 | 0.011216 | 0.009240 | 0.009028 |

## Ana ayrıştırma: kazancın kaynağı

| ufuk | dışsal değişkenler (HAR→HAR-X) | doğrusal olmayan model (HAR-X→XGBoost) |
| --- | --- | --- |
| 5 | −4.55% | +5.53% |
| 22 | −10.82% | +18.39% |
| 66 | −6.50% | +17.84% |
| 126 | −1.76% | +7.05% |

**Kazanç dışsal değişkenlerden (OVX, GPR) geliyor; doğrusal olmayan model aynı bilgiyle
dört ufukta da zarar veriyor.** Bu, projenin örtük varsayımını tersine çeviren ana
bulgudur.

## Veri eşitleme sağlamlık kontrolü

**Amaç:** XGBoost'un kaybının HAR ailesinin ~106 günlük (%4) veri avantajından
kaynaklanmadığını göstermek.

**Yöntem notu:** XGBoost'u HAR'ın penceresine (21. satır) indirmek MÜMKÜN DEĞİLDİR;
`brent_vol126` o satırda NaN'dır ve inmek için XGBoost'un özellik setini budamak
gerekirdi, bu da giderilmek istenen karıştırıcıyı daha büyüğüyle değiştirirdi. Bu yüzden
eşitleme ters yönde yapıldı: benchmark'lar XGBoost'un penceresine (127. satır)
indirildi. Hiçbir modelin özellik seti değişmedi. Eşitleme sonrası HAR/HAR-X train
büyüklükleri XGBoost'unkiyle birebir aynıdır.

| model | XGBoost'a göre fark, normal pencere | eşitlenmiş pencere |
| --- | --- | --- |
| HAR | −0.72 / −5.29 / −9.24 / −4.91 | −0.72 / −5.36 / −9.35 / −6.22 |
| HAR-X | −5.24 / −15.54 / −15.14 / −6.58 | −5.18 / −15.40 / −15.11 / −7.88 |
| HAR-X-log | −5.30 / −16.72 / −15.92 / −7.12 | −5.23 / −16.67 / −16.06 / −8.57 |

(h=5 / h=22 / h=66 / h=126 sırasıyla, % cinsinden; negatif = benchmark daha iyi.)

**Sonuç: veri farkı açıklama değildir.** Eşitleme sonrası HAR ailesinin üstünlüğü aynı
kaldı, h=126'da bir miktar ARTTI. Fazla erken veri (2008 krizi) HAR'a yardım etmiyormuş.
Bu bir sağlamlık kontrolüdür; birincil spesifikasyon sonuca göre değiştirilmemiştir.
Çıktılar `bench_*_aligned.*` dosyalarındadır.

## Geliştirme sırasında bulunan hatalar (keşif sırasıyla)

**1. `arch` kütüphanesinin `fit(last_obs=)` parametresi konumsaldır.** Getiri serisinin
indeksi 1'den başladığı için `last_obs=first_test` train dilimine test döneminin ilk
gününü katıyordu. Açık dilim (`ret[ret.index < first_test]`) ile karşılaştırılınca
parametrelerin farklı çıktığı görüldü ve açık dilimlemeye geçildi. `forecast(start=)`
de konumsaldır; konum açıkça hesaplanıp sonuç etiket bazında assert ile doğrulanıyor.
Ayrıca her fold'da "origin t−1'in birinci adımı = σ²_t" eşitliği kontrol ediliyor.

**2. `arch` tahmin sütun adları ufka göre sıfır dolguludur** (`h.1` / `h.01` / `h.001`).
h=5 çalışıp h=126'da KeyError alındı; sütun seçimi ada göre değil konuma göre yapıldı.

**3. HAR-log yanlış spesifiye edilmişti — keşif sırası önemlidir.** Planda "aynı
regresörler, bağımlı değişken log(target)" olarak tarif edilmişti. **İki fold'luk duman
testinde HAR-log RMSE'si h=5'te 0.040098 çıktı, seviyelerdeki HAR'ın (0.018221) iki
katından fazla.** Bu anormallik incelendiğinde bağımlı değişkeni loglayıp regresörleri
seviyede bırakmanın bir yanlış spesifikasyon olduğu görüldü; kanonik log-HAR her iki
tarafı da loglar. Log-log forma geçildi. **Bu değişiklik sonuç kovalayarak değil,
kanonik formu düzelterek yapılmıştır, ancak sıra şeffaflık adına aynen kaydedilir:
önce anormal sonuç görüldü, sonra spesifikasyon düzeltildi.** Düzeltme sonrası HAR-log,
HAR'a çok yakın çıktı (%0.2–1.2 daha kötü), yani anormallik gerçekten
misspecification'dan kaynaklanıyormuş.

`har_daily = |r_{t−1}|` 25 günde tam sıfırdır (kapanış değişmemiş) ve log tanımsız olur;
log spesifikasyonundaki regresörler önceden ilan edilmiş `LOG_FLOOR = 1e-4` tabanında
sınırlanır, 31 gözlem tabanlanır.

**4. HAR-X-log sonradan eklendi.** Başlangıçta 2×2 tasarımın (HAR/HAR-X × seviye/log)
yalnızca üç hücresi vardı. Dördüncü hücre tamamlandı. Log form ayrıca HAR-X'in negatif
volatilite tahmin edip tabana takılma sorununu yapısal olarak çözer: seviyelerde
h=66'da test gözlemlerinin %2.69'u, h=22'de %1.62'si tabana takılıyordu; log formda
bu sorun tanım gereği yoktur.

## GARCH notları

GARCH'a embargo uygulanmaz (ileriye bakan etiket kullanmaz), bu yüzden XGBoost'tan
131–252 gün fazla veri görür; fark fold bazında ve embargo/ısınma bileşenlerine
ayrılarak raporlanır. Avantaja rağmen GARCH her ufukta HAR'ın gerisindedir ve h=66 ile
h=126'da past-volatility baseline'ını bile geçemez.

GARCH ısrarcılığı (α+β) her fold'da ~1.000, yani neredeyse IGARCH. Varyans süreci birim
köke çok yakındır; bu, uzun ufuk tahminlerinin koşulsuz ortalamaya yakınsayıp bilgi
taşımamasını açıklar.

---

# Aşama 6: Attention BiLSTM

`scripts/06_attention_bilstm.py`. Amaç bir hipotez testi: XGBoost'un doğrusal
spesifikasyonlara kaybı modele mi özgü, yoksa bu problemde doğrusal olmayan
modellemenin genel sınırı mı?

Birincil spesifikasyonla aynı: fold yapısı, embargo, 2026 kuralı, ön işleme sırası,
ratio hedefi, train artıklarından Duan smearing, metrikler. Mimari, kapasite kuralının
BiLSTM karşılığı olan önceden ilan edilmiş kademeli tablodan seçilir. Lookback L=20,
ilan edilmiş, ayarlanmadı. Early stopping yok; gerekçesi Optuna deneyinde ölçülen
zayıf seçim sinyali.

## Tüm modeller (fold ortalaması RMSE)

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

BiLSTM R²_oos dört ufukta da negatif: −0.371, −0.249, −2.218, −3.094. Past-volatility'yi
hiçbir ufukta, train-mean'i yalnızca h=22'de geçebiliyor.

## Hipotez testinin cevabı

| ufuk | XGBoost vs HAR-X | BiLSTM vs HAR-X |
| --- | --- | --- |
| 5 | +5.53% | +37.58% |
| 22 | +18.39% | +37.07% |
| 66 | +17.84% | +53.03% |
| 126 | +7.05% | +41.83% |

**İki bağımsız doğrusal olmayan model ailesi de aynı bilgiyle doğrusal HAR-X'i
geçemiyor.** Sonuç XGBoost'a özgü değil.

## Dejenere tahmin kontrolü

0/60 fold dejenere (std(tahmin)/std(gerçek) < 0.1). Başarısızlığın biçimi sabit değere
çökme değil, **aşırı saçılım**: h=126'da oran ortalaması 1.486, yani model gerçekte
olduğundan ~1.5 kat oynak bir seri tahmin ediyor. Model gürültüyü sinyal sanıyor.

## Sağlamlık kontrolü: yetersiz eğitim

Birincil koşuda 8/60 fold "yetersiz eğitilmiş" çıktı (son %20 epoch diliminde eğitim
kaybı hâlâ >%10 düşüyor); hepsi yüksek kademede, h=5 ve h=22'de.

**Önceden ilan edilmiş kriter:** "son %10'luk epoch diliminde eğitim kaybındaki düşüş
%2'nin altına inene kadar eğit, üst sınır 200 epoch, alt sınır ilan edilmiş kademe
epoch sayısı." Yalnızca yüksek kademeye uygulandı. Kriter tamamen **eğitim kaybından**
türer, test performansına bakmaz; bu yüzden test setine bakarak yapılmış bir değişiklik
değildir.

**Fark notu:** bu modda kosinüs lr programının `T_max` değeri 200'dür, birincil koşuda
ilan edilmiş epoch sayısıydı. Karşılaştırma "aynı program, daha çok epoch" değil,
"yakınsayana kadar eğitilmiş" durumudur.

**Eğitim gerçekten uzadı ve kayıp gerçekten düştü.** h=5'te fold'lar 60 yerine ortalama
124 epoch koştu (aralık 60–198); geç fold'larda son eğitim kaybı 0.107'den 0.025'e,
yani ~4 kat düştü. h=22'de kriter hemen sağlandı, ortalama 62 epoch.

**Ama test performansı değişmedi.**

| ufuk | birincil | yakınsama kriterli | değişim | sağlamlık daha iyi olan fold |
| --- | --- | --- | --- | --- |
| 5 | 0.014227 | 0.014374 | +1.03% | 6/15 |
| 22 | 0.010512 | 0.010423 | −0.85% | 5/15 |
| 66 | 0.011589 | 0.011589 | 0.00% | 0/14 |
| 126 | 0.011431 | 0.011431 | 0.00% | 0/14 |

h=66 ve h=126 birebir aynı çıktı; o kademeler uyarlamalı değildi ve sonuçların altı
basamağa kadar aynı olması **determinizmin çalıştığının doğrulamasıdır**.

**Yorum: BiLSTM'in başarısızlığı yetersiz eğitim değil, aşırı uyumdur.** Eğitim kaybını
4 kat düşürmek test RMSE'sini iyileştirmedi, h=5'te hafifçe kötüleştirdi. HAR-X'e olan
açık pratikte aynı kaldı (+39.0 / +35.9 / +53.0 / +41.8).

**Bir tanı aracı sınırlılığı.** Durdurma kriteri (son %10, <%2) 24 yüksek kademe
fold'unun tamamında tetiklendi, ancak bağımsız raporlama sınıflandırıcısı (son %20)
sağlamlık koşusunda 12 fold'u hâlâ "yetersiz eğitilmiş" gösteriyor. Kuyruk düşüş
değerleri arasında negatifler de var (−12.9, −1.5), yani eğitim kaybı epoch'lar arası
gürültülü ve iki pencere de trendden çok gürültü ölçüyor. Kriter zaman zaman şansa bağlı
düz bir çift üzerinde tetiklenmiş olabilir. Düzleştirilmiş kayıp eğrisi daha iyi bir tanı
aracı olurdu; bu bir sınırlılık olarak kaydedilir.

**Sonuç birincil spesifikasyonu değiştirmez.** Çıktılar `bilstm_*_conv.*` dosyalarında.

---

# Aşama 7: Hibrit Modeller

`scripts/07_hybrid.py`. Gunnarsson vd. (2024) taramasının işaret ettiği hibrit
ekonometrik-ML yapısının doğrudan testi: ML'in HAR-X'e tamamlayıcı katkısı var mı?

Ağırlıklar 0.5/0.5 sabit, önceden ilan edilmiş, optimize edilmedi. H1 ve H2 yeni eğitim
gerektirmez (kaydedilmiş tahminlerin ortalaması); test satırlarının dört ufukta da
birebir aynı ve `y_true` değerlerinin tam eşleştiği assert ile doğrulandı. H3'ün taban
HAR-X'i, standalone HAR-X ile `atol=1e-10` toleransında aynı olduğu assert edilerek
yeniden fit edildi.

## Ana soru: hibritler HAR-X standalone'u geçti mi?

RMSE fold ortalaması, HAR-X'e göre yüzde fark. Negatif = hibrit daha iyi.

| hibrit | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| H1 XGBoost + BiLSTM | +12.28 | +22.43 | +30.59 | +20.94 |
| H2 HAR-X + XGBoost | −0.01 | +4.35 | +4.33 | +0.67 |
| H3 HAR-X artık modellemesi | +8.21 | +14.98 | +23.24 | +5.96 |

**Hiçbir hibrit ana metrikte HAR-X'i geçemiyor.** H1 beklendiği gibi çok kötü, iki zayıf
bileşenin ortalaması. H3 her ufukta kaybediyor. H2 h=5'te berabere, diğer üç ufukta
geride.

## RMSE (fold ortalaması)

| model | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR-X-log | 0.010335 | 0.007561 | 0.007504 | 0.008012 |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |
| H2 HAR-X+XGBoost | 0.010340 | 0.008002 | 0.007901 | 0.008113 |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| XGBoost | 0.010913 | 0.009080 | 0.008924 | 0.008627 |
| H3 HAR-X artık | 0.011190 | 0.008818 | 0.009333 | 0.008539 |
| H1 XGBoost+BiLSTM | 0.011611 | 0.009390 | 0.009890 | 0.009747 |
| BiLSTM | 0.014227 | 0.010512 | 0.011589 | 0.011431 |

## Tanı 1: artık aşaması bilgi çıkarıyor mu? (H3)

`R²_artık = 1 − SSE(e − ê)/SSE(e)`, tabanı "artık tahmin etmemek".

| ufuk | örneklem-içi | örneklem-dışı | dışı medyan | dışı pozitif fold |
| --- | --- | --- | --- | --- |
| 5 | +0.863 | −0.240 | −0.166 | 3/15 |
| 22 | +0.873 | −0.573 | −0.365 | 3/15 |
| 66 | +0.698 | −1.460 | −0.334 | 4/14 |
| 126 | +0.579 | −0.270 | +0.038 | 8/14 |

**Bu tablo hipotezin en doğrudan cevabıdır.** XGBoost, HAR-X'in örneklem-içi
artıklarının %58–87'sini açıklıyor, ama örneklem-dışında R² dört ufukta da **negatif**:
artık modeli hiç kullanmamak (yani HAR-X'i olduğu gibi bırakmak) daha iyi sonuç veriyor.
Fold'ların yalnızca 3–8'inde pozitif katkı var. **ML, HAR-X'in artıklarından
genelleşebilir hiçbir bilgi çıkaramıyor.**

**Öngörülen sapma gerçekleşmedi.** Planda artık modelinin örneklem-içi artıklarla
eğitilmesinin sapma yaratabileceğini not etmiştik. Artık standart sapmasının train/test
oranı 0.96–1.21 çıktı, yani 1'e yakın. Taban model 6 regresörlü OLS olduğu için bu
sapma önemsizmiş. Başarısızlığın nedeni bu değil; artık sinyali basitçe öğrenilebilir
değil.

## Tanı 2: bileşen hatalarının korelasyonu

| hibrit | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| H1 XGBoost–BiLSTM | 0.690 | 0.830 | 0.763 | 0.776 |
| H2 HAR-X–XGBoost | 0.875 | 0.800 | 0.814 | 0.871 |

Korelasyonlar 1'den uzak, yani çeşitlendirme potansiyeli teorik olarak var. Ama H2 yine
de kazanamıyor, çünkü XGBoost bileşeni HAR-X'ten sistematik olarak kötü ve ortalama onu
taşıyor.

## Bir toplulaştırma farkı — gizlenmiyor

H2 için havuzlanmış RMSE ile fold ortalaması **ters yönde** sonuç veriyor.

| ufuk | HAR-X havuz | H2 havuz | H2 havuz farkı | H2 fold ort. farkı | H2'nin kazandığı fold |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.011612 | 0.011381 | −1.99% | −0.01% | 6/15 |
| 22 | 0.009265 | 0.009247 | −0.20% | +4.35% | 3/15 |
| 66 | 0.009894 | 0.009879 | −0.15% | +4.33% | 5/14 |
| 126 | 0.009810 | 0.009534 | −2.81% | +0.67% | 7/14 |

Havuzlanmış ölçüte göre H2 dört ufukta da HAR-X'ten hafifçe iyi. Fold ortalamasına göre
üç ufukta kötü. Neden: H2 birkaç yüksek varyanslı yılda belirgin kazanıyor, havuz bu
yılları ağırlıklandırıyor; fold ortalaması ise her yıla eşit ağırlık veriyor ve H2
fold'ların çoğunluğunda kaybediyor (dört ufukta da yarıdan az).

**Ana metriğimiz fold ortalamasıdır** (CLAUDE.md "Metrik raporlama kuralı"), dolayısıyla
sonuç "H2 HAR-X'i geçemedi"dir. Havuzlanmış değer de raporlanır.

## Sonuç

Üç hibrit yapının hiçbiri HAR-X standalone'u ana metrikte geçemedi. Artık modellemesi
tanısı, ML'in HAR-X'in bıraktığı artıklardan genelleşebilir bilgi çıkaramadığını doğrudan
gösteriyor. Gunnarsson vd.'nin işaret ettiği hibrit yön, bu veri ve bu ufuklarda karşılık
bulmadı.

---

# Keşifsel / İkincil Analiz: volatilite rejimi ve toplulaştırma farkı

`scripts/07b_exploratory_vol_regime.py`.

**Bu bir keşifsel analizdir. Birincil bulguyu değiştirmez, model seçimi için
kullanılmaz.** Amacı tek bir soruyu aydınlatmak: H2 (HAR-X + XGBoost ortalaması) için
havuzlanmış RMSE ile fold ortalaması neden ters yönde sonuç veriyor?

**Bölme kriteri mekaniktir ve sonuçlara bakılarak seçilmemiştir.** Her ufuk için her
test yılının ortalama gerçekleşen volatilitesi hesaplanır; yıllar bu değerin medyanına
göre yüksek ve düşük rejim olarak ikiye ayrılır. Eşik veriden türer, performanstan
değil. Medyan ufuk başına ayrı hesaplanır çünkü ana metriğe dahil fold kümesi ufka göre
değişir.

## H2'nin HAR-X'e göre farkı, rejim bazında (%, negatif = H2 iyi)

| ölçüt | rejim | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- | --- |
| fold ortalaması | düşük | −0.31 | +5.23 | +8.89 | +3.95 |
| fold ortalaması | yüksek | +0.18 | +3.89 | +2.17 | −1.01 |
| havuzlanmış | düşük | +0.49 | +4.35 | +6.76 | +5.59 |
| havuzlanmış | yüksek | −2.75 | −1.16 | −1.24 | −4.56 |

Yüksek rejimin havuzlanmış ölçütteki ağırlığı (kareli hata payı): h=5 %76.8,
h=22 %82.9, h=66 %86.9, h=126 %83.5.

## Bulgu: rejim farkı açıklamanın yalnızca bir kısmı

Havuzlanmış ölçüt yüksek volatiliteli yılları %77–87 ağırlıkla tartıyor ve H2 o grupta
havuzlanmış ölçütte kazanıyor. Bu kadarı hipotezi destekliyor.

**Ama H2 yüksek volatiliteli yıllarda sistematik olarak kazanmıyor.** Aynı grup içinde
fold ortalaması dört ufkun üçünde hâlâ H2'yi geride gösteriyor, ve H2 yüksek rejim
fold'larının yalnızca 2/7, 1/7, 3/7 ve 4/7'sinde HAR-X'i geçiyor — yani çoğunlukta
kaybediyor.

## Asıl mekanizma: uç gözlem yoğunlaşması

| ufuk | en kötü %1 gözlemin havuzdaki payı | H2'nin o gözlemlerdeki farkı |
| --- | --- | --- |
| 5 | %38.8 | −10.8% |
| 22 | %44.8 | −12.3% |
| 66 | %40.6 | −9.3% |
| 126 | %24.3 | −13.6% |

Gözlemlerin en kötü %1'i havuzlanmış hatanın dörtte biri ile yarısı arasını oluşturuyor
ve H2 tam o gözlemlerde HAR-X'ten %9–14 daha iyi. Başka her yerde biraz daha kötü.

Fold düzeyinde de aynı örüntü. h=22'de H2, 2020 fold'unda %9.6 kazanıyor (o yılın RMSE'si
0.0252, tüm fold'ların en yükseği) ve diğer altı yüksek rejim yılının hepsinde kaybediyor.
h=126'da 2020 (−9.4%), 2019 (−4.1%) ve 2025 (−6.0%) kazanıyor, 2022'de %38.5 kaybediyor.

**Yorum:** XGBoost'un ortalamaya kattığı değer, nadir ve şiddetli volatilite
sıçramalarında hatayı sınırlamak. Bunlar havuzlanmış ölçütü domine ediyor ama fold
ortalamasında tek bir yıl olarak sayılıyor. Rejim değil, uç olay duyarlılığı.

Bu, ana metrik tercihimizi değiştirmez (fold ortalaması, CLAUDE.md "Metrik raporlama
kuralı") ve H2 birincil sonuçta HAR-X'i geçmemiş sayılır. Ancak makalede tartışılmaya
değer bir nüanstır: hangi ölçütün doğru olduğu, uygulamada nadir şok dönemlerindeki
hatanın mı yoksa tipik yıldaki hatanın mı önemli olduğuna bağlıdır.

Çıktılar: `explore_vol_regime_years.csv`, `explore_vol_regime_groups.csv`,
`explore_vol_regime_folds.csv`, `explore_vol_regime_summary.json`.

## Uç olay sağlamlık kontrolü: "uç olaylar" mı, "COVID" mi?

İki farklı güçte iddia söz konusu. Sağlamlık kontrolü hangisinin doğru olduğunu belirledi.

### En kötü %1 gözlemin takvim dağılımı

| yıl | h=5 | h=22 | h=66 | h=126 | toplam |
| --- | --- | --- | --- | --- | --- |
| 2020 | 25 | 32 | 35 | 16 | 108 |
| 2019 | 5 | 0 | 0 | 19 | 24 |
| 2022 | 3 | 4 | 0 | 0 | 7 |
| 2026 | 2 | 1 | 0 | 0 | 3 |
| 2015 | 2 | 0 | 0 | 0 | 2 |

2020'nin uç gözlemler içindeki payı: h=5 %67.6, h=22 %86.5, **h=66 %100**, h=126 %45.7.

### 2020 fold'u tamamen çıkarılınca (eşik yeniden hesaplandı)

H2'nin uç gözlemlerdeki farkı (%, negatif = H2 iyi):

| örnek | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| tüm yıllar | −10.78 | −12.33 | −9.35 | −13.58 |
| 2020 hariç | **+3.01** | **+5.12** | −2.79 | −3.58 |

Havuzlanmış ölçütte H2'nin HAR-X'e göre farkı 2020 hariç: +3.16, +8.81, +6.80, +0.68.
**Dört ufukta da işaret değiştirdi.** H2'nin havuzlanmış üstünlüğü tamamen 2020'den
geliyormuş.

Uç gözlemler dışında kalan %99'da H2 her iki örneklemde de tutarlı biçimde daha kötü
(+0.4 ile +9.5 arası).

### Kritik nüans: uzun ufuklarda 2020'yi çıkarmak yetmiyor

2020 hariç örneklemde uç gözlemlerin yıl dağılımı:

| ufuk | dağılım |
| --- | --- |
| 5 | 2014:3, 2015:5, 2016:2, 2019:5, 2021:6, 2022:4, 2026:10 |
| 22 | 2015:3, 2022:16, 2026:15 |
| 66 | 2019:21, 2022:7, 2025:5 |
| 126 | **2019:33 (hepsi)** |

h=126'daki 2019 uç gözlemlerinin tamamı 27.11.2019 – 31.12.2019 aralığından geliyor.
Bu satırların hedefi 126 işlem günü ileriye bakıyor, yani **yaklaşık Mayıs 2020'ye kadar
uzanıyor ve COVID çöküşünü ölçüyor.** Fold etiketi 2019 olsa da gözlem COVID gözlemidir.

Dolayısıyla uzun ufuklarda 2020 fold'unu çıkarmak COVID'i örneklemden çıkarmıyor.
h=66'da 33 uç gözlemin 21'i, h=126'da 33'ünün tamamı hâlâ COVID penceresi. Bu, h=66 ve
h=126'da 2020 hariç ölçülen küçük H2 avantajının (−2.79%, −3.58%) neden sürdüğünü
açıklıyor.

### Sonuç: hangi iddia yazılmalı

**Doğru iddia "COVID döneminde ML tamamlayıcı değer kattı"dır, "uç olaylarda" değil.**

- h=5 ve h=22'de COVID gerçekten çıkarılabiliyor ve H2'nin uç gözlem avantajı işaret
  değiştirip dezavantaja dönüşüyor. 2015–2016 petrol çöküşü, 2022 savaş dönemi ve 2026
  gibi diğer uç dönemler örneklemde mevcut, ama H2 oralarda kazanmıyor.
- h=66 ve h=126'da test sonuçsuzdur: 2020 fold'unu çıkarmak, hedef penceresi 2020'ye
  taşan 2019 satırlarını çıkarmadığı için COVID'i örneklemden gerçekten çıkarmaz.

Genel bir "uç olay" iddiası için gereken kanıt yok. Tek bir olaya dayanan bir bulgu tek
bir olay olarak raporlanmalıdır.

Çıktılar: `explore_tail_robustness.csv`, `explore_tail_year_distribution.csv`.

---

# Aşama 8: Diebold-Mariano Testleri

`scripts/08_dm_test.py`. Karesel hata kaybı, fold'lar arası havuzlanmış, Newey-West
(Bartlett) HAC düzeltmesi `L = h−1`, Harvey-Leybourne-Newbold küçük örneklem düzeltmesi,
Holm-Bonferroni çoklu test düzeltmesi (20 test). Ek olarak dağılımdan bağımsız fold
düzeyi işaret testi (binom).

## En önemli sonuç: ana bulgu istatistiksel olarak anlamlı DEĞİL

HAR-X vs XGBoost:

| ufuk | RMSE farkı | DM (HLN) | p | Holm p | işaret | işaret p | işaret Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | −2.52% | −0.604 | 0.546 | 1.000 | 10/15 | 0.302 | 1.000 |
| 22 | −9.46% | −1.384 | 0.167 | 1.000 | 13/15 | 0.007 | 0.089 |
| 66 | −9.15% | −1.468 | 0.142 | 1.000 | 11/14 | 0.057 | 0.631 |
| 126 | −0.98% | −0.142 | 0.887 | 1.000 | 10/14 | 0.180 | 1.000 |

Nokta tahminleri dört ufukta da tutarlı biçimde HAR-X lehine, ama **hiçbir ufukta
istatistiksel anlamlılık yok.** İşaret testi h=22'de ham p ile %5'i geçiyor, Holm
düzeltmesinden sonra o da düşüyor.

**Makalede bu bulgu "HAR-X, XGBoost'u anlamlı biçimde geçiyor" diye YAZILAMAZ.** Doğru
ifade: nokta tahminleri tutarlı olarak HAR-X lehine, ancak örtüşen pencerelerin yarattığı
otokorelasyon dikkate alındığında fark istatistiksel olarak ayırt edilemiyor.

## Holm düzeltmesinden sonra ayakta kalanlar

DM testi (3/20):

| ufuk | çift | fark | Holm p |
| --- | --- | --- | --- |
| 5 | HAR-X vs past-vol | −19.79% | <0.0001 |
| 5 | XGBoost vs past-vol | −17.71% | <0.0001 |
| 5 | HAR-X vs BiLSTM | −23.17% | <0.0001 |

İşaret testi (8/20): yukarıdakilere ek olarak h=5 HAR-X vs GARCH, h=22 HAR-X vs past-vol,
h=22 HAR-X vs GARCH, h=22 HAR-X vs BiLSTM, h=66 HAR-X vs GARCH.

Yani sağlam biçimde kanıtlanabilen tek şey: HAR-X ve XGBoost, naif ve zayıf
alternatifleri (past-volatility, BiLSTM, GARCH) kısa ufuklarda geçiyor. İki iyi model
arasındaki fark ayırt edilemiyor.

## HAC düzeltmesi neden zorunluydu

Varyans şişme çarpanı (HAC varyansı / bağımsızlık varsayımlı varyans):

| ufuk | min | ortalama | maks |
| --- | --- | --- | --- |
| 5 | 1.7 | 2.5 | 2.9 |
| 22 | 3.4 | 6.0 | 12.1 |
| 66 | 4.8 | 19.8 | 38.8 |
| 126 | 12.7 | 42.8 | 74.5 |

Düzeltmesiz bir DM testi standart hatayı h=126'da **8.6 kata kadar** küçük tahmin
ederdi. Gereklilik sayısal olarak doğrulanmıştır.

## Gecikme kuralının ampirik doğrulaması

Planda `L = h−1`'in bir ALT SINIR olduğunu, tahminler optimal olmadığı için
otokorelasyonun ötesine uzanabileceğini not etmiştik. ACF tanısı bu endişeyi
doğrulamadı — kayıp farkının otokorelasyonu `h−1`'de sıfıra iniyor:

| ufuk | ACF(1) | ACF(h−1) | ACF(h) | ort. ACF [h, 2h] |
| --- | --- | --- | --- | --- |
| 5 | 0.615 | −0.021 | −0.090 | −0.016 |
| 22 | 0.677 | −0.068 | −0.035 | −0.010 |
| 66 | 0.713 | −0.014 | −0.009 | +0.017 |
| 126 | 0.730 | +0.006 | +0.004 | −0.012 |

İlan edilmiş kural yeterli çıktı; daha uzun gecikme gerekmiyor.

## İki testin uyumu

Yön uyumu 17/20. Üç ayrışma da past-volatility karşılaştırmalarında ve uzun ufuklarda:

| ufuk | çift | RMSE farkı | DM | işaret |
| --- | --- | --- | --- | --- |
| 66 | XGBoost vs past-vol | −8.56% | −0.798 | 7/14 |
| 126 | HAR-X vs past-vol | −11.23% | −0.824 | 7/14 |
| 126 | XGBoost vs past-vol | −10.35% | −0.837 | 5/14 |

Havuzlanmış RMSE belirgin fark gösteriyor ama fold'ların yarısında veya azında
kazanılıyor. Bu, Aşama 7'nin keşifsel analizinde bulduğumuz uç gözlem yoğunlaşmasının
aynısı: havuzlanmış ölçüt birkaç şiddetli epizodun etkisinde.

## Değerlendirme

İşaret testi bu çalışmada DM'den daha çok anlamlı sonuç üretti (8'e karşı 3) ve HAC bant
genişliği tartışmasının hiçbirine bağlı değil. İki test aynı yönü gösterdiği 17 testte
bulgu bant genişliği tercihine karşı bağışıktır. Ana bulguda (HAR-X vs XGBoost) ise
ikisi de anlamlılık üretmiyor, yani sonucun zayıflığı yöntem tercihinden kaynaklanmıyor.

## Genişletilmiş test ailesi (32 test)

Üç çift eklendi: HAR vs HAR-X (ana ayrıştırma iddiası), HAR vs past-volatility,
HAR-X vs HAR-X-log. Holm düzeltmesi 8 çift × 4 ufuk = 32 test üzerinden **yeniden
hesaplandı**, dolayısıyla önceki 20 testli aileye göre eşik sıkılaştı.

### Ana ayrıştırma iddiası: HAR vs HAR-X

| ufuk | havuz farkı | DM (HLN) | p | Holm p | HAR-X kazanan fold | işaret p | işaret Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | +1.49% | +0.357 | 0.722 | 1.000 | 12/15 | 0.035 | 0.738 |
| 22 | +4.81% | +0.655 | 0.512 | 1.000 | 13/15 | 0.007 | 0.170 |
| 66 | −1.28% | −0.202 | 0.840 | 1.000 | 10/14 | 0.180 | 1.000 |
| 126 | −2.52% | −0.531 | 0.596 | 1.000 | 9/14 | 0.424 | 1.000 |

Pozitif fark = HAR daha kötü, yani HAR-X daha iyi.

**Dışsal değişkenlerin katkısı da istatistiksel olarak anlamlı değil.** İşaret testi kısa
ufuklarda ham p ile eşiği geçiyor (h=22'de 13/15 fold, p=0.007) ama 32 testlik Holm
düzeltmesinden sonra hiçbiri ayakta kalmıyor.

**Uzun ufuklarda ölçütler yön değiştiriyor:**

| ufuk | havuzlanmış | fold ortalaması |
| --- | --- | --- |
| 5 | +1.49% | +4.77% |
| 22 | +4.81% | +12.13% |
| 66 | −1.28% | +6.95% |
| 126 | −2.52% | +1.79% |

h=66 ve h=126'da havuzlanmış RMSE HAR'ı, fold ortalaması ve işaret testi HAR-X'i
gösteriyor. Yine uç gözlem yoğunlaşması: HAR birkaç şiddetli epizotta HAR-X'ten iyi,
tipik yılda kötü.

### Diğer iki çift

**HAR vs past-volatility:** h=5'te hem DM (−18.59%, Holm p<0.0001) hem işaret testi
(15/15, Holm p=0.002) anlamlı. Uzun ufuklarda anlamlılık yok.

**HAR-X vs HAR-X-log:** dört ufukta da hiçbir testte anlamlılık yok (DM p = 0.21, 0.14,
0.32, 0.38; işaret p = 0.61, 0.61, 0.79, 0.79). Seviyeler ile log-log spesifikasyonu
istatistiksel olarak ayırt edilemiyor. **Bu yararlı bir sıfır sonucudur:** birincil
spesifikasyon olarak seviyeleri seçmiş olmamız sonuçları değiştirmiyor.

### Holm sonrası ayakta kalanlar (32 testlik aile)

DM testi, 4/32 — hepsi h=5 ve hepsi zayıf alternatiflere karşı:

| çift | fark | Holm p |
| --- | --- | --- |
| HAR-X vs BiLSTM | −23.17% | <0.0001 |
| HAR-X vs past-vol | −19.79% | <0.0001 |
| HAR vs past-vol | −18.59% | <0.0001 |
| XGBoost vs past-vol | −17.71% | <0.0001 |

İşaret testi, 9/32: yukarıdaki dördün işaret karşılıkları, artı h=5 ve h=22'de HAR-X vs
GARCH, h=22'de HAR-X vs past-vol ve HAR-X vs BiLSTM, h=66'da HAR-X vs GARCH.

### Yön uyumu

24/32. Sekiz ayrışmanın yedisi h=66 ve h=126'da, altısı past-volatility veya HAR
karşılaştırmalarında. Hepsi aynı mekanizmadan: havuzlanmış ölçüt birkaç şiddetli
epizodun etkisinde, fold ortalaması ve işaret testi tipik yılı yansıtıyor.

### Genel değerlendirme

Çoklu test düzeltmesinden sonra istatistiksel olarak savunulabilen tek sınıf iddia,
iyi modellerin (HAR, HAR-X, XGBoost) naif ve zayıf alternatifleri (past-volatility,
BiLSTM, GARCH) **kısa ufuklarda** geçtiğidir. Çalışmanın iki ana iddiası —
"HAR-X, XGBoost'u geçiyor" ve "kazanç dışsal değişkenlerden geliyor" — nokta
tahminlerinde tutarlı yön göstermekle birlikte istatistiksel olarak ayırt edilemiyor.
Makale bu iddiaları anlamlılık dili yerine etki büyüklüğü ve yön tutarlılığı diliyle
kurmalıdır.

## Test ailelerinin ayrılması ve FWER / FDR

Testler iki aileye bölündü ve düzeltmeler **her aile içinde ayrı** hesaplandı. Her aile
için hem Holm (FWER, doğrulayıcı iddialar) hem Benjamini-Hochberg (FDR, keşifsel
bulgular) yan yana verilir; okuyucu kendi eşiğini seçebilir.

**Şeffaflık kaydı:** birincil aile, çalışmanın Aşama 5 ve 6'da ilan edilmiş iki ana
iddiasına karşılık gelir ve p-değerlerine bakılarak seçilmemiştir. Ancak aile tanımı
testlerden **sonra** resmileştirilmiştir; bu bir ön-kayıt değildir ve makalede böyle
belirtilmelidir.

### Birincil (doğrulayıcı) aile: 2 karşılaştırma × 4 ufuk = 8 test

| ufuk | karşılaştırma | fark | DM p | DM Holm | DM BH | işaret | işaret p | işaret Holm | işaret BH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | +1.49% | 0.722 | 1.000 | 0.887 | 3/15 | 0.035 | 0.211 | 0.094 |
| 22 | HAR vs HAR-X | +4.81% | 0.512 | 1.000 | 0.887 | 2/15 | 0.007 | 0.059 | **0.030** |
| 66 | HAR vs HAR-X | −1.28% | 0.840 | 1.000 | 0.887 | 4/14 | 0.180 | 0.718 | 0.239 |
| 126 | HAR vs HAR-X | −2.52% | 0.596 | 1.000 | 0.887 | 5/14 | 0.424 | 0.718 | 0.424 |
| 5 | HAR-X vs XGBoost | −2.52% | 0.546 | 1.000 | 0.887 | 10/15 | 0.302 | 0.718 | 0.345 |
| 22 | HAR-X vs XGBoost | −9.46% | 0.167 | 1.000 | 0.666 | 13/15 | 0.007 | 0.059 | **0.030** |
| 66 | HAR-X vs XGBoost | −9.15% | 0.142 | 1.000 | 0.666 | 11/14 | 0.057 | 0.287 | 0.115 |
| 126 | HAR-X vs XGBoost | −0.98% | 0.887 | 1.000 | 0.887 | 10/14 | 0.180 | 0.718 | 0.239 |

Ayakta kalan test sayısı (%5 eşiği):

| test | Holm (FWER) | BH (FDR) |
| --- | --- | --- |
| DM | 0/8 | 0/8 |
| işaret | 0/8 | **2/8** |

**En güçlü savunulabilir ifade:** FDR kontrolü altında, işaret testine göre h=22'de her
iki ana iddia da destekleniyor. HAR-X, XGBoost'u 15 fold'un 13'ünde geçiyor
(BH p = 0.030) ve HAR-X, HAR'ı 15 fold'un 13'ünde geçiyor (BH p = 0.030). Diğer üç
ufukta destek yok.

DM testi birincil ailede hiçbir koşulda anlamlılık üretmiyor; ham p değerleri
0.142–0.887 aralığında, yani aile küçültmek işe yaramıyor — testin bu veride gücü yok.

### İkincil / keşifsel aile: 6 karşılaştırma × 4 ufuk = 24 test

| test | Holm (FWER) | BH (FDR) |
| --- | --- | --- |
| DM | 4/24 | 6/24 |
| işaret | 9/24 | 9/24 |

DM'de BH'nin Holm'a eklediği iki test: h=22 HAR-X vs BiLSTM (BH p = 0.023) ve h=22
HAR-X vs past-volatility (BH p = 0.035). İkisi de zayıf alternatiflere karşı.

İşaret testinde iki düzeltme aynı 9 testi ayakta tutuyor; hepsi h=5, h=22 ve h=66'da ve
hepsi zayıf alternatiflere (past-volatility, BiLSTM, GARCH) karşı.

**HAR-X vs HAR-X-log** dört ufukta da hiçbir düzeltmede anlamlı değil (BH p = 0.27–0.43,
işaret BH p = 0.77–0.90). Seviyeler ile log-log arasındaki tercih sonuçları
etkilemiyor — yararlı bir sıfır sonucu.

### Sonuç

Doğrulayıcı çerçevede (birincil aile, FWER) çalışmanın iki ana iddiası da istatistiksel
olarak desteklenmiyor. FDR çerçevesinde yalnızca h=22'de ve yalnızca işaret testiyle
destekleniyor. İkincil ailedeki güçlü sonuçların tamamı, iyi modellerin naif ve zayıf
alternatifleri kısa ufuklarda geçtiği yönünde — bu beklenen ve düşük bilgi değerli bir
bulgudur.

Makale, ana iddialarını anlamlılık dili yerine **etki büyüklüğü ve yön tutarlılığı**
diliyle kurmalı, h=22'deki FDR desteğini ise nitelikli biçimde raporlamalıdır.

## Güç analizi: bu farklar bu veriyle tespit edilebilir miydi?

`scripts/09_power_analysis.py` → `outputs/power_analysis.csv`. Birincil ailedeki 8
karşılaştırma, %80 güç ve %5 iki yönlü eşik.

**Yöntem.** DM testi için örneklem birimi etkin bağımsız bloktur: `B = n/h` (h=126'da
3500/126 ≈ 28). Standartlaştırılmış etki `δ = |DM| / √B`; gerekli merkezi-olmayanlık
`z₀.₉₇₅ + z₀.₈₀ = 2.802`, dolayısıyla `B_gerekli = (2.802/δ)²`. İşaret testi için
örneklem birimi doğrudan fold (yıl); tam binom gücü %80'e ulaşana kadar fold sayısı
artırılır. Yıl başına işlem günü test döneminden ölçüldü: 244.1.

**Varsayımlar.** Gözlenen etki gerçek etki kabul edilir; hata yapısı örneklem büyüdükçe
değişmez; `B = n/h` kaba bir yaklaşımdır. "Yıl" test dönemi yılıdır.

**Uyarı.** Gözlenen etkiden hesaplanan "gerçekleşen güç", p-değerinin monoton bir
dönüşümüdür ve p-değerinin ötesinde bilgi taşımaz. Asıl cevap gerekli örneklem
sütunlarındadır.

### DM testi

| ufuk | karşılaştırma | etkin blok | gerçekleşen güç | gerekli yıl | kat artış |
| --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 732 | 0.065 | 927 | 62× |
| 22 | HAR vs HAR-X | 166 | 0.101 | 273 | 18× |
| 66 | HAR vs HAR-X | 53 | 0.055 | 2 764 | 193× |
| 126 | HAR vs HAR-X | 28 | 0.083 | 400 | 28× |
| 5 | HAR-X vs XGBoost | 732 | 0.093 | 323 | 22× |
| 22 | HAR-X vs XGBoost | 166 | 0.283 | 61 | 4× |
| 66 | HAR-X vs XGBoost | 53 | 0.312 | 52 | 4× |
| 126 | HAR-X vs XGBoost | 28 | 0.052 | 5 582 | 389× |

**DM testinin gerçekleşen gücü hiçbir testte 0.32'yi geçmiyor.** Yani bu tasarımda DM
testinin bu farkları yakalama şansı hiç olmadı. En iyimser durumda (h=66, HAR-X vs
XGBoost) 52 yıllık test dönemi gerekirdi; elimizde 14 yıl var.

### İşaret testi

| ufuk | karşılaştırma | fold | oran | gerçekleşen güç | gerekli yıl | kat artış |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 15 | 12/15 | 0.648 | 20 | 1.3× |
| 22 | HAR vs HAR-X | 15 | 13/15 | **0.871** | 15 | 1.0× |
| 66 | HAR vs HAR-X | 14 | 10/14 | 0.190 | 42 | 3.0× |
| 126 | HAR vs HAR-X | 14 | 9/14 | 0.076 | 94 | 6.7× |
| 5 | HAR-X vs XGBoost | 15 | 10/15 | 0.210 | 72 | 4.8× |
| 22 | HAR-X vs XGBoost | 15 | 13/15 | **0.871** | 15 | 1.0× |
| 66 | HAR-X vs XGBoost | 14 | 11/14 | 0.396 | 25 | 1.8× |
| 126 | HAR-X vs XGBoost | 14 | 10/14 | 0.190 | 42 | 3.0× |

**h=22'de her iki karşılaştırma da zaten yeterli güçte (0.871 > 0.80).** Bu önemli:
h=22'deki anlamlı sonuçlar düşük güçlü bir tesadüf değil, çalışmanın o ufukta yeterince
güçlü olduğu ve etkiyi tespit ettiği anlamına geliyor.

Diğer ufuklarda güç yetersiz: h=5'te HAR-X vs XGBoost için 0.21 (72 yıl gerekir),
h=126'da HAR vs HAR-X için 0.076 (94 yıl gerekir).

### Sonuç

İşaret testi bu tasarımda DM'den **bir ile iki mertebe daha verimli**. DM için gerekli
uzatma mevcut örneklemin 4–389 katı, işaret testi için 1–6.7 katı. Bu, Aşama 8'de
işaret testinin neden daha çok anlamlı sonuç ürettiğini sayısallaştırıyor: fold düzeyi
tutarlılık, örtüşen günlük gözlemlerden çok daha fazla bilgi taşıyor.

**Makale için çıkarım.** "Anlamlı fark bulunamadı" ifadesi bu çalışmada "fark yok"
anlamına gelmez; tasarım çoğu karşılaştırmada farkı tespit edecek güce sahip değildi.
h=22 istisnadır ve orada iki ana iddia da yeterli güçle desteklenmiştir. Sınırlılıklar
bölümü bu tabloyu içermelidir.

## Önsel (a priori) güç eğrileri

Bu bölüm **gözlenen sonuçları hiç kullanmaz**; hipotetik etki büyüklükleri üzerinden
"bu tasarım neyi tespit edebilirdi" sorusuna cevap verir. Dairesel değildir.
Çıktılar: `apriori_power_sign.csv`, `apriori_power_dm.csv`.

### İşaret testi: fold sayısı × gerçek kazanma olasılığı

%5 iki yönlü anlamlılık için gereken en az kazanma sayısı **n=15'te 12, n=14'te 12**.
Yani 15 yılda en az 12 (%80), 14 yılda en az 12 (%86) yıl kazanmak gerekiyor.

| gerçek p | n=14 | n=15 |
| --- | --- | --- |
| 0.60 | 0.040 | 0.092 |
| 0.65 | 0.084 | 0.173 |
| 0.70 | 0.161 | 0.297 |
| 0.75 | 0.281 | 0.461 |
| 0.80 | 0.448 | 0.648 |
| 0.85 | 0.648 | **0.823** |
| 0.90 | **0.842** | 0.944 |

**Bu tasarım, ancak yılların %85'ini (n=15) veya %90'ını (n=14) kazanan bir modeli
güvenilir biçimde tespit edebilir.** Gerçekten üstün ama yılların %70'ini kazanan bir
model, 15 fold ile yalnızca %30 olasılıkla tespit edilir.

### DM testi: etkin blok sayısı × hipotetik RMSE farkı

Dönüşüm: `δ_blok = k·|r²−1|`, `ncp = √B·δ_blok`. Katsayı `k`, verinin **gürültü**
yapısından (iki modelin hata korelasyonu ve kayıp dağılımı) kalibre edilir, gözlenen
**etki** büyüklüğünden değil. Ufuk başına birincil ailedeki iki çiftin ortalaması:
h=5 → 0.444, h=22 → 0.557, h=66 → 1.124, h=126 → 1.700.

| ufuk | etkin blok | %5 fark | %10 fark | %20 fark |
| --- | --- | --- | --- | --- |
| 5 | 732 | 0.216 | 0.626 | **0.991** |
| 22 | 166 | 0.108 | 0.276 | 0.733 |
| 66 | 53 | 0.126 | 0.343 | **0.838** |
| 126 | 28 | 0.142 | 0.401 | **0.899** |

Güç, blok sayısıyla monoton değil çünkü `k` ufka göre değişiyor: h=5 en çok bloğa
(732) ama en küçük `k`'ya (0.444) sahip; h=126 tersine.

`k` belirsizliğine duyarlılık dar: h=66 ve h=22'de ±0.02–0.06, yalnızca h=126'da
belirgin (%10 farkta 0.284–0.528).

### Sonuç: tasarımın tespit sınırı

| test | güvenilir tespit sınırı |
| --- | --- |
| DM | yaklaşık **%20** RMSE farkı (h=22 hariç, orada %20 bile 0.73) |
| işaret | yılların **%85–90**'ını kazanmak |

Gözlenen değerler bu sınırların altında kaldı: RMSE farkları %1–9, kazanma oranları
%64–87. **Çalışma, aradığı büyüklükteki etkiler için yapısal olarak yetersiz güçteydi.**

h=22'deki iki anlamlı sonuç (13/15 = %86.7 kazanma oranı) tam da tespit sınırının
üzerine düşüyor — bu, o sonuçların neden tek anlamlı sonuçlar olduğunu açıklıyor ve
tesadüf olmadıklarını destekliyor.

**Makale için:** bu tablo, "anlamlı fark bulunamadı" ifadesinin neden "fark yok"
anlamına gelmediğini önsel olarak, gözlenen sonuçlara başvurmadan gösterir. Sınırlılıklar
bölümünde bu iki tablo verilmelidir.

---

# Aşama 9: SHAP Analizi

`scripts/10_shap_analysis.py`. Soru: XGBoost ile HAR-X aynı değişkenleri mi kullanıyor?

TreeSHAP, XGBoost'un yerleşik `pred_contribs` özelliğiyle hesaplandı (`shap` paketi
gerekmedi); toplama özelliği her fold'da assert ile doğrulandı. Modeller birincil
spesifikasyonla birebir aynı şekilde yeniden eğitildi, SHAP her fold'un test dilimi
üzerinde. **Birim uyarısı:** model log-oran uzayında eğitildiği için SHAP değerleri de o
birimdedir.

## (1) Grup payları — asıl karşılaştırma

XGBoost (toplam |SHAP| payı, %):

| grup | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| brent_vol | 47.3 | 44.1 | 34.2 | 30.8 |
| gpr | 19.5 | 16.5 | 11.3 | 9.9 |
| ovx | 18.0 | 18.3 | 30.2 | 29.8 |
| brent_fiyat | 11.3 | 15.6 | 18.5 | 26.1 |
| takvim | 1.7 | 4.1 | 5.0 | 1.8 |
| etkileşim | 2.1 | 1.4 | 0.8 | 1.6 |

HAR-X (toplam |std beta| payı, %):

| grup | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| ovx | 72.2 | 72.8 | 66.1 | 59.1 |
| brent_vol | 19.9 | 18.8 | 14.4 | 19.6 |
| gpr | 8.0 | 8.5 | 19.5 | 21.2 |

**İki model çok farklı ağırlıklandırıyor.** HAR-X ezici biçimde OVX'e yaslanıyor
(%59–73). XGBoost'un birincil kaynağı Brent volatilitesi (%31–47), OVX'e yalnızca
%18–30 veriyor.

## (2) HAR-X dışı ağırlık

XGBoost'un, HAR-X'in hiç kullanmadığı gruplara (Brent fiyat/getiri, etkileşim, takvim)
verdiği toplam pay: **h=5 %15.1, h=22 %21.1, h=66 %24.3, h=126 %29.5.**

Ufuk uzadıkça artıyor. Takvim özellikleri h=22 ve h=66'da %4–5 pay alıyor; bunun
ekonomik bir gerekçesi yok ve büyük olasılıkla gürültüye uyumdur.

En önemli özellikler ufka göre doğru pencereyi seçiyor: h=5'te `vol5_vol60` (%14.5),
h=22'de `brent_vol20` (%18.7), h=66'da `brent_vol60` (%13.9), h=126'da `brent_vol126`
(%21.3). Model, ufka eşleşen volatilite penceresini kendiliğinden buluyor.

## (3) İşaret uyumu — iki tasarım kusuru bulundu ve düzeltildi

**Kusur 1 (kod hatası, düzeltildi).** Model bir özellik üzerinde hiç bölünme yapmadıysa
SHAP değerleri özdeş sıfırdır ve yön tanımsızdır. İlk uygulamada bunlar "zıt yön" olarak
sayılıyordu. Bu yanlış: "model o özelliği kullanmadı" demek. Düzeltildi ve ayrıca
raporlanıyor.

Kullanılmayan karşılaştırma oranı: h=5 %0, h=22 %2.7, **h=66 %44.3, h=126 %67.1.**
Uzun ufuklarda düşük kapasite kademesi (16 birim, derinlik 2) çoğu özelliği hiç
kullanmıyor.

**Kusur 2 (tasarım sınırı, belgelendi).** XGBoost log-oran hedefi üzerinde eğitiliyor;
`brent_vol5` ve `brent_vol20` paydadaki `past_vol_h` ile mekanik olarak ilintili.
h=5'te `brent_vol5` paydanın ta kendisi (korelasyon 1.000), h=22'de `brent_vol20` ile
korelasyon 0.991. Bu özelliklerde negatif SHAP yönü ortalamaya dönüşün mekanik
sonucudur, "zıt ilişki öğrenildi" anlamına gelmez. HAR-X'in betası ise seviye hedefi
üzerinde. **Bu iki özellikte karşılaştırma geçersizdir.**

Geçerli karşılaştırma yalnızca dışsal regresörlerle:

| ufuk | karşılaştırma | uyumlu | uyum |
| --- | --- | --- | --- |
| 5 | 45 | 24 | %53.3 |
| 22 | 43 | 31 | %72.1 |
| 66 | 15 | 9 | %60.0 |
| 126 | 7 | 5 | %71.4 |
| **toplam** | **110** | **69** | **%62.7** |

Regresör bazında:

| regresör | kullanılan fold | uyum | XGB yön kor. | HAR-X beta |
| --- | --- | --- | --- | --- |
| ovx_lag1 | 46 | **%82.6** | +0.373 | +0.581 |
| gprd_threat_lag1 | 30 | %50.0 | +0.166 | −0.026 |
| gprd_lag1 | 34 | %47.1 | −0.001 | +0.021 |

**OVX'te iki model güçlü biçimde hemfikir:** yön 46 fold'un 38'inde uyuşuyor, ikisinde
de pozitif. OVX aynı zamanda HAR-X'in baskın regresörü. GPR değişkenlerinde uyum
rastlantı düzeyinde (%47–50), ama HAR-X'in GPR betaları da zaten çok küçük
(|β| ≈ 0.02–0.03), yani orada karşılaştırılacak güçlü bir yön yok.

## (4) Fold'lar arası kararlılık

| ufuk | ardışık ρ ort. | min | maks | havuza ρ | ilk 5'e giren farklı özellik | en sık birinci | birinci kalma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.858 | 0.772 | 0.943 | 0.870 | 11 | vol5_vol60 | 10/15 |
| 22 | 0.866 | 0.788 | 0.955 | 0.895 | 12 | brent_vol20 | 11/15 |
| 66 | 0.863 | 0.670 | 0.926 | 0.861 | 12 | brent_vol60 | 5/14 |
| 126 | 0.845 | 0.657 | 0.993 | 0.802 | 12 | brent_vol126 | 6/14 |

**Sıralama savrulmuyor.** Ardışık fold'lar arası Spearman ortalaması 0.85–0.87, en
düşük değer 0.66. SHAP yorumu kırılgan değil. İlk sıradaki özellik kısa ufuklarda
fold'ların üçte ikisinde sabit, uzun ufuklarda daha oynak (5–6/14). İlk beş özelliğin
toplam atıftaki payı %42'den %62'ye çıkıyor, yani uzun ufuklarda önem daha yoğunlaşmış.

## (5) Spearman — uyarıldığı gibi bilgisiz

Grup düzeyi (3 öğe): ρ = −0.5, +0.5, −0.5, −0.5; hepsinde p = 0.667.
Ortak regresör düzeyi (5 öğe): ρ = −0.2, +0.5, −0.6, +0.5; p = 0.28–0.75.

Hiçbiri yorumlanabilir değil. Planda belirtildiği gibi bu ölçüt bu boyutlarda
anlamlılık üretemez; grup payları tablosu asıl kanıttır.

## Sonuç

**Hayır, aynı değişkenleri kullanmıyorlar.** HAR-X ağırlığının üçte ikisini OVX'e
veriyor; XGBoost birincil olarak Brent volatilite penceresine bakıyor ve dikkatinin
%15–30'unu HAR-X'in hiç kullanmadığı gruplara (fiyat seviyesi, takvim, etkileşim)
harcıyor. Uzun ufuklarda bu pay artıyor ve aynı zamanda özelliklerin %44–67'sini hiç
kullanmıyor.

Ortak zeminde, yani OVX'te, iki model aynı yönü buluyor (%83 uyum). Yani XGBoost temel
ilişkiyi yanlış öğrenmiyor; dikkatini farklı yerlere dağıtıyor ve bunun bir kısmı
(takvim değişkenleri gibi) muhtemelen gürültü.

Bu bulgu Aşama 7'yi de açıklıyor: artık modellemesi örneklem dışında hiçbir şey
çıkaramadı, çünkü XGBoost'un HAR-X'e ekleyebileceği bilgi büyük ölçüde gürültüden
oluşuyor.

## Kararlılık ölçütüne düzeltme: örtüşme uyarısı ve uzak fold çiftleri

Yukarıdaki ardışık fold korelasyonları **tek başına kararlılık kanıtı sayılamaz.**
Genişleyen pencerede fold k'nin eğitim seti fold k+1'inkinin alt kümesidir; ardışık
fold'lar eğitim verisinin ortalama **%88–89'unu paylaşır.** Bu koşulda yüksek Spearman
korelasyonu kısmen mekaniktir.

En dürüst ölçüt, örtüşmenin en az olduğu **ilk fold ile son fold** çiftidir.

| ufuk | ardışık ρ | ardışık örtüşme | ilk-son ρ | ilk-son örtüşme |
| --- | --- | --- | --- | --- |
| 5 | 0.858 | %89.1 | **0.773** | %19.4 |
| 22 | 0.866 | %89.0 | **0.785** | %19.1 |
| 66 | 0.863 | %88.3 | **0.686** | %19.4 |
| 126 | 0.845 | %87.9 | **0.529** | %18.2 |

Fold mesafesine göre korelasyon, örtüşmeyle birlikte düzenli olarak düşüyor:

| mesafe | örtüşme | ρ (h=5) | ρ (h=22) | ρ (h=66) | ρ (h=126) |
| --- | --- | --- | --- | --- | --- |
| 1 | %88–89 | 0.858 | 0.866 | 0.863 | 0.845 |
| 4 | %62–65 | 0.802 | 0.821 | 0.802 | 0.775 |
| 8 | %39–43 | 0.724 | 0.766 | 0.714 | 0.740 |
| 13 | %18–23 | 0.697 | 0.752 | 0.686 | 0.529 |

**Düzeltilmiş yorum.** Kısa ufuklarda kararlılık gerçek: örtüşme %19'a indiğinde bile
korelasyon 0.77–0.79'da kalıyor. Uzun ufuklarda ise kararlılık belirgin biçimde zayıf.
h=126'da ilk-son korelasyonu 0.529, yani ardışık ölçütün gösterdiğinin çok altında.

Bu, diğer bulgularla tutarlı: h=126'da etkin bağımsız gözlem sayısı ~28, model
özelliklerin %67'sini hiç kullanmıyor ve atıf sıralaması yıllar arasında gerçekten
oynuyor. **Uzun ufuklarda SHAP yorumu kırılgandır ve makalede öyle nitelenmelidir.**

Çıktı: `shap_stability_by_distance.csv`.


---

# Aşama 10: Dışsal Değişken Ablasyonu (OVX ve GPR ayrı ayrı)

`scripts/11_ablation_exogenous.py`. **Ne arandı:** Harici bir inceleme, HAR-X'teki GPR
bloğunun katkı sağlamadığını, hatta zarar verdiğini öne sürdü. İddia kendi hattımızda,
aynı fold yapısı, embargo, NaN maskesi, train-only taban ve metriklerle yeniden üretildi.
Dört iç içe OLS (seviye) spesifikasyonu: HAR, HAR + OVX, HAR + GPR, HAR-X (HAR + OVX +
GPR). Script, `har` ve `har_x`'in `bench_aggregate_all.csv` ile birebir aynı çıktığını ve
tüm varyantların aynı train/test satırlarını gördüğünü assert ile doğruluyor.

## Fold ortalaması RMSE

| varyant | h=5 | h=22 | h=66 | h=126 |
| --- | --- | --- | --- | --- |
| HAR | 0.010834 | 0.008600 | 0.008100 | 0.008203 |
| HAR + OVX | **0.010298** | **0.007603** | **0.007469** | **0.008003** |
| HAR + GPR | 0.010882 | 0.008668 | 0.008198 | 0.008262 |
| HAR-X | 0.010341 | 0.007669 | 0.007573 | 0.008059 |

(Bu dört varyant içinde en iyisi kalın. h=22'de HAR-X log-log, 0.007561 ile HAR + OVX'in
de önünde.)

## Ne bulundu

- **OVX'in katkısı:** HAR'a göre RMSE %4.9, %11.6, %7.8, %2.4 düşüyor. HAR + OVX, HAR'ı
  13/15, 14/15, 12/14, 10/14 fold'da geçiyor.
- **GPR'nin katkısı negatif:** Tek başına HAR'a eklendiğinde %0.4, %0.8, %1.2, %0.7;
  OVX'in üstüne eklendiğinde %0.4, %0.9, %1.4, %0.7 kötüleştiriyor. Yön dört ufukta da
  aynı.
- **Fold düzeyinde anlamlı değil:** HAR + OVX, HAR-X'i 9/15, 11/15, 9/14, 9/14 fold'da
  geçiyor; kesin işaret testi p = 0.61, 0.12, 0.42, 0.42. En kötü fold (h=5, 22, 66'da
  2024; h=126'da 2017) çıkarıldığında ortalama fark her ufukta hâlâ pozitif, yani sonuç
  tek bir fold'dan gelmiyor.
- **Katsayılar:** Standartlaştırılmış GPR katsayılarının fold ortalaması −0.06 ile +0.08
  arasında; OVX'inki 0.49–0.65. İki GPR bileşeni ters işaret alıyor (GPRD çoğunlukla
  pozitif, GPRD_THREAT çoğunlukla negatif) ve birbirini kısmen götürüyor. Uzun ufukta
  işaret kararsız: h=126'da GPRD 9/14, GPRD_THREAT 5/14 fold'da pozitif. h=5'te ise işaret
  büyük ölçüde sabit (13/15 ve 4/15 pozitif); "her ufukta savruluyor" demek doğru olmaz.

## Yoruma etkisi

v1.0.0 README'deki "OVX ve GPR katkı sağlıyor" ifadesi **yanlıştı** ve v1.1.0'da
düzeltildi. Doğru ifade: dışsal kazancın tamamı OVX'ten geliyor; GPR küçük ama yönü
tutarlı biçimde zarar veriyor ve bu zarar fold düzeyinde anlamlı değil. GPR günlük endeksi
haftalık güncellemelerle yayımlanıyor; `gprd_lag1` dünkü değerin tahmin anında bilindiğini
varsayıyor, bu da GPR lehine bir varsayım. Bu varsayıma rağmen katkı yok, dolayısıyla
bulgu muhafazakâr.

HAR-X ana karşılaştırmada ablasyondan önce belirlendiği haliyle kalıyor; sonradan daha iyi
çıkan HAR + OVX ile **değiştirilmedi** (CLAUDE.md Kural 5, cherry-picking yasağı). HAR +
OVX ve HAR + GPR ayrı satırlar olarak raporlanıyor.

Çıktılar: `ablation_exogenous*.csv`, `ablation_exogenous_summary.json`. Tahminler
sonradan `ablation_exogenous_predictions.csv` olarak eklendi (Aşama 13 için); mevcut
çıktılar yeniden çalıştırmada bayt düzeyinde aynı kaldı.

---

# Aşama 11: Tarih Boşluğu Tanısı

`scripts/14_date_gap_diagnostics.py`. **Ne arandı:** Veri dört serinin ortak tarihlerinde
birleştirildiği için, bir seride eksik olan gün tüm satırı düşürüyor. Bu durumda ardışık
satırlardan hesaplanan "günlük" getiri birden fazla işlem gününü kapsayabilir.

## Ne bulundu

- 4640 getirinin 3600'ünde (%77.6) ardışık satırlar arasındaki fark 1 takvim günü,
  898'inde (%19.4) 2-3 gün, 142'sinde (%3.1) 4 gün veya daha fazla.
- 209 satırda en az bir hafta içi gün atlanmış. Bunların 169'u tamamen NYSE tatilleriyle
  açıklanıyor. İşlem günlerine denk gelen 86 İngiltere tatilinin tamamı veride mevcut,
  yani birleşik seri ABD takvimini izliyor.
- **Gerçek boşluk: 40 satır, 55 atlanmış işlem günü.** Hepsi 2008–2016 arasında; 2017'den
  sonra hiç yok. En büyüğü Nisan 2009'daki 17 günlük boşluk. 2008–2013'teki tek günlük
  boşluklar çoğunlukla ayın 13–19'una denk geliyor; bu, vade devri günleriyle uyumlu ama
  doğrulanmadı. Hangi serinin eksik olduğu birleşik dosyadan ayırt edilemiyor.

## İki sayım: tutarsızlık değil, farklı payda

Hedef penceresinde gerçek boşluk bulunan gözlem sayısı iki farklı kümede sayıldı:

| ufuk | tam örneklem (2008–2026) | test fold'ları (2012–2026) |
| --- | --- | --- |
| 5 | 198 / 4636 (%4.3) | 80 / 3662 (%2.2) |
| 22 | 734 / 4619 (%15.9) | 328 / 3645 (%9.0) |
| 66 | 1182 / 4575 (%25.8) | 494 / 3500 (%14.1) |
| 126 | 1482 / 4515 (%32.8) | 614 / 3500 (%17.5) |

Önceki ad hoc tanıda h=5 için "198", doğrudan testte "80" rakamı geçmişti. İkisi de
doğru, ama farklı şeyleri sayıyorlar. 198, yalnızca eğitimde kullanılan 2008–2011 ısınma
dönemini de içeren tam örneklem sayımı. 80 ise ana örneklem dışı değerlendirmeye giren
test hedeflerinin sayımı. **Sonuçları etkileyen sayı test sayımıdır; README'de o
kullanılıyor.** Test sayımı iki script'te bağımsız olarak hesaplanıyor ve Aşama 13'te
assert ile eşleştiriliyor.

## Yoruma etkisi

Boşluklar sızıntı riski oluşturmuyor; tüm hesaplar satır sırasına dayalı ve geleceğe
bakmıyor. Sorun ölçüm tutarlılığıyla ilgili. Etkisi Aşama 12 ve 13'te ölçüldü.

Çıktılar: `date_gap_distribution.csv`, `date_gap_rows.csv`, `date_gap_by_year.csv`,
`date_gap_target_exposure.csv`, `date_gap_diagnostics_summary.json`.

---

# Aşama 12: Boşluksuz Alt Örneklem Sağlamlık Kontrolü (2017–2026)

`scripts/12_robustness_gapfree.py`. **Ne arandı:** Ana karşılaştırma, hiç tarih boşluğu
içermeyen 2017–2026 fold'larıyla tekrarlandı. Model yeniden eğitilmedi; mevcut fold
metrikleri yeniden ortalandı. Tam örneklem ana tabloyu birebir üretiyor (assert). Alt
örneklem test sonucuna bakılarak değil, Aşama 11'deki tanıyla önsel olarak belirlendi.

## Ne bulundu

| ufuk | fold (tam → 2017+) | Spearman RMSE sırası | Spearman MAE sırası |
| --- | --- | --- | --- |
| 5 | 15 → 10 | 0.955 | 0.964 |
| 22 | 15 → 10 | 0.982 | 0.991 |
| 66 | 14 → 9 | 0.836 | 0.873 |
| 126 | 14 → 9 | 0.673 | 0.700 |

h=5 ve h=22'de sıralama korunuyor. h=66'da ilk üç model aynı kalıyor, ama train-mean 9.
sıradan 4. sıraya çıkıyor. h=126'da train-mean 8. sıradan 1. sıraya çıkıyor ve bu dönemde
tüm modellerin R²_oos'u negatif; ilk beş model arasındaki fark yaklaşık %3.

**Ama bu değişim boşluklardan değil, dönemden kaynaklanıyor.** 2012–2016 sıralaması da tam
örneklemden farklı; train-mean orada h=66 ve h=126'da 11. sırada. Boşluk kaynaklı bir
bozulma tüm modellerin hedefini aynı biçimde kaydırırdı. Burada ise tek bir modelin
(sabit tahmin) göreli başarısı dönemler arasında büyük ölçüde değişiyor, yani iki dönemin
oynaklık rejimi farklı. Bu kontrol tek başına boşluk etkisini dönem etkisinden
ayıramıyor; bu yüzden Aşama 13 yapıldı.

## Yoruma etkisi

"Sıralama değişmiyorsa boşluklar etkisiz" argümanı bu kontrolle kurulamaz; uzun ufukta
sıralama değişiyor. Değişmeyen bulgu şu: 2017+ döneminde de HAR ailesi her ufukta XGBoost
ve BiLSTM'in önünde. h=126, 2017+ bulgusu (hiçbir modelin train-mean'i geçememesi)
Sınırlılıklar bölümünde yazılmalı.

Çıktılar: `robustness_gapfree_2017plus.csv`, `robustness_gapfree_2017plus_summary.json`.

---

# Aşama 13: Doğrudan Hedef Düzeltme Testi

`scripts/13_gap_target_test.py`. **Ne arandı:** Boşluk etkisini dönem etkisinden ayırmak.
Her gerçek boşluk getirisi r, 1 + k işlem gününü kapsıyorsa, r / sqrt(1 + k) ile tek
günlük eşdeğerine ölçeklendi (rastgele yürüyüş varsayımı: varyans zamanla doğrusal artar).
Hedef, düzeltilmiş getirilerden aynı formülle yeniden kuruldu. Tüm modellerin yayımlanmış
tahminleri **sabit tutularak** fold RMSE ve MAE yeniden hesaplandı. Yalnızca
değerlendirme hedefi değişiyor; özellikler ve eğitim hedefleri düzeltilmedi, model
yeniden eğitilmedi. HAR + OVX ve HAR + GPR dahil 12 model.

## Ne bulundu

| ufuk | değişen test hedefi | ort. hedef değişimi | RMSE değişimi | RMSE sıra değişimi | MAE sıra değişimi |
| --- | --- | --- | --- | --- | --- |
| 5 | 80 / 3662 | −%13.1 | −%0.15 … +%0.16 | 0 | 2 (train-mean ↔ BiLSTM) |
| 22 | 328 / 3645 | −%3.3 | −%0.10 … +%0.26 | 0 | 0 |
| 66 | 494 / 3500 | −%2.3 | +%0.20 … +%0.42 | 0 | 2 (XGBoost ↔ past-vol) |
| 126 | 614 / 3500 | −%2.0 | +%0.20 … +%0.49 | 0 | 0 |

Değişen hedef sayısı, Aşama 11'in test sayımıyla birebir aynı (assert). MAE'de yer
değiştiren iki çift zaten neredeyse eşitti (fark %0.2 ve %0.06). Düzeltilmiş hedefle de
HAR + OVX HAR-X'i, HAR da HAR + GPR'yi geçiyor; yani Aşama 10'un bulgusu da boşluklara
duyarlı değil.

## Yoruma etkisi

**Tarih boşlukları raporlanan sonuçları değiştirmiyor.** Metrikleri en fazla %0.5
kaydırıyor ve RMSE sıralaması hiçbir ufukta değişmiyor. Makalede birincil kanıt olarak
bu test, ek sağlamlık analizi olarak Aşama 12 verilmeli. Sınırlılık: 2017+ fold'larının
eğitim verisi de dahil, eğitim tarafındaki boşluklu getiriler düzeltilmedi.

Çıktılar: `gap_target_test.csv`, `gap_target_test_folds.csv`, `gap_target_test_summary.json`.
