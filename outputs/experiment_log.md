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


---

# Aşama 14: Ablasyon Katsayılarının Standartlaştırılması

`scripts/11_ablation_exogenous.py` (eklendi). **Ne arandı:** Ham katsayılar
karşılaştırılamıyor; regresörler ölçeklenmemiş düzeyde (OVX onlar, GPRD yüzler
mertebesinde), birim farkı etki farkı gibi okunabiliyor. Her fold ve ufuk için
beta_std = beta × sd(X) / sd(y) hesaplandı; sd'ler o fold'un train diliminden. `har_x`
satırları `harx_standardized_betas.csv` ile birebir aynı çıkıyor (assert).

**Fold sayısı:** Her varyant her ufukta 15 fold'da tahmin ediliyor (240 satır = 4 varyant
× 4 ufuk × 15 fold). Birincil metrik toplulaştırması h=66 ve h=126'da 14 fold (2026 kısmi
yıl kuralı). Yani 15 tahmin fold'u, uzun ufuklarda 14 metrik fold'u. Özet dosyası iki
kümeyi ayrı veriyor (`fold_set` = `main` / `all_estimated`).

## Ne bulundu (fold_set = main; ortalama, pozitif/negatif fold)

| ufuk | HAR-X: ovx_lag1 | HAR-X: gprd_lag1 | HAR-X: gprd_threat_lag1 | HAR+GPR: gprd_lag1 | HAR+GPR: gprd_threat_lag1 |
| --- | --- | --- | --- | --- | --- |
| 5 | +0.584 (15/0) | +0.028 (13/2) | −0.024 (4/11) | +0.040 (15/0) | −0.056 (0/15) |
| 22 | +0.650 (15/0) | +0.019 (11/4) | −0.028 (6/9) | +0.032 (11/4) | −0.063 (4/11) |
| 66 | +0.586 (14/0) | +0.077 (13/1) | −0.057 (3/11) | +0.091 (11/3) | −0.093 (3/11) |
| 126 | +0.487 (14/0) | +0.040 (9/5) | −0.007 (5/9) | +0.052 (9/5) | −0.035 (5/9) |

- GPR katsayıları OVX'inkinin kabaca onda biri ya da daha küçük.
- İki GPR bileşeni sistematik olarak ters işaretli. Train dilimlerinde korelasyonları
  0.82–0.89; ters işaret, bu yüksek eşdoğrusallıkla uyumlu ve net etkileri kısmen
  birbirini götürüyor.
- İşaret kısa ufukta kararlı: HAR+GPR'de h=5'te GPRD 15/15 pozitif, THREAT 15/15 negatif.
  Uzun ufukta kararsız: h=126'da 9/5 ve 5/9.
- OVX eklenince brent_vol20'nin standartlaştırılmış katsayısı 0.44–0.62'den 0.06–0.13'e
  düşüyor. OVX, kalıcılık bilgisinin büyük kısmını üstleniyor.

## Yoruma etkisi

README'deki ifadeyle tutarlı: GPR katsayıları OVX'e göre sıfıra yakın, iki bileşen ters
işaretli, işaret uzun ufukta kararsız. "İşaret her ufukta fold'dan fold'a savruluyor"
demek yanlış olur; h=5'te işaret tamamen kararlı.

Çıktılar: `ablation_exogenous_coefficients_standardized.csv`,
`ablation_exogenous_std_beta_summary.csv`. Mevcut ablasyon çıktıları bayt düzeyinde
değişmedi.

---

# Aşama 15: XGBoost-6 — fonksiyonel formu izole eden keşifsel koşu

`scripts/15_exploratory_xgb6.py`. **Statü: keşifsel ve post hoc**, ablasyon merdiveniyle
aynı. Birincil hipotez ailesine dahil değil, model seçiminde kullanılmıyor, birincil
spesifikasyonu değiştirmiyor. Ana sonuçlar bilindikten sonra tasarlandı.

**Ne arandı:** XGBoost ile HAR-X karşılaştırması üç şeyi birden değiştiriyor: model
ailesi, girdi kümesi (65 vs 6) ve hedef/ön işleme (log-oran + smearing + özelliklerde
log1p/winsorize/MinMax vs düzey, dönüşümsüz). XGBoost-6 yalnızca model ailesini
değiştiriyor: HAR-X'in altı ham regresörü, düzey hedef, smearing yok, ön işleme yok, HAR-X
ile aynı train/test satırları (assert), aynı train-min tabanı. Kapasite kuralı ve ortak
parametreler `03_walkforward.py`'dan import edildi. Not: winsorization birincil XGBoost'ta
yalnızca özelliklere uygulanıyor, hedefe değil.

## Ne bulundu (fold ortalaması RMSE, ana metriğe giren fold'lar)

| ufuk | XGBoost-6 | HAR-X | XGBoost birincil | XGB-6 vs HAR-X | birincil vs XGB-6 | XGB-6 / HAR-X kazanan fold | işaret p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.010645 | 0.010341 | 0.010913 | +2.94% | +2.51% | 3 / 12 | 0.035 |
| 22 | 0.008184 | 0.007669 | 0.009080 | +6.72% | +10.94% | 4 / 11 | 0.118 |
| 66 | 0.008015 | 0.007573 | 0.008924 | +5.84% | +11.34% | 2 / 12 | 0.013 |
| 126 | 0.008050 | 0.008059 | 0.008627 | −0.12% | +7.17% | 7 / 7 | 1.000 |

MAE'de XGBoost-6, HAR-X'ten %4.3, %10.3, %10.9, %2.6 kötü (işaret p = 0.035, 0.035,
0.057, 0.42). Kapasite katmanları: h=5'te 15 fold yüksek; h=22'de 9 yüksek ve 6 orta;
h=66'da 9 orta ve 5 düşük; h=126'da 2 orta ve 12 düşük. Tabana takılan tahmin yok.

- **Fonksiyonel form, girdi ve hedef sabitken:** h=5, 22, 66'da esnek form doğrusal formdan
  %3–7 kötü. h=126'da fark yok (−%0.12, 7'ye 7). İşaret testleri düzeltmesiz; 4 RMSE
  testine Holm uygulanınca hiçbiri 0.05 altında kalmıyor (en küçük 0.013 × 4 = 0.052).
- **Birincil XGBoost'un ek kaybı:** 65 özellik + log-oran hedef + smearing + ön işleme
  birlikte, XGBoost-6'ya göre %2.5–11.3 daha kötü. Birincil XGBoost ile HAR-X arasındaki
  farkın (%5.5, %18.4, %17.8, %7.0) bir kısmı fonksiyonel formdan, kalanı bu paketten
  geliyor.
- XGBoost-6, düz HAR'ı dört ufukta da geçiyor, çünkü OVX'i görüyor. En iyi HAR-ailesi
  modelini (HAR + OVX veya HAR-X-log) hiçbir ufukta geçmiyor.

## Yoruma etkisi

"Doğrusal olmayan modelleme katkı sağlamıyor" iddiası, girdi ve hedef sabitken de
geçerli, ama daha ölçülü: h=5–66'da esnek form küçük ama tutarlı bir kayıp veriyor,
h=126'da eşitlik. Birincil XGBoost'un büyük kaybı yalnızca fonksiyonel formdan değil,
zengin özellik seti ve hedef reparametrizasyonundan da geliyor. Bu ayrıştırma makalede
keşifsel olarak raporlanmalı; birincil karşılaştırmanın yerine geçmez.

Çıktılar: `exploratory_xgb6.csv`, `exploratory_xgb6_folds.csv`,
`exploratory_xgb6_predictions.csv`, `exploratory_xgb6_summary.json`.

---

# Aşama 16: GPR yayım tarihi hizalaması (erişilebilirlik düzeltmesi)

**Tetikleyici — şeffaflık kaydı.** Bu çalışma, projenin kodunu inceleyen **bağımsız bir dış
kod incelemesiyle tetiklendi**. İnceleme, GPR özelliklerinin tahmin anında erişilebilir
olmayan bilgiyi kullandığını işaret etti. Makalenin şeffaflık beyanında bu şekilde yer
alacak.

**Sorun.** Günlük GPR endeksi (Caldara ve Iacoviello) OVX ve Brent gibi gerçek zamanlı
gözlenmez. Toplu halde yayımlanır (haftalık, artı aylık güncelleme) ve sonraki sürümler
geçmiş değerleri revize eder. Model ise GPR'ı her gün bir gün gecikmeyle kullanıyordu:
satır t, t−1 tarihli gözlemi görüyordu. Bu `.shift(1)` kuralına biçimsel olarak uyuyor
ama bir **erişilebilirlik ihlali**. t−1 gözlemi çoğu gün t anında henüz yayımlanmamıştı.
Ölçüm (Aşama 16.2): zaman damgalı hizalamada satırların **%80.3'ü** yayımlanmamış bir
gözlem kullanıyor. Bu, Kritik Kural 6'nın ("t anında bilinmeyen bilgi kullanılamaz")
ruhuna aykırı bir sızıntıdır. Kod `.shift()` kontrolünden geçtiği için önceki
denetimlerde yakalanmadı.

Düzeltme üç aşamada yapıldı. Aşama 1 ve 2 commit `20cbfab` ve `a1482d6`'da, Aşama 3
commit `4ef8c39`'da.

## 16.1 Yayım kuralının tespiti (`scripts/16_gpr_vintages.py`)

**Kaynak (erişim 2026-09-24):** yazarların GitHub'daki sürüm arşivi
(`iacoviel/iacoviel.github.io/gpr_archive_files`). **289 günlük sürüm** var, 2022-02-24
ile 2026-09-21 arası. Sayfa beyanı: "The daily data are updated every Monday … If the
first day of the month or week falls on a federal holiday, data updates will take place
the next business day."

**Kural: D gününde yayımlanan dosya, D dahil D'ye kadarki gözlemleri içerir.** 289
sürümün **279'u** bu kuralı destekliyor (dosya tarihi = son gözlem tarihi). 10 istisnanın
7'si 1 gün, 1'i 2 gün, 1'i 3 gün geride. Bunların hepsi ay başı güncellemeleri; dosya bir
önceki ayın son gününe kadarki veriyi içeriyor. Bir istisna 124 gün geride
(2023-01-02 dosyası, son gözlem 2022-08-31). Bu bayat bir yükleme gibi görünüyor.

Sürüm günleri: Pazartesi 207, Salı 42, Çarşamba 14, Perşembe 12, Cuma 14.

**Gözlem başına yayım gecikmesi** (gözlem tarihinden onu ilk içeren sürüme kadar geçen
takvim günü): ortalama 2.84, medyan 3, en fazla 10. Gözlemin haftanın gününe göre medyan
gecikme: Pazartesi 0, Salı 6, Çarşamba 5, Perşembe 4, Cuma 3, Cumartesi 2, Pazar 1. Yıllar
arasında kararlı (2022–2026 ortalaması 2.76–2.88).

**Revizyonlar.** İlk yayım ile güncel sürüm arasındaki göreli fark GPRD'de ortalama −%5.2
(ortalama mutlak %12.7, medyan mutlak %10.0), GPRD_THREAT'te ortalama −%3.7 (ortalama
mutlak %14.6, medyan mutlak %11.3). Revizyonlar küçük değil ve sistematik bir işareti var.

**Kapsam uyarısı: 2022 öncesi karşı-olgusaldır.** Sürüm arşivi 2022-02-24'te başlıyor.
Günlük GPR, Caldara ve Iacoviello (2022) ile kamuya açıldı. 2008–2021 için "o gün
yayımlanmış değer" diye bir şey yok. Bu dönem için kural, bugünkü yayım rejiminin geçmişte
de geçerli olduğunu varsayar: gözlem d, d'den sonraki ilk Pazartesi (d dahil) yayımlanır;
o gün federal tatilse bir sonraki iş günü.

**Revizyonlar modellenmedi.** Değerler `data/veriseti.xlsx`'ten gelir (2026-09-01
sürümüyle birebir aynı), yani güncel sürüm değerleridir. Yayım hizalaması **zamanlamayı**
düzeltir, **revizyonu** düzeltmez. Tam gerçek zamanlı bir tasarım ilk yayım değerlerini
kullanırdı; bu yalnızca 2022 sonrası için mümkün. Sınırlılık olarak kaydedilir.

## 16.2 Yayım-hizalı özellikler (`scripts/02_build_features.py`, `scripts/gpr_publication.py`)

**Yöntem.** GPR özellikleri (lag, EMA, z-skoru, spike, momentum, threat oranı, etkileşimler)
endeksin **kendi gözlem dizisi** üzerinde hesaplanır, sonra yayım tarihine göre işlem
günlerine eşlenir. Satır t, t−1'e kadar yayımlanmış en son gözlemi görür: p(d*) ≤ t−1.
Düzey seriyi işlem takvimine ileri doldurup ondan türetmek, yayımlanmış gözlemlerin %78'ini
atardı. **Bir gün muhafazakâr:** sürümler ~13:30 UTC'de, Brent kapanışından önce
yayımlanıyor, yani aynı gün kullanım mümkün olurdu. Yine de diğer tahmin değişkenlerinin
`.shift(1)` kuralıyla tutarlılık için p(d*) ≤ t−1 istenir.

65 özelliğin **27'si** değişti (GPRD ve GPRD_THREAT'in tüm türevleri ve dört OVX × GPR
etkileşimi). İlk tam dolu satır iki sürümde de 127. Zaman damgalı `features.csv` bit
düzeyinde yeniden üretiliyor. Aynı commit'te bir düzeltme: sonraki bir sürüme eklenen
tarihler için ilk yayım takibi düzeltildi (2024-02-29).

**Etkin gecikme** (işlem günü t ile kullanılan GPR gözleminin tarihi arası):

| ölçü | zaman damgalı | yayım-hizalı |
| --- | --- | --- |
| takvim günü, medyan | 1 | 3 |
| takvim günü, ortalama | 1.47 | 3.69 |
| takvim günü, en fazla | 17 | 20 |
| işlem satırı, medyan / ortalama / en fazla | — | 3 / 2.97 / 8 |

Yayım-hizalı sürümde, işlem gününe göre medyan takvim günü: Pazartesi 7, Salı 1,
Çarşamba 2, Perşembe 3, Cuma 4. Pazartesi en eski bilgiyi görüyor, çünkü o günün yayımı t−1
kuralı gereği ancak Salı kullanılabiliyor.

**Testler (hepsi geçti):**

- **Prefix-invariance:** özellikler veri 3000. ve 4000. satırda kesilerek yeniden
  hesaplandı. Kesimden önceki tüm satırlar tam veriyle hesaplananla aynı (tolerans 0).
  Hiçbir özellik gelecekteki satırlara bakmıyor.
- **Yayım duyarlılığı, kontrol kollu:** rastgele 40 satırda, o satırın tarihinde henüz
  yayımlanmamış tüm GPR gözlemleri bozuldu (satır başına 136–4065 gözlem).
  **Yayım-hizalı kol: 40/40 satır değişmedi.** **Kontrol kolu (zaman damgalı): 33/40
  satır değişti**, yani test sızıntıyı yakalayabiliyor.

  **Kontrol kolunda değişmeyen 7 satırın açıklaması.** Zaman damgalı kol satır t'de her
  zaman t−1 tarihli gözlemi kullanır. Bozulma yalnızca t−1'e kadar yayımlanmamış
  gözlemlere uygulanır. Bu yüzden kontrol satırı ancak t−1 gözlemi t−1'e kadar
  yayımlanmamışsa değişir.
  - **6 Salı satırı.** t−1 gözlemi Pazartesi tarihli. Pazartesi, yayım gecikmesinin en
    küçük olduğu gözlem günü (medyan 0): Pazartesi gözlemi aynı gün Pazartesi dosyasında
    yayımlanır.
  - **1 Çarşamba satırı (2023-09-06).** İşçi Bayramı haftası. Güncelleme Salı'ya (09-05)
    kaydı ve Salı gözlemini içerdi, yani t−1 gözlemi yine aynı gün yayımlanmıştı.

  Doğrulama: yayım takviminden "t−1 gözlemi t−1'e kadar yayımlanmış mı?" sorusu 40 satırın
  **40'ında** kontrol kolunun sonucunu doğru öngörüyor (yayımlanmış 7 satırın 7'si
  değişmedi, yayımlanmamış 33 satırın 33'ü değişti). Dolayısıyla bu 7 satırı tespit
  edememek testin zayıflığı değil, ölçülmüş yayım gecikmesi yapısının beklenen sonucudur:
  bu satırlarda zaman damgalı hizalama zaten erişilebilir bir gözlem kullanıyor ve
  bozulacak bir şey yok.

  _Makale için önerilen cümle:_ "The seven control rows that did not change are exactly
  those whose t−1 observation had already been released by t−1 (six Tuesdays, whose t−1
  is a Monday released the same day, and one Wednesday in Labor Day week, when the update
  moved to Tuesday); the release calendar predicts the control outcome in 40 of 40 rows,
  so non-detection reflects the measured release schedule, not a weakness of the test."

## 16.3 GPR kullanan tüm modellerin yeniden koşulması

`scripts/alignment.py` ortak bir anahtar ekliyor: `--gpr-alignment publication`,
`features_publication_aligned.csv`'yi okur ve hem yazdığı hem okuduğu her çıktıya
`_publication_aligned` sonekini ekler (iki sürüm hiçbir zincirde karışmaz). Yeniden
koşulanlar: 03 (XGBoost), 04 (Optuna, büzülmüş ve ham smearing), 05 (benchmark'lar),
06 (BiLSTM), 07 (hibritler), 07b, 08 (DM), 10 (SHAP), 11 (ablasyon), 12, 13, 15 (XGB-6).
75 `_publication_aligned` dosyası üretildi. Karşılaştırma `scripts/17_gpr_alignment_comparison.py`
ile yapıldı.

**Aynı örneklem, saf hizalama etkisi.** Her model × ufuk × fold'da train/test satırları iki
sürümde aynı (assert). GPR kullanmayan modeller (HAR, HAR-log, HAR+OVX, GARCH, train-mean,
past-volatility) 360 fold satırında **bit düzeyinde aynı**. Fark yalnızca hizalamadan
geliyor.

**RMSE değişimi, fold ortalaması** (`100 × (yayım / zaman damgalı − 1)`, pozitif =
yayım gecikmesine uymanın maliyeti; parantezde yayım sürümünün daha iyi olduğu fold ve
düzeltmesiz işaret p):

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
| Optuna (büzülmüş) | +1.67% | +4.03% | +7.98% | −0.30% |
| Optuna (ham) | +1.82% | +3.08% | +0.50% | +1.69% |

- **Doğrusal modellerde maliyet çok küçük:** HAR-X +%0.02 ile +%0.80 arası; hiçbir işaret
  testi anlamlı değil. Bu, GPR'ın HAR-X'te zaten neredeyse sıfır ağırlık taşımasıyla
  tutarlı.
- **Esnek modeller daha oynak:** BiLSTM h=5'te −%5.5, h=66'da +%5.3 değişiyor, iki yönde de.
  Bu bir "sızıntı kazancının kaybı" değil, yüksek varyanslı bir modelin girdi değişikliğine
  duyarlılığı. XGBoost h=22'de yayım sürümünde **daha iyi** (−%2.3, 12/15). Düzeltmesiz
  p=0.035; 11 model × 4 ufuk içinde çoklu karşılaştırma altında anlamlı sayılmamalı.
- **Ana bulgular değişmedi:**
  - Dışsal katkı OVX'ten geliyor.
  - GPR, OVX'in üstüne dört ufukta da bir şey eklemiyor: HAR-X, HAR+OVX'ten
    +%0.43 / +%1.36 / +%2.21 / +%1.06 kötü.
  - Doğrusal olmayan modeller HAR-X'i hiçbir ufukta geçmiyor.
- **Birincil hipotez ailesi (8 test):** BH altında ayakta kalanlar **aynı iki test**: h=22'de
  HAR-X > HAR ve HAR-X > XGBoost, işaret testi 13/15, BH p = 0.030. DM 0/8 (HLN p
  0.113–0.876). Değişenler hiçbir sonucu çevirmiyor:
  - h=5 HAR vs HAR-X: HAR-X kazanımı 12/15 → 11/15 (ham p 0.035 → 0.118).
  - h=66 HAR-X vs XGBoost: 11/14 → 10/14 (0.057 → 0.180).
  - h=126 HAR-X vs XGBoost: 10/14 → 9/14.
- **İkincil aile (24 test):** işaret testinde ayakta kalan 9/24 (Holm ve BH) → 6/24 Holm,
  8/24 BH. DM 4/24 Holm, 6/24 BH, değişmedi.
- **Ayrıştırma** (HAR → HAR-X → XGB-6 → XGB), yayım-hizalı:
  - dışsal: −4.53 / −10.39 / −5.75 / −1.41%
  - fonksiyonel form: +4.78 / +9.11 / +5.16 / +0.63%
  - özellik paketi: +1.09 / +5.50 / +12.42 / +6.59%

  Fonksiyonel form maliyeti h=5 ve h=22'de arttı, h=126'da −0.12'den +0.63'e geçti. Yön
  değişmedi.
- **GPR'ın ağırlığı iki yönde farklı hareket ediyor:**
  - XGBoost SHAP'ında GPR grup payı **arttı**: %19.5 → 25.5, 16.5 → 29.8, 11.3 → 20.0,
    9.9 → 15.3.
  - HAR-X'te GPR betaları sıfıra **yaklaştı**: gprd_lag1 h=22'de +0.019 → +0.001,
    h=126'da +0.040 → +0.006.
  - Yorum: SHAP payı katkı değil kullanım ölçer. Haftalık basamaklı GPR serileri
    ağaçlara daha çok bölme noktası sunuyor, ama XGBoost'un doğruluğu artmıyor. GPR grubu
    23 özellikle en kalabalık grup; pay bu yüzden de şişik.
- **2026 kısmi yıl (h=66/126, bilgi amaçlı):** GPR kullanan modellerin çoğu yayım
  sürümünde %1–16 daha kötü, XGB-6 −%6.7 / −%6.2 daha iyi. Ana metriğe girmiyor; 41–101
  örtüşen gözlem, etkin gözlem ~1.

**Karar.** Birincil sonuçlar artık yayım-hizalı sürüm. Zaman damgalı sürüm makalenin
Ek A'sına taşınır ve iki sürüm karşılaştırması orada raporlanır. Model spesifikasyonu,
hiperparametre kuralı ve test aileleri değiştirilmedi. Değişen tek şey GPR özelliklerinin
zamanlaması.

## 16.4 Sağlamlık kontrollerinin yayım modunda yeniden koşulması

**Veri eşitleme (Aşama 5), `bench_*_aligned_publication_aligned`.** Benchmark'lar
XGBoost'un penceresine (satır 127) indirildi. XGBoost'a göre fark (h=5/22/66/126):

| model | normal pencere | eşitlenmiş pencere |
| --- | --- | --- |
| HAR | −1.11 / −3.05 / −10.25 / −5.44 | −1.12 / −3.13 / −10.36 / −6.74 |
| HAR-X | −5.60 / −13.12 / −15.41 / −6.77 | −5.58 / −13.01 / −15.37 / −8.29 |
| HAR-X-log | −5.89 / −14.21 / −16.66 / −8.04 | −5.81 / −14.19 / −16.81 / −9.48 |

**Sonuç değişmedi:** eşitleme sonrası HAR ailesinin üstünlüğü korunuyor, h=126'da yine
bir miktar artıyor. Zaman damgalı referans aynı komutla bit düzeyinde yeniden üretildi
(Aşama 5 tablosu). Süre 32 saniye.

**BiLSTM yakınsama kontrolü (Aşama 6), `bilstm_*_conv_publication_aligned`.** Aynı komut
(`--convergence-mode --suffix _conv`), aynı önceden ilan edilmiş kriter. Süre 29 dakika.

| ufuk | birincil | yakınsama kriterli | değişim | yakınsama daha iyi olan fold | (zaman damgalı: değişim, fold) |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.013440 | 0.013518 | +0.58% | 7/15 | +1.03%, 6/15 |
| 22 | 0.010386 | 0.010653 | +2.58% | 3/15 | −0.85%, 5/15 |
| 66 | 0.012207 | 0.012207 | 0.00% | 0/14 | 0.00% |
| 126 | 0.011585 | 0.011585 | 0.00% | 0/14 | 0.00% |

h=66 ve h=126'da yüksek kademe fold yok; sonuçlar tanım gereği birebir aynı (determinizm
doğrulaması yine geçerli). **Test hatası iyileşmedi**: h=5'te +%0.6, h=22'de +%2.6 kötü.

**Ancak kontrolün dayandığı öncül bu sürümde tekrarlanmadı.**
- **Eğitim uzamadı.** Zaman damgalı sürümde h=5'te fold'lar ortalama 124 epoch koşmuştu;
  geç fold'lar (2021–2026) 182–198 epoch koştu ve eğitim kaybı fold başına 4.1–6.1 kat
  düştü. Yayım sürümünde durdurma kriteri 15 fold'un 14'ünde 62–78 epoch'ta tetiklendi.
  Yalnızca 2025 fold'u 119 epoch koştu. Ortalama 72 epoch.
- **Kayıp düşüşü küçük.** Son eğitim kaybı oranı (birincil / yakınsama) medyan 1.36;
  zaman damgalı sürümde medyan 2.40, 8 fold'da ≥2. Yayım sürümünde yalnızca 1 fold ≥2.
- **Bağımsız sınıflandırıcı itiraz ediyor.** Son %20 penceresine göre h=5'te 11/15 fold
  hâlâ "yetersiz eğitilmiş" (birincil koşuda 4/15).

Bu, Aşama 6'da kaydedilen tanı sınırlılığının doğrudan sonucu. Durdurma kriteri (son %10,
<%2 düşüş) gürültülü kayıp eğrisinde şansa bağlı düz bir epoch çifti üzerinde
tetiklenebiliyor. GPR girdisinin değişmesi kayıp yörüngesini değiştirdi ve kriter bu kez
erken tetiklendi.

**Yorum.** "Yetersiz eğitim değil aşırı uyum" bulgusu yayım sürümünde **bu kontrolle
desteklenmiyor, ama çürütülmüyor da**. Yapılan ek eğitim test hatasını iyileştirmedi.
Fakat kontrol, zaman damgalı sürümdeki kanıtın dayandığı rejime (eğitim kaybında büyük
düşüş) ulaşmadı; yani eğitim kaybını gerçekten düşürmenin test hatasına etkisini
bu sürümde ölçmüyor. Aşırı saçılım göstergesi de bu sürümde zayıfladı:
std(tahmin)/std(gerçek) ortalaması h=5 ve h=22'de ~1.0 (zaman damgalı: 1.16 ve 0.96),
h=66 ve h=126'da 1.31 ve 1.68 (zaman damgalı: 1.23 ve 1.49). Aşırı saçılım argümanı
yalnızca uzun ufuklarda geçerli.

**Açık karar (kullanıcıda):** makalede bu iddia (a) zayıflatılarak mı yazılacak, yoksa
(b) daha güçlü bir kontrol mü koşulacak? (b) örneğin yüksek kademe fold'larda sabit 200
epoch ile yapılabilir; tahmini süre ~70 dakika. Bu, sonuç görüldükten sonra tasarlanmış
bir kontrol olur ve öyle kaydedilmelidir; birincil spesifikasyonu değiştirmez. Kriter yine
yalnızca eğitim kaybından türemeli, test performansına bakılmamalı.

**Karar: (b) seçildi.** Sonuç 16.5'te.

## 16.5 Keşifsel kontrol: BiLSTM sabit 200 epoch (`scripts/19_bilstm_fixed_epochs.py`)

**Statü: keşifsel ve post hoc.** 16.4'teki yakınsama sonucu görüldükten sonra
tasarlandı. Birincil spesifikasyonu değiştirmez, model seçiminde kullanılmaz. Epoch sayısı
(200), 06'da önceden ilan edilmiş üst sınırdır; test performansına bakılarak seçilmedi.
Yayım-hizalı sürüm, süre 69 dakika.

**Tasarım:**
- Erken durdurma tümüyle kapalı; sabit 200 epoch.
- Kapsam: yüksek kademe fold'lar, h=5'te 15, h=22'de 9 fold. h=66 ve h=126'da yüksek
  kademe yok.
- Kosinüs lr programı yakınsama koşusuyla aynı (`T_max=200`). Dolayısıyla bu koşunun
  k'ıncı epoch'u, k'da durmuş yakınsama koşusunun kendisi. Her fold'da assert edildi:
  k'daki eğitim kaybı ve k'daki test tahminleri bit düzeyinde aynı (24/24).
  "k epoch vs 200 epoch" karşılaştırması tek bir eğitim yolunun iki noktası; durdurma
  kuralı karışmıyor. Birincil koşu (60 epoch, `T_max=60`) ayrı bir yol, yan yana
  veriliyor.
- Epoch bazında eğitim kaybı kaydedildi (`bilstm_fixed200_loss_history`). Ayrıca k'da ve
  200'de, eval modunda (dropout kapalı) tüm eğitim setindeki MSE hesaplandı; bu daha temiz
  bir uyum ölçüsü.

**Önceden yazılmış yorum kuralı (sonuç görülmeden):**
- Kayıp ciddi düşer ve test hatası iyileşmezse, aşırı uyum argümanı durdurma kuralı
  karışmadan kurulur.
- Kayıp yine düşmezse, model epoch-kısıtlı değil kapasite-kısıtlıdır.

**Sonuç (ana fold'lar, yüksek kademe):**

| | h=5 (15 fold) | h=22 (9 fold) |
| --- | --- | --- |
| k (durdurma epoch'u), ortalama | 72 | 62 |
| eğitim kaybı k → 200, medyan kat | **2.48×** (1.51–4.65) | **3.09×** (2.58–3.50) |
| eval-modu eğitim MSE k → 200, medyan kat | 2.40× (1.41–6.29) | 3.08× (2.52–4.94) |
| eğitim kaybı birincil (60 ep) → 200, medyan kat | 3.63× | 2.21× |
| kaybı ≥2 kat düşen fold | 14/15 | 9/9 |
| test RMSE, birincil / k / 200 | 0.013440 / 0.013518 / **0.014493** | 0.012137 / 0.012583 / **0.013170** |
| test RMSE değişimi, 200 vs k | **+7.21%** | **+4.67%** |
| 200'ün k'dan iyi olduğu fold (işaret p, keşifsel, düzeltmesiz) | 2/15 (p = 0.007) | 3/9 (p = 0.51) |
| test MAE değişimi, 200 vs k | +5.22% | +3.32% |
| std(tahmin)/std(gerçek), k → 200 | 1.00 → 1.10 | 1.02 → 1.05 |

h=22'de 9 yüksek kademe fold'unun 200-epoch sonuçları diğer 6 fold'la birleştirildiğinde,
ufuk ortalaması RMSE 0.010653'ten 0.011006'ya çıkıyor (+%3.31). h=5'te tüm fold'lar
yüksek kademede; ufuk ortalaması +%7.21.

**Yorum.** Yorum kuralının ilk dalı gerçekleşti:
- Eğitim kaybı durdurma kuralı olmadan ciddi düştü: medyan 2.5× (h=5) ve 3.1× (h=22).
  Kayıp 24 fold'un 23'ünde en az 2 kat düştü. Eval-modu MSE de aynı ölçüde düştü; yani
  düşüş dropout gürültüsü değil, gerçek uyum artışı.
- Test hatası iyileşmedi, kötüleşti: h=5'te +%7.2 (15 fold'un 13'ünde kötü,
  p = 0.007), h=22'de +%4.7 (9'un 6'sında kötü, p = 0.51).

  **Bu p değerlerinin statüsü:** keşifsel bir teşhisten gelir, **birincil sekizlik
  hipotez ailesine DAHİL DEĞİLDİR** ve çoklu karşılaştırma için **düzeltilmemiştir**.
  Statüsü, HAR+OVX vs HAR-X'in düzeltmesiz p = 0.035'iyle aynıdır. Birincil aile
  (HAR vs HAR-X, HAR-X vs XGBoost × 4 ufuk) önceden sabitlendi; sonradan hiçbir test
  eklenmez. Bu p değerleri makalede çıkarım kanıtı olarak değil, teşhisin yönünü
  betimlemek için verilir.
- Tahmin saçılımı h=5'te arttı (1.00 → 1.10). Bu, modelin eğitim gürültüsünü öğrendiği
  okumasıyla tutarlı.

**"Yetersiz eğitim değil aşırı uyum" bulgusu yayım-hizalı sürümde bu kontrolle
destekleniyor**, ve erken durdurma karışması olmadan. Zaman damgalı sürümdeki "~4×"
düşüş burada medyan 2.5–3.1× (fold aralığı 1.5–4.7×). Makalede sayı bu dosyadan
verilmeli; zaman damgalı sürümün 4× rakamı Ek A'ya aittir.

Bu kontrol birincil spesifikasyonu değiştirmez; birincil BiLSTM 60/40/30 epoch'luk
ilan edilmiş kademe kuralıyla kalır.

Çıktılar: `bilstm_fixed200_folds_publication_aligned.csv`,
`bilstm_fixed200_loss_history_publication_aligned.csv`,
`bilstm_fixed200_predictions_publication_aligned.csv`,
`bilstm_fixed200_summary_publication_aligned.json`.

---

# Aşama 17: Sayı paketi (`scripts/18_paper_numbers.py`)

`outputs/paper_numbers_publication_aligned.md`, makalede ve kök README'de kullanılan tüm
sayıların tek kaynağı. Yalnızca kayıtlı çıktılardan üretilir; model eğitilmez. Başlığında
üretildiği commit, üretim tarihi ve çalışma ağacının temiz olup olmadığı yazar.

**Tutarlılık kontrolleri (assert):**
- Her fold'un RMSE'si tahmin dosyalarından yeniden hesaplanıp kayıtlı metrikle
  karşılaştırılır (iki sürümde 960 ve 1020 satır).
- Fold ortalamaları `gpr_alignment_comparison.csv` ile aynıdır.
- Train-mean R²_oos her fold'da tam 0'dır.
- 08'in kayıtlı BH değerleri pakette yeniden üretilir.

**Bulgu 1: R²_oos referansı kaynaklar arasında karışıyordu.** `hybrid_metrics_all`,
R²_oos'u XGBoost train penceresinin hedef ortalamasına göre hesaplıyor. `bench`,
`ablation`, `exploratory_xgb6` ve `opt_*` ise her modelin kendi train penceresine göre
hesaplıyor. HAR ailesi satır 21'den, XGBoost 127'den başladığı için bu referanslar farklı.
Aynı tabloda ikisi karışınca, modeller farklı sabit tahminlere karşı ölçülmüş oluyordu.

- **Düzeltme:** pakette tüm modellerin R²_oos'u fold başına tek bir ortak referansla
  (train-mean baseline'ının tahmini) tahminlerden yeniden hesaplanıyor. Hibrit kaynaklı
  modellerde kayıtlı değerle aynı çıkıyor (assert).
- **Etkisi:** tek işaret değişimi XGBoost-6'da, h=126'da +0.010 → −0.018. Diğer
  değişimler en fazla ~0.09, işaret değiştirmiyor.
- RMSE/MAE bundan etkilenmiyor. Kaynak dosyalar değiştirilmedi; pakette ayrı notla
  veriliyor.

**Bulgu 2: h=22'deki iki işaret testi bağımsız değil; Benjamini-Yekutieli eklendi.**
Birincil ailede BH altında ayakta kalan iki test (h=22: HAR-X > HAR ve HAR-X > XGBoost,
ikisi de 13/15, ham p 0.007) incelendi.

- **Sütun yeniden kullanımı yok.** İki fold farkı vektörü farklı (assert). HAR-X'in
  kaybettiği yıllar HAR'a karşı 2020 ve 2024, XGBoost'a karşı 2014 ve 2020. Beraberlik
  yok. Aynı p değeri, binom testinin yalnızca kazanma sayısına bağlı olmasından geliyor.
- **Ancak iki vektör 0.94 korelasyonlu.** İkisi de HAR-X'i içeriyor ve 2020 ortak kayıp
  yılı.
- **BY eklendi.** BH'nin FDR garantisi pozitif bağımlılık (PRDS) varsayımına dayanıyor.
  BY her bağımlılık yapısında geçerli: BH × c(m), c(8) = 2.718. Pakette Holm, BH ve BY
  yan yana, iki aile için de veriliyor.

**Birincil aile sonucu (yayım-hizalı):**
- **Holm:** hiçbir test ayakta kalmıyor.
- **BH:** h=22'de iki hipotez reddediliyor (p = 0.030). Bunlar iki ayrı hipotez, ama
  birbirinden bağımsız iki kanıt değil.
- **BY:** hiçbir test ayakta kalmıyor (en küçük p = 0.080).
- **DM:** hiçbir düzeltmede anlamlı değil.

İkincil ailede BY altında ayakta kalan: DM 4/24, işaret 6/24. BH altında DM 6/24, işaret
8/24.

**Kök README yayım-hizalı sayılarla güncellendi.** README'deki her sayı paketten gelir
(README'nin ihtiyaç duyduğu ek sayılar paketin Bölüm 9'unda). Değişen iki ifade:
- BiLSTM artık "tüm ufuklarda en kötü model" değil, "en kötü naif olmayan model"; h=5'te
  train-mean ile başa baş.
- HAR+OVX vs HAR-X işaret testi h=22'de düzeltmesiz p = 0.035 veriyor. Keşifsel ve
  düzeltmesiz olarak yazıldı.

GPR ile ilgili eski paragraf ("bir günlük gecikme varsayımı GPR lehine") kaldırıldı;
yerine ölçülmüş yayım kuralı ve as-of hizalama anlatıldı.

**Kapsam notu:** `09_power_analysis.py`'nin `--gpr-alignment` seçeneği yok; hâlâ zaman
damgalı DM sonuçlarını okuyor. Yayım modunda yeniden koşulmadı.

---

# Aşama 18 (2026-09-27): `--gpr-alignment` zorunlu; güç analizi yayım modunda

## Bayrak artık zorunlu — köken notu

**Bu tarihten önceki komutların kaydı olduğu gibi bırakıldı; geçmişe dönük bayrak
eklenmedi.** Günlük belge değil, köken kaydıdır.
- **Aşama 1–15:** komutlar bayraksız koşuldu. O dönemde bayrak yoktu; davranış,
  bugünkü `--gpr-alignment timestamp` ile aynıydı.
- **Aşama 16.3'ten 2026-09-27'ye kadar:** bayrak vardı ama isteğe bağlıydı, varsayılanı
  `timestamp` idi. `_publication_aligned` sonekli her çıktı, bayrak açıkça
  `publication` verilerek üretildi; sonek yalnızca bu şekilde oluşur. Bu dönemde
  bayraksız bir komut sessizce zaman damgalı sürümü üretirdi.
- **2026-09-27'den itibaren** (bu aşamanın commit'i): bayrak **zorunlu**, varsayılanı
  yok (`scripts/alignment.py`, `required=True`). Bayraksız çağrı hata verip durur ve her
  komutta bayrak açıkça yazılır. Kök README'deki çalıştırma talimatları bu biçime
  güncellendi.

**Gerekçe.** Birincil spesifikasyon `publication`. Varsayılan `timestamp` iken bayrağı
vermeyen biri ikincil sürümü üretir ve bunu fark etmez. Depo Zenodo üzerinden atıf
alacağı için bu sessiz yol kapatıldı.

**Risk kontrolü.**
- Argümanlar yalnızca `main()` içinde ayrıştırılıyor.
- 13, 15 ve 19 başka script'leri modül olarak içe aktarıyor, `main()`'lerini çağırmıyor.
- 17 ve 18 bayrak almıyor, iki sürümü kendileri okuyor.
- Depoda kabuk betiği veya CI yok.

**Sistematik tarama.** Model, metrik veya DM çıktısı okuyan script'lerin hepsinde
girdilerin `alignment.out()` / `features_path()` üzerinden okunduğu tek tek kontrol
edildi. Eksik olan yalnızca `09_power_analysis.py` idi. Bayrak gerekmeyenler:
- 01 (hedefler) ve 14 (tarih boşlukları): GPR'dan bağımsız.
- 02: iki özellik dosyasını birden üretir.
- 16: sürüm arşivi.
- `validate_data`.

## Güç analizi yayım modunda (`09_power_analysis.py --gpr-alignment publication`)

09'a bayrak eklendi. Yayım modunda okuduğu girdiler:
- DM sonuçları, yani HAC şişme çarpanları dahil 08'in yayım-hizalı çıktısı;
- tahmin dosyası.

Önsel DM eğrilerinin kalibrasyon katsayısı k bu DM sonuçlarından yeniden hesaplandı.

**Kırılmama kontrolü:** `--gpr-alignment timestamp` koşusu kayıtlı üç CSV'yi git blob
düzeyinde birebir yeniden üretti. JSON'da farklı olan yalnızca `gpr_alignment` (yeni
alan) ve `runtime_seconds`.

**İki sürüm karşılaştırması (Ek A):**

| ufuk | karşılaştırma | DM gerekli yıl (z.d. → yayım) | işaret gerekli yıl (z.d. → yayım) |
| --- | --- | --- | --- |
| 5 | HAR vs HAR-X | 927 → 760 | 20 → 37 |
| 22 | HAR vs HAR-X | 273 → 320 | 15 → 15 |
| 66 | HAR vs HAR-X | 2 764 → 1 680 | 42 → 42 |
| 126 | HAR vs HAR-X | 400 → 307 | 94 → 94 |
| 5 | HAR-X vs XGBoost | 323 → 233 | 72 → 72 |
| 22 | HAR-X vs XGBoost | 61 → 144 | 15 → 15 |
| 66 | HAR-X vs XGBoost | 52 → 45 | 25 → 42 |
| 126 | HAR-X vs XGBoost | 5 582 → 4 639 | 42 → 94 |

- **Önsel işaret testi eğrileri birebir aynı.** Yalnızca fold sayısına bağlılar. Anlamlılık
  için 15 fold'da en az 12, 14 fold'da en az 12 kazanma gerekiyor.
- **Önsel DM eğrileri k üzerinden hafifçe kaydı.** k: h=5 0.444 → 0.437, h=22 0.557 →
  0.542, h=66 1.124 → 1.137, h=126 1.700 → 1.688. %20 RMSE farkında güç h=5, 22, 66 ve
  126 için 0.989, 0.710, 0.846 ve 0.895 (zaman damgalı: 0.991, 0.733, 0.838, 0.899).
- **Tespit sınırı değişmedi.** DM için yaklaşık %20 RMSE farkı (h=22'de %20 bile 0.71),
  işaret testi için yılların %85–90'ını kazanmak.
- **Gözlenen etkiye dayalı gereksinimler, etkiler değiştiği ölçüde değişti.** DM için
  gereken uzatma mevcut dönemin 3.1–324 katı (zaman damgalı: 3.6–389). İşaret testi için
  1.0–6.7 katı (aynı aralık). h=22 HAR-X vs XGBoost'ta DM gereksinimi 61'den 144 yıla
  çıktı, çünkü o testte DM istatistiği 1.38'den 0.90'a düştü.
- **HAC şişme çarpanları** (08, birincil aile) iki sürümde neredeyse aynı: h=5 ~2.4,
  h=22 ~3.8–4.0, h=66 ~4.9–5.5, h=126 ~6.8–12.7.

**Yorum notu.** Aşama 8'deki güç bölümü, h=22'deki işaret testlerinin 0.871'lik
"gerçekleşen gücünü" o sonuçların "tesadüf olmadığını" destekleyen bir kanıt gibi
yorumluyor. Bu çıkarım geçerli değil. Gerçekleşen güç p değerinin monoton bir
dönüşümüdür ve bölüm de bunu kendisi söylüyor. Eski metin köken kaydı olarak bırakıldı.
Makalede ve sayı paketinde (Bölüm 10) bu çıkarım kullanılmaz. Bilgi taşıyan kısımlar
gerekli örneklem ve önsel eğrilerdir.

Çıktılar: `power_analysis_publication_aligned.csv`,
`apriori_power_sign_publication_aligned.csv`, `apriori_power_dm_publication_aligned.csv`,
`power_analysis_summary_publication_aligned.json`. Sayı paketi Bölüm 10.

## Düzeltme notu (2026-09-27, Aşama 18 sonrası)

Sayı paketine yöntem bölümü (Bölüm 11) eklenirken bu günlükteki iki ifadenin yanlış
olduğu görüldü. Özgün metinler köken kaydı olarak yerinde bırakıldı; doğrusu burada.

1. **16.1, yayım kuralı istisnaları.** "Bunların hepsi ay başı güncellemeleri" ifadesi
   yanlış. 10 istisnanın dökümü:
   - 6'sı ay başında yayımlanıp bir önceki ayın son gününde duran dosya.
   - 1'i bayat yükleme (2023-01-02 dosyası, son gözlem 2022-08-31).
   - 3'ü diğer: 2023-11-01 dosyası (son gözlem 2023-10-30), 2024-03-12 dosyası (son
     gözlem 2024-03-11), 2025-12-02 dosyası (son gözlem 2025-12-01).

   Kural desteği (279/289) değişmiyor. Kök README'deki aynı ifade düzeltildi.
2. **16.5 ve paketin eski sürümleri: "birincil aile önceden sabitlendi".** Bu ifade
   Aşama 8'in kaydıyla çelişiyor. Aile **testlerden sonra resmileştirildi ve ön-kayıt
   değildir**. Ancak p değerlerine bakılarak değil, Aşama 5–6'da ilan edilmiş iki iddiaya
   göre seçildi. O tarihten beri sabittir ve sonradan test eklenmez. Sayı paketi bu
   ifadeyle düzeltildi.

## Açıklama notu (2026-09-27): inceleme ifadesi, 7b ve fold RMSE korelasyonu

1. **"Bağımsız bir dış kod incelemesi" (16. aşama girişi).** "Bağımsız" üçüncü taraf bir
   insan incelemesini ima ediyor. Gerçekte bu, yazarın çalıştırdığı üçüncü taraf bir AI
   asistanıyla (GPT tabanlı bir araç) yapılmış harici bir kod denetimiydi. Özgün ifade
   köken kaydı olarak bırakıldı. Kök README "an external code audit performed with a
   third-party AI assistant, run by the author" olarak düzeltildi; makalenin AI kullanım
   beyanı aynı şeyi söylemeli.
2. **7b volatilite rejimi analizi post hoc'tur.**
   `scripts/07b_exploratory_vol_regime.py` docstring'i, Aşama 7'de ölçülen çelişkiyi
   (H2 havuzda HAR-X'ten iyi, fold ortalamasında kötü) alıntılıyor ve analizin amacını
   onu açıklamak olarak tanımlıyor. Yani hibrit sonuçları görüldükten sonra tasarlandı.
   Bölme kriteri (yılın ortalama gerçekleşen volatilitesinin medyanı) mekanik; performansa
   bakılarak seçilmedi. Git geçmişi zamanlamayı göstermiyor, çünkü iki script de ilk
   commit'te birlikte geliyor; kanıt metnin içeriği.
3. **HAR ile XGBoost'un fold RMSE korelasyonu (0.995, h=22) kendi başına bir bulgu
   değildir.** Fold RMSE yılın volatilite düzeyiyle ölçeklenir. h=22'de HAR ile naif
   past-volatility baseline'ı da 0.996 korelasyonlu. Değer yalnızca h=22 için geçerli
   (h=5: 0.981, h=66: 0.932, h=126: 0.931).

   İki h=22 işaret testinin neden bağımlı olduğunu açıklayan mekanizma olarak geçerli,
   ama "XGBoost HAR'ın bulmadığı bir şey bulmuyor" iddiasına kanıt değil. Ölçekten
   arındırılmış karşılaştırmalar sayı paketinde (Bölüm 11d); hepsi keşifsel.

---

# Aşama 19 (2026-09-29): Taban sıklığı, QLIKE ve fold başına smearing

**Amaç.** Makalenin Yöntem bölümündeki iki [PENDING] işaretini kapatmak. Yeniden eğitim
yok; her şey kayıtlı tahmin ve fold dosyalarından, yayım-hizalı sürüm. Kod:
`scripts/18_paper_numbers.py`, Bölüm 12 (commit `e7f308d`). Sayılar paketin Bölüm 12'sinde.
Burada yalnızca özet var.

## 19.1 HAR tahmin tabanının devreye girme sıklığı (paket 12a)

**Tespit yöntemi.** Tahmin dosyalarında taban işareti yok. Tabanlanmış satır, tahminin
fold'un kayıtlı tabanına (`pred_floor`, eğitim hedefinin minimumu) tam eşit olduğu satır
olarak tespit edildi. Üç kontrol yapıldı (assert):
- Fold başına eşitlik sayısı, uyum anında kaydedilen sayaçla birebir aynı
  (`n_clipped_har`, `n_clipped_har_x`, ablasyon `n_clipped`, H3 `n_floored`).
- Tabansız modellerde (HAR-log, HAR-X-log, GARCH, past-vol) tabana tam eşit tahmin sayısı
  0. HAR-log 18, HAR-X-log 100 satırda tabanın altında tahmin veriyor.
- Tabanlanmamış en yakın tahmin tabanın 3.8e−06 (göreli %0.04) üstünde.

Eşitlik tesadüfen oluşmuyor.

Ana fold'larda tabanlanan test gözlemi (h=5 / 22 / 66 / 126):

| model | sayı | oran | en yoğun fold |
| --- | --- | --- | --- |
| HAR | 0 / 0 / 0 / 0 | %0 | — |
| HAR + OVX | 0 / 22 / 46 / 2 | %0 / 0.60 / 1.31 / 0.06 | 2013, h=66'da %10.3 |
| HAR + GPR | 0 / 10 / 17 / 4 | %0 / 0.27 / 0.49 / 0.11 | 2014, h=22'de %2.8 |
| HAR-X | 0 / 70 / 106 / 21 | %0 / 1.92 / 3.03 / 0.60 | 2014, h=66'da %20.8 |
| HAR-log, HAR-X-log | taban yok | — | — |
| Hibrit H3 (ek) | 8 / 9 / 58 / 205 | %0.22 / 0.25 / 1.66 / 5.86 | 2014, h=126'da %44.8 |

## 19.2 QLIKE (paket 12b–12c)

`QLIKE = σ²/σ̂² − log(σ²/σ̂²) − 1`, varyans ölçeğinde (Patton 2011). Tüm modeller, dört
ufuk, fold ortalaması (birincil) ve havuzlanmış (ikincil).

**Kayıtlı seçim:** tabanlanan gözlemlerde QLIKE yayımlanan (tabanlanmış) tahmin üzerinden
hesaplandı. Değerlendirilen şey verilen tahmin. Pozitif olmayan tahmin veya hedef yok,
durdurma koşulu tetiklenmedi. **Statü: yalnızca betimleyici.** QLIKE kaybıyla DM veya
işaret testi koşulmadı; birincil aile 8 testle sabit.

**Sıralama RMSE'den farklı.** Kendall τ (17 model, fold ortalaması) h=5'te 0.632, h=22'de
0.765, h=66'da 0.662, h=126'da 0.603. Sırası değişen model sayısı 13, 12, 11 ve 14.

En iyi model:
- h=5: RMSE'de HAR+OVX, QLIKE'ta HAR-X-log.
- h=22: RMSE'de HAR+OVX, QLIKE'ta HAR-X-log.
- h=66: ikisinde de HAR+OVX.
- h=126: RMSE'de HAR-X-log, QLIKE'ta GARCH.

En büyük kaymalar:
- **GARCH** QLIKE'ta her ufukta yükseliyor: 12→2, 11→7, 11→4, 13→1.
- **HAR-X** h=5'te 3→9 düşüyor.
- **Past-volatility** h=66'da 9→16 düşüyor.

**Birincil ailenin iki karşılaştırmasında yön** (`100 × (a/b − 1)`, pozitif = a daha kötü;
fold ortalaması):

| a vs b | RMSE | QLIKE |
| --- | --- | --- |
| HAR vs HAR-X | +4.75 / +11.59 / +6.10 / +1.43% | **−12.59** / +25.62 / +6.90 / +0.69% |
| XGBoost vs HAR-X | +5.93 / +15.10 / +18.21 / +7.26% | +10.81 / +25.28 / +28.52 / +8.30% |

h=5'te HAR ile HAR-X arasındaki yön kayba göre değişiyor: RMSE'de HAR-X, QLIKE'ta HAR daha
iyi. Diğer yedi hücrede yön aynı.

**Yoğunlaşma.** h=5'te HAR-X'in havuzlanmış QLIKE'ının %30.0'ı satırların en büyük
%1'inden geliyor. En büyük tek satır 27.11.2013, σ/σ̂ = 11.2. HAR'da aynı pay %21.6,
HAR-X-log'da %19.0. QLIKE eksik tahmini sert cezalandırdığı için h=5 HAR-X sonucu birkaç
eksik tahmin gözlemine dayanıyor. Tabanlanmış satırların QLIKE payı:
- HAR-X'te küçük: en fazla %4.4, satırların %1.9'u.
- H3'te h=5'te büyük: satırların %0.22'si, QLIKE'ın %27.4'ü.

## 19.3 Fold başına smearing ve log-artık std'si (paket 12d)

Pakette daha önce yoktu. Kaynaklar:
- XGBoost: `wf_summary_all` (`smearing`, `resid_log_std`)
- BiLSTM: `bilstm_folds_all`
- HAR-log ve HAR-X-log: `bench_folds_all` (`har_smearing`, `har_x_smearing`)

Ana fold'larda medyan smearing (h=5 / 22 / 66 / 126):

| model | medyan smearing |
| --- | --- |
| XGBoost | 1.018 / 1.004 / 1.017 / 1.024 |
| BiLSTM | 1.035 / 1.005 / 1.011 / 1.027 |
| HAR-log | 1.130 / 1.055 / 1.049 / 1.058 |
| HAR-X-log | 1.111 / 1.039 / 1.039 / 1.052 |

XGBoost log-artık std'si medyan 0.190 / 0.097 / 0.181 / 0.204.

**Kapsam boşluğu.** Log-artık std'si yalnızca XGBoost için kayıtlı. 05 ve 06 yalnızca
katsayıyı kaydediyor. HAR-log ve HAR-X-log için OLS'nin yeniden tahmini, BiLSTM için
yeniden eğitim gerekir. Bu aşamada yapılmadı; pakette "—" olarak gösteriliyor.

---

# Aşama 20 (2026-09-29): Log-artık std'si ve h=5 HAR-X'in en büyük QLIKE satırı

Aşama 19'un iki açık ucu. Kod: `scripts/20_log_residual_std.py` (yeni) ve
`scripts/18_paper_numbers.py`, Bölüm 12c–12d. Commit `7e4084d`.

## 20.1 HAR-log ve HAR-X-log log-artık std'si

05 bu iki model için yalnızca smearing katsayısını kaydediyor. `20_log_residual_std.py
--gpr-alignment publication` OLS'yi yeniden tahmin ediyor. Veri hazırlığı, fold, embargo,
NaN maskeleri, `LOG_FLOOR` ve `ols_fit` 05'ten import ediliyor, kopyalanmıyor.

**Önce eşitlik kontrolü (assert):** her fold'da yeniden tahmin, kayıtlı test tahminlerini
(`pred_har_log`, `pred_har_x_log`) ve smearing katsayısını (`har_smearing`,
`har_x_smearing`) **bit düzeyinde** üretti: 120/120 model × ufuk × fold. Uyuşmazlık olsaydı
script dururdu.

Std ddof=1 ile hesaplandı (03'teki `resid_log_std` gibi). Ana fold'larda medyan (h=5 / 22 /
66 / 126):

| model | medyan log-artık std |
| --- | --- |
| HAR-log | 0.500 / 0.323 / 0.303 / 0.327 |
| HAR-X-log | 0.466 / 0.277 / 0.274 / 0.309 |

Bu değerler XGBoost'un 0.190 / 0.097 / 0.181 / 0.204'üyle doğrudan karşılaştırılamaz:
XGBoost'un hedefi `log(σ_h / past_vol_h)`, HAR-log ailesininki `log(σ_h)`.

## 20.2 BiLSTM log-artık std'si: kaydedilmedi

06 eğitilmiş ağırlıkları saklamıyor. Script'lerde `torch.save` yok, depoda
`.pt/.pth/.ckpt/.pkl` dosyası yok. Hesaplamak yeniden eğitim gerektirir; yapılmadı. Pakette
"kaydedilmedi" olarak duruyor.

## 20.3 h=5 HAR-X'in en büyük tek QLIKE satırı (27.11.2013)

27.11.2013 satırın kendi tarihi t'dir, yani **tahmin kökeni**; hedef penceresinin başı
değildir.
- **Özellikler:** `.shift(1)` ile t−1'e (26.11.2013) kadarki bilgiyi kullanır.
- **Hedef:** `std(r_{t+1}, …, r_{t+5})`. Pencere t'den sonraki beş işlem günü: 29.11.2013,
  02.12.2013, 03.12.2013, 04.12.2013, 05.12.2013. İlk getiri 27.11.2013 kapanışından
  29.11.2013 kapanışına; veri setinde 28.11.2013 satırı yok. t günü getirisi ne
  özelliklerde ne hedefte yer alır.

Sayılar:
- Gerçekleşen σ = 0.013080, HAR-X tahmini σ̂ = 0.001172. σ/σ̂ = 11.2, QLIKE = 118.7.
- Fold tabanı 0.001120; tahmin tabanın %4.67 üstünde, tabanlanmamış.
- Hedef, ham fiyatlardan yeniden hesaplanıp kayıtlı değerle karşılaştırıldı (assert).

---

# Aşama 21 (2026-09-29): h=5 QLIKE yön dönmesi — keşifsel kontroller

**Statü: keşifsel ve post hoc.** İki kontrol de Aşama 19'daki QLIKE sonucu (h=5'te HAR vs
HAR-X yönünün RMSE'ye göre ters dönmesi) görüldükten sonra tasarlandı. Eşikler (1.25 ×
taban, 0.5 × σ̂_HAR) sonuçlara bakılarak seçildi. Test yok. Kod `scripts/18_paper_numbers.py`,
Bölüm 12e (commit `f544558`). Hesaplar ana fold'lar üzerinden, havuzlanmış.

## 21.1 Tabana yakınlık (h=5)

h=5'te fold tabanı her fold'da 0.00111966.

| küme | n | HAR-X σ̂/taban, medyan (min–maks) | HAR-X ≤ 1.25 × taban | HAR σ̂/taban, medyan (min–maks) | HAR ≤ 1.25 × taban |
| --- | --- | --- | --- | --- | --- |
| HAR-X'in en büyük %1 QLIKE satırı | 36 | 3.77 (1.05–25.94) | 2 | 11.02 (5.35–20.89) | 0 |
| bütün h=5 test gözlemleri | 3662 | 15.64 (1.05–182.79) | 2 | 14.93 (2.79–77.25) | 0 |

- En büyük %1 satır, HAR-X'in h=5 QLIKE toplamının %30.0'ını oluşturuyor.
- σ̂ ≤ 1.25 × taban olan 2 satır (27.11.2013 ve 25.07.2014) toplamın %5.5'ini oluşturuyor.
- Bu 2 satır hariç ortalama QLIKE: HAR-X 0.6292, HAR 0.5751. Tüm satırlarla 0.6654 ve
  0.5749.

## 21.2 Mekanik kural: σ̂_HAR-X < 0.5 × σ̂_HAR ("HAR-X belirgin düşük")

"Pay", kural satırlarının Σ(QLIKE_HAR-X − QLIKE_HAR) içindeki payı. Toplam fark
negatifse pay işaretiyle okunur.

| ufuk | kural satırı | yıllar | Σ fark, tümü | Σ fark, kural satırları | pay | hariç ort. QLIKE, havuz (HAR-X / HAR) | hariç, fold ort. (HAR-X / HAR) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 82 / 3662 | 2013 (46), 2014 (34), 2017 (2) | +331.56 | +580.01 | %174.9 | 0.5072 / 0.5766 | 0.5094 / 0.5777 |
| h=22 | 9 / 3645 | 2014 (9) | −238.44 | +4.63 | %−1.9 | 0.2825 / 0.3494 | 0.2837 / 0.3576 |
| h=66 | 0 / 3500 | — | −86.00 | 0 | — | 0.3602 / 0.3847 | 0.3581 / 0.3828 |
| h=126 | 0 / 3500 | — | −9.10 | 0 | — | 0.4005 / 0.4031 | 0.3992 / 0.4020 |

h=5'te kural satırları çıkarıldığında ortalama QLIKE'ta HAR-X daha düşük (havuzda ve fold
ortalamasında). Kural h=66 ve h=126'da hiçbir satırda tetiklenmiyor.

---

# Aşama 22 (2026-09-29): Clark–West ek ailesi; XGB-6 vs HAR fold sayımları; 2026 dipnotu

Kod: `scripts/21_clark_west.py` (yeni) ve `scripts/18_paper_numbers.py`, Bölüm 13. Commit
`3bb6ead`. Hepsi `--gpr-alignment publication`.

## 22.1 Clark–West, HAR ⊂ HAR-X (ek aile, 4 test)

**Aile kuralı.** Birincil aile 8 testle sabit kalıyor. CW oraya eklenmedi ve DM
testlerinin yerine geçmiyor. CW ayrı, 4 testlik bir ek aile; Holm, BH ve BY kendi içinde.

**Etiket:** "iç içe yapıya uygun istatistik; birincil DM testleri görüldükten sonra, CW
sonuçları görülmeden eklendi".

**Tarih kaydı.** Aile, CLAUDE.md'de commit `72106e7` ile ilan edildi (2026-09-29
16:15:34 +0300). Depoda bundan önce hiçbir CW hesabı yok: script'lerde, çıktılarda ve
commit mesajlarında "clark" araması yalnızca bu ilanı buluyor.

**Kurulum.**
- `f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`, tek yanlı, H1: HAR-X daha iyi.
- Kayıtlı yayımlanan (tabanlanmış) tahminler kullanıldı: `hybrid_predictions_all`, ana
  fold'lar, havuzlanmış seri, tarihe göre sıralı.
- HAC: Newey-West, Bartlett, L = h−1. Fonksiyonlar 08'den import edildi.
- Çıkarım birincil DM ailesiyle aynı: HLN çarpanı ve t(n−1). Holm, BH ve BY, HLN p
  değerine uygulandı.

| ufuk | n | ort. (e²_HAR − e²_HARX) | ort. düzeltme | CW (HLN) | p HLN (ham) | Holm | BH | BY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h=5 | 3662 | 4.49e−06 | 3.02e−05 | 5.336 | <0.001 | <0.001 | <0.001 | <0.001 |
| h=22 | 3645 | 7.89e−06 | 2.60e−05 | 3.587 | <0.001 | <0.001 | <0.001 | <0.001 |
| h=66 | 3500 | −3.24e−06 | 1.80e−05 | 2.975 | 0.001 | 0.003 | 0.002 | 0.004 |
| h=126 | 3500 | −5.56e−06 | 1.10e−05 | 1.194 | 0.116 | 0.116 | 0.116 | 0.242 |

Kesin değerler `clark_west_publication_aligned.csv`'de.
- CW, h=5, h=22 ve h=66'da üç düzeltmenin üçünde de H0'ı reddediyor; h=126'da
  reddetmiyor.
- h=66'da havuzlanmış ham MSE farkı negatif (HAR'ın MSE'si daha düşük); CW buna karşın
  reddediyor.
- Fold düzeyinde f ortalaması pozitif olan fold sayısı (betimleyici): 15/15, 15/15,
  12/14, 10/14.

## 22.2 XGBoost-6 vs HAR, fold bazında (keşifsel; p değeri yok)

| ufuk | XGB-6 kazandığı yıl / fold | fold ort. farkı (XGB-6 / HAR − 1) | sayım ile ortalama yönü |
| --- | --- | --- | --- |
| h=5 | 5/15 | +0.03% | uyuşuyor (ikisi de HAR) |
| h=22 | 8/15 | −2.23% | uyuşuyor (XGB-6) |
| h=66 | 8/14 | −0.89% | uyuşuyor (XGB-6) |
| h=126 | 8/14 | −0.79% | uyuşuyor (XGB-6) |

**Aşama 15 ile ilişkisi.** Aşama 15'teki "XGBoost-6, düz HAR'ı dört ufukta da geçiyor"
ifadesi zaman damgalı sürüme ait (h=5: 0.010645 vs 0.010834). Yayım-hizalı sürümde h=5'te
XGB-6 fold ortalamasında HAR'dan %0.03 kötü (0.010838 vs 0.010834). Kazandığı yıl 5/15.
İfade yayım-hizalı sürüm için h=5'te geçerli değil. Aşama 15'in metni köken kaydı olarak
olduğu gibi bırakıldı.

## 22.3 2026 kısmi yıl, h=66 ve h=126

Pakette zaten var: Bölüm 1e. Tüm modellerin o fold'daki RMSE, MAE ve R²_oos değerleri;
n = 101 (h=66) ve 41 (h=126); birincil toplulaştırmadan dışlanmış. Yeni hesap yapılmadı;
Bölüm 13c yalnızca oraya işaret ediyor.

---

# Aşama 23 (2026-09-29): Tablo 1 ve veri doğrulamanın sertleştirilmesi (dış inceleme, bulgu 7)

Commit'ler: `664a8fb` (doğrulama ve SHA-256), `62a65a4` (Tablo 1). Sayılar paketin
Bölüm 14'ünde.

## 23.1 Tablo 1: Tanımlayıcı istatistikler (`scripts/22_descriptive_stats.py`)

**Değişkenler:** Brent günlük log getirisi, dört hedef, OVX, GPRD ve GPRD_THREAT.

**Örneklem:** modelin örneklemi, 4641 satır, işlem takvimi.
- GPR serileri **modelin gördüğü haliyle**, yani yayım-hizalı `gprd_lag1` ve
  `gprd_threat_lag1`. N = 4637; ilk yayımdan önceki 4 satır tanımsız.
- Getiri N = 4640; hedefler N = 4641 − h.

**İstatistikler:** N, ortalama, std, min, maks, çarpıklık, fazla basıklık, ADF (sabitli,
AIC gecikmesi) ve Ljung–Box Q(20). Tüm serilerde ADF birim kökü ve Ljung–Box
otokorelasyonsuzluğu reddediyor (p < 0.001).

**Hedeflerde Ljung–Box reddi mekanik.** Ardışık hedefler h getirinin h−1'ini paylaşan
örtüşen pencerelerden hesaplanıyor. Tablo notunda bu açıkça yazılı; red kalıcılık kanıtı
olarak sunulmuyor. Yayım-hizalı GPR iki yayım arasında sabit kaldığı için onun
otokorelasyonu da kısmen yapıdan geliyor. Q(20) değeri 23 236; gözlem tarihli seride
22 379.

**GPR'ın kendi takvimi belirgin farklı; dipnota alındı.** Kaynak, tüm takvim günleri
(6818 gün, hafta sonları dahil). Yayım-hizalı model girdisiyle karşılaştırma:

| ölçü | kendi takvimi | yayım-hizalı model girdisi |
| --- | --- | --- |
| GPRD ortalaması | 103.4 | 114.0 |
| min | 0 | 24.8 |
| fazla basıklık | 7.6 | 11.1 |

Dipnottaki diğer satırlar gözlem tarihli işlem günü serileri: GPRD ortalaması 115.1,
fazla basıklık 8.1.

Kendi takvimi serisinin kaynağı yerel sürüm dosyası
`data_gpr_daily_recent_accessed_2026-09-24.dta`. Dosya git dışında; SHA-256'sı JSON'da
kayıtlı. Bu sürüm veri setinin eşleştiği 2026-09-01 sürümünden yeni: 44 işlem günü değeri
farklı (2025-06-02 – 2026-09-01), en büyük mutlak fark 33.8. Paket notunda yazılı.

`statsmodels` artık doğrudan import ediliyor; `requirements.txt` notu güncellendi.

## 23.2 `validate_data.py` hatada duruyor (dış inceleme, bulgu 7)

Önceden script hiçbir koşulda sıfır dışı çıkış vermiyordu. Şimdi şu kritik koşullarda
raporu yazıp **çıkış kodu 1** ile duruyor:
- beklenen kolon eksik;
- zorunlu kolonda (beşi de) eksik değer;
- `DD.MM.YYYY` olarak ayrıştırılamayan tarih;
- artan sırası bozuk tarih;
- yinelenen tarih.

**Belgelenmiş 40 tarih boşluğu hata değil**; raporlanıyor, doğrulamayı düşürmüyor.

Uyarı olarak kalanlar:
- fazladan veya sırası farklı kolon;
- 4641'den farklı satır sayısı;
- değişmiş OVX rekor değeri;
- SHA-256 uyuşmazlığı veya kayıtlı özetin bulunmaması.

**SHA-256.** `data/veriseti.xlsx` özeti
`f13956e7d0eef3dfa49dee1ac83e098f0d1cea872774eb2c72f3fe33661ebd26`. Özet
`data/veriseti.xlsx.sha256` dosyasında, `sha256sum -c` biçiminde; git dışı bırakılmadı.
README'nin veri bölümünde ve `data/README.md`'de de var. Özet dosya baytları üzerinden
hesaplandığı için aynı değerleri başka bir programla kaydedilmiş bir dosya eşleşmez. Bu
yüzden uyuşmazlık hata değil uyarı.

**Mevcut veriyle sonuç:** GEÇTİ, çıkış kodu 0, uyarı yok, SHA-256 eşleşiyor. Tarih
boşluğu ve ufuk CSV'leri değişmedi; rapor JSON'una `errors`, `warnings` ve `sha256`
alanları eklendi.

**Hata yollarının testi.** Scratchpad'deki bozuk kopyalarla denendi:

| senaryo | çıkış kodu |
| --- | --- |
| kolon eksik | 1 |
| GPRD'de eksik değer | 1 |
| iki satırın sırası değişmiş | 1 |
| yinelenen tarih | 1 |
| ISO biçimli tarih | 1 |
| 10 satır silinerek eklenen boşluk | 0 (satır sayısı ve SHA uyarısı) |
| yeniden kaydedilmiş özdeş kopya | 0 (SHA uyarısı) |

## Düzeltme notu (2026-09-29, Aşama 23 sonrası): Tablo 1 dipnotunun GPR sürümü

Aşama 23.1'de "kendi takvimi" satırları 2026-09-24 tarihli yerel dosyadan hesaplanmıştı.
O sürüm veri setinden 44 işlem gününde farklıydı. Özgün metin köken kaydı olarak yerinde
bırakıldı; doğrusu burada.

**Eşleşen sürüm.** Sürüm önbelleği (`data/gpr_vintages/vintages/`) 16'nın `--cleanup`
seçeneğiyle silinmişti; 289 sürüm yerelde yoktu. 16'nın kullandığı arşivden üç dosya
indirildi ve veri setiyle karşılaştırıldı:

| sürüm | eksik işlem günü | en büyük mutlak fark | sonuç |
| --- | --- | --- | --- |
| 2026-08-31 | 1 | 45.8 | eşleşmiyor |
| **2026-09-01** | 0 | **5.7e−14** | eşleşiyor (4641 işlem gününün hepsinde; kayan nokta yuvarlaması) |
| 2026-09-08 | 0 | 33.8 | eşleşmiyor |

Bu, 02'nin raporundaki "identical to the 2026-09-01 vintage" kaydıyla tutarlı.

Kendi takvimi satırları bu sürümden yeniden hesaplandı (script 22, commit `cb2166e`).
Script eşleşmeyi assert ediyor; uyuşmazlık uyarısı kaldırıldı. Değişen değerler:

| seri | ortalama | fazla basıklık |
| --- | --- | --- |
| GPRD | 103.4 → 103.5 | 7.64 → 7.60 |
| GPRD_THREAT | 112.1 → 112.2 | 10.46 → 10.42 |

**Sıfır değerli günler** (bu sürüm, 2008-01-02 – 2026-09-01):
- GPRD yalnızca 2025-02-09'da (pazar) 0; aynı gün GPRD_THREAT de 0.
- GPRD_THREAT 8 günde 0: 2009-04-19, 2016-08-21, 2019-05-12, 2020-08-30, 2023-07-30,
  2023-10-22, 2024-09-22, 2025-02-09. Hepsi pazar.

Pakette yalnızca tarihler yazılı.

---

# Aşama 24 (2026-09-29): Şekil 2 ve Ek A kaynak haritası

Commit `f6f2173`. Yeni hesap yok; kayıtlı çıktılardan.

**Şekil 2 (`scripts/23_figure2.py`).**
- **(a) Ana metin:** fold bazında RMSE oranı, dört ufuk, ana fold'lar.
  - Oran **XGBoost / HAR-X** olarak çizildi. İstemdeki eksen okuması ("1'in üstü XGBoost
    daha kötü") ancak bu sırayla doğru; HAR-X / XGBoost'ta 1'in üstü HAR-X'in daha kötü
    olduğu anlamına gelirdi. Sıra, %-fark konvansiyonuyla da uyumlu (a = XGBoost).
  - 1'de referans çizgisi var; eksen etiketi "(above 1: XGBoost worse)".
  - Oranın 1'in üstünde olduğu fold sayısı, birincil ailedeki HAR-X kazanma sayısıyla
    assert edildi: 10/15, 13/15, 10/14, 9/14.
  - h=66 ve h=126'da 2026 fold'u çizimde yok; CSV'de `include_in_main = False` ile
    duruyor.
- **(b) Ek:** SHAP grup payları; her grubun özellik sayısı etiketin yanında. Toplam 65
  özellik (assert).
- **Biçim:** gri tonlama, Arial, vektör PDF ve SVG (metin metin olarak kalıyor). Dosyalar
  koşudan koşuya bayt düzeyinde aynı; oluşturma tarihi gömülmüyor, SVG kimlik tuzu sabit.

**Ek A kaynak haritası** (`outputs/appendix_a_source_map.md`). Methodology v3'ün atıf
yaptığı A1, A3, A6 ve A8 maddeleri kaynak dosyaya, script'e ve paket bölümüne eşlendi.

| durum | maddeler |
| --- | --- |
| eksik | A6 roll-over (koşulmadı) |
| kaynak var ama pakette tablo yok | A1 özellik listesi; A6 Optuna seçim sinyali; A6 ham smearing korelasyonu; A6 07b volatilite rejimi; A8 eğitim uzunluğu asimetrisi |

Paketin §7b'si (BiLSTM yakınsama kontrolü) ile 07b script'i (volatilite rejimi) farklı
şeyler; haritada not düşüldü.

---

# Aşama 25 (2026-09-29): Roll-over sağlamlık analizi — KOŞUDAN ÖNCE yazılan tasarım ve statüler

**Bu bölüm koşudan önce yazıldı ve koşudan önce commit edildi.** Sonuçlar Aşama 25.2'de
olacak. Statüler, sonuç ne çıkarsa çıksın değişmeyecek. Üç varyantın üçü de sonuç ne
olursa olsun raporlanacak. Luo vd. (2024) tasarıma temel yapılmadı.

## 25.1 Tasarım

**Vade kuralı.** Kaynak ICE Brent sözleşme spesifikasyonu (circular 13165 Attach 6) ve
Circular 15/235.

| kontratlar | işlemin durduğu gün |
| --- | --- |
| Şubat 2016'ya kadar | Kontrat ayının ilk gününden 15 takvim günü önceki günden bir önceki iş günü. O gün iş günü değilse önce bir önceki iş gününe gidilir. |
| Mart 2016'dan itibaren | Kontrat ayından iki önceki ayın son iş günü. Noel veya Yılbaşı öncesi iş gününe denk gelirse bir önceki iş günü. |

- Kural örneklem içinde değişti. Mart 2016 kontratı 29.01.2016'da, Şubat 2016 kontratı
  14.01.2016'da vadelendi.
- İş günü: İngiltere ve Galler'de resmi tatil olmayan işlem günü.
- Kodlanan kural, ICE'ın resmi tablosundaki 88 vadeyi (Aralık 2015 – Mart 2023) birebir
  üretiyor.
- 2008–2015 için resmi tablo bulunamadı. Bu dönemin tarihleri kuraldan ve tatil
  takviminden türetildi.

**Veri kaynağı varsayımı.** Yahoo `BZ=F`, NYMEX'teki Brent Last Day Financial (BZ)
kontratı. Bu kontratın ICE vade takvimine uyduğu birincil CME belgesinden doğrulanmadı;
varsayım olarak kullanılıyor. Yahoo'nun sürekli seride hangi gün geçiş yaptığı
belgelenmemiş.

**Geçiş satırı.** Vade gününden sonraki ilk veri satırı. O satırın getirisi eski
kontratın kapanışından yeni kontratın kapanışına uzanıyor.

**Varyantlar ve statüleri:**

| varyant | statü | ne yapılıyor |
| --- | --- | --- |
| **A** | birincil sağlamlık varyantı | Geçiş satırının getirisi çıkarılır (NaN). Hedef, aynı h günlük penceredeki kalan getirilerin std'si; pencerenin tamamlanmış olma şartı korunur. Getiri özellikleri temiz getirilerle yeniden hesaplanır: `brent_ret_lag1-5`, `brent_vol5/20/60/126`, `vol_ratio`, `vol5_vol60`, `vol20_vol126`, HAR'ın `har_daily` terimi ve past-vol baseline (XGBoost'un oran paydası). Getiri gecikmeleri ve `har_daily`, son temiz getirileri kullanır. |
| **A′** | duyarlılık kontrolü | A ile aynı, ama vade sonrası iki satır çıkarılır. Geçiş gününün bir gün belirsiz olmasına karşı. |
| **B** | duyarlılık kontrolü, yalnızca h=5 | Getiri çıkarılmaz. Hedef penceresinde geçiş satırı bulunan satırlar örneklemden atılır; yaklaşık %24. |

**A ve tarih boşlukları.** A, vade günleriyle örtüşen tarih boşluğu satırlarının
getirisini de çıkarır. Belgelenmiş 40 boşluğun kaçının A'da (ve A′'de) çıkarılan
satırlarla örtüştüğü raporlanacak.

*Hipotez (kanıtlanmış değil):* boşlukların bir kısmı geçiş kaynaklı olabilir. Yahoo, vade
günü veya ertesi günü satırını atlıyor olabilir. Bu analiz yalnızca örtüşmeyi sayar,
nedeni göstermez.

**Sınırlılık.** Kontrol, hedefteki ve getiri özelliklerindeki geçiş etkisini temizliyor.
XGBoost'un fiyat düzeyi özelliklerindeki (`brent_lag1-5`, `brent_ema5/10/20`) etkiyi
temizlemiyor. Bu özellikler bilerek değiştirilmedi; o etkiyi temizlemek geri ayarlanmış
bir sürekli seri gerektirir ve verisi yok.

**Testler ve raporlama:**
- Birincil ailenin 8 testi her varyantta kendi Holm/BH/BY düzeltmeleriyle koşulacak
  (CLAUDE.md: varyant veri üzerindeki tekrarlar yeni aile değildir ve birincil aileyle
  havuzlanmaz). Kurulum 08 ile aynı: DM (HLN) ve işaret testi.
- Varyant içi model farkları raporlanacak.
- Hedef değiştiği için mutlak RMSE birincil sonuçla karşılaştırılmayacak.

**Modeller:** HAR, HAR-X, XGBoost; `--gpr-alignment publication`.

**Uygulama doğrulaması.** Script, maskesiz kurulumda şunları bit düzeyinde yeniden
üretmedikçe varyantları koşmaz:
- hedefleri, özellik dosyasını ve HAR, HAR-X, XGBoost tahminlerini;
- birincil ailenin kayıtlı test değerlerini.

Tek kod değişikliği 03'teki `run_horizon`'a eklenen isteğe bağlı `past_vol` parametresi.
Varsayılanı mevcut davranışı koruyor; birincil çıktılar yeniden üretilip değişmediği
kontrol edilecek.

## 25.2 Sonuçlar (2026-09-29, koşudan sonra)

Commit `a4d990f`; `scripts/24_rollover_robustness.py --gpr-alignment publication`, süre
93 sn. Sayılar paketin §15'inde. Statüler 25.1'deki gibi, değiştirilmedi.

**Doğrulama (assert):**
- Vade takvimi ICE'ın resmi tablosundaki 88 vadenin 88'ini üretiyor.
- Boş maskeyle hedefler, getiri özellikleri ve HAR, HAR-X, XGBoost tahminleri bit
  düzeyinde yeniden üretiliyor; 8 testin değerleri kayıtlı birincil değerlerle aynı.
- 03'teki `past_vol` parametresinin varsayılan yolu: 03 yayım modunda yeniden koşuldu;
  tahmin ve metrik dosyaları birebir aynı, yalnızca `runtime_seconds` değişti ve kayıtlı
  değer geri alındı.

**Örneklem:**
- 225 vade; 14 vade günü veride yok.
- A 225, A′ 450 getiri çıkarıyor.
- B, h=5'te 1122 satırı atıyor (penceresinde geçiş olan satırlar).

**Varyant içi farklar**, fold ortalaması RMSE, `100 × (a/b − 1)`, h=5 / 22 / 66 / 126.
Birincil sonuç yalnızca yüzdeler için referans olarak verildi; mutlak RMSE
karşılaştırılmıyor.

| varyant | HAR vs HAR-X | XGBoost vs HAR-X |
| --- | --- | --- |
| birincil (referans) | +4.75 / +11.59 / +6.10 / +1.43% | +5.93 / +15.10 / +18.21 / +7.26% |
| A | +4.88 / +12.10 / +5.86 / +1.13% | +5.83 / +18.80 / +14.48 / +8.56% |
| A′ | +5.19 / +12.43 / +6.09 / +0.96% | +8.06 / +19.24 / +15.50 / +8.53% |
| B (yalnızca h=5) | +4.90% | +3.99% |

Üç varyantta da HAR-X her ufukta hem HAR'dan hem XGBoost'tan düşük RMSE'ye sahip. Birincil
sonuçta da durum bu.

**8 test** (DM ve işaret testi; Holm/BH/BY varyantın kendi testleri içinde):

| varyant | %5'te ayakta kalan | ham değerler |
| --- | --- | --- |
| A | hiçbir test, hiçbir düzeltmede | DM HLN p 0.122–0.937. İşaret testinde h=22'de iki test 12/15, ham p 0.035; BH 0.141. |
| A′ | hiçbir test | h=22 HAR vs HAR-X 13/15, ham p 0.007; Holm ve BH 0.059, BY 0.161. |
| B (2 test) | hiçbir test | — |

Birincil ailede BH altında reddedilen iki h=22 işaret testi (13/15, BH 0.030) A'da 12/15'e
iner ve BH altında ayakta kalmaz. A′'de biri 13/15'te kalır ama BH 0.059 ile %5 eşiğinin
üstündedir.

**Tarih boşluklarıyla örtüşme:** belgelenmiş 40 boşluk satırının 18'i A'da, 27'si A′'de
getirisi çıkarılan satırlar.
- A'da: 2008–2013 arası 18 satır.
- A′'de ek 9 satır: 2013-03-19 – 2014-02-19 arası.

*Hipotez, kanıtlanmış değil:* boşlukların bir kısmı geçiş kaynaklı olabilir. Örtüşme
nedeni göstermez; A′ iki satır çıkardığı için şans örtüşmesi de artar.

**Sınırlılık:** XGBoost'un fiyat düzeyi özelliklerindeki geçiş etkisi temizlenmedi (25.1).

---

# Aşama 26 (2026-09-29): Clark–West, roll-over varyantlarında — KOŞUDAN ÖNCE yazılan tasarım

**Bu bölüm koşudan önce yazıldı ve koşudan önce commit edildi.** Sonuçlar 26.2'de
olacak. Üç varyant da sonuç ne çıkarsa çıksın raporlanacak.

**Ne:** Clark–West ek ailesinin (HAR ⊂ HAR-X, 4 test; Aşama 22) Aşama 25'in roll-over
varyantlarında tekrarı. CLAUDE.md (commit `0a33dd4`): ilan edilmiş herhangi bir ailenin
varyant veri üzerindeki tekrarı yeni aile değildir. Ek A'da kendi Holm/BH/BY
düzeltmeleriyle raporlanır ve birincil aileyle havuzlanmaz.

**Statüler (roll-over'daki gibi):**

| varyant | statü |
| --- | --- |
| A | birincil sağlamlık varyantı |
| A′ | duyarlılık kontrolü |
| B | duyarlılık kontrolü, yalnızca h=5 |

**Kurulum Aşama 22 ile aynı:**
- `f_t = e_HAR,t² − [e_HARX,t² − (ŷ_HAR,t − ŷ_HARX,t)²]`, tek yanlı, H1: HAR-X daha iyi.
- Newey-West Bartlett, L = h−1; HLN çarpanı ve t(n−1). Düzeltmeler HLN p değerine
  uygulanır; normal p yan sütunda.
- Tahminler Aşama 25'in kayıtlı varyant tahminleri: `rollover_predictions_publication_aligned.csv`,
  ana fold'lar, havuzlanmış.

**Düzeltmeler varyant içinde:** A ve A′'de 4 test; B'de tek test (h=5). Tek testte Holm,
BH ve BY ham p'ye eşittir.

**Doğrulama:** CW istatistiğinin hesabı 21'den bir fonksiyona taşınacak. 21'in çıktısının
değişmediği kontrol edilecek. Maskesiz roll-over tahminleri kayıtlı
`clark_west_publication_aligned.csv` değerlerini yeniden üretmeli (assert).
