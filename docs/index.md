# AI-CLS Teknik Dokümantasyonu

**Proje:** TÜBİTAK 1001 — 125E748, "Temel Analiz Öğretimine Yönelik Yapay Zeka Destekli-Etkileşimli Bir Platformun Tasarımı ve Geliştirilmesi (AI-CLS)".
**Süre:** 01.05.2026 → 01.05.2028 (24 ay). **1. Gelişme Raporu:** 01.05.2027. **Model dondurma:** 2027 yazı (pilot uygulama 2027 güz dönemi).
**Dijital ekip:** Çağrı Şahin (koordinasyon, modeller, metin→komut katmanı, ortam), Mehmet Murat Turhan (matematik motoru İP-4, veri boru hatları, kitap şeması/API İP-8), Şahiner Güler (İP-2 web: React arayüz, arka uç API, veri toplama arayüzü).

Bu site, AI-CLS kod deposunun (`ai-cls/`) mimarisini, kararlarını, geliştirme kurallarını ve TÜBİTAK iş paketlerine (İP) göre raporlama düzenini anlatır. Düzyazı Türkçe; kod tanımlayıcıları, dosya adları, yapılandırma anahtarları ve commit mesajları İngilizcedir. Kısaltmalar metin içinde ilk geçtikleri yerde açılır; ayrı bir sözlük tutulmaz.

## Sistem tek cümleyle

Tarayıcıda el yazısı (iz verisi veya görüntü) ve ses (mikrofon) alınır; el yazısı OCR (optik karakter tanıma) modeliyle LaTeX'e, ses ASR (otomatik konuşma tanıma) modeliyle Türkçe metne çevrilir; Türkçe metin kural tabanlı **metin→komut** katmanıyla tipli bir komuta dönüşür; LaTeX ifadesi kendi yazdığımız **Lark dilbilgisi** ile SymPy'a çevrilir; sonuç yalnızca **SymPy** (deterministik CAS, bilgisayarlı cebir sistemi) tarafından üretilir ve LaTeX/KaTeX ile ekrana çizilir; "türev tanımını getir" gibi komutlar **kitap bilgi grafiğinden** (JSON-LD/RDF, MySQL) içerik çeker.

Sözleşmeye bağlı sayısal hedefler ve bu depoda nasıl ölçüldükleri: [ml/degerlendirme.md](ml/degerlendirme.md). İş paketi çıktıları ile depo artefaktlarının eşlemesi: [raporlama/ip-eslestirme.md](raporlama/ip-eslestirme.md).

## Tasarım ilkeleri

1. **Deterministik çekirdek, olasılıksal kenarlar.** Kullanıcının gördüğü her sonuç SymPy'dan gelir. OCR, ASR ve (kapalı gelen) büyük dil modeli yedeği yalnızca *aday* üretir; adaylar dilbilgisi ve komut envanteriyle doğrulanmadan hesaplamaya giremez. Bu, öneri metnindeki "halüsinasyon üretmez" iddiasının teknik karşılığıdır.
2. **Her rapor sayısı yeniden hesaplanabilir.** Her değerlendirme çıktısı (`reports/metrics/*.json`) git SHA'sı, veri kümesi parmak izi (SHA-256), normalizasyon sürümü ve model özetiyle yazılır; `poe report-check` bu zinciri doğrular; rapor bölümleri sayıları elle değil makroyla alır.
3. **Onay kapıları koddadır.** Veri seti indirme, ağırlık indirme, CUDA kurulumu ve eğitim koşusu; CLI bayrağı + ortam değişkeni + git'teki onay defteri üçlüsü olmadan çalışmaz; CI bunların reddedildiğini test eder.
4. **Windows-önce, OneDrive-dışı.** Sanal ortam, veri kökü, model ağırlıkları, koşu kayıtları ve `node_modules` OneDrive dışında; görev çalıştırıcı `poethepoet` (make yok).
5. **Test-önce, katmanlı bağımlılık.** `import-linter` sözleşmesi torch'u deterministik çekirdekten uzak tutar; OCR/ASR arayüzlerinin yapay (stub) uygulamaları sayesinde API, arayüz ve değerlendirme koşumu veri seti, ağırlık ve GPU olmadan bugün test edilir.
6. **Literatür-önce ekip, kayıtlı deneme alanı.** `experiments/` altında 1 sayfalık notlu küçük denemeler; işe yarayan sonuç normal PR ile `src/`'a taşınır.

## Belge haritası

| Bölüm | Belge | İçerik |
|---|---|---|
| Mimari | [mimari/genel-bakis.md](mimari/genel-bakis.md) | Bileşen diyagramı, veri akışı, modül sınırları, tam depo ağacı, çalışma zamanı topolojisi |
| Mimari | [mimari/moduller.md](mimari/moduller.md) | Her paketin sorumluluğu, açık API imzaları, bağımlılıkları, sahibi |
| Mimari | [mimari/api.md](mimari/api.md) | REST uç noktaları, şemalar, hata modeli (RFC 9457), sürümleme, WebSocket akışları |
| Mimari | [mimari/veri-modeli.md](mimari/veri-modeli.md) | Tablolar, SQLite-dev / MySQL-prod, Alembic, kitap içerik şeması (JSON-LD/RDF) ve sorgulama |
| Mimari | [mimari/latex-ayristirici.md](mimari/latex-ayristirici.md) | LaTeX→SymPy Lark dilbilgisi, AST, hata politikası, gidiş-dönüş doğrulama, 1000 problemlik test seti |
| Mimari | [mimari/metin-komut-katmani.md](mimari/metin-komut-katmani.md) | Komut envanteri biçimi, normalizasyon, niyet/yuva ayrıştırma, SymPy eşleme, yedek politika |
| ML | [ml/egitim-altyapisi.md](ml/egitim-altyapisi.md) | Yapılandırma, veri kümeleri, model havuzu, eğitici, kontrol noktaları, deney takibi, tekrarlanabilirlik, GPU planı, onay kapılı betikler |
| ML | [ml/egitim-el-kitabi.md](ml/egitim-el-kitabi.md) | Operatör el kitabı: GPU'ya çıkış (WSL2 + nvidia-docker, ADR-0020), kapı istekleri, veri hazırlama, eğitim, sorun giderme çizelgesi |
| ML | [ml/degerlendirme.md](ml/degerlendirme.md) | Metrik tanımları (CER, ExpRate, sembolik eşdeğerlik, konuşmacı-ayrık WER, gecikme, SUS), test kümeleri, rapor biçimi |
| ML | [ml/veri-yonetimi.md](ml/veri-yonetimi.md) | MathWriting ve özgün ses verisi, lisanslar, dizin düzeni, sürümleme, KVKK/rıza |
| Geliştirme | [gelistirme/is-akisi.md](gelistirme/is-akisi.md) | Dallanma, commit kuralı, inceleme, CI işleri, kalite kapıları, pre-commit, sürümleme, poe görevleri |
| Geliştirme | [gelistirme/test-stratejisi.md](gelistirme/test-stratejisi.md) | Seviyeler, araçlar, işaretler, kapsam hedefleri, özellik tabanlı testler, fikstürler, somut test adları |
| Geliştirme | [gelistirme/kurulum.md](gelistirme/kurulum.md) | Windows kurulumu, onay kapıları, ortam değişkenleri |
| Geliştirme | [gelistirme/onay-defteri.md](gelistirme/onay-defteri.md) | Onay kapısı defteri (`require_gate()` bu dosyayı okur) |
| Güvenlik | [guvenlik-gizlilik.md](guvenlik-gizlilik.md) | Roller, KVKK, sırlar, SSL, yükleme doğrulama |
| Raporlama | [raporlama/ip-eslestirme.md](raporlama/ip-eslestirme.md) | İP çıktısı → depo artefaktı tablosu, metriklerin gelişme raporuna akışı |
| Deneme | [deneme-alani.md](deneme-alani.md) | `experiments/` düzeni ve 1 sayfalık not şablonu |
| Kararlar | [adr/README.md](adr/README.md) | Mimari karar kayıtları (ADR, MADR biçimi) ve süreç |

## Sözleşmeye bağlı hedefler (özet)

| Bileşen | Metrik | Hedef | Ölçen kod |
|---|---|---|---|
| OCR (el yazısı→LaTeX) | CER, test kümesi | < %10 | `aicls.eval.ocr_eval` |
| ASR (Türkçe matematik komutu) | WER, konuşmacı-ayrık test | < %10 | `aicls.eval.asr_eval` |
| LaTeX→SymPy | geçerli girdide ayrıştırma başarısı | > %99 | `aicls.eval.parser_eval` |
| CAS uçtan uca | ≥1000 problemlik müfredat setinde doğru cevap | > %99 | `aicls.eval.cas_eval` |
| Yanıt süresi | komut → ekranda sonuç (ortalama) | < 3 s | `aicls.eval.latency_eval` + `latency_samples` |
| Kullanılabilirlik | SUS (System Usability Scale) | > 75 | `sus_responses` tablosu |
| Kitap entegrasyonu | etiketli içerik / komutla getirme doğruluğu | %100 / > %95 | `aicls book coverage`, `aicls.eval.book_eval` |
| Komut iletimi (İP-2) | komutun doğru niyet/yuva ile CAS'a ulaşması | > %95 | `aicls.eval.nlu_eval`, `command_intent_accuracy` |

## Onay kapıları (özet)

Şu dört eylem, `docs/gelistirme/onay-defteri.md` içinde onay satırı olmadan **kod tarafından reddedilir**: (1) veri seti indirme, (2) model ağırlığı indirme, (3) CUDA/GPU PyTorch kurulumu, (4) gerçek veriyle eğitim koşusu. Ayrıntı: [gelistirme/kurulum.md](gelistirme/kurulum.md#3-onay-kapıları) ve [ADR-0012](adr/0012-onay-kapilari.md).
