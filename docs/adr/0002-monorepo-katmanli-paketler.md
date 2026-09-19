# ADR-0002: Tek depo, tek Python dağıtımı, import-linter ile katmanlı paketler

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan, Ş. Güler
- **İlgili İP:** İP-2 (modüler mimari, kod depoları), İP-3, İP-4, İP-8

## Bağlam ve problem

Üç geliştirici, altı iş paketi, iki yıl. Öneri İP-2 risk planı "modüler bir mimari" ister ve "Frontend ve Backend Kod Depoları (Git)" ara çıktısı verir. Deterministik CAS çekirdeğinin torch/transformers'a bağımlı olmaması, API'nin eğitim yığınını import etmemesi ve NLU'nun veritabanına dokunmaması gerekir; hakem değerlendirmelerinde iki öneride döngüsel bağımlılık (ocr↔ml, api→ml) tespit edilmiştir.

## Değerlendirilen seçenekler

1. **Birden çok pip paketi / İP başına depo** — sınırlar keskin ama üç kişi için sürüm/CI yükü; ortak sürüm numarası zor.
2. **Tek depo, tek dağıtım `aicls`, alt paketler, bağımlılık yönü araçla zorlanmış** — tek CI, tek sürüm, tek kilit dosyası; sınırlar `import-linter` ile.
3. **Tek depo, düz paket** — sınır yok; torch her yere sızar.

## Karar

Seçenek 2. `src/` düzeni, 14 alt paket; `web/` ayrı npm projesi; `training/`, `datasets/`, `reports/`, `experiments/`, `docs/` aynı depoda. Katman sözleşmesi (`[tool.importlinter]`):

```
cli > (api | ml) > eval > (data | collect | book) > nlu > mathengine > latex2sympy > (ocr | asr) > db > core
```

Yasak sözleşmeleri: `no-torch-in-core`, `api-no-training` (api → ml/eval/data yasak), `nlu-no-db` (nlu → db/book yasak; kitap niyeti `BookFetch` komutu olarak döner), `eval-no-ml`, `experiments-isolated`. OCR/ASR paketleri `Recognizer`/`Transcriber` protokolü + stub + tembel torch sarmalayıcı barındırır; `ml` bunlara bağımlıdır, tersi asla. Frontend ve backend "kod depoları" tek git deposunun `web/` ve `src/aicls/api` dizinleridir; İP-2 ara çıktısı bu yollarla raporlanır. Lisans: yürütücü kararı beklenir; `LICENSE` dosyası yer tutucu, açık kaynak bileşen kullanımı öneri A.2.2 ("açık kaynaklı olması ... esneklik") ile uyumludur.

## Sonuçlar

- Olumlu: API, arayüz ve değerlendirme torch/veri/GPU olmadan test edilir; hakemlerin döngü bulguları yapısal olarak imkânsızlaşır; tek `poe ci`.
- Olumsuz: `import-linter` ihlali PR'ı kırar (istenen); paketler arası veri nesneleri `core.types`/pydantic ile paylaşılır, bazı yinelenen DTO'lar olur.

## Öneri metniyle uyum

İP-2 risk planı: "Proje başlangıcından itibaren modüler bir mimari benimsenecektir. Servisler arası iletişim, iyi tanımlanmış RESTful API'ler veya WebSocket üzerinden sağlanacaktır. ... sürekli entegrasyon (CI) pratikleri uygulanacaktır." İP-2 ara çıktısı "Frontend ve Backend Kod Depoları (Git)".

## Doğrulama

`poe imports` (`lint-imports`) CI'da; `test_api_does_not_import_torch` (`sys.modules` denetimi `create_app` sonrası).

## İlgili

[mimari/genel-bakis.md §3](../mimari/genel-bakis.md#3-modül-sınırları), ADR-0010.
