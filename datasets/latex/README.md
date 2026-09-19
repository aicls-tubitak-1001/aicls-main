# datasets/latex — ayrıştırıcı değerlendirme kümeleri

Bu dizin `aicls.latex2sympy` için **sürümlü değerlendirme varlıklarını** tutar
([ADR-0004](../../docs/adr/0004-latex-ayristirici-lark-dilbilgisi.md),
[mimari/latex-ayristirici.md §6](../../docs/mimari/latex-ayristirici.md)). İP-4'ün sözleşmeye
bağlı ölçütü — *geçerli girdilerin > %99'unun hatasız SymPy'a çevrilmesi* — bu dosyalar
üzerinden ölçülür ve `tests/unit/latex2sympy/test_datasets.py` içinde regresyon kapısı olarak
çalışır.

| Dosya | Satır | İçerik |
|---|---|---|
| `valid_inputs.jsonl` | 3709 | ayrıştırılması **beklenen** girdiler (başarı oranı ölçümü) |
| `unsupported.jsonl` | 106 | kapsam dışı / bozuk girdiler (hata davranışı ölçümü) |
| `generate_valid_inputs.py` | — | her iki dosyayı deterministik olarak yeniden üreten betik |

## 1. `valid_inputs.jsonl` satır şeması

Her satır tek bir JSON nesnesidir (anahtarlar alfabetik sıralı, UTF-8, LF):

```json
{"expect_ok": true, "id": "curriculum-0042", "latex": "\\frac{x + 1}{x - 2}", "source": "curriculum", "topic": "rasyonel"}
```

| Alan | Tip | Anlam |
|---|---|---|
| `id` | `str` | `"<source>-<4 hane>"`; tekildir ve pytest kimliği olarak kullanılır |
| `latex` | `str` | girdi dizesi (en fazla 2048 karakter, `MAX_INPUT_CHARS`) |
| `source` | `str` | `curriculum` \| `sympy_latex` \| `ocr_output` \| `mathwriting_label` |
| `expect_ok` | `bool` | her zaman `true` — bu küme "başarılı olmalı" kümesidir |
| `topic` | `str` | konu etiketi (`limit`, `turev`, `belirli_integral`, …); kırılım raporu için |
| `variant` | `int` | yalnızca `ocr_output`: uygulanan bozulma türünün numarası |
| `synthetic` | `bool` | yalnızca `mathwriting_label`: satırın elle yazıldığını belirtir |

### Kaynak dağılımı ve üretim biçimi

- **`curriculum`** — Temel Analiz şablonlarının küçük tam sayı parametreleriyle açılımı
  (polinom, rasyonel, kök, trigonometri, üstel/logaritma, limit, türev, belirsiz/belirli
  integral, diziler-seriler, denklem-eşitsizlik, klasik ders tahtası yazımları: türev tanımı,
  `f(x) = …`, `\begin{pmatrix}…`).
- **`sympy_latex`** — `sympy_to_latex()` çıktıları. Bu satırlar kritiktir: CAS sonuçları
  kullanıcıya bu biçimde gider ve kullanıcı bunları **yeniden girdi olarak** gönderebilir.
- **`ocr_output`** — müfredat satırlarının gerçekçi el yazısı/OCR yazımlarına dönüştürülmüş
  hâlleri: `\dfrac`, `\left(`/`\right)`, sonda eksik tek `}`, `\displaystyle`, `\mathrm{d}x`,
  `|x|`, Türkçe ondalık virgül (`1,5`), unicode `\times`, çoklu boşluk.
- **`mathwriting_label`** — MathWriting etiket *biçiminde* (belirteçler arası boşluklu:
  `\frac { 1 } { 2 }`) elle yazılmış satırlar. `synthetic: true` taşırlar. **Gerçek MathWriting
  verisi indirilmemiştir** (onay kapısı 1, ADR-0012); veri seti kapısı açıldığında normalize
  edilmiş gerçek etiketler bu kaynağa eklenecek ve `synthetic` alanı kalkacaktır.

### Başarı tanımı (sıkı)

Bir satır ancak şu üçü birden sağlanırsa başarılı sayılır:

1. `parse_latex(latex)` istisna atmaz,
2. `to_sympy(node)` istisna atmaz,
3. `roundtrip_check(latex)` **ok** döner (kanonik LaTeX / `srepr` / sayısal örnekleme
   seviyelerinden biri).

Üçüncü koşul sayesinde **sessiz yanlış ayrıştırma başarısızlık sayılır**: ifade ayrışsa bile
`sympy_to_latex` çıktısı yeniden ayrıştırıldığında farklı bir ifadeye dönüşüyorsa satır düşer.

Son ölçüm (2026-09-07, onarım dalgası sonrası): `3709/3709 = %100.0`; kırılım
`curriculum 2315/2315`, `sympy_latex 272/272`, `ocr_output 972/972`,
`mathwriting_label 150/150`.

Bu dalgada eklenen satırlar (`topic: kismi_turev`, 118 satır + türetilmiş `ocr_output`
varyantları) Leibniz **kısayolunu** kapsar: `\frac{\partial f}{\partial x}`,
`\frac{\partial^{2} f}{\partial x^{2}}`, `\frac{d y}{d x}`, `\frac{d^{n}y}{dx^{n}}`.
Bunlardan ilki `aicls.ocr.CURRICULUM_LATEX` içinde birebir bulunur ve düzeltmeden önce
`ParseError` veriyordu; `\frac{d^{2}y}{dx^{2}}` ise **sessizce** `d y / x^{2}` olarak
ayrışıyordu (korpusta zaten bulunan 21 satır bu yanlış okumayı "başarı" sayıyordu).
`_corrupt` varyant 7 artık `d^{` içeren satırlara uygulanmaz; aksi hâlde bozulma Leibniz
belirtecini kırıp aynı sessiz yanlış ayrıştırmayı geri getiriyordu.

> Bu korpus dilbilgisiyle birlikte geliştirildiği için oran doğal olarak yüksektir; asıl işlevi
> **regresyon kapısı** olmaktır. Bağımsız ölçüm, OCR/ASR boru hattı çalışmaya başladığında
> eklenecek gerçek `ocr_output` ve `mathwriting_label` satırlarıyla yapılacaktır.

## 2. `unsupported.jsonl` satır şeması

```json
{"construct": "\\vec", "expect_error": "unsupported", "id": "unsupported-0005", "latex": "\\vec{v} + \\vec{w}"}
```

| Alan | Tip | Anlam |
|---|---|---|
| `id` | `str` | `"unsupported-<4 hane>"` |
| `latex` | `str` | girdi dizesi |
| `expect_error` | `str` | `unsupported` → `UnsupportedConstruct`, `parse_error` → `ParseError`, `semantic` → `SemanticError` |
| `construct` | `str` | `unsupported` satırlarında beklenen yapı adı (`\vec`, `\begin{cases}`); diğerlerinde boş |

Test, satırı `to_sympy(parse_latex(latex))` boru hattından geçirir ve **tam olarak** beklenen
hata sınıfının atıldığını, `UnsupportedConstruct` ise `construct` alanının birebir eşleştiğini
doğrular. Böylece "desteklenmiyor" mesajı kullanıcıya her zaman yapının adıyla gider.

Son ölçüm (2026-09-07): 106 satırın **106'sı** beyan edilen hata sınıfıyla reddedildi
(`unsupported` 80 — 70 adlandırılmış yapı + 10 bilinmeyen makro, `parse_error` 20,
`semantic` 6). Hiçbir satır sessizce ayrışmıyor. Bu dalgada eklenen 3 satır **karma kısmi
türev**dir (`\frac{\partial^{2} f}{\partial x \partial y}`): şekil olarak tanınır ama v1
AST'sinde karşılığı yoktur, bu yüzden anonim `ParseError` yerine
`construct = "\frac{\partial^{2}}{\partial x \partial y}"` adıyla reddedilir
(`aicls.latex2sympy.parser.MIXED_PARTIAL_CONSTRUCT`).

## 3. Yeniden üretim

```powershell
$env:PYTHONPATH="src"; C:\Users\cagri\.venvs\ai-cls\Scripts\python.exe datasets\latex\generate_valid_inputs.py
```

Betik ağ erişimi yapmaz, veri indirmez ve deterministiktir; aynı sürümde aynı satırları üretir.
Dosyalar git'te tutulur (toplam < 1 MB, `datasets/` için 5 MB sınırı geçerlidir).

## 4. Lisans ve kaynak

Satırların tamamı bu proje için yazılmıştır (Ç. Şahin / M. M. Turhan, İP-4). Üçüncü taraf veri
seti içermez; MathWriting'in kendisi bu dizinde **yoktur** ve manifest yoluyla temsil edilir
([ADR-0008](../../docs/adr/0008-veri-yonetimi-manifest-onedrive-disi.md)).
