# datasets/curriculum — İP-4 müfredat test seti (1000+ problem)

Bu dizin, sözleşmedeki **"CAS uçtan uca doğruluk > %99"** ölçütünün üzerinde ölçüldüğü *Temel
Analiz* problem kümesidir (İP-4 "test veri seti ve analiz raporu";
[docs/mimari/latex-ayristirici.md §7](../../docs/mimari/latex-ayristirici.md),
[docs/ml/degerlendirme.md §3](../../docs/ml/degerlendirme.md)).

| Yol | İçerik |
|---|---|
| `cards/<konu>.yaml` | Kaynak kartlar (konu başına bir dosya, 1204 kart) — **derlenen dosyanın kaynağı budur** |
| `problems.jsonl` | Derlenmiş küme: 1. satır köken başlığı (`_header`), ardından kimliğe göre sıralı 1204 problem |
| `FROZEN.sha256` | Her satırın `sha256_json` özeti; `test_frozen_curriculum_rows_unchanged` bunu doğrular |
| `generate_cards.py` | Kartları **yazan** üretici (müfredat şablonları + kapalı-form yanıtlar) |
| `../../scripts/build_curriculum_testset.py` | Kartları doğrulayan ve derleyen derleyici |

## 1. KÖKEN (rapor için zorunlu uyarı)

> **KÖKEN.** Bu küme **proje içinde yazılmıştır**: elle yazılmış müfredat şablonlarından makine
> ile genişletilmiştir. **Bağımsız bir derlemden ÖRNEKLENMEMİŞTİR** ve hiçbir satır telifli bir
> ders kitabından alınmamıştır. Dolayısıyla burada ölçülen CAS doğruluk oranı **bağımsız bir
> ölçüm değil, güçlü bir regresyon ölçüsüdür**. 2027-05-01 ilerleme raporunda bu sayı verilirken
> bu cümle birlikte verilmelidir (bkz. `KNOWN_ISSUES.md` "Metric provenance",
> `datasets/book/README.md` aynı disiplini uygular).

Ölçümü boş bir totolojiden ayıran nokta şudur: **beklenen yanıtlar CAS'a sorulmamıştır.**
`generate_cards.py` hiçbir zaman `sympy.diff` / `sympy.integrate` / `sympy.limit` /
`sympy.solveset` / `sympy.series` çağırmaz; her şablon, ders kitabındaki kuralın verdiği
kapalı-form yanıtı (ör. `(a x^n)' = a n x^{n-1}`, `int dx/(x^2+a^2) = arctan(x/a)/a`,
`e^{ax}` Maclaurin katsayıları) SymPy yapıcılarıyla **kurar**. Derleyici ise motoru çalıştırıp
sonucu bu yanıtla karşılaştırır. İki taraf birbirinden bağımsız olduğu için karşılaştırma
gerçekten başarısız olabilir — nitekim ilk tam koşumda 3 satır (`\int 21 \, dx` gibi hatalı bir
şablon çıktısı) tam da bu yolla yakalanmıştır.

**Hâlâ eksik olan:** hiçbir satır bir insan tarafından incelenmemiştir (`reviewed_by: ""`).
Pilot öncesinde M. M. Turhan / Ç. Şahin incelemesi ve gerçek ders kitabı bölümlerinden alınmış
bağımsız bir alt küme gerekir.

## 2. Konu kotaları

Kotalar `docs/mimari/latex-ayristirici.md §7.1`'deki ağırlık tablosudur ve **alt sınırdır**;
derleyici hem kotayı hem de dağılımın kota oranından ±0,02'den fazla sapmamasını denetler
(`check_quotas`, `test_curriculum_cards_compile_and_quotas_match`).

| Konu | Bölüm | Kota | v1 | Pay | Alt konular |
|---|---|---|---|---|---|
| `limit` | `aicls:bolum-1` | 120 | 144 | 0,120 | polinom_limiti, belirsizlik_0_0, sonsuzda_limit, trigonometrik_limit, tek_yonlu_limit, e_limiti, ustel_logaritmik_limit, kok_belirsizligi |
| `sureklilik` | `aicls:bolum-1` | 60 | 72 | 0,060 | kaldirilabilir_sureksizlik, surekli_genisletme, tek_yonlu_sureklilik, deger_kontrolu, ara_deger_teoremi |
| `turev` | `aicls:bolum-2` | 200 | 240 | 0,199 | kuvvet_kurali, toplam_kurali, carpim_kurali, bolum_kurali, zincir_kurali, trigonometrik_turev, ustel_logaritmik_turev, **kapali_turev**, yuksek_mertebeden_turev, ters_trigonometrik_turev, hiperbolik_turev, turev_tanimi |
| `turev_uygulamalari` | `aicls:bolum-3` | 120 | 144 | 0,120 | kritik_noktalar, teget_egimi, artan_azalan, ikinci_turev_bukeylik, optimizasyon, hiz_ivme, l_hospital, ortalama_deger_teoremi |
| `belirsiz_integral` | `aicls:bolum-4` | 150 | 180 | 0,150 | kuvvet_kurali, ustel_integral, trigonometrik_integral, rasyonel_integral, **degisken_degistirme**, **kismi_integrasyon**, **basit_kesirlere_ayirma**, **trigonometrik_teknikler**, ters_trigonometrik_sonuc |
| `belirli_integral` | `aicls:bolum-4` | 120 | 144 | 0,120 | kuvvet_belirli, trigonometrik_belirli, ustel_belirli, rasyonel_belirli, ters_trigonometrik_belirli, degisken_degistirme_belirli, kismi_integrasyon_belirli, has_olmayan_integral |
| `integral_uygulamalari` | `aicls:bolum-5` | 80 | 96 | 0,080 | egri_alti_alan, iki_egri_arasi_alan, donel_cisim_hacmi, yay_uzunlugu, ortalama_deger, is_ve_kuvvet |
| `diziler_seriler` | `aicls:bolum-6` | 80 | 100 | 0,083 | dizi_limiti, geometrik_seri, teleskopik_seri, kuvvet_toplamlari, **taylor_maclaurin** (36 satır), yakinsaklik_testleri |
| `denklem_esitsizlik` | `aicls:bolum-0` | 50 | 60 | 0,050 | ikinci_derece_denklem, birinci_derece_denklem, carpanlara_ayirma, acilim, sadelestirme, ustel_logaritmik_denklem |
| `grafik` | `aicls:bolum-0` | 20 | 24 | 0,020 | polinom_grafigi, trigonometrik_grafik, ustel_grafik, rasyonel_grafik |
| **Toplam** | | **1000** | **1204** | | |

Müfredatın "kapalı türev", "integral teknikleri" ve "Taylor/Maclaurin" başlıkları belge
tablosunda ayrı konu değildir; bunlar yukarıda **kalın** yazılan alt konulardır.

## 3. Satır şeması

`aicls.data.curriculum.Problem` ile birebir aynıdır; her kart on beş alanın **hepsini** yazar
(yükleyici varsayılanına güvenilmez).

```yaml
- id: turev-0042            # <konu>-NNNN, kart dosyaları arasında tekil
  topic: turev              # kota tablosundaki konu
  subtopic: zincir_kurali   # şablon ailesi
  chapter_ref: aicls:bolum-2
  difficulty: 2             # 1 alıştırma | 2 standart | 3 çok adımlı
  input:
    kind: latex             # latex | text_tr  (text_tr satırları NLU'yu da sınar)
    value: \frac{d}{dx} \left(\sin(2x^{2})\right)
  command: {op: diff, expr: \sin(2x^{2}), var: x, order: 1}
  expected:
    latex: 4 \cdot x \cdot \cos{\left(2 \cdot x^{2} \right)}
    srepr: Mul(Integer(4), Symbol('x', real=True), cos(...))
  checker: symbolic         # symbolic | numeric | set | bool | exact_latex | plot_ok
  checker_args: {}
  source: sablon:turev/zincir_sinus
  author: generator:datasets/curriculum/generate_cards.py
  reviewed_by: ''           # boş: henüz insan incelemesi yok
  status: frozen            # draft | reviewed | frozen  (aşağıya bakınız)
  notes_tr: 'Zincir kurali: (sin(u))'' = u'' cos(u), burada u = a x^n.'
```

Notlar:

- `expected.latex`, `expected.srepr`'ın **kanonik basımıdır**; derleyici
  `sympy_to_latex(sympify(srepr)) == expected.latex` eşitliğini her satırda doğrular, böylece
  yanıtın iki yarısı birbirinden sapamaz.
- `checker_args` serbest bir sözlüktür. Belirsiz integrallerde
  `{up_to_constant: true, var: "x"}` kullanılır: bir ilkel yalnızca **bir sabit farkla**
  belirlidir, dolayısıyla doğru karşılaştırma `d/dx(motor) ≡ d/dx(beklenen)`'dir. Derleyici
  ayrıca `d/dx(motor) ≡ integrand` özelliğini de sınar. `numeric` için
  `{tolerance: 1e-9, domain: [-3, 3]}`.
- Sonsuz değerli tek yönlü limitlerde `checker: exact_latex` kullanılır; nedeni
  `aicls.mathengine.symbolic_equivalent(oo, oo)` çağrısının `simplify(oo - oo) = nan` yüzünden
  `"different"` dönmesidir (derleyici kendi `compare()` işlevinde önce kimlik denetimi yapar).
- `text_tr` satırları (70 satır) Türkçe komut cümlesidir; `aicls.nlu` yazılana kadar yalnızca
  yapısal olarak doğrulanır ve **ayrıştırma başarısı paydasına girmez**. NLU altın kümesi
  `datasets/nlu/` dizinindedir.
- Ölçüt dağılımı: `symbolic` 1071, `set` 102, `plot_ok` 24, `exact_latex` 8.
  `numeric` ve `bool` derleyicide uygulanmış ve birim testleriyle sınanmıştır, ancak v1'de
  bunları kullanan bir satır **yoktur** — belirli integrallerin tümü tam (sembolik) değer
  verdiği için sayısal karşılaştırmaya gerek kalmamıştır.
- Komut dağılımı: `integrate` 420, `limit` 278, `diff` 241, `solve` 102, `evaluate` 58,
  `series` 36, `simplify` 26, `plot` 24, `factor` 10, `expand` 10 (on işlemin hepsi).
- Zorluk dağılımı: 1 → 231, 2 → 583, 3 → 391.

## 4. Derleme, dondurma ve değişiklik kuralları

```bash
python datasets/curriculum/generate_cards.py          # kartları yeniden yaz
python scripts/build_curriculum_testset.py --check    # yalnızca doğrula
python scripts/build_curriculum_testset.py            # problems.jsonl + FROZEN.sha256 yaz
python scripts/build_curriculum_testset.py --sample 40 --check   # hızlı kontrol
```

- **Sıralama** kimliğe göredir; `problems.jsonl` satır satır JSON'dur (UTF-8, LF, anahtarlar
  sıralı), böylece `fingerprint_jsonl` kararlıdır.
- **`_header` satırı.** Dosyanın 1. satırı köken başlığıdır: `id: _header`, `topic: _meta`,
  `_header: true`. Geçerli (ve doğru yanıtlı, önemsiz) bir problem olarak yazılmıştır, böylece
  onu süzmeyi unutan bir tüketici hiçbir ölçütü bozamaz. Tüketiciler `row["id"] == "_header"`
  veya `row.get("_header")` ile süzmelidir.
- **Dondurma (sapma, bkz. Uygulama notları).** Belge yalnızca `status: frozen` satırların
  dondurulmasını söyler; v1'de **tüm satırlar** `FROZEN.sha256`'ya yazılır, çünkü satırlar makine
  ile üretilmiştir ve sessiz bir değişiklik ölçümü fark ettirmeden kaydırabilir. Buradaki
  `frozen`, "bayt düzeyinde dondurulmuş" demektir; **"incelenmiş" demek değildir** — inceleme
  bilgisi `reviewed_by` alanındadır.
- Bir satır düzeltilecekse: `DATASET_VERSION` yükseltilir, `FROZEN.sha256` yeniden üretilir ve
  değişiklik aşağıdaki günlüğe yazılır. Satır **silinmez**; yeni problem yeni kimliktir.
- SymPy sürümü değişirse beklenen yanıtların kanonik basımı kayabilir; `sympy_version`
  başlıkta kayıtlıdır ve `pyproject.toml` sürümü `>=1.14,<2` ile sınırlar.

## 5. Ölçülen değerler (v1, sympy 1.14, 2026-09-07)

| Ölçüt | Değer | Tanım |
|---|---|---|
| CAS doğruluğu | **1204/1204 = 1,0000** | `checker` başarılı / toplam satır |
| Ayrıştırma başarısı (sıkı) | **1134/1134 = 1,0000** | `latex` girdilerde parse + `to_sympy` + seviye-2 gidiş-dönüş |
| Desteklenmeyen | 0 | `UnsupportedProblem` |
| Karar verilemeyen | 0 | eşdeğerlik ölçütünün çekimser kaldığı satır |
| Motor süresi | ort. ~5 ms, p95 ~16 ms | `InProcessExecutor`, tek çekirdek |

**Bu %100 bağımsız bir başarı değildir.** Şablonlar, motor ile uyuşana kadar düzeltilmiştir;
düzeltilen 14 uyuşmazlığın tamamı **kümenin kendi hatasıydı** (3 hatalı şablon çıktısı,
11 gidiş-dönüş uyuşmazlığı), motorun değil. Yani sayı şunu kanıtlar: *motorun bugünkü davranışı,
müfredat kurallarının verdiği kapalı-form yanıtlarla 1204 problemde çelişmiyor.* Bir sonraki
sürümde motorun gerçekten zorlanacağı satırların (bilerek desteklenmeyen problemler, uzun
simplify'lar, gerçek kitap soruları) eklenmesi gerekir.

## 6. Lisans ve kişisel veri

İçerik proje ekibinin kendi metnidir; telifli bir kaynaktan alıntı yoktur. Kişisel veri yoktur.
Lisans: depo kökündeki `LICENSE`.

## 7. Değişiklik günlüğü

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| `curriculum-2026.09-v1` | 2026-09-07 | İlk sürüm: 1204 kart, 10 konu, 73 alt konu, 149 şablon. |
