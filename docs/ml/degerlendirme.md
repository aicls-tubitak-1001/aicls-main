# Değerlendirme ve Metrik Tanımları

Paket: `aicls.eval`. Karar: [ADR-0015](../adr/0015-metrik-tanimlari-dondurulmus.md). Her metrik burada bir kez tanımlanır, kodda bir kez uygulanır (`aicls.eval.metrics`), normalizasyon sürümü her raporda yazılır. 2027 ve 2028 raporlarında aynı sayı aynı anlama gelir; tanım değişirse yeni sürüm (`latex_v2`) açılır, eskisi silinmez.

## 1. Normalizasyon sürümleri (`aicls.eval.normalizers`)

### `latex_v1` (OCR çıktıları ve ifadeler için)

1. `normalize_latex` ile aynı yüzey temizliği: NFC, `\left`/`\right`, `\,` `\;` `\!` `\quad` `~` `\displaystyle` silme, `\dfrac`/`\tfrac` → `\frac`, Unicode → komut.
2. Token'lara ayır (komut `\[a-zA-Z]+`, tek karakter, sayı dizisi).
3. Tek token'lık üs/alt indis kanonik parantez: `x^2` → `x^{2}`, `x_1` → `x_{1}`.
4. Gereksiz `{}` düzleştirme: `{x}` → `x` (yalnızca tek token içeriyorsa ve üs/alt indis argümanı değilse).
5. Token'ları tek boşlukla birleştir.

`latex_v1` idempotenttir (özellik testi). MathWriting'in `normalized_label` alanına uygulanır; ham `label` kullanılmaz (rapor bunu belirtir).

### `tr_v1` (ASR transkriptleri için)

Yalnızca şu kurallar — **sözlük katlama, ASR hata tablosu veya matematik terimi birleştirme yoktur** (hakemlerin işaret ettiği "metriği test setine göre eniyileme" riskini kapatır; bu katlamalar yalnızca `aicls.nlu.normalizer` içindedir ve etkileri `command_intent_accuracy` ile ayrıca ölçülür):

1. NFC.
2. Türkçe küçük harf: `İ→i`, `I→ı`, ardından `casefold`.
3. Kesme işareti ve tire dahil tüm noktalama kaldırılır (`x'in` → `xin`; iki taraf da aynı kurala tabi olduğu için tutarlı).
4. Rakam dizileri Türkçe sayı sözcüklerine çevrilir (`"0'dan 1'e"` → `"sıfırdan bire"` değil, `"sıfır dan bir e"` — ek eklemesi yapılmaz; kural: rakam → sayı sözcüğü, ekler ayrı token kalır). Belgelenmiş tek sözcük-sayısı değiştiren kuraldır.
5. Boşluk sıkıştırma.

`test_tr_v1_normalizer_changes_word_count_only_by_digit_rule` bu sınırı sabitler. Raporlar **ham WER** ve **tr_v1 WER**'i yan yana verir.

## 2. Metrikler

| Metrik | Tanım | Uygulama | Hedef |
|---|---|---|---|
| **CER** (karakter hata oranı, İP-3) | Korpus düzeyi: `Σ(S+D+I) / Σ len(ref)` — `latex_v1` sonrası dizelerin **karakterleri** (boşluklar çıkarılır) üzerinde Levenshtein; kaynak: Michael ve ark. 2019 (öneri A.2.1). | `corpus_cer(pairs, norm="latex_v1")`; saf Python Levenshtein; `jiwer` yalnızca çapraz denetim testinde | < 0.10, MathWriting resmi test bölmesi |
| **TER** (token hata oranı) | Aynı hesap `latex_v1` token'ları üzerinde | `corpus_ter` | ikincil sütun |
| **ExpRate** (ifade düzeyi tam eşleşme) | `latex_v1(ref) == latex_v1(hyp)` oranı | `exact_match` | ikincil |
| **Sembolik eşdeğerlik** | İkisi de `latex2sympy` ile ayrışıyorsa `simplify(a−b)==0` veya 5 rastgele reel noktada `|a−b| < 1e-6·(1+|a|)`; sonuç üç durumlu: `equal` / `different` / `unparsable` (ayrıştırılamayan asla "yanlış" sayılmaz, ayrı raporlanır) | `symbolic_equivalence` | OCR ikincil, CAS birincil |
| **Ayrıştırma başarısı** (İP-4) | `valid_inputs.jsonl` satırlarında parse + to_sympy istisnasız **ve** gidiş-dönüş `ok` | `parse_success_rate` | > 0.99 |
| **CAS doğruluğu** (İP-4) | `problems.jsonl` satırlarında `checker` başarılı / toplam; `unsupported` ayrı sayılır (payda içinde kalır) | `run_cas_eval` | > 0.99 |
| **WER** (İP-3) | Korpus düzeyi `Σ(S+D+I) / Σ words(ref)`; **konuşmacı-ayrık** test bölmesi (`audio_commands_splits.json`); ham ve `tr_v1` | `corpus_wer` | < 0.10 (tr_v1), ham da raporlanır |
| **Komut niyet doğruluğu** (İP-3/İP-4 sınırı, İP-2 "komut iletimi") | ASR transkripti → `parse_command` → niyet == altın **ve** komut tam eşleşme | `run_asr_eval` içinde `command_intent_accuracy`, `command_exact` | > 0.95 |
| **NLU doğruluğu** | Yazılı altın kümede niyet doğruluğu, yuva F1, komut tam eşleşme, açıklama oranı | `run_nlu_eval` | ≥ 0.95 |
| **Gecikme** (İP-2/İP-4/İP-9) | **İstemci** ölçer: komut gönderim anından KaTeX render tamamlanmasına (`performance.now()`); sunucu `Server-Timing` aşamaları eklenir. Raporlanan: **ortalama** (sözleşme ifadesi), p50, p95, n, modalite bazında | `latency_samples` tablosu + `latency_stats`; CI'da stub tabanlı sentetik `run_latency_eval` | ortalama < 3000 ms |
| **Kitap etiketleme kapsamı** (İP-8) | `coverage(TagResult)` = etiketlenen birim / tanınan başlık | `aicls book coverage` | 1.00 |
| **Kitap getirme doğruluğu** (İP-8) | `fetch_test.jsonl`: en iyi aday == beklenen birim | `run_book_eval` | > 0.95 |
| **SUS** (İP-2/İP-9) | 10 madde 1–5; tek maddeler `(x−1)`, çift maddeler `(5−x)`, toplam × 2,5 | `sus_score`; `sus_responses` tablosu | > 75 |

Bütün başlık metriklerinde `bootstrap_ci` (1000 yeniden örnekleme, seed 1234) ile %95 güven aralığı raporlanır.

## 3. Test kümeleri

| Küme | Yol | Boyut / sürümleme | Kullanım |
|---|---|---|---|
| Müfredat problemleri | `datasets/curriculum/problems.jsonl` (+ `cards/`, `FROZEN.sha256`) | 300 (8. hafta) → 1000 (6. ay); `status` alanı; dondurulmuş satır değişmez | CAS doğruluğu, ayrıştırma |
| Geçerli LaTeX girdileri | `datasets/latex/valid_inputs.jsonl` | ≥ 2000; kaynak etiketli | Ayrıştırma başarısı |
| Desteklenmeyen yapılar | `datasets/latex/unsupported.jsonl` | | Hata politikası |
| Türkçe komutlar | `datasets/nlu/commands_test.jsonl`, `asr_noisy_tr.jsonl` | ≥ 300 / türetilmiş | NLU, gürültü dayanımı |
| MathWriting test | `AICLS_DATA_ROOT/mathwriting/shards/test` (resmi bölme; ~7,6 bin) | manifest sha | OCR CER |
| Ses komutları test | `AICLS_DATA_ROOT/audio/manifest.jsonl` + `datasets/manifests/audio_commands_splits.json` | konuşmacı-ayrık, tabakalı; her dışa aktarımda sürüm tarihi | WER |
| Kitap getirme | `datasets/book/fetch_test.jsonl` + `book/sample/` | bölüm başına büyür | İP-8 |
| Gecikme komutları | `datasets/nlu/commands_test.jsonl` alt kümesi (`tags: latency`) | 200 komut | CI sentetik gecikme |

Her küme `datasets/README.md` içinde kaynak, lisans, sürüm ve dondurma kuralıyla listelenir; JSONL dosyalarının SHA-256'sı `fingerprint_jsonl` ile hesaplanıp rapora yazılır.

### 3.1 Uygulama notları — müfredat test seti (2026-09-07)

`datasets/curriculum/` ilk kez derlendi (`curriculum-2026.09-v1`, 1204 problem). Bölüm 3'ün
tasarımına göre **saptığımız** ve **eklediğimiz** noktalar; ayrıntı ve ölçülen değerler
`datasets/curriculum/README.md` içindedir (depo kökünde, `docs/` ağacı dışında).

1. **Köken (rapor için bağlayıcı).** Küme proje içinde yazılmıştır: elle yazılmış müfredat
   şablonlarından makine ile genişletilmiştir; bağımsız bir derlemden örneklenmemiştir. Bu
   yüzden aşağıdaki CAS doğruluk sayısı **bağımsız bir ölçüm değil, regresyon ölçüsüdür** ve
   2027-05-01 raporunda bu cümle ile birlikte verilmelidir (KNOWN_ISSUES "Metric provenance").
   Ölçümü totolojiden ayıran nokta, beklenen yanıtların **CAS'a sorulmamış** olmasıdır:
   `datasets/curriculum/generate_cards.py` yalnızca müfredat kuralının verdiği kapalı formu
   kurar, `scripts/build_curriculum_testset.py` motoru bununla karşılaştırır.
2. **Dondurma kapsamı (sapma).** `docs/mimari/latex-ayristirici.md §7.2` yalnızca
   `status: frozen` satırların `FROZEN.sha256`'ya yazılmasını söyler; v1'de **tüm satırlar**
   yazılır, çünkü satırlar makine ile üretilmiştir ve tek bir sessiz değişiklik ölçümü fark
   ettirmeden kaydırabilir. Buradaki `frozen` "bayt düzeyinde dondurulmuş" demektir;
   **"incelenmiş" demek değildir** — inceleme durumu `reviewed_by` alanındadır ve v1'de tüm
   satırlarda boştur.
3. **Köken başlığı satırı (ekleme).** `problems.jsonl`'ın 1. satırı `id: "_header"`,
   `topic: "_meta"`, `_header: true` taşıyan bir köken kaydıdır. `aicls.data.load_curriculum`
   her satırı `Problem` olarak doğruladığı için başlık, geçerli **ve doğru yanıtlı** önemsiz bir
   problem (`simplify 1`) olarak yazılmıştır: onu süzmeyi unutan bir tüketici hiçbir ölçütü
   bozamaz. Tüketiciler `row["id"] == "_header"` ile süzmelidir.
4. **Belirsiz integral ölçütü (ekleme).** Bir ilkel yalnızca bir sabit farkla belirlidir;
   `checker: symbolic` + `checker_args: {up_to_constant: true, var: "x"}` ile karşılaştırma
   `d/dx(motor) ≡ d/dx(beklenen)` üzerinden yapılır ve ayrıca `d/dx(motor) ≡ integrand`
   doğrulanır. Enum'a yeni bir `checker` adı **eklenmemiştir**
   (`aicls.data.curriculum.CheckerName` bir `Literal`'dır).
5. **Sonsuz değerli limitler (ekleme).** `symbolic_equivalent(oo, oo)`,
   `simplify(oo - oo) = nan` yüzünden `"different"` döner. Derleyicinin `compare()` işlevi bu
   yüzden önce kimlik denetimi (`actual == expected`) yapar; sonsuz sonuçlu 8 satırda ayrıca
   `checker: exact_latex` kullanılır. `aicls.mathengine.equivalence` düzeltilirse bu satırlar
   `symbolic`'e çevrilebilir.
6. **Ayrıştırma ölçütünün ayrıştırılması.** Gidiş-dönüş uyuşmazlığı bir *basım* gözlemidir,
   yanlış yanıt değildir; derleyici bunu `parse_success` (sıkı tanım) içinde sayar, CAS
   doğruluk oranına karıştırmaz. `text_tr` satırları ayrıştırma paydasına girmez.
7. **Bulgu: çözüm kümeleri dilbilgisi dışında.** Motorun ürettiği `\left\{-2, 2\right\}`
   biçimi v1 dilbilgisinde yoktur (§3.2), yani kullanıcı kendi sonucunu girdi kutusuna geri
   yapıştırırsa ayrıştırma hatası alır. 102 satır bu durumdadır;
   `test_expected_latex_stays_inside_the_grammar_except_for_solution_sets` bunu sabitler.
8. **Bulgu: `\sum` değerlendirilmiyor.** `Simplify` işlemi `\sum_{k=1}^{n} k` girdisini
   değerlendirmez ve sonuç `kind="number"` etiketiyle döner (yakınsak seriler için).
   Bu yüzden seri toplamları kümeye **kısmi toplamların limiti** olarak yazılmıştır (sınıfta
   öğretilen tanım); `Sum` desteklenirse ayrı bir aile eklenebilir.

Ölçülen değerler (sympy 1.14, `InProcessExecutor`): CAS doğruluğu **1204/1204 = 1,0000**;
ayrıştırma başarısı (sıkı) **1134/1134 = 1,0000**; desteklenmeyen 0; ortalama motor süresi
~5 ms, p95 ~16 ms. İlk tam koşumda 14 uyuşmazlık çıkmış, **tamamı kümenin kendi hatası**
olduğu görülüp düzeltilmiştir (3 hatalı şablon çıktısı, 11 gidiş-dönüş uyuşmazlığı); motorda
hata bulunmamıştır.

## 4. Rapor biçimi

`aicls eval <kind>` → `reports/metrics/<kind>_<split>_<gitsha7>.json`:

```json
{
  "name": "ocr_test", "kind": "ocr",
  "git_sha": "3f9c1e2...", "dirty": false, "citable": true,
  "dataset_fingerprint": "sha256:...", "dataset_version": "mathwriting-2024.09",
  "model_version": "ocr-0.3.0", "model_sha256": "...", "config_hash": "...",
  "normalizer_versions": {"latex": "v1"},
  "n": 7644,
  "metrics": {"cer": 0.087, "ter": 0.112, "exprate": 0.61, "symeq_equal": 0.66, "symeq_unparsable": 0.03},
  "ci95": {"cer": [0.083, 0.091]},
  "breakdown": {"by_label_length": {...}, "by_writer_split": {...}},
  "failures_path": "reports/metrics/ocr_test_3f9c1e2_worst100.jsonl",
  "timestamp": "2027-03-14T10:22:31Z",
  "definitions_ref": "docs/ml/degerlendirme.md#2-metrikler"
}
```

- `reports/metrics/latest.json` işaretçi dosyasıdır; `poe metrics-promote reports/metrics/ocr_test_3f9c1e2.json` inceleme sonrası günceller.
- `poe report-check`: her terfi edilmiş dosyanın `git_sha`'sı `HEAD`'in atası mı, `dirty=false` ve `citable=true` mi, `dataset_fingerprint` `datasets/manifests/*.json` veya JSONL parmak iziyle eşleşiyor mu, `normalizer_versions` güncel mi — biri tutmazsa CI kırmızı. **Bağlantı denetimi (wiring check) olarak işaretlenmiş bir girdi de reddedilir**: ölçütleri zaten `null`'dır, ama girdinin kendisi belgelerin okuduğu dosyada duramaz (`check_promoted` → `check_report(..., reject_wiring_check=True)`; `aicls report check --json` reddedilen türleri `wiring_checks` altında sayar).
- Aynı JSON `eval_runs` tablosuna `is_reportable` ile yazılır (`aicls report promote`), böylece yönetici sayfası ve MkDocs makroları tek kaynaktan okur.
- MkDocs makroları (`mkdocs-macros-plugin`) rapor sayfalarında `{{ metrics.ocr.cer }}`, `{{ metrics.asr.wer_tr_v1 }}`, `{{ metrics.cas.correct_rate }}`, `{{ metrics.latency.mean_ms }}` değerlerini `latest.json` üzerinden çeker ve dipnotta git SHA + parmak izi + tarih yazar; rapor bölümlerinde elle yazılmış sayı yoktur.
- `render_markdown(reports)` sözleşme hedefi sütunlu tablo üretir (`docs/raporlama/sonuclar.md`, otomatik; elle düzenlenmez).
- **Stub arkalı bir koşum `latest.json`'a sayı yazmaz.** Ölçütün öznesi öğrenilmiş bir model olan türlerde (`ocr`, `asr` — `MODEL_BACKED_KINDS`) `promote()` her ölçütü `null` yapar ve girdiye `measurement: "wiring_check"` ile İngilizce bir gerekçe yazar; koşumun gerçekten gözlediği değerler aynı girdinin `wiring_check.metrics` alanında ve koşum dosyasının kendisinde durur. Ölçüt **anahtarları** silinmez, çünkü makrolar `on_undefined: strict` ile çalışır — `{{ metrics.ocr.cer }}` yine çözülür, yalnızca alıntılanacak bir sayı yoktur. `latency` de `stub` işaretlidir ama ölçtüğü NLU → CAS hattı gerçek koddur (`stub` orada "tarayıcı değil, sunucu tarafı" demektir), o yüzden onun sayısı terfi eder.
- **`wiring_check` tek bir boolean değil, gerekçe listesidir** (2026-09-08 eki). Bayrak eskiden yalnızca "veri sentetikti" demek olduğu için, RTX 5090'daki ilk gerçek veri koşusu — 100 örnek üzerinde 20 adım, rastgele başlatılmış model, 16 örneklik doğrulama — `fit_report.json`'a `wiring_check: false` yazdı. Bir bayrağın yanlış olabileceği en kötü yön budur: fazla iddia. `aicls.eval.report.assess_wiring_check(...)` artık yukarıdaki tablonun her satırını bağımsız olarak denetler ve bir `WiringCheckVerdict` döndürür; boolean bu listeden **türetilir**, liste de sayının yanında yolculuk eder (`fit_report.wiring_check_reasons`, `latest.json`'da `wiring_check.reasons`), böylece okuyan kişi *neden* olduğunu görür. Eşikler keyfi değildir: `MIN_VAL_SAMPLES = 100`, `CER < 0,10` hedefini iki katından ayırabilmek için gereken en küçük n'dir (en kötü durum `1,96·0,5/√n < 0,10` ⇒ `n > 96`); `MIN_REPORTABLE_STEPS = 100` ise ihtiyatlı bir tabandır — geçmek kanıt değildir, geçmemek kesin olarak kanıtsızlıktır. Bilinmeyen olgu asla koşumun aleyhine sayılmaz (her argüman `None`/`False` varsayılanıyla "bunu yargılama" demektir), çünkü meşru bir final koşusunu işaretleyen bir bayrağa bir daha güvenilmez.
- **OCR koşumunda her örnek, modele verilmeden önce körleştirilir** (`aicls.eval.runners.blind_sample`): `Ink.label` alanı kopyadan silinir, geometri ve `sample_id` korunur. Stub tanıyıcı etiketi bulduğunda onu aynen geri verdiği için, körleştirme olmadan ölçüm kendi cevap anahtarını eline almış bir adayın sınavıdır. ASR yolunda ayrı bir adım gerekmez: koşucu çeviriciye yalnızca dalga biçimini verir, `text` / `speaker` / `intent` taşıyan örneğin kendisini hiç vermez.

### 4.1 Koşudan rapora: bir eğitim koşusu nasıl alıntılanabilir sayıya dönüşür

Bir sayının rapora girebilmesi için **kesintisiz bir zincir** gerekir. Zincirin her halkası bir dosya üretir ve bir sonraki halka o dosyanın kimliğini içine yazar; herhangi bir halka kopuksa sayı alıntılanamaz. Zincir beş adımdır.

```text
1. EĞİTİM      run_manifest.json      git SHA + kirli bayrağı + config hash + veri parmak izleri
       ↓                              → citable: git yanıt verdi VE ağaç temiz
2. AĞIRLIK     checkpoint.json        model.safetensors'ın sha256'sı
       ↓       aicls models import    → model_versions (path, sha256, config_hash, git_sha, run_id)
       ↓       aicls models activate  → is_active
3. ÖLÇÜM       reports/metrics/       aicls eval ocr --model ocr-0.3.0 --split test
       ↓       <ad>_<gitsha7>.json    → dataset_fingerprint, model_version, model_sha256,
       ↓                                normalizer_versions, n, ci95, citable
4. KAYIT       eval_runs tablosu      is_reportable = report.citable (raporun kendi kararı)
       ↓
5. TERFİ       aicls report check     git ataliği + dirty=false + citable=true + parmak izi eşleşmesi
               aicls report promote   → latest.json → MkDocs makroları → docs/raporlama/*.md
```

**Adım 1 — koşu kimliği.** `Trainer.fit()` ilk optimizer adımından **önce** `run_manifest.json` yazar: git SHA, dal, kirli bayrağı, kirliyse diff'in SHA-256'sı, çözümlenmiş yapılandırmanın hash'i, veri manifestlerinin parmak izleri, `pip freeze` SHA-256'sı (tam liste `env.json`), Python/torch/CUDA/cuDNN sürümleri, GPU adı ve sürücüsü, hostname, başlangıç zamanı. `citable` alanı bir doğrulayıcıyla **yeniden hesaplanır**: git yanıt verdi **ve** ağaç temiz. Kirli ağaç `train.allow_dirty: false` iken koşuyu tümüyle reddeder; `--allow-dirty` ile koşu çalışır ama `citable: false` damgasını yer ve zincir orada biter.

**Adım 2 — ağırlığın kimliği.** Kontrol noktası dizinindeki `checkpoint.json`, `model.safetensors`ın SHA-256'sını taşır. `aicls models import` bu özeti diskteki baytlarla **doğrular** (uyuşmazlıkta dosyanın tamamını reddeder) ve `model_versions` satırına yazar; `config_hash`/`git_sha`/`run_id` kontrol noktasının kendi `run_manifest.json`'undan okunur. `models/registry.yaml` girdisindeki `gate_refs` alanı, bu ağırlığı mümkün kılan onay defteri satırlarını adlandırır — hakemin "bu model hangi onayla üretildi" sorusunun yanıtı budur.

**Adım 3 — ölçümün kimliği.** `EvalReport`, ölçümün kime ve neye ait olduğunu kendi içinde taşır: veri kümesinin parmak izi (`sha256:<hex>`; manifest, parça manifesti ve bölme dosyası aynı biçimi kullanır), veri sürümü, model sürümü ve model özeti, normalizasyon sürümleri (`latex_v1`, `tr_v1`), örnek sayısı `n`, `bootstrap_ci` ile %95 güven aralığı, git SHA ve kirli bayrağı. ASR tarafında **konuşmacı-ayrık bölme dosyası zorunludur**: `run_asr_eval` bölme parmak izi olmadan WER üretmeyi reddeder.

**Adım 4 — kayıt.** Aynı belge `eval_runs` tablosuna yazılır; `is_reportable` doğrudan `report.citable` değerinden gelir, yani çağıranın isteğinden değil raporun kendi hesabından. Böylece yönetici sayfası, MkDocs makroları ve gelişme raporu tek kaynaktan okur.

**Adım 5 — terfi.** `aicls report check`, terfi edilmiş her dosya için şunları arar: `git_sha` `HEAD`'in **atası** mı; `dirty == false` ve `citable == true` mi; `dataset_fingerprint` `datasets/manifests/*.json` veya JSONL parmak iziyle eşleşiyor mu; `normalizer_versions` güncel mi. Biri tutmazsa çıkış kodu 1 ve terfi yapılmaz. `aicls report promote` `latest.json`'ı günceller; rapor sayfaları sayıyı makroyla (`{{ metrics.ocr.cer }}`) çeker ve dipnotta git SHA + parmak izi + tarih yazar. **Rapor bölümlerinde elle yazılmış sayı yoktur.** `scripts/reproduce_report.ps1`, `report-YYYY-MM` etiketinden temiz bir klonla zinciri baştan koşturup sonuçları karşılaştırır.

#### Zinciri kesen durumlar

| Durum | Nerede yakalanır | Sonuç |
|---|---|---|
| Kirli çalışma ağacı | `RunManifest` doğrulayıcısı | `citable: false`; `report check` reddeder |
| Git yanıt vermiyor (klon değil, git yok) | aynı yerde | `citable: false` — belgedeki "kirli ⇒ alıntılanamaz" kuralından daha sıkı |
| Stub model | `EvalReport.breakdown.stub: true` → `stub_backend` | Sayı **kanıt değildir**; rapora giremez |
| Sentetik veri | `assess_wiring_check(synthetic_data=True)` → `synthetic_data` | Aynı |
| Adım bütçesi eğitim sayılamayacak kadar küçük | `steps < MIN_REPORTABLE_STEPS` (100) → `step_budget_too_small` | Aynı |
| Öğrenme oranı ısınma rampasını hiç geçmemiş | `steps <= optim.warmup_steps` → `warmup_not_completed` | Aynı |
| Doğrulama alt kümesi ölçüte anlam veremeyecek kadar küçük | `n_val < MIN_VAL_SAMPLES` (100) → `val_subset_too_small` | Aynı |
| Ön eğitimli olması gereken ağırlıklar rastgele başlatılmış | `assess_wiring_check(random_init=True)` → `random_init` | Aynı |
| Alıntılanabilir olmayan (kirli) ağaç | `RunManifest.citable is False` → `not_citable` | Aynı |
| Veri parmak izi yok | `report check` | Terfi reddedilir |
| Rapor `HEAD`'in atası olmayan bir commit'te üretilmiş | `report check` | Terfi reddedilir |
| Ağırlık özeti diskle uyuşmuyor | `aicls models import` | İçe aktarma tümüyle reddedilir |
| Bölme dosyası yok (ASR) | `run_asr_eval` | WER üretilmez |

#### 2026-09-08 durumu: zincir hiç tamamlanmadı

!!! danger "Bugün elimizde OCR CER sayısı da ASR WER sayısı da YOK"

    Zincirin ilk halkası bile gerçek anlamda kurulmadı: **hiçbir model eğitilmedi, hiçbir veri
    kümesi indirilmedi, hiçbir ağırlık indirilmedi**, kurulu torch CPU derlemesidir
    (`2.11.0+cpu`) ve `model_versions` tablosunda tek satır yoktur.

    `reports/metrics/` içinde bugün duran, model içeren her dosya bir **bağlantı denetimidir**:

    * `ocr_test_*.json` — `cer: 0,9679`, `exprate: 0,0156` (2026-09-08). Bu bir OCR sonucu
      **değildir**: stub tanıyıcı sabit müfredat tablosundan tahmin eder, yani bu sayı bir tablo
      aramasının CER'idir. Aynı dosyanın `label_leak_control` bloğu, körleştirme kapatıldığında
      aynı stub'ın **0,0000** aldığını gösterir — 2026-09-08'e kadar `latest.json`'a terfi eden
      sayı buydu ve stub kendi etiketini geri verdiği için ortaya çıkmıştı. Koşucu artık her
      örneği modele vermeden önce körleştiriyor (`blind_sample`); `training/scripts/evaluate.py`
      aynı şeyi kendi `blind_ink`'iyle kaynakta zaten yapıyordu.
    * `asr_test_*.json` — `wer_tr_v1: 1,11`. Stub, dalga biçimini hash'leyip bir cümle havuzundan
      seçim yapar; bu, rastgele çekilişin WER'idir.
    * `latest.json`'ın `ocr` ve `asr` blokları **hiçbir sayı yayımlamaz**: her ölçüt `null`,
      `measurement: "wiring_check"` ve gerekçe yazılıdır. Stub'ın gerçekte aldığı değerler
      `wiring_check.metrics` altındadır; saklanan bir şey yok, yalnızca yayımlanan bir şey yok.
    * Her ikisi de `citable: false`, `dirty: true`, `model_version: "*-stub-*"` taşır.

    Sözleşmeye bağlı **CER < %10** ve **WER < %10** hedefleri için kanıt, ancak dört kapı
    (`DATASET_DOWNLOAD`, `WEIGHTS_DOWNLOAD`, `CUDA_INSTALL`, `TRAINING`) açıldıktan sonra,
    gerçek MathWriting test bölmesi ve gerçek konuşmacı-ayrık ses test bölmesi üzerinde
    üretilebilir. 2027-05-01 gelişme raporunda bu iki satır **ölçülmedi** olarak yazılmalı ya
    da yukarıdaki zincirden geçmiş gerçek sayılarla doldurulmalıdır; arada üçüncü bir seçenek
    yoktur. Ayrıntı: [egitim-altyapisi.md §11](egitim-altyapisi.md#11-uygulama-notları-aiclsml-aiclscli-training-2026-09-08),
    işletim: [egitim-el-kitabi.md](egitim-el-kitabi.md).

    Aynı uyarı, kökeni proje içinde olan kümeler için de geçerlidir (§3.1): `parser` ve `cas`
    sayıları **regresyon ölçüsüdür**, bağımsız bir derlemden örneklenmiş değildir.

## 5. Değerlendirme koşumları

| Komut | Girdi | Çıktı metrikleri |
|---|---|---|
| `aicls eval parser` | `valid_inputs.jsonl` | rate, n, ci95, kaynak kırılımı |
| `aicls eval cas` | `problems.jsonl` + üretim sandbox'ı | parse_success, correct_rate, unsupported, konu kırılımı, ortalama/p95 ms, işçi yeniden başlatma sayısı |
| `aicls eval nlu` | `commands_test.jsonl`, `asr_noisy_tr.jsonl` | intent_accuracy, slot_f1, command_exact, clarification_rate |
| `aicls eval ocr --model ocr-0.3.0 --split test` | parçalar + `Recognizer` | cer (+ci), ter, exprate, symeq üçlü, uzunluk kırılımı, en kötü 100 |
| `aicls eval asr --model asr-0.2.0 --split test` | manifest + split + `Transcriber` | wer_raw, wer_tr_v1 (+ci), cer, command_intent_accuracy, üniversite/cinsiyet kırılımı |
| `aicls eval book` | örnek bölüm + `fetch_test.jsonl` | fetch_top1, fetch_top3, tagging_coverage |
| `aicls eval latency --n 200` | stub modeller + `TestClient` | mean/p50/p95 aşama bazında (CI); gerçek rapor için `--from-db --since` ile `latency_samples` |

`tests/perf/test_latency_budget.py` aşama bütçelerini sabitler (stub modellerle): `nlu ≤ 50 ms p50`, `parse ≤ 10 ms`, `cas ≤ 150 ms` (müfredat ortanca problemleri), `book_resolve ≤ 20 ms`, metin yolu toplam `≤ 300 ms p50`; bütçe tablosu `docs/mimari/api.md` ile bağlantılı olarak `docs/raporlama/ip-02-web.md` içinde raporlanır. Gerçek modellerle bütçe: OCR çıkarımı ≤ 800 ms, ASR (whisper-small, 5 s ses) ≤ 1200 ms, toplam ortalama < 3000 ms.

## 6. Testler

`test_cer_reference_values_hand_computed`, `test_cer_identical_zero_and_bounds`, `test_cer_uses_normalized_label_not_raw`, `test_ter_vs_cer_differ_on_brace_case`, `test_wer_corpus_matches_jiwer_reference`, `test_wer_raw_and_tr_v1_both_reported`, `test_tr_v1_normalizer_changes_word_count_only_by_digit_rule`, `test_tr_v1_is_idempotent`, `test_latex_v1_is_idempotent`, `test_symbolic_equivalence_accepts_factored_vs_expanded`, `test_symbolic_equivalence_returns_unparsable_not_different`, `test_parse_success_rate_counts_silent_misparse_as_failure`, `test_bootstrap_ci_contains_point_estimate`, `test_sus_score_reference_values`, `test_latency_stats_mean_p50_p95`, `test_eval_report_has_fingerprint_git_sha_and_normalizer_versions`, `test_report_check_rejects_non_ancestor_sha`, `test_report_check_rejects_dirty_run`, `test_asr_eval_reports_command_intent_accuracy`, `test_asr_eval_refuses_without_speaker_split_file`.
