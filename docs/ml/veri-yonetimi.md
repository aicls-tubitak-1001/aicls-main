# Veri Yönetimi

Karar: [ADR-0008](../adr/0008-veri-yonetimi-manifest-onedrive-disi.md). İlke: büyük veri git'e girmez, OneDrive'a girmez; küçük değerlendirme setleri git'te sürümlenir; her veri kökü SHA-256 manifestiyle tanımlanır; kişisel veri takma adla ve rıza sürümüyle işlenir.

## 1. Veri kümeleri

| Küme | Kaynak / lisans | Boyut | Kullanım | Kapı |
|---|---|---|---|---|
| **MathWriting** (Gervais ve ark., 2024) | Google Research; **CC BY-NC-SA 4.0** (ticari kullanım yasak, türev aynı lisansla); InkML; ~230 bin gerçek + ~400 bin sentetik ifade + sembol alt kümesi; resmi train/valid/test bölmeleri | `mathwriting-2024.tgz` **3 096 141 721 B** (2,88 GiB), ~20 GB açık, ~6 GB `npz` parça | OCR eğitimi/testi; `valid_inputs.jsonl` için normalize etiket örnekleri | `DATASET_DOWNLOAD` |
| **MathWriting örneklem** (resmi excerpt) | Aynı kaynak ve **aynı lisans**; aynı bölme düzeni | `mathwriting-2024-excerpt.tgz` **1 620 634 B** (1,55 MiB) | Kapı açıldığında ilk indirilecek olan: tüm hattı 3 GB harcamadan kanıtlar | `DATASET_DOWNLOAD` |
| **AI-CLS-TR-Speech** (özgün) | Proje; "Model Gelişimine Katkıda Bulun" sekmesi; Muş Alparslan, Mersin, Amasya; hedef 50 saat ham (~25 bin cümle) → artırma ile 200+ saat | 50 saat ≈ 5,8 GB (16 kHz PCM16) | ASR ince ayar ve konuşmacı-ayrık test | rıza + etik kurul (dış bağımlılık) |
| **Okuma Senaryoları Kütüphanesi** | Proje; `datasets/nlu/reading_scenarios.yaml`; ≥ 600 cümle, senaryo etiketi, niyet | metin | cümle istemleri, NLU altın kümesi tohumu | yok |
| **Müfredat problem seti** | Proje; YAML kartları | ≥ 1000 satır | CAS/ayrıştırma | yok |
| **Kitap örnek bölümü** | Kitap ekibi taslağı (İP-7, dış bağımlılık); etiketli Markdown | 10–30 birim | İP-8 geliştirme/test | yok |
| **Whisper ağırlıkları** | `openai/whisper-small` (MIT), `openai/whisper-large-v3-turbo` (MIT) | 0,5 GB / 1,6 GB | ASR taban modeli | `WEIGHTS_DOWNLOAD` |
| **Wav2Vec2 XLS-R** | `facebook/wav2vec2-xls-r-300m` (Apache-2.0) | 1,2 GB | ensemble B planı | `WEIGHTS_DOWNLOAD` |
| **Sentetik fikstürler** | `scripts/make_synthetic_fixtures.py`, seed 1234 | < 2 MB (git'te) | CI | yok |

Lisans metinleri ve alıntı biçimleri `datasets/README.md` ve veri kartlarında (`docs/raporlama/ip-03-yapay-zeka.md` altında MathWriting kartı ve AI-CLS-TR-Speech kartı) tutulur. MathWriting'in dağıtım koşulları (yeniden dağıtım, atıf) rapora aynen alınır.

!!! warning "MathWriting lisansı: CC BY-NC-SA 4.0 — ticari kullanım yasak"
    2026-09-08'de resmi depo (`google-research/mathwriting`) ve arşivin kendisi doğrulandı;
    katalogda daha önce yazan **Apache-2.0 yanlıştı**. Gerçek lisans
    **Creative Commons Attribution-NonCommercial-ShareAlike 4.0**:

    - **NonCommercial** — veri ve ondan türetilen çalışma ticari amaçla kullanılamaz. Projede
      konuşulan patent/ticarileşme aşaması için bu veriyle eğitilmiş ağırlıklar doğrudan
      taşınamaz; hukuki görüş alınmadan ticari bir sürümde kullanılmamalıdır.
    - **ShareAlike** — türev çalışma aynı lisansla dağıtılmalıdır.
    - **Attribution** — Gervais ve ark. (2024) atfı zorunludur.

    Aynı kısıt resmi örneklem (`mathwriting-2024-excerpt.tgz`) için de geçerlidir. Kısıt
    `src/aicls/data/catalog.yaml` içindeki `notes_tr` alanında da yazılıdır ve
    `tests/unit/data/test_catalog.py` ile sabitlenmiştir.

## 2. Dizin düzeni (OneDrive dışı)

`AICLS_DATA_ROOT` = `D:\aicls-data` (ikinci disk varsa) ya da `C:\aicls-data`; satın alınan iş istasyonunda NVMe. `scripts/check_env.py` yol OneDrive altındaysa uyarır ve `poe ci` bunu hata sayar.

```
D:\aicls-data\
├── aicls-dev.db                      # SQLite dev
├── cache\lark\                       # dilbilgisi önbelleği
├── mathwriting\
│   ├── mathwriting-2024.tgz          # indirilen arşiv; VARSAYILAN OLARAK AÇILMAZ
│   ├── raw\                          # yalnızca `download_mathwriting.py --extract` ile (~20 GB)
│   ├── shards\{train,valid,test,symbols,synthetic}\*.npz  # prepare_mathwriting.py + shards_manifest.json
│   ├── vocab.json
│   └── manifest.json
├── audio\
│   ├── raw\<speaker_pseudonym>\<ulid>.wav          # kabul edilen kayıtlar (toplama)
│   ├── manifest.jsonl                # aicls contrib export-manifest (yalnızca takma ad)
│   ├── noise_bank\                   # gürültü enjeksiyonu için (lisanslı/kayıtlı)
│   └── tts\                          # TTS artırma çıktıları (kapılı motor)
├── models\
│   ├── pretrained\whisper-small\     # download_weights.py; from_pretrained(local_dir)
│   ├── ocr-0.3.0\                    # kontrol noktası dizini (bkz. egitim-altyapisi.md §5)
│   └── asr-0.2.0\
├── runs\<run_name>-<gitsha7>-<ulid>\ # metrics.jsonl, params.json, env.json, run_manifest.json, ckpt\
├── mlruns\                           # MlflowTracker (opsiyonel)
├── uploads\{ink,images,plots}\       # API varlıkları (assets tablosu)
└── backups\
```

`AICLS_MODELS_ROOT` varsayılan `AICLS_DATA_ROOT/models`. Git deposunda `/data/` yoktur; `datasets/` (küçük) vardır.

## 3. Manifestler ve sürümleme

- **Büyük veri:** `aicls data manifest <root> --name mathwriting --license CC-BY-NC-SA-4.0 --source <url>` → `manifest.json` (`files[] {path, size, sha256}`, `fingerprint` = dosya listesinin SHA-256'sı, `retrieved_at`, `version`). `aicls data verify mathwriting` doğrular; `tests/data` işaretli testler manifest doğrulanmadan koşmaz.
- **Parçalar:** `shards_manifest.json` (parça başına sha256, örnek sayısı, split, `source_root`); parmak izi OCR raporlarına yazılır. `source_root` parçaların hangi kaynaktan üretildiğini söyler: açılmış dizin (`D:\aicls-data\mathwriting\raw\train`) ya da doğrudan arşiv akışı (`tar:D:\aicls-data\mathwriting\mathwriting-2024.tgz!train`). Örnek sırası kaynağa bağlıdır (dizinde yola göre sıralı, arşivde arşiv sırası). `tar` sıralı bir dizin gezintisi yazdığından iki sıra pratikte çakışır ve parçalar bayt bayt aynı çıkar: 2026-09-08'de 20 bin dosyalık sentetik bir fikstürde beş bölmenin beşinde de parmak izleri **aynı** ölçüldü. Bu bir gözlemdir, **garanti değildir** — başka bir sırayla paketlenmiş bir arşiv parça sınırlarını kaydırır ve parmak izini değiştirir (`test_shard_order_follows_the_archive_so_the_fingerprint_can_change`). Bu yüzden iki rapor karşılaştırılırken `source_root` da karşılaştırılmalıdır.
- **Ses:** her dışa aktarım `manifest.jsonl` + `datasets/manifests/audio_commands.json` (özet: saat, konuşmacı, üniversite/cinsiyet/yaş dağılımı, rıza sürümleri, parmak izi) + `audio_commands_splits.json` (konuşmacı-ayrık, tabakalı, seed). Split dosyası bir kez üretilir; yeni konuşmacılar eklendiğinde **eski test konuşmacıları test'te kalır** (`make_splits(..., frozen=previous_splits)`), böylece rapor dönemleri arasında test kümesi kayması olmaz.
- **Git'teki setler:** `datasets/**/*.jsonl` parmak izleri raporlara yazılır; `datasets/curriculum/FROZEN.sha256`; rapor etiketleri (`report-2027-05`) tüm setleri dondurur. DVC/Git LFS kullanılmaz (manifest tabanlı yaklaşım yeterli, araç yok; ADR-0008 sonradan DVC'ye geçişi manifestleri bozmadan mümkün kılar).
- **Sürüm adlandırma:** veri seti `<ad>-<YYYY.MM>` (`mathwriting-2024.09`, `aicls-tr-speech-2027.02`); model `<component>-<semver>`; rapor `report-<YYYY-MM>`.

## 4. Kişisel veri, KVKK ve rıza

Etik kurul başvurusu ve KVKK süreci başka bir ekibin sorumluluğundadır (dış bağımlılık). Bu depo süreci **teknik olarak zorlar**:

| Konu | Uygulama |
|---|---|
| Rıza metni | `consent_versions` tablosu; sürüm, `text_tr`, `sha256`, yürürlük tarihi, `ethics_approval_ref`. Arayüz `GET /contribute/consent/current` ile gösterir; kabul edilmeden kayıt ucu 403. |
| Kapı | `AICLS_ETHICS_APPROVAL_REF` boşsa toplama yönlendiricisi 503 `GATE_BLOCKED`; ekip içi boru hattı doğrulaması ayrı rıza ile `AICLS_ENV=dev`'de yapılır ve bu kayıtlar dışa aktarıma `source=internal` ile girer, raporlanan WER'e girmez. |
| Takma ad | `speakers.pseudonym` (P00001); kullanıcı bağı `speakers.user_id` yalnızca ilerleme takibi için ve **yalnızca admin**; eğitim manifesti ve tüm dışa aktarımlar bu sütunu içermez (`test_export_manifest_has_no_identity_fields`). |
| Demografi | isteğe bağlı, yalnızca bant: yaş bandı, cinsiyet, üniversite, aksan bölgesi. Ad, doğum tarihi, e-posta kayıtlarla saklanmaz. |
| IP/tarayıcı | tuzlu hash; 90 gün sonra NULL. |
| Depolama | `audio/raw` NTFS izinleriyle yalnızca servis hesabı; halka açık sunum yok; yönetici dinlemesi 5 dk'lık imzalı URL. Toplama VM'inde gece senkronizasyonundan sonra silinir. |
| Geri çekme | `POST /contribute/withdraw` ve `aicls contrib withdraw --speaker P00012`: ses, türev özellikler, `recordings` satırları silinir (`deleted_at`), rıza satırı ve denetim izi (içerik olmadan) kalır — KVKK "rıza/geri çekme kanıtı". Sonraki dışa aktarımlar bu konuşmacıyı içermez; daha önce eğitilmiş modeller model kartında "geri çekilen konuşmacı sayısı" ile not edilir. |
| Yedek | şifreli (AES), 30 gün; geri yükleme tatbikatı üç ayda bir. |
| Toplama operasyonu | `docs/raporlama/ip-03-yapay-zeka.md` altında "Veri toplama protokolü": senaryo kütüphanesi, hedef dağılım (üniversite, cinsiyet, yaş), QC eşikleri, inceleme kuyruğu, haftalık ilerleme panosu (`GET /contribute/stats`). |

### Uygulama notları (aicls.collect)

§4'teki kuralların koddaki karşılığı `aicls.collect` paketidir (`src/aicls/collect/`, kapsam %99,9; testler `tests/unit/collect/` ve `tests/integration/test_collect_contracts.py`).

| §4 satırı | Kod |
|---|---|
| Rıza metni | `resources/consent_texts.yaml` (sürüm 1.0, yürürlük 2026-05-01) → `current_consent()`; `accept_consent` metni ve SHA-256'sını `consent_versions` tablosuna bir kez yazar, sonraki kabuller aynı satırı paylaşır. |
| Kapı | `require_contribution_enabled()`; `AICLS_ETHICS_APPROVAL_REF` boşsa `ContributionGateError` (503 `GATE_BLOCKED`). `withdraw` bilinçli olarak kapısızdır. |
| Takma ad | `accept_consent` → `RecordingRepository.create_speaker` (P00001…); dosya yolları yalnızca `^P\d{5}$` deseninden kurulur, istemci girdisi hiçbir dizin bileşenine dönüşmez. |
| Demografi | `Demographics` (pydantic, `extra="forbid"`): ad/e-posta/doğum tarihi taşıyan bir istek doğrulamada reddedilir. |
| IP/tarayıcı | `hash_ip()` tuzlu SHA-256 üretir; `accept_consent` ham IP adresini (IPv4/IPv6) ve hash olmayan değerleri reddeder. |
| Depolama | `AICLS_DATA_ROOT/audio/raw/<takma ad>/<ulid>.wav`; kayıt 16 kHz mono PCM16'ya normalize edilerek `.part` geçici dosyası üzerinden atomik yazılır. QC'yi geçemeyen yükleme **hiç yazılmaz** ve satır açılmaz. |
| Geri çekme | `withdraw()` → `mark_withdrawn` (satırlar `deleted_at`, `ip_hash`/`user_agent_hash` NULL, rıza satırı korunur) + `audio/raw` ve `audio/features` altındaki dosyaların silinmesi. Yinelenen çağrı ilk geri çekme zaman damgasını korur. |
| Dışa aktarım | `export_manifest()` yalnızca `MANIFEST_FIELDS` alanlarını yazar; `speaker_id`, `user_id`, e-posta ve hash sütunları manifeste **girmez**. |

**Bölme dosyası (§3 "Ses" satırının uygulanışı).** `export_manifest`, `out.parent/audio_commands_splits.json` dosyasını okur; orada bulunan her konuşmacı-bölme ataması aynen korunur, yalnızca yeni konuşmacılar atanır. Yeni konuşmacılar `(üniversite, cinsiyet, yaş bandı)` tabakalarına ayrılır; tabaka içinde sıralama `sha256(f"{seed}:{takma ad}")` ile belirlenir (küresel RNG durumu yok) ve her konuşmacı, o tabakada kotasına göre en açık bölmeye gider. Böylece `make_splits(..., frozen=previous_splits)` davranışı sağlanır: **önceki test konuşmacıları test'te kalır** ve rapor dönemleri arasında WER karşılaştırılabilir olur.

**Kaynak etiketi.** Manifest satırlarındaki `source` alanı varsayılan olarak `contrib`'tir; ekip içi doğrulama kayıtları `export_manifest(..., source="internal")`, TTS artırma çıktıları `source="tts"` ile ayrı dışa aktarılır (§5: TTS örnekleri test bölmesine girmez).

**Eksik dosya denetimi.** `verify_files=True` (varsayılan) diskte bulunmayan kayıtları manifeste yazmaz ve `ManifestReport.missing_files` alanında raporlar; böylece eğitim betiği okunamayan bir yolla karşılaşmaz.

## 5. Artırma (öneri A.2.3)

`aicls.asr.augment`: SpecAugment (zaman/frekans maskeleri, log-Mel üzerinde), perde kaydırma (yeniden örnekleme tabanlı, ±2 yarım ton), gürültü enjeksiyonu (`noise_bank`, 10–25 dB SNR), hız bozma (0,9–1,1), `TtsAugmenter` protokolü (motor seçimi ayrı ADR ve kapı; TTS çıktıları `source=tts` ile manifestte ayrı tutulur ve **test bölmesine asla girmez**). Artırma yalnızca eğitim bölmesine uygulanır; 50 saat ham → 200+ saat hedefine giden oran `data.augment` yapılandırmasıyla belgelenir ve model kartına yazılır.

## 6. Testler

`test_data_manifest_build_and_verify`, `test_data_manifest_verify_detects_tampered_sha`, `test_check_env_warns_on_onedrive_path`, `test_speaker_disjoint_split_has_no_overlap`, `test_speaker_disjoint_split_is_stratified_within_5pct`, `test_split_freezes_previous_test_speakers`, `test_export_manifest_has_no_identity_fields`, `test_export_manifest_excludes_withdrawn_and_qc_failed`, `test_tts_samples_never_in_test_split`, `test_withdraw_deletes_files_keeps_consent_audit`, `test_prepare_mathwriting_shard_roundtrip_on_synthetic_inkml`.

## 7. Uygulama notları (`aicls.data`, 2026-09-07)

Bu bölüm paketin gerçekte ne yaptığını belgeler; sözleşmedeki adlar korunmuş, yalnızca aşağıdaki noktalar netleştirilmiş veya eklenmiştir.

- **Manifest.** `DataManifest.build(root, name, *, license, source_url, version=None, retrieved_at=None)`; `manifest.json`, `__pycache__`/`.git` gibi dizinler ve manifestin kendisi listeye alınmaz. `fingerprint = "sha256:" + sha256_json([[path, size, sha256], ...])` (yol sırasına göre; `aicls.core.hashing` ile aynı biçim). `verify()` **istisna atmaz**, `VerifyReport(ok, checked, missing[], changed[], extra[], fingerprint_matches)` döner; `summary()` tek satırlık İngilizce özet verir. `save(dir|path)` ve `load_manifest(path|dir)` JSON gidiş-dönüşünü sağlar.
- **Katalog.** `src/aicls/data/catalog.yaml` iki bölümdür: `datasets` (MathWriting — `mathwriting-2024.09`, `mathwriting-2024.tgz`, 3 096 141 721 B, **CC-BY-NC-SA-4.0**, kapı `DATASET_DOWNLOAD`; `mathwriting_excerpt` — resmi 1 620 634 B'lık örneklem, aynı lisans, aynı kapı; `audio_commands` — özgün, URL yok, **kapı yok**) ve `weights` (whisper-small, whisper-large-v3-turbo, wav2vec2-xls-r-300m). Yükleyici `load_catalog()` (süreç başına önbellekli) ve `catalog_entry(name)`; `CatalogEntry.describe()` kapılı indirme betiğinin yazdıracağı satırı üretir, `CatalogEntry.archive_name` ise indirme ve parçalama betiklerinin ortak dosya adı kaynağıdır (URL'nin son parçası — uzantı tahmin edilmez). **Pakette indirme kodu yoktur** (`test_catalog_module_contains_no_download_helper`); `sha256` alanı indirme sonrası doldurulur.
- **Katalog düzeltmesi (2026-09-08).** MathWriting girdisinin üç alanı yanlıştı ve resmi kaynağa karşı doğrulanarak düzeltildi: URL `.zip` → **`.tgz`** (eski adres HTTP 403 veriyordu), `expected_bytes` 3 113 851 289 → **3 096 141 721**, lisans Apache-2.0 → **CC-BY-NC-SA-4.0**. Sonuncusu teknik değil hukuki bir olgudur; §1'deki uyarıya bakın.
- **Bölmeler.** `make_splits(..., frozen=None)` sözleşmesine ek olarak `frozen` bir `Splits`, bir sözlük veya bir dosya yolu olabilir. Algoritma: grup başına tek tabaka (uyuşmazlık hata), tabaka içinde seed'li karıştırma + boyuta göre sıralama, ardından **en büyük açık (deficit) açgözlü** atama; eşitlikte önce genel açık, sonra `test > valid > train` sırası gözetilir (test kümesi korunur). Sonuç kimlik sırasından bağımsızdır (`sorted`). `Splits` bir pydantic modelidir: `train/valid/test`, MathWriting için `extra{symbols, synthetic}`, `seed`, `ratios`, `source`, `fingerprint` (özellik), `sizes()`, `split_of()`, `index()`; `save_splits`/`load_splits` JSON'a yazar/okur. Oranlar pozitif olmalı ve 1'e toplanmalıdır.
- **`mathwriting_official(root)`** iki düzeni de okur: split başına dizin (`<root>/train/*.inkml`) veya liste dosyası (`<root>/train.txt`). Kimlikler dosya kökleridir (`stem`).
- **Parçalar.** `prepare_shards(inkml_root, out, *, split, shard_size=4096, render=False, image_size=(96,384), stroke_width=2, is_synthetic=None, limit=None, skip_invalid=False)`; çıktılar `<out>/<split>/<split>-00000.npz` ve `<out>/<split>/shards_manifest.json`. Parça içeriği: `points (P,3) float32`, `stroke_ptr`, `sample_ptr`, `labels`, `writer_ids`, `sample_ids`, `is_synthetic`, isteğe bağlı `images (n,H,W) uint8`. Etiketler `normalize_label()` (tokenizer'ın kanonik biçimi) ile saklanır. Boş izler atlanır ve manifestte `n_skipped` olarak sayılır.
- **Akışlı parçalama (2026-09-08).** `prepare_shards` artık `prepare_shards_from_source(source, out, *, splits, ...)` üzerine ince bir sarmalayıcıdır; kaynak soyutlaması `aicls.data.sources` içindedir: `DirectoryInkmlSource` (açılmış ağaç, yola göre sıralı) ve `TarInkmlSource` (`.tgz` akışı, hiçbir dosya açılmaz). Gzip akışı geri sarılamadığı için istenen **tüm bölmeler tek geçişte** üretilir — bölme başına bir tampon. Üye yükleri `MAX_INKML_BYTES` (1 MiB) + 1 bayta kadar okunur, böylece bozuk bir arşiv sınırsız tampon ayırtamaz; `skip_invalid=True` okunamayan bir belgeyi `n_skipped`'a yazıp koşuyu sürdürür (varsayılan kapalı). `training/scripts/prepare_mathwriting.py --archive` bu yolu açar.
- **Ölçüm (2026-09-08, GPU konteyneri, `drvfs` veri kökü, 20 bin dosyalık sentetik fikstür).** Açma 166 s + dizinden parçalama 135 s = **301 s**; arşivden akıtma **6,5 s** ve diske tek bir InkML dosyası yazılmadı. Beş bölmenin beşinde örnekler ve parmak izleri aynıydı. Bu iki kod yolunun bu dosya sistemindeki **G/Ç ölçümüdür**; sentetik fikstür üzerinde alınmıştır, MathWriting'in içeriği hakkında hiçbir şey söylemez ve hiçbir sözleşme metriğinin kanıtı değildir.
- **Bellek eşlemesi.** `npz` **sıkıştırmasız** yazılır; `ShardReader(path, mmap=True)` zip üyesinin veri konumunu bulup `np.memmap` ile eşler (`test_shard_arrays_are_memory_mapped`), eşleme mümkün değilse sessizce `np.load` ile okur. `np.load(..., mmap_mode=...)` `.npz` için çalışmadığından bu yol gerekliydi.
- **Sözlük.** `build_vocab_from_shards(root, out, splits=("train",))` parça etiketlerinden `vocab.json` üretir (`aicls.ocr.LatexTokenizer`).
- **Veri kümeleri.** `MathWritingDataset` `torch.utils.data.Dataset`'ten **türemez**; `__len__`/`__getitem__` taşıyan bir map-style nesnedir ve `to_torch()` çağrısı torch'u tembel içe aktararak sarmalar (`import aicls.data` torch'u yüklemez — `test_importing_aicls_data_does_not_import_torch_or_matplotlib`). Öğe tipi `InkSample(features, label_ids, label, sample_id, writer_id, is_synthetic)`; belgelenen `(features, label_ids)` demet açılımı da çalışır. Ek anahtar argümanlar: `max_label_len`, `add_special`, `stroke_width`.
- **Sentetik oran.** Karışım örnekleyici düzeyinde ve **rastgeleliksiz**dir: `i` konumu sentetiktir ⇔ `floor((i+1)·r) > floor(i·r)`. Epoch uzunluğu `round(n_real/(1−r))`, `r=1.0` iken `n_synthetic`'tir; her gerçek örnek epoch başına tam bir kez görülür, sentetik örnekler gerekiyorsa döngüye alınır. `r>0` ise `<root>/synthetic` parçaları zorunludur.
- **Ses.** `AudioCommandDataset(manifest, split, *, feature_fn, splits=None, max_seconds=30)`. Bölme ataması zorunludur: satırdaki `split` alanı, komşu `audio_commands_splits.json` ya da açık `splits=` argümanı; hiçbiri yoksa yükleme `speaker_split_missing` ile reddedilir. Kimlik alanı taşıyan satır (`user_id`, `email`, ...) reddedilir. WAV okuma `aicls.asr.audio_io` ile, gerekirse 16 kHz'e yeniden örnekleme yapılır. Öğe: `AudioSample(features, text, speaker, sample_id, sample_rate, duration_s)`.
- **Müfredat.** `load_curriculum(path)` hem `problems.jsonl` hem `*.yaml` kart dosyasını okur; `Problem` pydantic modeli (`extra="allow"`, böylece kart alanları kaybolmaz), tekrar eden kimlik hatadır, hatalı satırın numarası `details["line"]` içindedir.
- **Sentetik üreteçler.** `synth_ink(rng, latex)` (Bezier; `rng` bir `np.random.Generator` **veya** tamsayı seed olabilir; şerit sayısı ve eğim etiketin `crc32`'sinden gelir, böylece aynı seed farklı ifadeyi farklı çizer), `synth_image(rng, latex, size)` (matplotlib **mathtext**, TeX kurulumu gerekmez; ayrıştırılamayan ifadede iz çizimine düşer), `synth_audio(rng, seconds, *, snr_db=None)`, `synth_inkml_dataset(out, n, seed)`. Ek: `synth_image_dataset(out, n, seed)`, `SyntheticInkDataset`, `SyntheticAudioDataset` (CI duman yapılandırmaları için).
- **Collate.** `collate_ink/collate_image/collate_audio` numpy sözlükleri döner (`features`, `feature_lengths`, `feature_mask`, `labels`, `label_lengths`, `labels_text`, `sample_ids`, `writer_ids`, `is_synthetic`; ses tarafında `texts`, `speakers`). `to_torch=True` ile tensöre çevrilir (`functools.partial(collate_ink, to_torch=True)` `DataLoader` için). Etiket dolgusu `<pad>` (id 0).
- **Kapsam.** `tests/unit/data` + `tests/integration/test_data_contracts.py`: 143 test, `aicls.data` satır kapsamı %94,2.
