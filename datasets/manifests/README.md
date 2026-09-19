# datasets/manifests/ — büyük veri için parmak izi ve bölme dosyaları

Bu dizin **veri içermez**. Büyük veri (MathWriting, ses kayıtları, model ağırlıkları) `AICLS_DATA_ROOT`
altında, git ve OneDrive dışında durur ([ADR-0008](../../docs/adr/0008-veri-yonetimi-manifest-onedrive-disi.md));
burada yalnızca o verinin **kimliği** sürümlenir: dosya listeleri, SHA-256 özetleri, lisans/kaynak bilgisi
ve konuşmacı-ayrık bölme listeleri. Raporlardaki her sayı buradaki bir parmak izine dayanır
([docs/ml/veri-yonetimi.md](../../docs/ml/veri-yonetimi.md) §3).

## Dosyalar

| Dosya | Üreten | İçerik |
|---|---|---|
| `mathwriting.json` | `aicls data manifest <root> --name mathwriting --license CC-BY-NC-SA-4.0 --source <url>` → `DataManifest.build()` | `files[] {path, size, sha256}`, `license`, `source_url`, `retrieved_at`, `version`, `fingerprint` |
| `audio_commands.json` | `aicls contrib export-manifest` + `DataManifest.build()` | ses dışa aktarımının özeti ve parmak izi (yalnızca takma ad; kimlik alanı yok) |
| `audio_commands_splits.json` | `aicls data split ...` → `aicls.data.make_splits()` → `save_splits()` | `train/valid/test` kimlik listeleri, `seed`, `ratios`, `fingerprint` |
| `mathwriting_shards.json` (isteğe bağlı kopya) | `aicls.data.prepare_shards()` | parça başına `sha256`, örnek sayısı, split, `fingerprint` |

Manifest dosyalarının kendisi veri kökünde de (`<root>/manifest.json`, `<shards>/<split>/shards_manifest.json`)
durur; buradaki kopyalar **git'te sürümlenen** kanıt nüshalarıdır.

## Kurallar

1. **Elle düzenlenmez.** Her dosya bir komutun çıktısıdır; elle değiştirilen bir parmak izi kanıt değeri taşımaz.
2. **Kişisel veri yoktur.** Ses manifestlerinde yalnızca takma ad (`P00012`), süre, bant düzeyinde demografi ve
   bölme bilgisi bulunur; `user_id`, e-posta, ad, doğum tarihi ve IP alanları `AudioCommandDataset` tarafından
   yükleme anında reddedilir (`FORBIDDEN_FIELDS`).
3. **Test kümesi kaymaz.** Yeni konuşmacılar eklendiğinde bölme dosyası
   `make_splits(..., frozen=<önceki bölme>)` ile yeniden üretilir: önceki test konuşmacıları test'te kalır
   (`test_split_freezes_previous_test_speakers`). Yeni dosya eskisinin üstüne yazılır ve commit mesajında
   eski/yeni `fingerprint` birlikte anılır.
4. **Rapora giriş.** Her `EvalReport.dataset_fingerprint` alanı buradaki bir `fingerprint` değeridir
   (`"sha256:<hex>"`). Rapor etiketi (`report-2027-05`) anında tüm dosyalar dondurulur.
5. **Boyut.** Bu dizindeki dosyalar birkaç yüz KB'ı geçmemelidir; MathWriting manifesti büyükse
   (630 bin dosya) yalnızca veri kökünde tutulur ve buraya `fingerprint` + `n_files` özeti işlenir.

## Doğrulama

```powershell
aicls data verify mathwriting          # DataManifest.verify(): eksik / değişmiş / fazla dosya raporu
```

`tests/data` işaretli testler yalnızca doğrulanmış bir manifest varsa koşar (`AICLS_TEST_DATA=1`);
manifest yoksa atlanırlar, böylece CI hiçbir zaman indirilmemiş veriye bağlı kalmaz.
