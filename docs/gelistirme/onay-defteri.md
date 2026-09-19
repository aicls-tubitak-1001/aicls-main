# Onay Defteri (Approval Ledger)

Bu dosya `aicls.core.gates.require_gate()` tarafından **makine olarak okunur**. Bir kapılı eylem, yalnızca aşağıdaki tabloda ilgili `KAPI` için `approved` durumunda bir satır varsa **ve** CLI bayrağı (`--i-have-approval`) **ve** ortam değişkeni (`AICLS_GATE_<KAPI>=approved`) birlikte sağlanmışsa çalışır. Bkz. [kurulum.md §3](kurulum.md#3-onay-kapıları), [ADR-0012](../adr/0012-onay-kapilari.md), işletim için [ml/egitim-el-kitabi.md](../ml/egitim-el-kitabi.md).

**Durum (2026-09-15): tablo dört kapının dördü için de `approved` satır içerir** (2026-09-08 satırları ve CUDA_INSTALL için 2026-09-15 Windows-yerli satırı). Kapıların gerçekten *kullanıldığının* kaydı `AICLS_DATA_ROOT/gate_invocations.log` dosyasındadır; bu tablo yalnızca izni kaydeder.

## Kapı adları

Yalnızca şu dört ad geçerlidir; başka bir ad yazılırsa defter ayrıştırma hatası verir ve **her kapı kapanır**:

`DATASET_DOWNLOAD` · `WEIGHTS_DOWNLOAD` · `CUDA_INSTALL` · `TRAINING`

## Satır eklemeden önce

1. **Maliyeti belgeleyin.** İlgili betiği `--dry-run --json` ile koşturun ve çıktıyı PR açıklamasına ekleyin. Kuru koşu kapıyı hiç sormaz; indirme boyutunu, gerekli diski, lisansı ve kaynak URL'yi yazar. Onay talebi hazırlamak kapısız bir iştir.
2. **Yazılı onayı alın.** Satırı ekleyen kişi, proje yürütücüsünün veya AI koordinatörünün (Ç. Şahin) **yazılı** onayına dayanmalıdır: e-posta, toplantı kararı veya PR yorumu.
3. **Kapsamı daraltın.** "MathWriting indirilebilir" değil, "MathWriting 2024.09, `D:\aicls-data` altına, tek seferlik" yazın. Kapsam, iptal tarihini kendiliğinden belirler.

## Satır biçimi

| Sütun | İçerik | Kural |
|---|---|---|
| `KAPI` | dört addan biri | başka bir değer defteri geçersiz kılar |
| `durum` | `approved` veya `revoked` | başka değer yok |
| `tarih` | `YYYY-MM-DD` | ISO; başka biçim ayrıştırma hatası |
| `kim` | satırı ekleyen kişinin kullanıcı adı | boş olamaz |
| `kapsam` | onayın **kaynağı** (e-posta tarihi, toplantı kararı) ve **sınırları** (hangi veri seti, hangi sürüm, hangi disk, hangi yapılandırma) | boş olamaz |

Örnek satırlar. Bunlar **örnektir**; satır başındaki `>` işareti kasıtlıdır ve kopyalarken silinir — böylece hiçbir araç bu blokları gerçek bir onay satırı sanamaz:

```text
> | DATASET_DOWNLOAD | approved | 2027-01-15 | csahin | MathWriting 2024.09 (mathwriting-2024.tgz, 2,88 GiB; açılmaz), LİSANS CC BY-NC-SA 4.0 — ticari kullanım yasak, D:\aicls-data, tek seferlik; onay: e-posta 2027-01-14 |
> | DATASET_DOWNLOAD | revoked  | 2027-03-01 | csahin | indirme tamamlandı ve doğrulandı; kapsam kapandı |
```

## Kurallar

- **Satır eklemek bir PR gerektirir.** Doğrudan `main` üzerine yazılmaz; defterin değeri, git geçmişinde incelemeden geçmiş olmasından gelir.
- `durum` yalnızca `approved` veya `revoked` olabilir. **İptal edilen onay silinmez**, yeni bir `revoked` satırıyla kapatılır; `require_gate` yalnızca ilgili kapının **en son** satırına bakar.
- **Biçim değişmez:** tablo başlığı ve sütun sırası `test_gate_ledger_format` ile korunur. Beş hücre, bilinen kapı adı, `approved`/`revoked`, ISO tarih, dolu `kim` ve `kapsam`. Bozuk tek bir satır `GateLedgerError` fırlatır ve dört kapıyı birden kapatır — bu kasıtlıdır.
- **Tabloyu bir günlük gibi kullanın:** en yeni satır en alta eklenir, aradaki satırlar düzenlenmez.
- Her kapılı çalıştırma ayrıca `AICLS_DATA_ROOT/gate_invocations.log` dosyasına bir JSON satırı olarak yazılır (kapı, zaman, kim, git SHA, ayar hash'i, kullanılan defter satırı). Defter **iznin**, bu günlük **kullanımın** kaydıdır; ikisi bir arada denetlenir.
- **CI'da kapı her zaman reddeder** (`CI` veya `GITHUB_ACTIONS` ayarlıysa), üç koşul sağlansa bile.
- Bu dosyaya satır eklemek, ilgili eylemi de yapmak anlamına gelmez: satır üç koşuldan yalnızca biridir.

## Bir onayı kapatmak

Kapsamı tamamlanan onay bekletilmez. Sırasıyla: (1) tabloya `revoked` satırı ekleyin, (2) ortam değişkenini düşürün (`Remove-Item Env:\AICLS_GATE_<KAPI>`) ve `.env` içinde kaldıysa silin, (3) `gate_invocations.log` dosyasını gözden geçirip beklenmedik bir kullanım olup olmadığına bakın. İptal, geçmişte üretilmiş ağırlıkları veya raporları geçersiz kılmaz; onlar `models/registry.yaml` içindeki `gate_refs` alanı üzerinden hangi onaya dayandıklarını taşımaya devam eder.

| KAPI | durum | tarih | kim | kapsam |
|---|---|---|---|---|
| CUDA_INSTALL | approved | 2026-09-08 | csahin | GPU yalnızca WSL2 + nvidia-docker üzerinden kullanılır (codex-research-cu130 imajı, /mnt/d/aicls-venv); Windows ana makinesine CUDA yığını KURULMAZ, scripts/install_ml.ps1 bu onay kapsamında değildir. Bkz. ADR-0020. Onay: Ç. Şahin yazılı talimatı 2026-09-08 ("rtx kullanılabilir, eğitime doğru yol alalım" / "koşuyu gpu da başlat"). |
| DATASET_DOWNLOAD | approved | 2026-09-08 | csahin | Yalnızca MathWriting 2024 (mathwriting-2024-excerpt.tgz 1,55 MiB ve mathwriting-2024.tgz 2,88 GiB), hedef D:\aicls-data, arşiv açılmaz (tar'dan akışla npz parçalarına dönüştürülür). LİSANS CC BY-NC-SA 4.0: TİCARİ KULLANIM YASAK, ShareAlike; bu onay yalnızca 125E748 kapsamındaki akademik araştırma kullanımını kapsar, ticari/patent aşaması için ayrı hukuki görüş gerekir. Onay: Ç. Şahin yazılı talimatı 2026-09-08. |
| TRAINING | approved | 2026-09-08 | csahin | OCR hattı: training/configs/ocr/* yapılandırmaları, MathWriting verisi, tek RTX 5090, konteyner içinde. Önce 200 adımlık `aicls profile`, bütçe ölçümden türetilir. ASR eğitimi bu onayın DIŞINDADIR (Türkçe konuşma korpusu henüz toplanmadı). Onay: Ç. Şahin yazılı talimatı 2026-09-08 ("koşuyu gpu da başlat verileri elde ederek"). |
| WEIGHTS_DOWNLOAD | approved | 2026-09-08 | csahin | Önceden eğitilmiş MODEL AĞIRLIKLARI (veri kümesi değil; veri indirmesi DATASET_DOWNLOAD satırına tabidir): ses tanıma ve yerel dil modeli değerlendirmesi kapsamında, hedef D:\aicls-data\models\pretrained, kod hiçbir yerde hub kimliği tutmaz. Model adı, sürümü, boyutu ve LİSANSI her indirmede catalog.yaml'a ve gate_invocations.log'a yazılır; ticari aşamaya taşınamayacak bir lisans (ör. non-commercial, araştırma-amaçlı) çıkarsa kullanılmadan önce ayrıca bildirilir. Onay: Ç. Şahin yazılı talimatı 2026-09-08 ("WEIGHTS_DOWNLOAD kapısı açılabilir. serbest."). |
| CUDA_INSTALL | approved | 2026-09-15 | csahin | İş istasyonu AICLS (MSI MS-7E86, RTX 5090, sürücü 591.86 / CUDA 13.1) için WINDOWS-YERLİ kurulum: scripts/install_ml.ps1 ile cu128 indeksinden torch 2.11.0 + torchvision 0.26.0 + torchaudio 2.11.0, ardından peft ve accelerate; hedef venv C:\Users\AICLS\.venvs\ai-cls (Python 3.11.9). 2026-09-08 satırındaki "install_ml.ps1 bu onay kapsamında değildir" sınırı bu makine için kaldırılır; gerekçe: bu iş istasyonunda WSL2 ve ADR-0020'nin dayandığı codex-research-cu130 imajı yoktur, konteyner yolu yeniden üretilemez (ADR-0020 Sonuçlar 2 ve 4). Veri kökü C:\aicls-data (D: bir USB bellektir). Onay: Ç. Şahin yazılı talimatı 2026-09-15 ("saltokunurdan çık ve kurulumu yap"). |
