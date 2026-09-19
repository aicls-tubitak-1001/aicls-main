# Mimari Karar Kayıtları (ADR)

ADR (Architecture Decision Record), bir mimari kararı, bağlamını, değerlendirilen seçenekleri ve sonuçlarını kısa ve kalıcı biçimde kaydeder. Biçim MADR (Markdown Any Decision Records) uyarlamasıdır; ek olarak her ADR'de **"Öneri metniyle uyum"** bölümü zorunludur: karar, TÜBİTAK öneri metninin hangi maddesine (A.x, İP-x görev/ölçüt/risk planı) dayanıyor ya da nereden sapıyor, sapıyorsa öneri metninin hangi cümlesi buna izin veriyor. Gelişme raporunda her sapma bu bölüme atıfla açıklanır.

## Süreç

1. `python scripts/new_adr.py <slug>` → `docs/adr/NNNN-<slug>.md` şablondan (`0000-sablon.md`).
2. Durum `önerildi` ile PR açılır (`docs/adr-NNNN` dalı, `.github/ISSUE_TEMPLATE/adr-proposal.md` ile tartışma).
3. Üç geliştiriciden en az ikisi onaylar → `kabul edildi`. Öneri metninden sapan kararlarda proje yürütücüsü bilgilendirilir (PR açıklamasında not).
4. Kabul edilmiş ADR değiştirilmez; yeni ADR eskisini `yerini aldı` yapar (eskisi silinmez).
5. Bekleyen kararlar (ör. 0013) `önerildi` durumunda kalır ve karar tarihi/deneme girdisi belirtilir.
6. PR şablonundaki "ADR gerekli mi?" sorusu şu durumlarda **evet**: öneri metnindeki teknoloji listesinden sapma; yeni dış bağımlılık; veri/gizlilik etkisi; metrik tanımı; kapsam eşiği değişimi; deploy topolojisi.

## Dizin

| No | Başlık | Durum |
|---|---|---|
| [0001](0001-web-catisi-fastapi.md) | Web çatısı: Flask yerine FastAPI | kabul edildi |
| [0002](0002-monorepo-katmanli-paketler.md) | Tek depo, tek dağıtım, katmanlı paketler (import-linter) | kabul edildi |
| [0003](0003-veritabani-sqlite-dev-mysql-prod.md) | SQLAlchemy 2 + Alembic; SQLite geliştirme, MySQL üretim | kabul edildi |
| [0004](0004-latex-ayristirici-lark-dilbilgisi.md) | LaTeX → SymPy için kendi Lark dilbilgisi ve tipli AST | kabul edildi |
| [0005](0005-metin-komut-katmani-ip4.md) | Kural tabanlı metin→komut katmanı, İP-4 altında raporlama | kabul edildi |
| [0006](0006-ml-altyapisi-pytorch-hf-yapilandirma.md) | PyTorch + Hugging Face; pydantic yapılandırma; OCR için kendi Trainer, Whisper için Seq2SeqTrainer sarmalayıcısı | kabul edildi |
| [0007](0007-deney-takibi-jsonl-mlflow-manifest.md) | Deney takibi: JSONL varsayılan, MLflow opsiyonel, zorunlu koşu manifesti | kabul edildi |
| [0008](0008-veri-yonetimi-manifest-onedrive-disi.md) | Veri yönetimi: SHA-256 manifestleri, OneDrive dışı kök, git'te küçük setler | kabul edildi |
| [0009](0009-dokumantasyon-mkdocs-turkce.md) | Dokümantasyon: MkDocs Material (Türkçe), mkdocstrings, metrik makroları | kabul edildi |
| [0010](0010-test-stratejisi.md) | Test stratejisi: pytest işaretleri, hypothesis, paket kapsam eşikleri | kabul edildi |
| [0011](0011-windows-onedrive-gelistirme-ortami.md) | Windows/OneDrive geliştirme ortamı: venv konumu, poethepoet, junction | kabul edildi |
| [0012](0012-onay-kapilari.md) | Onay kapıları: bayrak + ortam değişkeni + onay defteri | kabul edildi |
| [0013](0013-el-yazisi-girdi-bicimi.md) | El yazısı girdi biçimi: çevrimiçi iz birincil, çevrimdışı görüntü destekli | önerildi (karar 6. hafta) |
| [0014](0014-gecikme-mimarisi-websocket.md) | Gecikme mimarisi: WebSocket akışı, Server-Timing, istemci telemetrisi | kabul edildi |
| [0015](0015-metrik-tanimlari-dondurulmus.md) | Dondurulmuş metrik tanımları ve sürümlü normalizasyon | kabul edildi |
| [0016](0016-cas-yurutme-sandbox.md) | CAS yürütme: ısıtılmış işçi havuzu, zaman aşımında öldürme | kabul edildi |
| [0017](0017-ses-girdi-bicimi-wav.md) | Ses girdisi: tarayıcıda 16 kHz PCM16 WAV, sunucuda ffmpeg yok | kabul edildi |
| [0018](0018-calisma-zamani-topolojisi.md) | Çalışma zamanı topolojisi: iş istasyonu + küçük toplama VM'i | kabul edildi |
| [0019](0019-kitap-bilgi-grafigi-iliskisel-depolama.md) | Kitap bilgi grafiği: JSON-LD belge + üçlü tablosu, MySQL | kabul edildi |
| [0020](0020-gpu-calisma-ortami-wsl-nvidia-docker.md) | GPU çalışma ortamı: WSL2 + nvidia-docker konteyneri; Windows'a CUDA torch kurulmadı | kabul edildi |

Bekleyen kararlar (ADR açılmadı): lisans (yürütücü kararı; 0002'de not), üniversite SSO, Redis hız sınırı, TTS motoru seçimi, DVC'ye geçiş.
