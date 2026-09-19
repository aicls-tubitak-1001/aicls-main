# deploy/ — iş istasyonu servisi, toplama VM'i, yedekleme

Topoloji kararı: [ADR-0018](../docs/adr/0018-calisma-zamani-topolojisi.md); kurulum özeti [docs/gelistirme/kurulum.md §4](../docs/gelistirme/kurulum.md); güvenlik [docs/guvenlik-gizlilik.md](../docs/guvenlik-gizlilik.md).

## Planlanan düzen

```text
deploy/
├── docker-compose.yml          # api (profile=full) + mysql + caddy (geliştirme/pilot)
├── Dockerfile.api              # python:3.11-slim, --no-deps -e ., uvicorn
├── Dockerfile.web              # node:24 build -> caddy static
├── Caddyfile                   # üniversite sertifikası, /api ve /ws reverse proxy, güvenlik başlıkları
├── mysql/init.sql              # aicls_app / aicls_migrate / aicls_readonly kullanıcıları, utf8mb4
├── workstation/                # NSSM Windows servisi: install_service.ps1, nssm-aicls-api.xml, README.md
├── collect-vm/                 # Ubuntu toplama VM'i: docker-compose.yml (profile=collect), Caddyfile (Let's Encrypt), .env.example
└── backup/                     # backup.ps1 / backup.sh (mysqldump --single-transaction, 7-zip AES-256, 30 gün rotasyon)
```

## İlkeler

- Sırlar yalnızca servis hesabının ortamında / `env_file` (600); depoda örnek dosyalar.
- İş istasyonu (İP-1 sunucusu) tam profil ve GPU çıkarımı; küçük toplama VM'i yalnızca `collect` profili (kimlik, rıza, kayıt yükleme, sağlık).
- Yedekler 10 TB HDD + üniversite içi ikinci kopya; üç ayda bir geri yükleme tatbikatı `docs/raporlama/ip-01-altyapi.md` günlüğüne.
- Compose dosyaları sürüm etiketini (`vX.Y.Z`) referans alır; `scripts/release.py` günceller.
