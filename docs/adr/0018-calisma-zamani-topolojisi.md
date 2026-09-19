# ADR-0018: Çalışma zamanı topolojisi — iş istasyonu (İP-1 sunucusu, profil full) + küçük toplama VM'i (profil collect), tek kod tabanı

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, Ş. Güler, M. M. Turhan
- **İlgili İP:** İP-1 (sunucu, ağ, SSL), İP-2 (dağıtım), İP-3 (veri toplama)

## Bağlam ve problem

İP-1 tek bir sunucu bilgisayarı (satın alma şartnamesi: 16 çekirdek, 128 GB, 4 TB NVMe + 10 TB HDD, KGK, 7/24) ve üniversite ağından bir port öngörür. Aynı makine GPU ile eğitim yapacak, modelleri sunacak, MySQL barındıracaktır. Ses toplama sekmesi üç üniversiteden gönüllülere 7/24 internetten ulaşmalı; ham KVKK verisinin halka açık yüzeyde uzun süre kalmaması gerekir. Hakemler iki host'un üç kişilik ekibe işletme yükü getirdiğini ve raw sesin halka açık hostta durmasını sorun olarak işaretledi; buna karşın toplama yüzeyinin GPU makinesini doğrudan dışa açmaması da bir güvenlik gereğidir.

## Değerlendirilen seçenekler

1. **Her şey iş istasyonunda** — en az işletme; GPU makinesi doğrudan internete açık; eğitim sırasında toplama kesintileri; üniversite portu tek nokta.
2. **Her şey bulutta (İP-1 B planı VPS)** — GPU yok; öneri metni fiziksel sunucuyu tercih eder.
3. **İş istasyonu (`profile=full`: API, sandbox, MySQL, Caddy, GPU eğitim/çıkarım) + küçük Ubuntu VM (`profile=collect`: auth/contribute/feedback/health, MySQL toplama şeması, Caddy Let's Encrypt; gece çekme ve silme); tek kod tabanı, aynı migrasyonlar, tek compose dosyası** — güvenlik ve erişilebilirlik; işletme yükü sınırlı.

## Karar

Seçenek 3, aşamalı: Faz 0–1 yalnızca geliştirici makinesi; Faz 2'de iş istasyonu NSSM servisi + MySQL + Caddy; toplama sekmesi yayına alınırken (etik kurul onayı dış bağımlılık) VM devreye girer — üniversite VM'i yoksa İP-1 B planındaki bulut VPS. VM'siz senaryo (`collect` profili iş istasyonunda ikinci uvicorn) de desteklenir. `scripts/sync_recordings.py` gece VM → iş istasyonu kopyalar ve VM'de siler. Eğitim sırasında iş istasyonu API'si deterministik yolu sürdürür; model çıkarımı `ready` ucunda `training` bildirir. Yedekler iş istasyonunda (10 TB HDD) + üniversite içi kopya.

## Sonuçlar

- Olumlu: GPU makinesi doğrudan dışa açılmaz; toplama 7/24; ham ses halka açık hostta en fazla bir gün; tek kod/migrasyon.
- Olumsuz: ikinci host işletmesi (compose + sync + sertifika yenileme) — `deploy/collect-vm/` tek dosya ve `docs/raporlama/ip-01-altyapi.md` runbook ile sınırlı; iş istasyonundaki compute proxy'si üniversite portuna bağlı.

## Öneri metniyle uyum

İP-1 görevleri: sunucu temini, üniversite ağına entegrasyon, port, SSL. İP-1 risk planı B: "yüksek performanslı bir bulut sunucu (Cloud Server/VPS) hizmeti satın alınacaktır" — VM bu planın küçültülmüş, toplama-özel uygulamasıdır. A.2.3: "Web arayüzündeki veri toplama sekmesinde ... üç farklı üniversitedeki ... gönüllü katılımcılar". B.1.2 gizlilik.

## Doğrulama

`test_api_collect_profile_mounts_only_allowed_routers`, `deploy/collect-vm/docker-compose.yml` `docker compose config` CI adımı (gece), `scripts/sync_recordings.py --dry-run` testi.

## İlgili

[mimari/genel-bakis.md §5](../mimari/genel-bakis.md#5-çalışma-zamanı-topolojisi), [gelistirme/kurulum.md §4](../gelistirme/kurulum.md#4-sunucu--vm-kurulumu-özet), ADR-0012.
