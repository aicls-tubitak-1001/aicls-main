# ADR-0003: SQLAlchemy 2.0 + Alembic; geliştirmede SQLite, üretimde MySQL 8

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** M. M. Turhan, Ç. Şahin, Ş. Güler
- **İlgili İP:** İP-2 (kullanıcı yönetimi, veritabanı), İP-8 (sorgulanabilir veritabanı, MySQL tabanlı API)

## Bağlam ve problem

Öneri A.1.2 "Veritabanı olarak MySQL tercih edilecektir", A.3.9 "MySQL tabanlı bir API" der ve SQL enjeksiyonuna karşı önlem ister. Geliştiriciler Windows/OneDrive makinelerinde çalışır; her makinede MySQL servisi kurmak ve CI'da MySQL beklemek test döngüsünü yavaşlatır.

## Değerlendirilen seçenekler

1. **Her yerde MySQL** (geliştirmede Docker) — üretimle birebir; Windows'ta Docker/WSL yükü, CI'da servis; testler yavaş.
2. **SQLite geliştirme/test, MySQL üretim; lehçe tarafsız şema; gece MySQL CI** — sıfır kurulumlu testler; sapma riski kural ve testlerle kapatılır.
3. **PostgreSQL** — öneri metniyle çelişir.

## Karar

Seçenek 2. SQLAlchemy 2.0 tipli ORM (parametreli sorgular = SQL enjeksiyon önlemi), Alembic (`render_as_batch=True`), sorgu üreten tek yer `aicls.db.repositories`. Şema kuralları: ULID `CHAR(26)`, uzunluklu `String`, `native_enum=False`, `sa.JSON`, UTC naive `DateTime`, sunucu fonksiyonu yok. Tek MySQL'e özgü öğe: `book_units.search_text` FULLTEXT indeksi, yalnızca `dialect.name == "mysql"` korumalı migrasyonda; SQLite'ta `LIKE` + `difflib` yolu. CI her PR'da SQLite head→base→head + tüm migrasyonların MySQL lehçesiyle derlenmesi; gece `mysql:8` servisiyle tam döngü; üretim öncesi ve her rapor etiketinde `mysql` işaretli takım iş istasyonunda koşar. Üç MySQL kullanıcısı (`aicls_migrate`, `aicls_app`, `aicls_ro`).

## Sonuçlar

- Olumlu: `poe test` saniyeler içinde; migrasyonlar her testte sınanır; lehçe sapması gece yakalanır.
- Olumsuz: SQLite'ta FULLTEXT yok → arama davranışı iki yolda test edilir; `mysql` işaretli testler geliştirici makinesinde `AICLS_TEST_MYSQL_URL` ister.

## Öneri metniyle uyum

A.1.2: "Veritabanı olarak MySQL tercih edilecektir. ... SQL enjeksiyonu gibi zafiyetlere karşı önlemler alınacak". A.3.9: "MySQL tabanlı bir API". Üretim ve İP-8 teslimatı MySQL'dir; SQLite yalnızca geliştirme kolaylığıdır ve raporda "üretim veritabanı" olarak anılmaz.

## Doğrulama

`test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile`, `test_migrations_head_base_head_sqlite`, `test_migrations_mysql_service` (`mysql`), `test_schema_rules_all_strings_have_length`, `tools/lint_no_raw_sql.py`.

## İlgili

[mimari/veri-modeli.md](../mimari/veri-modeli.md), ADR-0019.
