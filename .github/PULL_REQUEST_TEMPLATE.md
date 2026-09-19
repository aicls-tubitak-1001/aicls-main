<!-- Başlık Conventional Commit biçiminde: type(scope): summary  (İngilizce) -->

## Özet

<!-- Ne değişti, neden? "Neden" gövdeye; bağlantılı ADR / deneme notu varsa yaz. -->

- **İP:** İP-N
- **ADR:** ADR-00NN / gerekmiyor
- **Refs:** #

## Kontrol listesi

- [ ] Testler önce yazıldı; `poe test-fast` ve `poe ci` yerelde yeşil
- [ ] Belgeler güncellendi (`docs/`, ilgili modül bölümü, `CHANGELOG.md` → Unreleased)
- [ ] ADR gerekli mi? (`docs/adr/README.md` §6 ölçütleri) — evetse taslak açıldı
- [ ] `openapi.json` yenilendi (API değiştiyse: `poe openapi`)
- [ ] Migrasyon elle incelendi (`alembic revision --autogenerate` çıktısı; MySQL uyumu)
- [ ] **Kapılı bir komut çalıştırılmadı** (veri/ağırlık indirme, CUDA kurulumu, eğitim) — ADR-0012
- [ ] Metrik dosyası değiştiyse `poe report-check` geçti
- [ ] PR < 400 satır ya da önce tasarım notu paylaşıldı
- [ ] Türkçe kullanıcı metinleri yalnızca `aicls.core.messages_tr` içinde
- [ ] Ham SQL yok (`tools/lint_no_raw_sql.py`); torch yalnızca tembel import (`tools/check_gates.py`)

## Nasıl test edildi

<!-- Komutlar ve çıktı özeti; sentetik fikstür / stub model kullanıldıysa belirt. -->
