# ADR-0012: Onay kapıları — CLI bayrağı + ortam değişkeni + git'teki onay defteri; statik kapı denetimi

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin (proje sahibi kararı 2026-09-07)
- **İlgili İP:** İP-3 (veri, ağırlık, eğitim), İP-1 (GPU kurulumu)

## Bağlam ve problem

Proje sahibi dört eylemi kesin onaya bağladı: veri seti indirme, model ağırlığı indirme, CUDA/GPU PyTorch kurulumu, her türlü eğitim koşusu. Kapı "insan disiplini" yerine kod olmalı, denetlenebilir ve test edilebilir olmalı; ayrıca torchvision varsayılan ağırlıkları ve HF hub kimlikleri gibi **örtük** indirmeler de engellenmeli.

## Değerlendirilen seçenekler

1. **Yalnızca belge/uyarı** — denetlenemez.
2. **Yalnızca ortam değişkeni** — yanlışlıkla `.env`'de kalır.
3. **Üçlü kilit: `--i-have-approval` bayrağı AND `AICLS_GATE_<NAME>=approved` AND git'teki `docs/gelistirme/onay-defteri.md` satırı; CI hiçbirini ayarlamaz ve reddi test eder; `tools/check_gates.py` kaynak ağacında örtük indirme desenlerini arar** — denetlenebilir (defter git geçmişinde), test edilebilir, örtük ihlal statik olarak yakalanır.

## Karar

Seçenek 3. `aicls.core.gates.require_gate(name, cli_flag)`; kapı adları `DATASET_DOWNLOAD`, `WEIGHTS_DOWNLOAD`, `CUDA_INSTALL`, `TRAINING`; ret → `GateError`, çıkış kodu 3, eksik koşullar mesajda. Defter satır biçimi `| KAPI | approved | YYYY-MM-DD | kim | kapsam |`; iptal `revoked` satırıyla. Her kapılı çalıştırma `AICLS_DATA_ROOT/gate_invocations.log`'a yazılır. Sentetik veriyle eğitim (`--synthetic`, `data.dataset: synthetic_*`) kapısızdır. `tools/check_gates.py` (CI): `urllib`, `requests.get`, `hf_hub_download`, `snapshot_download`, `from_pretrained("openai/`/`"facebook/`, `weights=` (None dışı), `torch.hub` kullanımını kapılı modüller (`training/scripts/download_*.py`, `scripts/install_ml.ps1`) dışında yasaklar. `poe_tasks.toml` kapılı komut içermez. `install_ml.ps1` cu128 pin + `get_device_capability()==(12,0)` doğrulaması.

## Sonuçlar

- Olumlu: yanlışlıkla indirme/eğitim yapısal olarak zor; onay kaydı denetlenebilir; hakem "kim onayladı" sorusuna deftere bakar.
- Olumsuz: meşru koşular üç adım ister (istenen sürtünme); defter PR'ı onay yazışmasına atıf taşır.

## Öneri metniyle uyum

Öneri metni onay süreci tanımlamaz; bu proje sahibi kuralıdır. Veri toplama tarafında B.1.2 (etik kurul onayı, gönüllü olur formu) ile paraleldir: `AICLS_ETHICS_APPROVAL_REF` boşken toplama sekmesi de kapalıdır.

## Doğrulama

`test_gated_download_refuses_without_flag_and_env`, `test_gated_download_refuses_without_ledger_line`, `test_gated_train_refuses_in_ci`, `test_gated_train_allows_synthetic_without_gate`, `test_install_ml_script_contains_sm120_check`, `test_check_gates_tool_flags_hub_id_usage`, `test_torchvision_encoders_built_with_weights_none`, `test_gate_ledger_format`.

## İlgili

[gelistirme/kurulum.md §3](../gelistirme/kurulum.md#3-onay-kapıları), [gelistirme/onay-defteri.md](../gelistirme/onay-defteri.md), ADR-0006.
