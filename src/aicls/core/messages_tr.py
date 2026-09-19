"""Single source of Turkish user-facing strings.

Every Turkish text that can reach a user (API problem details, CLI output, WebSocket envelopes)
is defined here and nowhere else (docs/mimari/moduller.md, ortak kurallar). Other packages import
the ``MSG_*`` constants or call :func:`msg` with a key and keyword arguments.

The contract text from the proposal's İP-4 risk plan is :data:`MSG_UNSUPPORTED_PROBLEM`; its
exact wording is protected by a unit test and must not change.

Keys of :data:`MESSAGES` are either an :class:`aicls.core.errors.ErrorCode` value (the default
detail text for that code) or a more specific templated message whose placeholders are filled by
:func:`msg`. :data:`TITLES` holds the short RFC 9457 ``title`` per error code.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------------------------
# Contract text (İP-4 risk plan) — wording frozen, tested by test_messages_tr.py
# ---------------------------------------------------------------------------------------------
MSG_UNSUPPORTED_PROBLEM = "Bu problem tipi şu an için desteklenmemektedir."

# ---------------------------------------------------------------------------------------------
# Default detail text per ErrorCode (no placeholders)
# ---------------------------------------------------------------------------------------------
MSG_PARSE_ERROR = "LaTeX ifadesi ayrıştırılamadı; lütfen ifadeyi kontrol edin."
MSG_UNSUPPORTED_CONSTRUCT = "Bu LaTeX yapısı henüz desteklenmiyor."
MSG_CAS_TIMEOUT = "Hesaplama zaman aşımına uğradı; lütfen ifadeyi sadeleştirip tekrar deneyin."
MSG_NLU_NO_INTENT = "Komut anlaşılamadı; lütfen farklı bir ifadeyle tekrar deneyin."
MSG_OCR_INPUT_INVALID = "El yazısı girdisi geçersiz veya desteklenmeyen biçimde."
MSG_ASR_INPUT_INVALID = "Ses girdisi geçersiz veya desteklenmeyen biçimde."
MSG_UPLOAD_INVALID = "Yüklenen dosya geçersiz."
MSG_CONSENT_REQUIRED = "Devam etmek için güncel katılım onamını kabul etmeniz gerekiyor."
MSG_AUTH_INVALID = "Kimlik doğrulama başarısız; lütfen yeniden giriş yapın."
MSG_FORBIDDEN = "Bu işlem için yetkiniz yok."
MSG_NOT_FOUND = "İstenen kayıt bulunamadı."
MSG_RATE_LIMITED = "Çok fazla istek gönderildi; lütfen kısa bir süre sonra tekrar deneyin."
MSG_TRAINING_BUSY = "GPU şu anda başka bir eğitim işi tarafından kullanılıyor."
MSG_VALIDATION_ERROR = "İstek doğrulanamadı; lütfen alanları kontrol edin."
MSG_GATE_BLOCKED = "Bu işlem onay kapısı tarafından engellendi."
MSG_INTERNAL = "Beklenmeyen bir sunucu hatası oluştu; istek kimliğiyle destek ekibine başvurun."

# ---------------------------------------------------------------------------------------------
# Templated messages (placeholders filled by msg())
# ---------------------------------------------------------------------------------------------
MSG_PARSE_ERROR_AT = "Konum {position}: {reason}"
MSG_UNSUPPORTED_CONSTRUCT_NAMED = "Bu LaTeX yapısı henüz desteklenmiyor: {construct}"
MSG_CAS_TIMEOUT_SECONDS = (
    "Hesaplama {timeout_s} saniye içinde tamamlanamadı; lütfen ifadeyi sadeleştirip tekrar deneyin."
)
MSG_NLU_NO_INTENT_SUGGEST = "Komut anlaşılamadı. Şunu mu demek istediniz: {suggestion}"
MSG_INPUT_TOO_LONG = "Girdi çok uzun: en fazla {max_len} karakter olabilir."
MSG_RATE_LIMITED_RETRY = "Çok fazla istek gönderildi; {retry_after} saniye sonra tekrar deneyin."
MSG_ETHICS_APPROVAL_MISSING = (
    "Ses toplama sekmesi, etik kurul onayı kaydedilene kadar (AICLS_ETHICS_APPROVAL_REF) kapalıdır."
)
MSG_MODEL_NOT_READY = "Model şu anda hazır değil; lütfen daha sonra tekrar deneyin."
# Single-GPU training queue (aicls.ml.queue, docs/ml/egitim-altyapisi.md §8). The operator is not
# an API client: nothing was rate limited, the one RTX 5090 is simply taken.
MSG_TRAINING_QUEUE_BUSY = (
    "İş istasyonundaki tek GPU'da hâlihazırda bir eğitim işi çalışıyor; "
    "mevcut iş bitene kadar bekleyin (eğitim kuyruğu: training/queue.lock)."
)

# Recording quality control for the "Model Gelişimine Katkıda Bulun" tab (İP-3). One template per
# ``aicls.asr.qc.QcCheck`` name; placeholders: {value}, {threshold}, {value_pct}, {threshold_pct}.
MSG_QC_DURATION_MIN = "Kayıt çok kısa: {value:.2f} saniye (en az {threshold:.2f} saniye olmalı)."
MSG_QC_DURATION_MAX = "Kayıt çok uzun: {value:.2f} saniye (en fazla {threshold:.2f} saniye olmalı)."
MSG_QC_CLIPPING_RATIO = (
    "Kayıtta kırpılma var: örneklerin %{value_pct:.2f}'si üst sınırda "
    "(en fazla %{threshold_pct:.2f}). Mikrofon kazancını düşürüp tekrar deneyin."
)
MSG_QC_SNR_DB = (
    "Ortam gürültüsü yüksek: sinyal/gürültü oranı {value:.1f} dB "
    "(en az {threshold:.1f} dB olmalı). Daha sessiz bir ortamda tekrar deneyin."
)
MSG_QC_SILENCE_RATIO = (
    "Kaydın büyük bölümü sessiz: sessizlik oranı %{value_pct:.0f} "
    "(en fazla %{threshold_pct:.0f}). Cümleyi baştan sona okuduğunuzdan emin olun."
)
MSG_QC_DC_OFFSET = (
    "Mikrofon sinyalinde doğru akım kayması var: {value:.3f} "
    "(en fazla {threshold:.3f}). Farklı bir mikrofon veya tarayıcı deneyin."
)

# Approval gates (ADR-0012). The refusal message lists every missing condition.
MSG_GATE_REFUSED = (
    "'{gate}' kapısı kapalı; eksik koşullar: {missing}. "
    "Bkz. docs/gelistirme/kurulum.md §3 ve ADR-0012."
)
MSG_GATE_MISSING_FLAG = "--i-have-approval bayrağı verilmedi"
MSG_GATE_MISSING_ENV = "AICLS_GATE_{gate}=approved ortam değişkeni ayarlı değil"
MSG_GATE_MISSING_LEDGER = (
    "docs/gelistirme/onay-defteri.md dosyasında '{gate}' için geçerli bir 'approved' satırı yok"
)
MSG_GATE_CI_REFUSED = "CI ortamında kapılı eylemler her zaman reddedilir"
MSG_GATE_UNKNOWN = "Bilinmeyen kapı adı: {gate}"
MSG_GATE_LEDGER_INVALID = "Onay defteri biçimi geçersiz ({path}, satır {line}): {reason}"

# Generic infrastructure messages
MSG_FILE_NOT_FOUND = "Dosya bulunamadı: {path}"
MSG_RESOURCE_NOT_FOUND = "Paket kaynağı bulunamadı: {package}:{name}"
MSG_REPO_ROOT_NOT_FOUND = "Depo kökü bulunamadı (pyproject.toml aranıyor): {start}"
MSG_INVALID_STAGE_NAME = "Geçersiz aşama adı: {name}"
MSG_INVALID_DURATION = "Süre negatif olamaz: {value}"
MSG_INVALID_LOG_LEVEL = "Geçersiz günlük düzeyi: {level}"
MSG_INVALID_HASH_LENGTH = "Kısa hash uzunluğu 1 ile 64 arasında olmalı: {n}"
MSG_INVALID_TIMESTAMP = "ULID zaman damgası 48 bitlik aralığın dışında: {value}"
MSG_INVALID_ULID = "Geçersiz ULID: {value}"
MSG_JSON_NOT_SERIALISABLE = "Nesne JSON'a dönüştürülemedi: {reason}"

# ---------------------------------------------------------------------------------------------
# RFC 9457 titles per ErrorCode (short, no trailing period)
# ---------------------------------------------------------------------------------------------
TITLES: dict[str, str] = {
    "PARSE_ERROR": "LaTeX ifadesi ayrıştırılamadı",
    "UNSUPPORTED_CONSTRUCT": "Desteklenmeyen LaTeX yapısı",
    "UNSUPPORTED_PROBLEM": "Desteklenmeyen problem tipi",
    "CAS_TIMEOUT": "Hesaplama zaman aşımı",
    "NLU_NO_INTENT": "Komut anlaşılamadı",
    "OCR_INPUT_INVALID": "Geçersiz el yazısı girdisi",
    "ASR_INPUT_INVALID": "Geçersiz ses girdisi",
    "UPLOAD_INVALID": "Geçersiz yükleme",
    "CONSENT_REQUIRED": "Onam gerekli",
    "AUTH_INVALID": "Kimlik doğrulama başarısız",
    "FORBIDDEN": "Erişim reddedildi",
    "NOT_FOUND": "Bulunamadı",
    "RATE_LIMITED": "Çok fazla istek",
    "TRAINING_BUSY": "GPU meşgul",
    "VALIDATION_ERROR": "Geçersiz istek",
    "GATE_BLOCKED": "Onay kapısı kapalı",
    "INTERNAL": "Sunucu hatası",
}

# ---------------------------------------------------------------------------------------------
# Registry: every MSG_* constant keyed by its name without the MSG_ prefix
# ---------------------------------------------------------------------------------------------
MESSAGES: dict[str, str] = {
    name.removeprefix("MSG_"): value
    for name, value in dict(globals()).items()
    if name.startswith("MSG_") and isinstance(value, str)
}


def msg(code: Any, **kwargs: Any) -> str:
    """Return the Turkish message registered under ``code`` with placeholders filled.

    Args:
        code: A message key (``"PARSE_ERROR_AT"``) or an :class:`~aicls.core.errors.ErrorCode`
            member (its ``value`` is used).
        **kwargs: Values for the ``{placeholder}`` fields of the template.

    Returns:
        The formatted Turkish text.

    Raises:
        KeyError: If the key is not registered or a placeholder is missing. Both are
            programming errors, surfaced early so tests catch them.
    """
    key = getattr(code, "value", code)
    if not isinstance(key, str) or key not in MESSAGES:
        raise KeyError(f"unknown Turkish message key: {key!r}")
    template = MESSAGES[key]
    if not kwargs and "{" not in template:
        return template
    return template.format(**kwargs)


__all__ = [
    "MESSAGES",
    "MSG_ASR_INPUT_INVALID",
    "MSG_AUTH_INVALID",
    "MSG_CAS_TIMEOUT",
    "MSG_CAS_TIMEOUT_SECONDS",
    "MSG_CONSENT_REQUIRED",
    "MSG_ETHICS_APPROVAL_MISSING",
    "MSG_FILE_NOT_FOUND",
    "MSG_FORBIDDEN",
    "MSG_GATE_BLOCKED",
    "MSG_GATE_CI_REFUSED",
    "MSG_GATE_LEDGER_INVALID",
    "MSG_GATE_MISSING_ENV",
    "MSG_GATE_MISSING_FLAG",
    "MSG_GATE_MISSING_LEDGER",
    "MSG_GATE_REFUSED",
    "MSG_GATE_UNKNOWN",
    "MSG_INPUT_TOO_LONG",
    "MSG_INTERNAL",
    "MSG_INVALID_DURATION",
    "MSG_INVALID_HASH_LENGTH",
    "MSG_INVALID_LOG_LEVEL",
    "MSG_INVALID_STAGE_NAME",
    "MSG_INVALID_TIMESTAMP",
    "MSG_INVALID_ULID",
    "MSG_JSON_NOT_SERIALISABLE",
    "MSG_MODEL_NOT_READY",
    "MSG_NLU_NO_INTENT",
    "MSG_NLU_NO_INTENT_SUGGEST",
    "MSG_NOT_FOUND",
    "MSG_OCR_INPUT_INVALID",
    "MSG_PARSE_ERROR",
    "MSG_PARSE_ERROR_AT",
    "MSG_QC_CLIPPING_RATIO",
    "MSG_QC_DC_OFFSET",
    "MSG_QC_DURATION_MAX",
    "MSG_QC_DURATION_MIN",
    "MSG_QC_SILENCE_RATIO",
    "MSG_QC_SNR_DB",
    "MSG_RATE_LIMITED",
    "MSG_RATE_LIMITED_RETRY",
    "MSG_REPO_ROOT_NOT_FOUND",
    "MSG_RESOURCE_NOT_FOUND",
    "MSG_TRAINING_BUSY",
    "MSG_TRAINING_QUEUE_BUSY",
    "MSG_UNSUPPORTED_CONSTRUCT",
    "MSG_UNSUPPORTED_CONSTRUCT_NAMED",
    "MSG_UNSUPPORTED_PROBLEM",
    "MSG_UPLOAD_INVALID",
    "MSG_VALIDATION_ERROR",
    "TITLES",
    "msg",
]
