"""Turkish strings of the generated results table, until they move into ``aicls.core``.

The project rule is that every Turkish user-facing string lives in
:mod:`aicls.core.messages_tr`. The generated report table (:func:`aicls.eval.render_markdown`)
needs a handful of column headings and status words that were not in the core table when this
package was written, and this package may not edit ``core``.

:func:`tr` resolves a key against :mod:`aicls.core.messages_tr` **first** and falls back to
:data:`PENDING_CORE_MESSAGES` only while the key is still missing there, so the day the
constants are added to core this module keeps working and starts serving the core wording. The
same shim pattern is used by :mod:`aicls.mathengine.messages` and
:data:`aicls.asr.qc.QC_REASONS_TR`.
"""

from __future__ import annotations

from typing import Any

from aicls.core import messages_tr

#: Turkish strings owned by ``eval`` until they move into :mod:`aicls.core.messages_tr`.
PENDING_CORE_MESSAGES: dict[str, str] = {
    # --- generated results table (docs/raporlama, MkDocs) ---
    "REPORT_TABLE_TITLE": "Ölçüm sonuçları",
    "REPORT_COL_KIND": "Koşum",
    "REPORT_COL_METRIC": "Ölçüt",
    "REPORT_COL_VALUE": "Değer",
    "REPORT_COL_CI": "%95 GA",
    "REPORT_COL_N": "n",
    "REPORT_COL_TARGET": "Hedef",
    "REPORT_COL_STATUS": "Durum",
    "REPORT_STATUS_PASS": "tuttu",
    "REPORT_STATUS_FAIL": "tutmadı",
    "REPORT_STATUS_NONE": "—",
    "REPORT_STATUS_STUB": "stub · kanıt değil",
    "REPORT_PROVENANCE": (
        "Kaynak: {name} · commit {sha} · veri parmak izi {fingerprint} · {timestamp}"
    ),
    "REPORT_NOT_CITABLE": (
        "UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); "
        "rapora yazılamaz."
    ),
    "REPORT_WIRING_WARNING": (
        "UYARI: bu koşum bir bağlantı denetimidir; sözleşme ölçütü için kanıt değildir. "
        "Gerekçe(ler) aşağıda."
    ),
    "REPORT_STUB_WARNING": (
        "UYARI: bu sayı stub (taklit) model ile üretilmiştir; bağlantı denetimidir, "
        "sözleşme ölçütü için kanıt değildir."
    ),
    "REPORT_DEFINITIONS": "Tanımlar: {ref}",
}


def tr(key: str, **kwargs: Any) -> str:
    """Return the Turkish text registered under ``key`` with its placeholders filled.

    Args:
        key: Name without the ``MSG_`` prefix (``"REPORT_COL_VALUE"``).
        **kwargs: Placeholder values for ``str.format``.

    Returns:
        The core message when it exists, otherwise the pending one; the key itself when neither
        is registered (a missing string must never crash a report run).
    """
    text = getattr(messages_tr, f"MSG_{key}", None)
    if not isinstance(text, str):
        text = PENDING_CORE_MESSAGES.get(key, key)
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):  # pragma: no cover - defensive: a template/key mismatch
        return text


__all__ = ["PENDING_CORE_MESSAGES", "tr"]
