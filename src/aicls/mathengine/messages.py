"""Turkish text resolver for the strings ``aicls.core.messages_tr`` does not carry yet.

The project rule is that every Turkish user-facing string lives in
:mod:`aicls.core.messages_tr`. ``mathengine`` needs six strings that were not part of the
core message table when this package was written (step titles and plot labels), and this
package may not edit ``core``.

:func:`tr` therefore resolves a key against :mod:`aicls.core.messages_tr` **first** and falls
back to :data:`PENDING_CORE_MESSAGES` only while the key is still missing there. Once the
constants below are added to ``core`` (``MSG_STEP_PRODUCT_RULE`` ...), this module keeps
working unchanged and starts serving the core wording; :data:`PENDING_CORE_MESSAGES` can then
be deleted. The same shim pattern is used by :data:`aicls.asr.qc.QC_REASONS_TR`.
"""

from __future__ import annotations

from typing import Any

from aicls.core import messages_tr

#: Turkish strings owned by ``mathengine`` until they move into ``aicls.core.messages_tr``.
#: Keys are the ``MSG_``-less names the core table would use.
PENDING_CORE_MESSAGES: dict[str, str] = {
    # --- solution step titles (docs/mimari/moduller.md, aicls.mathengine "adımlar") ---
    "STEP_DERIVATIVE": "Türev",
    "STEP_PRODUCT_RULE": "Çarpım kuralı",
    "STEP_CHAIN_RULE": "Zincir kuralı",
    "STEP_POWER_RULE": "Kuvvet kuralı",
    "STEP_RESULT": "Sonuç",
    # --- plot labels (Matplotlib Agg output, ADR-0016) ---
    "PLOT_TITLE": "{expr} fonksiyonunun grafiği",
    "PLOT_X_LABEL": "{var}",
    "PLOT_Y_LABEL": "f({var})",
}


def tr(key: str, **kwargs: Any) -> str:
    """Return the Turkish text registered under ``key`` with its placeholders filled.

    Args:
        key: A :mod:`aicls.core.messages_tr` message key (``"CAS_TIMEOUT"``) or one of
            :data:`PENDING_CORE_MESSAGES`.
        **kwargs: Values for the ``{placeholder}`` fields of the template.

    Returns:
        The formatted Turkish text.

    Raises:
        KeyError: If the key is registered in neither table -- a programming error.
    """
    if key in messages_tr.MESSAGES:
        return messages_tr.msg(key, **kwargs)
    template = PENDING_CORE_MESSAGES[key]
    if not kwargs and "{" not in template:
        return template
    return template.format(**kwargs)


__all__ = ["PENDING_CORE_MESSAGES", "tr"]
