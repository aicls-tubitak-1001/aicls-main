"""AI-CLS: AI-supported interactive Calculus teaching platform (TÜBİTAK 1001 #125E748).

Single distribution with layered subpackages (ADR-0002):
``cli > api|ml > eval > data|collect|book > nlu > mathengine > latex2sympy > ocr|asr > db > core``.
``__version__`` is the single source of truth for the package version (pyproject reads it).
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
