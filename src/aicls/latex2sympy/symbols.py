"""Symbol table and parser configuration (docs/mimari/latex-ayristirici.md 3.5).

A :class:`SymbolTable` maps AST symbol names to *stable* ``sympy.Symbol`` objects with the
assumptions the curriculum expects (everything real by default).  Stability matters twice
over: SymPy treats ``Symbol('k')`` and ``Symbol('k', integer=True)`` as different symbols, so a
summation index must carry the same assumptions inside and outside the body - that is what
:meth:`SymbolTable.bound` guarantees.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import sympy


@dataclass(frozen=True, slots=True)
class ParserConfig:
    """Switches that change the meaning (not the syntax) of an expression.

    Attributes:
        log_base: Base of a bare ``\\log``; ``"e"`` (classroom usage, identical to ``\\ln``)
            or ``"10"``. ``\\log_{b}`` is always explicit.
        imaginary_i: When True a bare ``i`` is the imaginary unit; otherwise it is a symbol.
        real_symbols: When True free symbols are created with ``real=True``.
    """

    log_base: Literal["e", "10"] = "e"
    imaginary_i: bool = False
    real_symbols: bool = True


#: Configuration used when the caller passes none.
DEFAULT_CONFIG = ParserConfig()


class SymbolTable:
    """Name -> ``sympy.Symbol`` with assumptions, plus the reserved constant names.

    Args:
        config: Parser configuration; only :attr:`ParserConfig.real_symbols` is read here.
    """

    #: Names that denote constants unless the caller declares them.
    RESERVED: ClassVar[frozenset[str]] = frozenset({"e", "pi", "infty", "i", "oo"})

    def __init__(self, *, config: ParserConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._declared: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, sympy.Symbol] = {}
        self._reserved: set[str] = set(self.RESERVED)

    @property
    def reserved(self) -> frozenset[str]:
        """Reserved names that are still constants in this table."""
        return frozenset(self._reserved)

    def declare(self, name: str, **assumptions: Any) -> None:
        """Declare ``name`` with explicit SymPy assumptions.

        Declaring a reserved name (``e``, ``i``, ``pi``) turns it back into a symbol.

        Args:
            name: Symbol name as it appears in the AST (``x``, ``x_{1}``, ``alpha``).
            **assumptions: Passed straight to ``sympy.Symbol`` (``integer=True``, ...).
        """
        self._declared[name] = dict(assumptions)
        self._reserved.discard(name)
        self._cache.pop(name, None)

    def is_declared(self, name: str) -> bool:
        """Whether ``name`` was explicitly declared."""
        return name in self._declared

    def symbol(self, name: str) -> sympy.Symbol:
        """Return (and memoise) the ``sympy.Symbol`` for ``name``."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        assumptions = self._declared.get(name)
        if assumptions is None:
            assumptions = {"real": True} if self.config.real_symbols else {}
        created = sympy.Symbol(name, **assumptions)
        self._cache[name] = created
        return created

    @contextmanager
    def bound(self, name: str, **assumptions: Any) -> Iterator[sympy.Symbol]:
        """Temporarily declare ``name`` (summation/product index) and restore afterwards.

        Args:
            name: The bound variable.
            **assumptions: Assumptions for the duration of the block.

        Yields:
            The symbol that is active inside the block.
        """
        had_declaration = name in self._declared
        previous_assumptions = self._declared.get(name)
        previous_cache = self._cache.get(name)
        was_reserved = name in self._reserved
        self.declare(name, **assumptions)
        try:
            yield self.symbol(name)
        finally:
            self._cache.pop(name, None)
            self._declared.pop(name, None)
            if had_declaration and previous_assumptions is not None:
                self._declared[name] = previous_assumptions
            if previous_cache is not None:
                self._cache[name] = previous_cache
            if was_reserved:
                self._reserved.add(name)


__all__ = ["DEFAULT_CONFIG", "ParserConfig", "SymbolTable"]
