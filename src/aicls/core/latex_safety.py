"""LaTeX sanitiser: allow-list filtering before anything reaches KaTeX.

Every LaTeX string that is rendered — user input, OCR/ASR output, CAS results, book content —
passes through :func:`sanitize_latex` (docs/guvenlik-gizlilik.md §6, ADR-0019). The policy is an
**allow-list** of KaTeX math commands and environments:

* allowed commands are kept verbatim;
* forbidden commands (``\\href``, ``\\url``, ``\\includegraphics``, ``\\input``, ``\\write``,
  ``\\def``, ``\\csname``, HTML extensions, macro definitions, ...) are removed **together with
  their arguments**;
* any other command is removed but its brace groups stay (``\\foo{x}`` → ``{x}``);
* unknown environments are unwrapped; stray ``$``/``#``, control characters and a trailing
  backslash are dropped; braces are balanced.

The function is idempotent and never returns a forbidden command (property-tested). Length limits
are the API layer's job; pass ``max_len`` to enforce one here.
"""

from __future__ import annotations

import re
import unicodedata

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError

#: Contract limit for LaTeX inputs (docs/guvenlik-gizlilik.md §6).
MAX_LATEX_LENGTH = 2048

_GREEK = (
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu "
    "xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega "
    "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega"
)
_FUNCTIONS = (
    "sin cos tan cot sec csc arcsin arccos arctan arccot sinh cosh tanh coth sech csch "
    "log ln lg exp lim limsup liminf sup inf max min det dim gcd deg arg ker hom Pr "
    "operatorname"
)
_STRUCTURE = (
    "frac dfrac tfrac cfrac sqrt sum prod coprod int iint iiint oint binom dbinom tbinom "
    "left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr "
    "begin end substack overset underset stackrel overbrace underbrace overline underline "
    "hat widehat bar vec tilde widetilde dot ddot dddot mathring overrightarrow overleftarrow "
    "displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits "
    "mathrm mathbf mathit mathsf mathtt mathbb mathcal mathfrak mathscr boldsymbol bm "
    "text textbf textit textrm textsf texttt textnormal "
    "phantom hphantom vphantom smash not quad qquad space thinspace medspace thickspace "
    "enspace negthinspace mathbin mathrel mathop mathord mathpunct mathopen mathclose "
    "color textcolor"
)
_SYMBOLS = (
    "cdot cdots ldots vdots ddots dots dotsb dotsc times div pm mp ast star circ bullet "
    "infty partial nabla prime to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow "
    "Leftrightarrow longrightarrow longleftarrow Longrightarrow Longleftarrow "
    "longleftrightarrow Longleftrightarrow mapsto longmapsto uparrow downarrow implies iff "
    "impliedby leq le geq ge neq ne approx equiv sim simeq cong propto ll gg prec succ "
    "subset subseteq supset supseteq subsetneq supsetneq in notin ni cup cap setminus "
    "emptyset varnothing forall exists nexists neg lnot land lor wedge vee oplus otimes "
    "ominus odot angle measuredangle perp parallel nparallel mid nmid langle rangle lfloor "
    "rfloor lceil rceil vert Vert lvert rvert lVert rVert therefore because aleph hbar ell "
    "Re Im wp top bot triangle square diamond degree checkmark dagger ddagger S P "
    "lbrace rbrace lbrack rbrack backslash colon ldotp cdotp"
)

#: Commands kept verbatim (KaTeX math subset).
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    f"{_GREEK} {_FUNCTIONS} {_STRUCTURE} {_SYMBOLS}".split()
)

#: One-character control symbols kept verbatim (``\\,`` ``\\{`` ``\\%`` ...).
ALLOWED_SYMBOL_COMMANDS: frozenset[str] = frozenset("\\,;:! {}%&#$_|")

#: Environments kept; others are unwrapped.
ALLOWED_ENVIRONMENTS: frozenset[str] = frozenset(
    (
        "matrix",
        "pmatrix",
        "bmatrix",
        "Bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
        "cases",
        "rcases",
        "dcases",
        "aligned",
        "gathered",
        "alignedat",
        "array",
        "darray",
        "split",
        "subarray",
    )
)

#: Commands removed together with their arguments (number of brace groups to drop).
FORBIDDEN_COMMAND_ARGS: dict[str, int] = {
    # linking / resources
    "href": 2,
    "url": 1,
    "includegraphics": 1,
    "input": 1,
    "include": 1,
    "usepackage": 1,
    "documentclass": 1,
    "openin": 0,
    "openout": 0,
    "closeout": 0,
    "closein": 0,
    "read": 0,
    "write": 1,
    "immediate": 0,
    "special": 1,
    "pdfliteral": 1,
    "shipout": 0,
    "message": 1,
    "errmessage": 1,
    "scantokens": 1,
    "jobname": 0,
    # macro machinery
    "def": 1,
    "gdef": 1,
    "edef": 1,
    "xdef": 1,
    "let": 0,
    "futurelet": 0,
    "newcommand": 2,
    "renewcommand": 2,
    "providecommand": 2,
    "DeclareMathOperator": 2,
    "newenvironment": 3,
    "renewenvironment": 3,
    "csname": 0,
    "endcsname": 0,
    "expandafter": 0,
    "noexpand": 0,
    "relax": 0,
    "catcode": 0,
    "uccode": 0,
    "lccode": 0,
    "char": 0,
    "chardef": 0,
    "mathchardef": 0,
    "countdef": 0,
    "global": 0,
    "long": 0,
    "outer": 0,
    "afterassignment": 0,
    "aftergroup": 0,
    "makeatletter": 0,
    "makeatother": 0,
    "meaning": 0,
    "string": 0,
    # KaTeX HTML extensions (trust: false already blocks them; belt and braces)
    "html": 0,
    "htmlClass": 2,
    "htmlId": 2,
    "htmlStyle": 2,
    "htmlData": 2,
    "includegraphicsx": 1,
    "rule": 2,
}

#: Names of forbidden commands (never present in sanitised output).
FORBIDDEN_COMMANDS: frozenset[str] = frozenset(FORBIDDEN_COMMAND_ARGS)

#: Forbidden commands that may be followed by control sequences to consume (``\\def\\x``).
_CONTROL_SEQ_BUDGET: dict[str, int] = {
    "def": 1,
    "gdef": 1,
    "edef": 1,
    "xdef": 1,
    "let": 2,
    "futurelet": 2,
    "newcommand": 1,
    "renewcommand": 1,
    "providecommand": 1,
    "DeclareMathOperator": 1,
    "chardef": 1,
    "mathchardef": 1,
    "countdef": 1,
}

_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_ENV_RE = re.compile(r"\s*\{([A-Za-z*]+)\}")
_CS_RE = re.compile(r"\\([A-Za-z]+|.)", re.DOTALL)
_DIGITS_RE = re.compile(r"\d+")

# fmt: off
_DELIMITERS: tuple[tuple[str, str], ...] = (
    ("$$", "$$"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
    ("$", "$"),
)
# fmt: on


def _clean_chars(s: str) -> str:
    """NFC-normalise, unify newlines and drop control/format characters (keep ``\\n``, ``\\t``)."""
    text = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    )


def _trim(s: str) -> str:
    """Strip surrounding whitespace but keep a trailing escaped space (``\\ ``)."""
    s = s.lstrip()
    while s and s[-1].isspace():
        j = len(s) - 2
        backslashes = 0
        while j >= 0 and s[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 1:
            break  # the space is the control symbol "\ "
        s = s[:-1]
    return s


def _prepare(s: str) -> str:
    return _strip_delimiters(_trim(_clean_chars(s)))


def _strip_delimiters(s: str) -> str:
    for start, end in _DELIMITERS:
        if (
            len(s) >= len(start) + len(end)
            and s.startswith(start)
            and s.endswith(end)
            and not (start == end and s == start)
        ):
            return _trim(s[len(start) : -len(end)])
    return s


def _find_group_end(s: str, start: int) -> int:
    """Index just past the brace group opening at ``start`` (``s[start] == "{"``).

    Escaped braces do not count. An unterminated group extends to the end of the string.
    """
    depth = 0
    i = start
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _skip_arguments(s: str, i: int, name: str) -> int:
    """Skip the arguments of forbidden command ``name`` starting at ``i``; return the new index."""
    groups_left = FORBIDDEN_COMMAND_ARGS.get(name, 1)
    cs_left = _CONTROL_SEQ_BUDGET.get(name, 0)
    n = len(s)
    while i < n:
        j = i
        while j < n and s[j] in " \t":
            j += 1
        if j >= n:
            break
        ch = s[j]
        if ch == "[":
            close = s.find("]", j + 1)
            if close == -1:
                break
            i = close + 1
            continue
        if ch == "{" and groups_left > 0:
            i = _find_group_end(s, j)
            groups_left -= 1
            continue
        if ch == "\\" and cs_left > 0:
            m = _CS_RE.match(s, j)
            if m is None:  # pragma: no cover - a lone trailing backslash is dropped elsewhere
                break
            i = m.end()
            cs_left -= 1
            continue
        if ch == "#" and name in _CONTROL_SEQ_BUDGET:
            # macro parameter text (\def\cmd#1#2{...})
            i = j + 2 if j + 1 < n and s[j + 1].isdigit() else j + 1
            continue
        if ch.isdigit() and name in {"write", "openout", "openin", "closeout", "closein", "read"}:
            m_digits = _DIGITS_RE.match(s, j)
            i = m_digits.end() if m_digits is not None else j + 1
            continue
        if ch == "=" and name in {"openout", "openin", "catcode", "uccode", "lccode"}:
            i = j + 1
            continue
        break
    return i


def _scan(s: str, *, collect: list[str] | None = None) -> str:
    """Single pass over ``s``: keep allowed tokens, drop or unwrap the rest, balance braces."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(s)
    # True right after an alphabetic command was emitted: a following letter would otherwise
    # fuse with it into a new command name (\frac + a -> \fraca) once removed tokens vanish.
    pending_cmd = False
    while i < n:
        ch = s[i]
        if ch == "\\":
            m = _CMD_RE.match(s, i)
            if m is not None:
                name = m.group(1)
                i = m.end()
                if name in ("begin", "end"):
                    env_match = _ENV_RE.match(s, i)
                    if env_match is not None:
                        env = env_match.group(1)
                        i = env_match.end()
                        if env in ALLOWED_ENVIRONMENTS:
                            out.append(f"\\{name}{{{env}}}")
                            pending_cmd = False
                        elif collect is not None:
                            collect.append(f"{name}{{{env}}}")
                    elif collect is not None:
                        collect.append(name)
                elif name in ALLOWED_COMMANDS:
                    out.append(f"\\{name}")
                    pending_cmd = True
                    if name == "operatorname" and i < n and s[i] == "*":
                        out.append("*")
                        pending_cmd = False
                        i += 1
                elif name in FORBIDDEN_COMMANDS:
                    if collect is not None:
                        collect.append(name)
                    i = _skip_arguments(s, i, name)
                elif collect is not None:
                    collect.append(name)
                continue
            if i + 1 < n:
                sym = s[i + 1]
                if sym in ALLOWED_SYMBOL_COMMANDS:
                    out.append(f"\\{sym}")
                    pending_cmd = False
                elif collect is not None:
                    collect.append(sym)
                i += 2
                continue
            i += 1  # trailing lone backslash
            continue
        if ch in "$#":
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                i += 1
                continue
            depth -= 1
        if pending_cmd and ch.isalpha():
            out.append(" ")
        pending_cmd = False
        out.append(ch)
        i += 1
    # Trim whitespace at the token level so an escaped space ("\ ") at either end survives.
    while out and out[0].isspace():
        out.pop(0)
    while out and out[-1].isspace():
        out.pop()
    return "".join(out) + "}" * depth


def sanitize_latex(s: str, *, max_len: int | None = None) -> str:
    """Return ``s`` reduced to the KaTeX allow-list (see module docstring).

    Args:
        s: Raw LaTeX (math mode; ``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]`` wrappers
            are stripped).
        max_len: When given, reject inputs longer than this many characters (the API layer uses
            :data:`MAX_LATEX_LENGTH`).

    Returns:
        Sanitised LaTeX; empty string for empty/whitespace input.

    Raises:
        InputValidationError: If ``max_len`` is set and exceeded.
    """
    if max_len is not None and len(s) > max_len:
        raise InputValidationError(
            messages_tr.msg("INPUT_TOO_LONG", max_len=max_len),
            details={"max_len": max_len, "length": len(s)},
        )
    return _scan(_prepare(s))


def find_disallowed_commands(s: str) -> list[str]:
    """List (in order, without duplicates) the command names that sanitising would remove.

    Useful for diagnostics and for the parser's ``UNSUPPORTED_CONSTRUCT`` details.
    """
    found: list[str] = []
    _scan(_prepare(s), collect=found)
    seen: set[str] = set()
    unique: list[str] = []
    for name in found:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


__all__ = [
    "ALLOWED_COMMANDS",
    "ALLOWED_ENVIRONMENTS",
    "ALLOWED_SYMBOL_COMMANDS",
    "FORBIDDEN_COMMANDS",
    "FORBIDDEN_COMMAND_ARGS",
    "MAX_LATEX_LENGTH",
    "find_disallowed_commands",
    "sanitize_latex",
]
