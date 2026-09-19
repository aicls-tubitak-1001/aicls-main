"""sanitize_latex: allow-list filtering before KaTeX (docs/guvenlik-gizlilik.md §6)."""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st
import pytest

from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.latex_safety import (
    ALLOWED_COMMANDS,
    ALLOWED_ENVIRONMENTS,
    FORBIDDEN_COMMANDS,
    MAX_LATEX_LENGTH,
    find_disallowed_commands,
    sanitize_latex,
)


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{1}{3}",
        r"\int_{0}^{1} x^{2} \, dx",
        r"\lim_{x \to 0} \frac{\sin x}{x}",
        r"\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}",
        r"\left( \frac{a}{b} \right)",
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
        r"\begin{cases} x & x \geq 0 \\ -x & x < 0 \end{cases}",
        r"f'(x) = 2x",
        r"\sqrt[3]{x}",
        r"\operatorname{sinh}(x)",
        r"\mathbb{R}^{n}",
        r"\text{türev}",
        r"\alpha + \beta \cdot \gamma \leq \Omega",
        r"\vec{v} \times \hat{n}",
        r"\frac{\partial f}{\partial x}",
        r"a \, b \; c \: d \! e",
        r"\{ x \mid x > 0 \}",
        r"100\%",
        r"\left. \frac{df}{dx} \right|_{x=0}",
        r"\ln x \approx x - 1",
    ],
)
def test_sanitize_keeps_allowed_math(latex: str) -> None:
    assert sanitize_latex(latex) == latex


@pytest.mark.parametrize(
    ("bad", "forbidden"),
    [
        (r"\href{http://evil}{x^2}", r"\href"),
        (r"\url{http://evil}", r"\url"),
        (r"\includegraphics[width=1cm]{/etc/passwd}", r"\includegraphics"),
        (r"\input{/etc/passwd}", r"\input"),
        (r"\include{secret}", r"\include"),
        (r"\write18{rm -rf /}", r"\write"),
        (r"\immediate\write18{ls}", r"\write"),
        (r"\def\x{\input{y}} x", r"\def"),
        (r"\csname input\endcsname{z}", r"\csname"),
        (r"\newcommand{\evil}{\href{a}{b}}", r"\newcommand"),
        (r"\let\a\input", r"\let"),
        (r"\catcode`\%=0", r"\catcode"),
        (r"\htmlClass{x}{y}", r"\htmlClass"),
        (r"\expandafter\csname input\endcsname", r"\expandafter"),
        (r"\usepackage{shellesc}", r"\usepackage"),
        (r"\openout1=x.txt", r"\openout"),
    ],
)
def test_sanitize_removes_forbidden_commands(bad: str, forbidden: str) -> None:
    out = sanitize_latex(bad)
    assert forbidden not in out
    assert "\\input" not in out
    assert "\\endcsname" not in out
    assert find_disallowed_commands(out) == []


def test_forbidden_set_covers_security_table() -> None:
    for name in ("href", "url", "includegraphics", "input", "write", "def", "csname"):
        assert name in FORBIDDEN_COMMANDS
        assert name not in ALLOWED_COMMANDS


def test_href_removed_with_its_arguments_but_rest_kept() -> None:
    out = sanitize_latex(r"x^2 + \href{http://evil}{click} + 1")
    assert out == "x^2 +  + 1"
    assert "evil" not in out and "click" not in out


def test_unknown_command_removed_but_argument_text_kept() -> None:
    assert sanitize_latex(r"\foo{x}") == "{x}"
    assert sanitize_latex(r"\xyzzy{5cm}{1cm} y") == "{5cm}{1cm} y"


def test_rule_is_forbidden_with_dimensions_dropped() -> None:
    assert sanitize_latex(r"\rule{5cm}{1cm} y") == "y"


def test_disallowed_environment_unwrapped() -> None:
    out = sanitize_latex(r"\begin{tikzpicture} \draw (0,0) \end{tikzpicture}")
    assert "tikzpicture" not in out
    assert "\\draw" not in out
    assert "(0,0)" in out
    assert "matrix" in ALLOWED_ENVIRONMENTS and "tikzpicture" not in ALLOWED_ENVIRONMENTS


def test_begin_without_name_is_dropped() -> None:
    assert sanitize_latex(r"\begin x \end") == "x"


def test_math_delimiters_stripped() -> None:
    assert sanitize_latex(r"$x^2$") == "x^2"
    assert sanitize_latex(r"$$x^2$$") == "x^2"
    assert sanitize_latex(r"\(x^2\)") == "x^2"
    assert sanitize_latex(r"\[x^2\]") == "x^2"


def test_stray_dollar_and_hash_removed() -> None:
    assert sanitize_latex(r"a $ b # c") == "a  b  c"
    assert sanitize_latex(r"\$ \#") == r"\$ \#"


def test_braces_balanced() -> None:
    assert sanitize_latex(r"\frac{1}{2") == r"\frac{1}{2}"
    assert sanitize_latex(r"x}") == "x"
    assert sanitize_latex(r"{{x}") == "{{x}}"


def test_control_characters_and_whitespace_normalised() -> None:
    assert sanitize_latex("x\x00\x07y") == "xy"
    assert sanitize_latex("  x \n\n y  ") == "x \n\n y"  # trimmed, inner newlines kept
    assert sanitize_latex("é") == "é"  # NFC


def test_trailing_lone_backslash_removed() -> None:
    assert sanitize_latex("x^2\\") == "x^2"
    assert sanitize_latex("x \\' y") == "x  y"


def test_max_length_enforced_when_requested() -> None:
    long = "x" * (MAX_LATEX_LENGTH + 1)
    assert len(sanitize_latex(long)) == MAX_LATEX_LENGTH + 1
    with pytest.raises(AiclsError) as ei:
        sanitize_latex(long, max_len=MAX_LATEX_LENGTH)
    assert ei.value.code is ErrorCode.VALIDATION_ERROR
    assert sanitize_latex("x" * MAX_LATEX_LENGTH, max_len=MAX_LATEX_LENGTH)


def test_empty_and_whitespace_only() -> None:
    assert sanitize_latex("") == ""
    assert sanitize_latex("   ") == ""


def test_find_disallowed_commands_reports_names() -> None:
    assert find_disallowed_commands(r"\frac{1}{2} \href{a}{b} \foo") == ["href", "foo"]
    assert find_disallowed_commands(r"\begin{tikz} x \end{tikz}") == ["begin{tikz}", "end{tikz}"]


def test_allowed_and_forbidden_are_disjoint() -> None:
    assert not (ALLOWED_COMMANDS & FORBIDDEN_COMMANDS)


_latex_alphabet = st.sampled_from(
    [
        *list("abcxyz0123456789 {}[]^_+-=()*'%ş\n\t"),
        "\\",
        "\\frac",
        "\\href",
        "\\input",
        "\\def",
        "\\let",
        "\\write18",
        "\\csname",
        "\\operatorname",
        "\\begin{matrix}",
        "\\end{matrix}",
        "\\begin{foo}",
        "\\end{foo}",
        "\\ ",
        "\\\\",
        "\\{",
        "\\}",
        "$",
        "$$",
        "\\(",
        "\\)",
        "#",
        "&",
    ]
)


def _command_tokens(s: str) -> list[str]:
    """Independent TeX-style tokenizer: control words/symbols actually present in ``s``."""
    names: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:], re.DOTALL)
            if m is None:
                break
            names.append(m.group(1))
            i += m.end()
            continue
        i += 1
    return names


@given(st.lists(_latex_alphabet, max_size=40).map("".join))
def test_sanitize_is_idempotent_and_never_leaks_forbidden(s: str) -> None:
    once = sanitize_latex(s)
    assert sanitize_latex(once) == once
    assert find_disallowed_commands(once) == []
    assert not (set(_command_tokens(once)) & FORBIDDEN_COMMANDS)


def test_escaped_backslash_followed_by_text_is_not_a_command() -> None:
    # "\\" is the row separator; "href" after it is plain text, not \href
    assert sanitize_latex("\\\\href") == "\\\\href"
    assert _command_tokens("\\\\href") == ["\\"]


def test_operatorname_star_kept_but_other_stars_are_text() -> None:
    assert sanitize_latex(r"\operatorname*{argmax}_x") == r"\operatorname*{argmax}_x"
    assert sanitize_latex(r"\alpha*2") == r"\alpha*2"


def test_forbidden_command_with_unterminated_group_is_dropped_to_end() -> None:
    assert sanitize_latex(r"x \href{http://evil") == "x"


def test_forbidden_command_with_unclosed_bracket_keeps_bracket_text() -> None:
    out = sanitize_latex(r"\includegraphics[width x")
    assert "includegraphics" not in out
    assert out == "[width x"


def test_def_with_control_sequence_and_parameter_text() -> None:
    out = sanitize_latex(r"\def\cmd#1{\input{#1}} \cmd{x}")
    assert "\\input" not in out and "\\def" not in out and "\\cmd" not in out
    assert out == "{x}"


def test_let_consumes_two_control_sequences() -> None:
    assert sanitize_latex(r"\let\a\frac x") == "x"


def test_write_with_stream_number_and_openout_assignment() -> None:
    assert sanitize_latex(r"\openout3=\jobname.txt \write3{hi} y") == ".txt  y"


def test_removed_token_between_command_and_letters_does_not_fuse_names() -> None:
    assert sanitize_latex("\\frac\\0a") == "\\frac a"
    assert sanitize_latex(r"\alpha\href{a}{b}x") == r"\alpha x"
    assert sanitize_latex(r"\alpha x") == r"\alpha x"


def test_escaped_space_at_edges_survives_trimming() -> None:
    assert sanitize_latex("\\ \\") == "\\ "
    assert sanitize_latex("  \\ x \\   ") == "\\ x \\ "
