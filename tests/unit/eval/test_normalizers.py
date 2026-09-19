"""Frozen normalisation rules (ADR-0015, docs/ml/degerlendirme.md §1).

Every assertion here is a *definition*, not an observation: if one of them has to change, the
metric changed, and ADR-0015 requires a new version (``latex_v2`` / ``tr_v2``) next to the old
one — never an edit of ``v1``.
"""

from __future__ import annotations

import pytest

from aicls.eval.normalizers import (
    MAX_LATEX_PASSES,
    NORMALIZER_VERSIONS,
    NORMALIZERS,
    apply_norm,
    latex_tokens,
    latex_v1,
    tr_v1,
    tr_words,
    turkish_number_words,
)


# ---------------------------------------------------------------------------------- latex_v1
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # single-token exponent gets canonical braces (rule 3)
        ("x^2", "x ^ { 2 }"),
        ("x_1", "x _ { 1 }"),
        ("x^{2}", "x ^ { 2 }"),
        # a redundant group around a single token is flattened (rule 4) ...
        ("{x}", "x"),
        # ... to a fixed point, so nesting cannot survive one pass
        ("{{x}}", "x"),
        ("{{{2}}}", "2"),
        # ... but never when it is the argument of a script
        ("x^{y}", "x ^ { y }"),
        # \left / \right and spacing macros disappear (rule 1, via normalize_latex)
        (r"\left( x \right)", "( x )"),
        (r"x \, + \; y", "x + y"),
        # \dfrac and \tfrac collapse onto \frac (rule 1)
        (r"\dfrac{1}{2}", r"\frac 1 2"),
        (r"\tfrac{1}{2}", r"\frac 1 2"),
        # unicode becomes the command (rule 1)
        ("x²", "x ^ { 2 }"),
        ("π", r"\pi"),
        # digit runs stay one token (rule 2)
        ("x^{12}", "x ^ { 12 }"),
    ],
)
def test_latex_v1_reference_forms(raw: str, expected: str) -> None:
    """The frozen surface form of ``latex_v1`` for the documented rules."""
    assert latex_v1(raw) == expected


def test_latex_v1_is_idempotent() -> None:
    """Applying the normaliser twice never changes the result (ADR-0015 §1)."""
    samples = [
        r"\dfrac{x^2}{\left(y+1\right)}",
        r"\int_0^1 x^2 \, dx",
        r"\lim_{x\to0}\frac{\sin x}{x}=1",
        "{x}^{2}",
        "α + β²",
        "",
    ]
    for sample in samples:
        once = latex_v1(sample)
        assert latex_v1(once) == once, sample


def test_latex_v1_does_not_flatten_a_braced_script_operator() -> None:
    """Rule 4 removes *redundant* braces only; ``{^}`` braces are not redundant.

    Found by ``test_latex_v1_is_idempotent`` (hypothesis, counterexample ``"{^}x"``). Flattening
    ``{ ^ }`` to a bare ``^`` hands the operator the next token as its argument, so a second
    pass produces ``^ { x }`` — a different string, and a different expression. Rule 3 already
    refuses to brace an operator as a script argument (``_is_operand``); rule 4 has to refuse
    the mirror case, or the two rules fight each other and the normaliser stops being a
    function of its input.
    """
    assert latex_v1("{^}x") == "{ ^ } x"
    assert latex_v1("{_}y") == "{ _ } y"
    assert latex_v1("x{^}2") == "x { ^ } 2"
    for sample in ("{^}x", "{_}y", "x{^}2", "{{^}}", "{^}"):
        once = latex_v1(sample)
        assert latex_v1(once) == once, sample
    # a group around a real operand is still redundant and still goes away
    assert latex_v1("{x}") == "x"
    assert latex_v1(r"{\frac}{a}{b}") == r"\frac a b"


def test_latex_v1_tokens_are_commands_digit_runs_and_single_characters() -> None:
    """Rule 2: a command, a digit run and any other single character are one token each."""
    assert latex_tokens(r"\frac{12}{ab}") == ["\\frac", "{", "12", "}", "{", "a", "b", "}"]


def test_latex_v1_empty_input_is_empty() -> None:
    """An empty hypothesis normalises to the empty string, not to an error."""
    assert latex_v1("") == ""
    assert latex_v1("   ") == ""


# ------------------------------------------------------------------------------------- tr_v1
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Turkish casefold: İ -> i and I -> ı (never the Unicode default)
        ("İNTEGRAL", "integral"),
        ("IRMAK", "ırmak"),
        # punctuation, apostrophes and hyphens are deleted, not replaced by a space
        ("x'in türevi", "xin türevi"),
        ("türev, integral; limit.", "türev integral limit"),
        ("bir-iki", "biriki"),  # the hyphen simply disappears, it is not a word boundary
        # digits become Turkish number words; suffixes stay separate tokens (rule 4)
        ("0'dan 1'e", "sıfır dan bir e"),
        ("12 sayısı", "on iki sayısı"),
        # whitespace is squeezed (rule 5)
        ("  iki   x  ", "iki x"),
    ],
)
def test_tr_v1_reference_forms(raw: str, expected: str) -> None:
    """The frozen surface form of ``tr_v1`` for the five documented rules."""
    assert tr_v1(raw) == expected


def test_tr_v1_is_idempotent() -> None:
    """Applying the normaliser twice never changes the result (ADR-0015 §1)."""
    samples = [
        "X'in 2. türevini al",
        "SIFIRDAN 1'E KADAR",
        "İki  virgül,  beş",
        "2026 yılı",
        "",
    ]
    for sample in samples:
        once = tr_v1(sample)
        assert tr_v1(once) == once, sample


def test_tr_v1_normalizer_changes_word_count_only_by_digit_rule() -> None:
    """Rule 4 is the *only* rule that may change the number of words (ADR-0015).

    Lexicon folding, ASR-confusion tables and math-term joining are deliberately absent, so a
    text without digits keeps exactly as many words as it had.
    """
    without_digits = [
        "x karenin türevini al",
        "sinüs x'in integralini hesapla",
        "İKİ ÜÇ dört",
        "limit, x sıfıra giderken; sinüs x bölü x",
    ]
    for text in without_digits:
        assert len(tr_words(text)) == len(text.split()), text

    # the digit rule, and only the digit rule, moves the count: "0'dan" is one word before and
    # two after ("sıfır" + the suffix "dan").
    assert tr_words("0'dan") == ["sıfır", "dan"]
    assert len(tr_words("0'dan 1'e")) == 4
    assert len(["0'dan", "1'e"]) == 2


def test_tr_v1_does_not_fold_asr_variants_or_math_words() -> None:
    """ADR-0015: no lexicon folding lives in the metric normaliser.

    ``aicls.nlu.normalizer`` may map "sinus" to "sinüs"; the WER normaliser may not, otherwise
    the metric would be tuned to the test set.
    """
    assert tr_v1("sinus") != tr_v1("sinüs")
    assert tr_v1("x kare") != tr_v1("x karesi")
    assert tr_v1("türevini") != tr_v1("türev")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, ["sıfır"]),
        (7, ["yedi"]),
        (10, ["on"]),
        (12, ["on", "iki"]),
        (90, ["doksan"]),
        (100, ["yüz"]),
        (200, ["iki", "yüz"]),
        (1000, ["bin"]),
        (2026, ["iki", "bin", "yirmi", "altı"]),
        (1_000_000, ["bir", "milyon"]),
        (
            1_234_567_890,
            [
                "bir",
                "milyar",
                "iki",
                "yüz",
                "otuz",
                "dört",
                "milyon",
                "beş",
                "yüz",
                "altmış",
                "yedi",
                "bin",
                "sekiz",
                "yüz",
                "doksan",
            ],
        ),
    ],
)
def test_turkish_number_words_reference_values(value: int, expected: list[str]) -> None:
    """The digit-to-word table is frozen with the metric."""
    assert turkish_number_words(value) == expected


def test_very_long_digit_runs_are_read_digit_by_digit() -> None:
    """A digit run longer than the named range is read out digit by digit (frozen fallback)."""
    assert tr_words("9" * 13) == ["dokuz"] * 13


# -------------------------------------------------------------------------------- dispatching
def test_normalizer_versions_table_is_the_reported_one() -> None:
    """Every report writes this table; it names the versions the numbers were produced with."""
    assert NORMALIZER_VERSIONS == {"latex": "v1", "tr": "v1"}
    assert set(NORMALIZERS) == {"latex_v1", "tr_v1", "none"}


def test_apply_norm_none_is_the_identity() -> None:
    """``norm="none"`` is the raw column every report prints next to the normalised one."""
    assert apply_norm("X'in  TÜREVİ", "none") == "X'in  TÜREVİ"
    assert apply_norm("X'in TÜREVİ", "tr_v1") == "xin türevi"


def test_apply_norm_rejects_an_unknown_version() -> None:
    """An unknown normaliser name is a validation error, never a silent fallback."""
    from aicls.core.errors import InputValidationError

    with pytest.raises(InputValidationError):
        apply_norm("x", "latex_v2")


def test_turkish_number_words_rejects_a_negative_value() -> None:
    """The digit rule only ever sees digit runs, so a negative value is a programming error."""
    from aicls.core.errors import InputValidationError

    with pytest.raises(InputValidationError):
        turkish_number_words(-1)


# ---------------------------------------------------------------------------------------------
# latex_v1 reaches a fixed point (wave-7 integration gate, 2026-09-08)
# ---------------------------------------------------------------------------------------------
def test_latex_v1_settles_on_inputs_whose_first_pass_moves_a_function_power() -> None:
    r"""These two inputs were the counterexamples ``test_latex_v1_is_idempotent`` produced.

    ``normalize_latex``'s function-power repair is not whitespace-insensitive: it did not
    recognise ``\sin{xx}^\alpha`` but did recognise the spaced, brace-balanced form that
    ``latex_v1`` itself produces, so a second application moved the exponent and changed the
    string a CER had already been computed on. The parser fix removed the first case; iterating
    to a fixed point removes the class.
    """
    for raw in (r"\sin{xx}^\alpha", r"\sin{xx}^\alpha{x", r"\cos{x}^2", r"\ln{x}^{2}"):
        once = latex_v1(raw)
        assert latex_v1(once) == once, raw


def test_latex_v1_reports_when_it_cannot_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pass cap is a bound, not a silent truncation: it returns the last value and warns."""
    from aicls.eval import normalizers

    seen: list[int] = []

    def never_settles(text: str) -> str:
        seen.append(len(text))
        return text + " x"

    monkeypatch.setattr(normalizers, "_latex_v1_pass", never_settles)
    result = normalizers.latex_v1("x")

    assert len(seen) == normalizers.MAX_LATEX_PASSES
    assert result.endswith("x")


def test_latex_v1_pass_count_is_documented_and_positive() -> None:
    """A cap of zero would make ``latex_v1`` return its raw input."""
    assert MAX_LATEX_PASSES >= 2
