"""The spoken-mathematics grammar: Turkish dictation -> canonical LaTeX."""

from __future__ import annotations

import pytest

from aicls.latex2sympy import parse_latex
from aicls.nlu.errors import SpokenMathError
from aicls.nlu.spoken_math import spoken_to_ast, spoken_to_latex
from aicls.nlu.spoken_math.parser import MAX_SPOKEN_CHARS, grammar_source


class TestDocumentedExamples:
    def test_spoken_math_x_kare_arti_iki_x_to_latex(self) -> None:
        """The architecture page's first example."""
        assert spoken_to_latex("x kare artı iki x") == "x^{2} + 2 x"

    def test_spoken_math_bir_bolu_x_to_frac(self) -> None:
        """ "bölü" becomes a fraction."""
        assert spoken_to_latex("bir bölü x") == r"\frac{1}{x}"

    def test_x_kare_carpi_sinus_x(self) -> None:
        assert spoken_to_latex("x kare çarpı sinüs x") == r"x^{2} \sin\left(x\right)"

    def test_bolu_binds_only_the_neighbouring_powers(self) -> None:
        """Documented rule: "x kare bölü x artı bir" is ``x^2/x + 1``, not ``x^2/(x+1)``."""
        assert spoken_to_latex("x kare bölü x artı bir") == r"\frac{x^{2}}{x} + 1"
        assert (
            spoken_to_latex("x kare bölü parantez aç x artı bir parantez kapa")
            == r"\frac{x^{2}}{x + 1}"
        )


class TestGrammarCoverage:
    @pytest.mark.parametrize(
        ("spoken", "latex"),
        [
            ("x kare", "x^{2}"),
            ("x küp", "x^{3}"),
            ("iki x", "2 x"),
            ("x üssü dört", "x^{4}"),
            ("iki x üzeri üç", "2 x^{3}"),
            ("x kare eksi dört", "x^{2} - 4"),
            ("eksi x", "-x"),
            ("sinüs x", r"\sin\left(x\right)"),
            ("kosinüs x", r"\cos\left(x\right)"),
            ("tanjant x", r"\tan\left(x\right)"),
            ("logaritma x", r"\log\left(x\right)"),
            ("ln x", r"\ln\left(x\right)"),
            ("e üzeri x", "e^{x}"),
            ("e üzeri eksi x kare", "e^{-x^{2}}"),
            ("karekök x", r"\sqrt{x}"),
            ("küpkök x", r"\sqrt[3]{x}"),
            ("mutlak değer x", r"\lvert x \rvert"),
            ("beş faktöriyel", "5!"),
            ("pi", r"\pi"),
            ("sonsuz", r"\infty"),
            ("eksi sonsuz", r"-\infty"),
            ("alfa artı beta", r"\alpha + \beta"),
            ("üç x artı beş eşittir on", "3 x + 5 = 10"),
            ("parantez aç x artı bir parantez kapa kare", r"\left(x + 1\right)^{2}"),
            ("sinüs x bölü x", r"\frac{\sin\left(x\right)}{x}"),
        ],
    )
    def test_translates(self, spoken: str, latex: str) -> None:
        assert spoken_to_latex(spoken) == latex

    @pytest.mark.parametrize(
        "spoken",
        [
            "x kare artı iki x",
            "sinüs x bölü x",
            "e üzeri eksi x kare",
            "üç x artı beş eşittir on",
            "parantez aç x artı bir parantez kapa küp",
            "mutlak değer x eksi iki",
            "küpkök sekiz",
        ],
    )
    def test_output_is_accepted_by_parse_latex(self, spoken: str) -> None:
        """The whole point of building AST nodes: the printer's output is always re-parsable."""
        parse_latex(spoken_to_latex(spoken))

    def test_ast_nodes_are_returned_by_spoken_to_ast(self) -> None:
        from aicls.latex2sympy.ast import Node

        assert isinstance(spoken_to_ast("x kare"), Node)


class TestFailures:
    @pytest.mark.parametrize("spoken", ["", "   ", "türev", "hava durumu", "tahtayı temizle"])
    def test_non_expressions_raise_spoken_math_error(self, spoken: str) -> None:
        with pytest.raises(SpokenMathError):
            spoken_to_latex(spoken)

    def test_a_trailing_case_ending_is_tolerated(self) -> None:
        """ "x kare nin" is "x karenin" dictated slowly: the ending is dropped, not parsed."""
        assert spoken_to_latex("x kare nin") == "x^{2}"

    def test_over_long_input_is_refused(self) -> None:
        with pytest.raises(SpokenMathError) as excinfo:
            spoken_to_latex("x artı " * MAX_SPOKEN_CHARS)
        assert excinfo.value.details["reason"] == "too_long"

    def test_error_carries_a_position(self) -> None:
        with pytest.raises(SpokenMathError) as excinfo:
            spoken_to_latex("x kare artı", normalized=True)
        assert excinfo.value.position >= 1


class TestGrammarResource:
    def test_grammar_ships_inside_the_package(self) -> None:
        source = grammar_source()
        assert "start: relation" in source
        assert "%ignore WS" in source
