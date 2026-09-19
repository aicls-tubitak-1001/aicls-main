"""Deterministic Matplotlib output and the product/chain-rule narration."""

from __future__ import annotations

import hashlib

import pytest
import sympy

from aicls.mathengine import Diff, Plot, compute, diff_steps, render_plot
from aicls.mathengine.errors import UnsupportedProblem
from aicls.mathengine.messages import PENDING_CORE_MESSAGES, tr
from aicls.mathengine.plotting import (
    DETERMINISTIC_RC,
    FIGURE_DPI,
    MAX_SAMPLES,
    MIN_SAMPLES,
    load_backend,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
X = sympy.Symbol("x", real=True)

#: ADR-0016 budgets ~300 kB for moving ``plot_png`` back through the sandbox pipe.
MAX_PLOT_BYTES = 300_000


# ---------------------------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------------------------
def test_plot_renders_deterministic_png_hash() -> None:
    first = render_plot(X**2, X, -10.0, 10.0)
    second = render_plot(X**2, X, -10.0, 10.0)
    assert first[:8] == PNG_SIGNATURE
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert len(first) < MAX_PLOT_BYTES


def test_plot_bytes_differ_for_a_different_expression() -> None:
    assert render_plot(X**2, X, -1.0, 1.0) != render_plot(X**3, X, -1.0, 1.0)


def test_plot_png_carries_no_version_or_timestamp_metadata() -> None:
    """The Software tag would embed the Matplotlib version and break the hash across machines."""
    payload = render_plot(X, X, -1.0, 1.0)
    assert b"Software" not in payload
    assert b"matplotlib" not in payload.lower()


def test_svg_output_is_also_supported_and_stable() -> None:
    first = render_plot(X, X, -1.0, 1.0, fmt="svg")
    assert first.lstrip()[:5] in {b"<?xml", b"<svg "}
    assert first == render_plot(X, X, -1.0, 1.0, fmt="svg")


def test_labels_come_from_the_message_table() -> None:
    for key in ("PLOT_TITLE", "PLOT_X_LABEL", "PLOT_Y_LABEL"):
        assert key in PENDING_CORE_MESSAGES
    assert tr("PLOT_TITLE", expr="x**2").endswith("fonksiyonunun grafiği")
    assert tr("PLOT_X_LABEL", var="t") == "t"
    assert tr("PLOT_Y_LABEL", var="t") == "f(t)"


def test_tr_prefers_the_core_message_table() -> None:
    assert tr("UNSUPPORTED_PROBLEM") == "Bu problem tipi şu an için desteklenmemektedir."
    with pytest.raises(KeyError):
        tr("NO_SUCH_MESSAGE_KEY")


def test_sample_count_is_clamped_into_range() -> None:
    small = render_plot(X, X, -1.0, 1.0, samples=1)
    large = render_plot(X, X, -1.0, 1.0, samples=MAX_SAMPLES * 10)
    assert small == render_plot(X, X, -1.0, 1.0, samples=MIN_SAMPLES)
    assert large == render_plot(X, X, -1.0, 1.0, samples=MAX_SAMPLES)


def test_plot_survives_poles_and_undefined_regions() -> None:
    """``1/x`` on a range containing 0 and ``ln(x)`` on a range containing negatives."""
    assert render_plot(1 / X, X, -5.0, 5.0)[:8] == PNG_SIGNATURE
    assert render_plot(sympy.log(X), X, -5.0, 5.0)[:8] == PNG_SIGNATURE


def test_plot_of_a_constant_still_renders() -> None:
    assert render_plot(sympy.Integer(3), X, -1.0, 1.0)[:8] == PNG_SIGNATURE


def test_plot_without_any_finite_value_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        render_plot(sympy.log(X), X, -5.0, -1.0)
    assert excinfo.value.details["reason"] == "no_finite_values"


def test_plot_with_a_degenerate_range_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        render_plot(X, X, 1.0, 1.0)
    assert excinfo.value.details["reason"] == "empty_plot_range"


def test_plot_of_an_unlambdifiable_expression_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem):
        render_plot(sympy.Function("f")(X), X, -1.0, 1.0)


def test_matplotlib_is_not_imported_by_importing_the_package() -> None:
    """``aicls.nlu`` imports the command models on every request; it must not pay for plotting."""
    from pathlib import Path
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    script = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import aicls.mathengine;"
        "print('matplotlib' in sys.modules)"
    ).format(src=repo_root / "src")
    # Fixed argv, no shell: the only variable part is this repository's own src path.
    output = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert output.stdout.strip() == "False", output.stdout


def test_the_backend_loader_is_cached_and_selects_agg() -> None:
    first = load_backend()
    assert load_backend() is first
    matplotlib, figure_class, canvas_class = first
    assert matplotlib.get_backend().lower() == "agg"
    assert figure_class.__name__ == "Figure"
    assert canvas_class.__name__ == "FigureCanvasAgg"


def test_rc_context_pins_the_parameters_that_could_drift() -> None:
    for key in ("font.family", "svg.hashsalt", "path.simplify", "savefig.dpi"):
        assert key in DETERMINISTIC_RC
    assert DETERMINISTIC_RC["savefig.dpi"] == FIGURE_DPI


def test_plot_command_and_render_plot_agree() -> None:
    result = compute(Plot(op="plot", expr="x^{2}", xmin=-10.0, xmax=10.0))
    assert result.plot_png == render_plot(X**2, X, -10.0, 10.0)


# ---------------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------------
def test_product_rule_is_narrated() -> None:
    expr = X * sympy.sin(X)
    steps = diff_steps(expr, X, sympy.diff(expr, X))
    assert [step.rule for step in steps] == ["product_rule", "product_rule_applied", "result"]
    assert steps[0].title_tr == "Çarpım kuralı"
    assert r"\sin" in steps[0].latex
    assert steps[-1].latex


def test_chain_rule_is_narrated() -> None:
    expr = sympy.sin(X**2)
    steps = diff_steps(expr, X, sympy.diff(expr, X))
    assert [step.rule for step in steps] == ["chain_rule", "chain_rule_applied", "result"]
    assert steps[0].title_tr == "Zincir kuralı"
    assert "u" in steps[0].latex


def test_chain_rule_on_a_power_of_a_compound_base() -> None:
    expr = (X**2 + 1) ** 3
    steps = diff_steps(expr, X, sympy.diff(expr, X))
    assert steps and steps[0].rule == "chain_rule"


def test_constant_multiple_is_not_a_product_rule() -> None:
    assert diff_steps(2 * X, X, sympy.Integer(2)) == []


def test_three_factor_product_is_not_narrated() -> None:
    expr = X * sympy.sin(X) * sympy.cos(X)
    assert diff_steps(expr, X, sympy.diff(expr, X)) == []


def test_product_rule_folds_constants_into_the_first_factor() -> None:
    expr = 3 * X * sympy.sin(X)
    steps = diff_steps(expr, X, sympy.diff(expr, X))
    assert steps and steps[0].rule == "product_rule"
    assert "3" in steps[0].latex


def test_plain_power_and_simple_function_are_not_narrated() -> None:
    assert diff_steps(X**2, X, 2 * X) == []
    assert diff_steps(sympy.sin(X), X, sympy.cos(X)) == []
    assert diff_steps(sympy.Integer(5), X, sympy.Integer(0)) == []


def test_composition_of_a_constant_inner_is_not_a_chain_rule() -> None:
    assert diff_steps(sympy.sin(sympy.Integer(2)), X, sympy.Integer(0)) == []


def test_steps_reach_the_result_of_a_diff_command() -> None:
    result = compute(Diff(op="diff", expr=r"x\cos(x)"))
    assert result.steps
    assert all(step.title_tr and step.latex for step in result.steps)
