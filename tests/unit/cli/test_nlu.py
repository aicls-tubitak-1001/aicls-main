"""``aicls nlu try``: the proposal's own example, end to end through the CLI.

"x karenin türevini al" is the utterance the proposal uses to describe the text-to-command layer,
so it is the one this command is asserted on. The CLI prints the typed command; it never computes
a result — only :mod:`aicls.mathengine` does that.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


def test_nlu_try_resolves_the_proposal_example(runner: CliRunner, app_module: object) -> None:
    """The headline utterance becomes a differentiate command with the expression filled in."""
    result = runner.invoke(app_module, ["nlu", "try", "x karenin türevini al", "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["outcome"] == "matched"
    assert payload["intent"] == "differentiate"
    assert payload["command"]["op"] == "diff"
    assert "x" in payload["command"]["expr"]
    assert payload["trace"], "the trace is what the UI shows next to a voice command"


def test_nlu_try_uses_the_session_context(runner: CliRunner, app_module: object) -> None:
    """``--last-latex`` is what makes a bare "türevini al" resolvable."""
    result = runner.invoke(
        app_module, ["nlu", "try", "türevini al", "--last-latex", "x^{2}", "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["outcome"] == "matched"
    assert payload["command"]["expr"] == "x^{2}"


def test_nlu_try_reports_a_refusal_without_failing(runner: CliRunner, app_module: object) -> None:
    """ "I did not understand" is an outcome of the layer, not a process error."""
    result = runner.invoke(app_module, ["nlu", "try", "bugün hava nasıl", "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["outcome"] in {"no_match", "clarification"}
    if payload["outcome"] == "no_match":
        assert payload["suggestions_tr"], "a refusal must guide, not just refuse"


def test_nlu_try_prints_human_readable_lines(runner: CliRunner, app_module: object) -> None:
    """Without ``--json`` the command prints the intent, the slots and the trace."""
    result = runner.invoke(app_module, ["nlu", "try", "x karenin türevini al"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "intent differentiate" in result.stdout
    assert "trace:" in result.stdout
