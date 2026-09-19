"""The ``aicls`` console command: the root Typer application (docs/mimari/moduller.md#aiclscli).

``pyproject.toml`` declares ``aicls = "aicls.cli.main:app"``; every command group lives in its own
module next to this one and is mounted here. Two conventions hold across the whole tree:

* **One error boundary.** :class:`AiclsGroup` catches every :class:`aicls.core.errors.AiclsError`
  raised anywhere below it, prints the Turkish user message and exits with the error's own
  ``exit_code`` — ``3`` for a refused approval gate (ADR-0012), ``1`` for everything else. No
  command has to remember to do this.
* **Lazy service imports.** Command bodies import :mod:`aicls.api`, :mod:`aicls.ml`,
  :mod:`aicls.eval` and their dependencies *inside* the function, so ``aicls --help`` does not pay
  for torch, SymPy or FastAPI and ``import aicls.cli.main`` stays free of torch.

The approval gates are enforced here and nowhere else in the CLI: ``aicls train`` and
``aicls profile`` call :func:`aicls.ml.trainer.require_training_gate`, which refuses unless the
run is synthetic (``--synthetic``/``data.dataset: synthetic_*``) or all three conditions of the
triple lock hold. Nothing in this package ever sets ``AICLS_GATE_*`` or writes the ledger.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from aicls import __version__
from aicls.cli import book as book_cmd
from aicls.cli import contrib as contrib_cmd
from aicls.cli import data as data_cmd
from aicls.cli import db as db_cmd
from aicls.cli import evaluate as eval_cmd
from aicls.cli import models as models_cmd
from aicls.cli import nlu as nlu_cmd
from aicls.cli import report as report_cmd
from aicls.cli import serve as serve_cmd
from aicls.cli import train as train_cmd
from aicls.cli.common import CliState, configure_cli_logging, render_error
from aicls.core.errors import AiclsError

#: Short description shown by ``aicls --help``.
HELP = """AI-CLS platform command line (TUBITAK 1001 #125E748).

Serve the API, migrate and seed the database, build and verify datasets, run the gated training
and profiling loops, evaluate every component, load the textbook graph, manage the model registry
and check the reports that feed the progress report.
"""


class AiclsGroup(TyperGroup):
    """Root command group that turns an :class:`AiclsError` into its documented exit code."""

    def invoke(self, ctx: Any) -> Any:
        """Run the requested command, converting a typed platform error into an exit code.

        ``ctx`` is a click ``Context``; it is annotated ``Any`` because typer 0.27 vendors click
        as ``typer._click`` and its ``Context`` is a different class from ``click.Context``,
        while typer 0.21 uses click's own. One annotation has to type-check against both.
        """
        try:
            return super().invoke(ctx)
        except AiclsError as exc:
            raise typer.Exit(render_error(exc)) from exc


def _version_callback(value: bool) -> None:
    """Print the package version and exit (``--version``)."""
    if value:
        typer.echo(f"aicls {__version__}")
        raise typer.Exit


app = typer.Typer(
    cls=AiclsGroup,
    name="aicls",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the AI-CLS version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Set up the invocation state, refresh the settings and send the logs to stderr.

    ``Settings`` is cached per process (:func:`aicls.core.settings.get_settings`); clearing the
    cache here means an ``AICLS_*`` variable exported just before the call is honoured, which is
    also what makes the gate-refusal tests hermetic. Logging is reconfigured onto **stderr** so
    that stdout carries only the command's own output and ``--json`` stays pipeable.
    """
    from aicls.core.settings import get_settings

    get_settings.cache_clear()
    ctx.obj = CliState()
    configure_cli_logging(get_settings().log_level)


app.command("serve", help=serve_cmd.HELP)(serve_cmd.serve)
app.add_typer(db_cmd.app, name="db")
app.add_typer(data_cmd.app, name="data")
app.command("train", help=train_cmd.TRAIN_HELP)(train_cmd.train)
app.command("profile", help=train_cmd.PROFILE_HELP)(train_cmd.profile)
app.command("eval", help=eval_cmd.HELP)(eval_cmd.evaluate)
app.add_typer(book_cmd.app, name="book")
app.add_typer(models_cmd.app, name="models")
app.add_typer(contrib_cmd.app, name="contrib")
app.add_typer(nlu_cmd.app, name="nlu")
app.add_typer(report_cmd.app, name="report")


__all__ = ["HELP", "AiclsGroup", "app", "main"]

if __name__ == "__main__":  # pragma: no cover - "python -m aicls.cli.main"
    app()
