"""Shared plumbing for the ``aicls`` console command (docs/mimari/moduller.md#aiclscli).

Four things live here so that no command module has to repeat them:

* :class:`CliState` and :func:`emit` — every read command accepts ``--json``; the flag switches
  the same payload between a JSON document on stdout and a few human-readable English lines.
* :func:`render_error` — the single place that maps an :class:`aicls.core.errors.AiclsError` to a
  process exit code (``3`` for a refused approval gate, ADR-0012) and prints the Turkish user
  message the error carries from :mod:`aicls.core.messages_tr`.
* :func:`db_session` — opens an engine and brings the schema to ``head`` **through the
  migrations** (ADR-0003 forbids ``metadata.create_all``) before yielding a session.
* :func:`jsonable` — one JSON encoder for pydantic models, dataclasses, ``Path``, ``datetime``
  and enums, so ``--json`` output has the same shape everywhere.

Nothing here imports :mod:`aicls.api`, :mod:`aicls.ml`, :mod:`aicls.eval` or torch: the command
modules import their services lazily inside the command bodies, which keeps ``aicls --help`` fast
and keeps ``import aicls.cli.main`` free of torch.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import dataclasses
import datetime as dt
from enum import Enum
import importlib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, NoReturn, TextIO, cast

from structlog.typing import FilteringBoundLogger
import typer

from aicls.core.errors import AiclsError
from aicls.core.logging import configure_logging, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps sqlalchemy out of the import path
    from sqlalchemy.orm import Session

#: Process exit codes (docs/mimari/moduller.md#aiclscli, ADR-0012).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GATE = 3

#: Marker printed instead of a value that was deliberately not measured.
UNKNOWN = "unknown"


@dataclasses.dataclass(slots=True)
class CliState:
    """Per-invocation state shared between the root callback and the commands.

    Attributes:
        json_output: Whether the running command was asked for machine-readable output. The
            error renderer reads it too, so a ``--json`` run fails with a JSON problem document
            rather than a prose line.
    """

    json_output: bool = False


#: Where the running command's click context lives, in the order to ask. typer 0.27 vendors
#: click as ``typer._click`` and runs every command on that copy, so the *click package's* own
#: context stack is empty while a typer command runs; asking it made ``cli_state()`` hand out a
#: fresh, empty state on every call and ``--json`` was silently ignored (found by the CI run
#: of 2026-09-10, which installed typer 0.27 while the workstation still had 0.21). Older typer
#: has no ``typer._click`` and runs on click itself, hence the fallback.
_CONTEXT_MODULES = ("typer._click.globals", "click.globals")


def _current_context() -> Any:
    for module_name in _CONTEXT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - depends on the installed typer
            continue
        return module.get_current_context(silent=True)
    return None  # pragma: no cover - click is always installed


def cli_state() -> CliState:
    """Return the :class:`CliState` of the current invocation (a fresh one outside a context)."""
    ctx = _current_context()
    if ctx is None:  # pragma: no cover - only when a command body is called directly
        return CliState()
    root = ctx.find_root()
    if not isinstance(root.obj, CliState):
        root.obj = CliState()
    return root.obj


def use_json(flag: bool) -> bool:
    """Record the command's ``--json`` flag on the invocation state and return it."""
    state = cli_state()
    state.json_output = bool(flag)
    return state.json_output


class StderrStream:
    """A file-like object that resolves ``sys.stderr`` at write time.

    :func:`aicls.core.logging.configure_logging` binds its destination once, but the CLI must not
    capture whichever stream happened to be installed at start-up: ``stdout`` belongs to the
    command's own output (``--json`` must stay pipeable) and a test runner replaces both streams
    for the duration of one invocation.
    """

    def write(self, message: str) -> int:
        """Write ``message`` to the current ``sys.stderr``."""
        return sys.stderr.write(message)

    def flush(self) -> None:
        """Flush the current ``sys.stderr``."""
        sys.stderr.flush()


def cli_logger(name: str) -> FilteringBoundLogger:
    """Bind a logger **now**, after :func:`configure_cli_logging` has redirected the stream.

    ``structlog``'s :func:`~structlog.get_logger` binds the logger factory eagerly, so a
    module-level logger created at import time keeps writing to whichever stream was configured
    then. The CLI reconfigures logging in its root callback, so every log site here binds at call
    time instead — otherwise a command's own log lines would land on stdout and corrupt
    ``--json`` output.
    """
    return get_logger(name)


def configure_cli_logging(level: str | None = None) -> None:
    """Send structured log lines to stderr so stdout carries only the command's output."""
    # StderrStream implements the two methods structlog's PrintLogger uses (write, flush);
    # the parameter is typed TextIO, which a full file object would be.
    configure_logging(level, stream=cast("TextIO", StderrStream()))


# --------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Convert ``value`` into something :func:`json.dumps` accepts.

    Handles pydantic models (``model_dump``), dataclasses, ``Path``, ``datetime``/``date``,
    enums, sets and nested containers. Unknown objects degrade to ``str(value)`` rather than
    raising: a CLI must not fail while reporting a result.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return jsonable(dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return jsonable(as_dict())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [jsonable(item) for item in value]
    return str(value)


def dump_json(payload: Any) -> str:
    """Serialise ``payload`` the way every ``--json`` command writes it (UTF-8, two spaces)."""
    return json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)


def emit(payload: Any, *lines: str) -> None:
    """Print ``payload`` as JSON under ``--json``, otherwise the given English lines.

    Args:
        payload: The machine-readable result of the command.
        *lines: Human-readable lines; printed one per line when ``--json`` was not given.
    """
    if cli_state().json_output:
        typer.echo(dump_json(payload))
        return
    for line in lines:
        typer.echo(line)


def echo(line: str) -> None:
    """Print one human-readable line, suppressed under ``--json`` (which owns stdout)."""
    if not cli_state().json_output:
        typer.echo(line)


def fail(payload: Any, *lines: str, code: int = EXIT_ERROR) -> NoReturn:
    """Emit a result that represents a failed check and exit with ``code``.

    Used where the outcome is a *report* rather than an exception — a data root that does not
    match its manifest, an untagged heading, a report that may not be cited.
    """
    emit(payload, *lines)
    raise typer.Exit(code)


def render_error(exc: AiclsError) -> int:
    """Print ``exc`` on stderr and return the process exit code it maps to.

    Under ``--json`` the RFC 9457 problem document is printed as **one line**, so a script gets
    the same ``code``/``details`` it would get from the HTTP API and can pick it out of the log
    lines that share the stream; otherwise the Turkish user message
    (:mod:`aicls.core.messages_tr`) is printed with the machine-readable code in front of it.
    """
    if cli_state().json_output:
        line = json.dumps(jsonable(exc.to_problem()), ensure_ascii=False, sort_keys=True)
        typer.echo(line, err=True)
    else:
        typer.echo(f"{exc.code.value}: {exc.message_tr}", err=True)
        for key, value in sorted(exc.details.items()):
            typer.echo(f"  {key}: {value}", err=True)
    cli_logger(__name__).warning("cli.error", code=exc.code.value, exit_code=exc.exit_code)
    return int(exc.exit_code)


# --------------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------------
def split_csv(value: str | None) -> tuple[str, ...]:
    """Split ``"university,gender"`` into ``("university", "gender")`` (empty parts dropped)."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_ratios(value: str | None, default: tuple[float, float, float]) -> tuple[float, ...]:
    """Parse ``"0.8,0.1,0.1"`` into three floats.

    Raises:
        aicls.core.errors.InputValidationError: If the value is not three numbers.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR

    if not value:
        return default
    parts = split_csv(value)
    try:
        numbers = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise InputValidationError(
            MSG_VALIDATION_ERROR, details={"reason": "ratios_not_numeric", "value": value}
        ) from exc
    if len(numbers) != 3:
        raise InputValidationError(
            MSG_VALIDATION_ERROR, details={"reason": "ratios_need_three_values", "value": value}
        )
    return numbers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects (blank lines skipped).

    Raises:
        aicls.core.errors.NotFoundError: If the file does not exist.
        aicls.core.errors.InputValidationError: If a line is not a JSON object.
    """
    from aicls.core.errors import InputValidationError, NotFoundError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR, msg

    target = Path(path)
    if not target.is_file():
        raise NotFoundError(msg("FILE_NOT_FOUND", path=str(target)), details={"path": str(target)})
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    MSG_VALIDATION_ERROR,
                    details={"reason": "jsonl_not_json", "path": str(target), "line": line_no},
                ) from exc
            if not isinstance(row, dict):
                raise InputValidationError(
                    MSG_VALIDATION_ERROR,
                    details={
                        "reason": "jsonl_row_not_object",
                        "path": str(target),
                        "line": line_no,
                    },
                )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------------------------
def resolve_db_url(url: str | None) -> str:
    """Return ``url`` or the configured ``AICLS_DB_URL`` (``Settings.db_url``)."""
    if url:
        return url
    from aicls.core.settings import get_settings

    return get_settings().db_url


@contextmanager
def db_session(url: str | None = None, *, migrate: bool = True) -> Iterator[Session]:
    """Yield a session on ``url``, migrating the schema to ``head`` first.

    Args:
        url: Database URL; the configured one when omitted.
        migrate: Run ``alembic upgrade head`` when the database is not already at head. The
            schema always comes from the migrations (ADR-0003), never from ``create_all``.

    Yields:
        An open :class:`~sqlalchemy.orm.Session`; the transaction is committed on a clean exit
        and rolled back on an exception by :func:`aicls.db.session_scope`.
    """
    from aicls.db import current_revision, head_revision, make_engine, session_scope, upgrade_head

    resolved = resolve_db_url(url)
    engine = make_engine(resolved)
    try:
        if migrate and current_revision(engine) != head_revision():
            upgrade_head(engine)
            cli_logger(__name__).info("cli.db.migrated", revision=head_revision())
        with session_scope(engine) as session:
            yield session
    finally:
        engine.dispose()


__all__ = [
    "EXIT_ERROR",
    "EXIT_GATE",
    "EXIT_OK",
    "UNKNOWN",
    "CliState",
    "StderrStream",
    "cli_logger",
    "cli_state",
    "configure_cli_logging",
    "db_session",
    "dump_json",
    "echo",
    "emit",
    "fail",
    "jsonable",
    "parse_ratios",
    "read_jsonl",
    "render_error",
    "resolve_db_url",
    "split_csv",
    "use_json",
]
