"""``aicls db`` — schema migrations and reference data (ADR-0003).

The schema is created by Alembic in every environment, including the in-memory test database, so
that every pull request exercises the migrations; ``metadata.create_all`` is never called here.

``aicls db seed`` inserts the **reference data the platform cannot run without**, and only that:

* every published Turkish consent version (:func:`aicls.collect.consent_texts`) — the voice
  contribution flow refuses a recording whose consent version is not stored;
* the prompt sentences a contributor is asked to read, taken from the packaged NLU command
  inventory (:func:`aicls.nlu.load_inventory`), because those utterances are exactly the command
  vocabulary İP-3 needs recordings of.

Seeding is idempotent: consent texts are immutable once stored and a prompt whose text already
exists is skipped, so running it twice changes nothing.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Annotated, Any

import typer

from aicls.cli.common import cli_logger, db_session, emit, resolve_db_url, use_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

app = typer.Typer(
    name="db",
    help="Create, migrate and seed the AI-CLS database.",
    no_args_is_help=True,
)

#: Prompt sentences seeded per intent (the inventory carries several worked examples each).
DEFAULT_PROMPTS_PER_INTENT = 8


@app.command("upgrade")
def upgrade(
    url: Annotated[str, typer.Option("--url", help="Database URL; AICLS_DB_URL by default.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Upgrade the database to the newest revision (``alembic upgrade head``)."""
    from aicls.db import current_revision, head_revision, make_engine, upgrade_head

    use_json(json_output)
    target = resolve_db_url(url)
    engine = make_engine(target)
    try:
        before = current_revision(engine)
        upgrade_head(engine)
        after = current_revision(engine)
    finally:
        engine.dispose()
    payload = {"url": target, "revision_before": before, "revision": after, "head": head_revision()}
    emit(payload, f"upgraded {before or 'base'} -> {after} ({target})")


@app.command("downgrade")
def downgrade(
    revision: Annotated[str, typer.Argument(help="Target revision, or 'base' to drop the schema.")],
    url: Annotated[str, typer.Option("--url", help="Database URL; AICLS_DB_URL by default.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Downgrade the database to ``revision``."""
    from aicls.db import current_revision, make_engine
    from aicls.db import downgrade as db_downgrade

    use_json(json_output)
    target = resolve_db_url(url)
    engine = make_engine(target)
    try:
        before = current_revision(engine)
        db_downgrade(engine, revision)
        after = current_revision(engine)
    finally:
        engine.dispose()
    payload = {"url": target, "revision_before": before, "revision": after, "target": revision}
    emit(payload, f"downgraded {before or 'base'} -> {after or 'base'} ({target})")


@app.command("revision")
def revision(
    message: Annotated[str, typer.Option("-m", "--message", help="Revision message.")],
    url: Annotated[str, typer.Option("--url", help="Database URL; AICLS_DB_URL by default.")] = "",
    autogenerate: Annotated[
        bool,
        typer.Option(
            "--autogenerate/--no-autogenerate",
            help="Diff the ORM metadata against the database to draft the migration.",
        ),
    ] = True,
    version_path: Annotated[
        Path | None,
        typer.Option("--version-path", help="Write the script here instead of the packaged dir."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Create a new migration script.

    ``--autogenerate`` needs a database already at ``head``: Alembic compares the live schema with
    the ORM metadata. A generated script is always reviewed by hand before it is committed.
    """
    from contextlib import redirect_stdout

    from alembic import command as alembic_command

    from aicls.db.migrate import alembic_config

    use_json(json_output)
    target = resolve_db_url(url)
    config = alembic_config(target)
    # Alembic narrates on stdout ("Generating ..."); stdout belongs to this command's output.
    config.stdout = sys.stderr
    if version_path is not None:
        directory = Path(version_path)
        directory.mkdir(parents=True, exist_ok=True)
        # Alembic refuses a version_path that is not a configured version location.
        config.set_main_option("version_locations", str(directory))
    with redirect_stdout(sys.stderr):
        scripts = alembic_command.revision(
            config,
            message=message,
            autogenerate=bool(autogenerate),
            version_path=str(version_path) if version_path is not None else None,
        )
    written = scripts if isinstance(scripts, list) else [scripts]
    paths = [str(getattr(script, "path", "")) for script in written if script is not None]
    payload = {"message": message, "autogenerate": bool(autogenerate), "files": paths}
    emit(payload, *(f"wrote {path}" for path in paths))


@app.command("seed")
def seed(
    url: Annotated[str, typer.Option("--url", help="Database URL; AICLS_DB_URL by default.")] = "",
    prompts_per_intent: Annotated[
        int,
        typer.Option("--prompts-per-intent", help="Prompt sentences taken per inventory intent."),
    ] = DEFAULT_PROMPTS_PER_INTENT,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Insert the reference data (consent versions, prompt sentences); safe to repeat."""
    use_json(json_output)
    target = resolve_db_url(url)
    with db_session(target) as session:
        report = seed_reference_data(session, prompts_per_intent=prompts_per_intent)
    payload = {"url": target, **report}
    emit(
        payload,
        f"consent versions: {payload['consent_versions']} stored ({payload['consent_added']} new)",
        f"prompt sentences: {payload['prompts_added']} added, {payload['prompts_total']} active",
    )


def seed_reference_data(session: Session, *, prompts_per_intent: int) -> dict[str, Any]:
    """Store consent versions and prompt sentences; return what changed.

    Args:
        session: An open session on a migrated database.
        prompts_per_intent: How many worked examples per intent become prompt sentences.

    Returns:
        Counts of stored and newly added rows.
    """
    from sqlalchemy import func, select

    from aicls.collect import consent_texts
    from aicls.core.settings import get_settings
    from aicls.db import RecordingRepository
    from aicls.db.models import ConsentVersion, PromptSentence

    settings = get_settings()
    repo = RecordingRepository(session)
    before = int(session.scalar(select(func.count()).select_from(ConsentVersion)) or 0)
    texts = consent_texts()
    for text in texts:
        repo.upsert_consent_version(
            text.version,
            text.text_tr,
            sha256=text.sha256,
            effective_from=text.effective_from,
            ethics_approval_ref=settings.ethics_approval_ref,
        )
    after = int(session.scalar(select(func.count()).select_from(ConsentVersion)) or 0)
    added = repo.add_prompts(prompt_rows(prompts_per_intent))
    total = int(session.scalar(select(func.count()).select_from(PromptSentence)) or 0)
    cli_logger(__name__).info(
        "cli.db.seeded", consent=after, prompts_added=added, prompts_total=total
    )
    return {
        "consent_versions": after,
        "consent_added": after - before,
        "prompts_added": added,
        "prompts_total": total,
    }


def prompt_rows(per_intent: int) -> list[dict[str, Any]]:
    """Build prompt-sentence rows from the packaged command inventory.

    The reading prompts of the voice-contribution tab are the command utterances the platform
    must understand, so the inventory's worked examples are their natural source: no separate
    sentence list can drift away from the intents the NLU layer actually supports.
    """
    from aicls.nlu import load_inventory

    inventory = load_inventory()
    rows: list[dict[str, Any]] = []
    for intent in inventory.intents:
        for example in intent.examples[: max(int(per_intent), 0)]:
            rows.append(
                {
                    "text_tr": example.text,
                    "scenario_tag": intent.kind,
                    "intent": intent.id,
                    "difficulty": 1 + min(len(example.slots), 2),
                    "is_active": True,
                }
            )
    return rows


__all__ = [
    "DEFAULT_PROMPTS_PER_INTENT",
    "app",
    "downgrade",
    "prompt_rows",
    "revision",
    "seed",
    "seed_reference_data",
    "upgrade",
]
