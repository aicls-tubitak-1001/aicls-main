"""The public contract of ``aicls.db`` (modules/db.json ``public_api``).

Other packages are written against these names in parallel, so a rename must fail here first.
"""

from __future__ import annotations

import inspect

import pytest

from aicls import db

#: Every name modules/db.json promises, with the callables' required parameters.
CONTRACT: dict[str, tuple[str, ...]] = {
    "Base": (),
    "TimestampMixin": (),
    "make_engine": ("url",),
    "session_scope": (),
    "get_session": (),
    "upgrade_head": ("url",),
    "UserRepository": ("get_by_email", "create", "set_role"),
    "WorkspaceRepository": ("list_for_user", "create", "get_owned"),
    "OperationRepository": ("add", "page", "delete"),
    "RecognitionRepository": ("add", "set_correction"),
    "RecordingRepository": ("add", "next_prompts", "stats", "mark_withdrawn"),
    "BookRepository": ("upsert_units", "get", "search_alias", "search_text", "related"),
    "ModelVersionRepository": ("register", "active", "activate"),
    "EvalRunRepository": ("add", "list"),
    "LatencyRepository": ("add", "stats"),
    "FeedbackRepository": ("add_feedback", "add_sus", "page"),
    "AuditRepository": ("log",),
}


@pytest.mark.parametrize("name", sorted(CONTRACT))
def test_public_api_exports_the_contract(name: str) -> None:
    """Every promised symbol is importable from ``aicls.db`` and listed in ``__all__``."""
    assert hasattr(db, name), name
    assert name in db.__all__, name


@pytest.mark.parametrize(
    ("name", "members"), [(k, v) for k, v in sorted(CONTRACT.items()) if k.endswith("Repository")]
)
def test_repositories_expose_their_methods_and_take_a_session(
    name: str, members: tuple[str, ...]
) -> None:
    """Repositories are constructed with a session and expose their contract methods."""
    repository = getattr(db, name)
    assert list(inspect.signature(repository.__init__).parameters)[:2] == ["self", "session"]
    for member in members:
        assert callable(getattr(repository, member)), f"{name}.{member}"


def test_engine_and_migration_helpers_have_the_promised_signatures() -> None:
    """``make_engine(url, *, echo=False)``, ``session_scope(engine)``, ``upgrade_head(url)``."""
    make_engine = inspect.signature(db.make_engine)
    assert list(make_engine.parameters) == ["url", "echo"]
    assert make_engine.parameters["echo"].default is False
    assert list(inspect.signature(db.session_scope).parameters) == ["engine"]
    assert list(inspect.signature(db.get_session).parameters) == []
    assert list(inspect.signature(db.upgrade_head).parameters) == ["url"]


def test_all_is_unique_and_fully_resolvable() -> None:
    """``__all__`` is the documented surface: unique, resolvable, private-free."""
    assert len(db.__all__) == len(set(db.__all__))
    for name in db.__all__:
        assert hasattr(db, name), name
        assert not name.startswith("_"), name


def test_db_only_imports_core_and_never_torch() -> None:
    """Layering rule (ADR-0002): ``aicls.db`` imports only ``aicls.core`` and third parties."""
    import ast
    from pathlib import Path

    package = Path(db.__file__).parent
    forbidden = ("torch", "transformers", "torchvision", "peft", "accelerate")
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in forbidden, f"{path.name}: {name}"
                if name.startswith("aicls."):
                    assert name.startswith(("aicls.core", "aicls.db")), f"{path.name}: {name}"
