"""Static layering contracts (ADR-0002), asserted inside the normal pytest run.

Why this file exists
--------------------
``lint-imports`` already enforces the contracts declared in ``pyproject.toml``, but it is a
separate command that only the integration gate runs. Wave 2 shipped a real violation that got
all the way to the gate: :mod:`aicls.collect` legitimately consumes ``aicls.asr.audio_io`` and
``aicls.asr.qc``, both of which used to import ``aicls.asr.interfaces``, which reaches ``torch``
through a lazy in-function import. The module owner *did* test that importing
:mod:`aicls.collect` leaves ``torch`` out of ``sys.modules`` — and that test passed, because the
offending import never executes. Runtime absence is not the contract; the contract is that no
**static** import path exists at all, so that nothing can ever accidentally start executing it.

These tests therefore read the contracts straight out of ``pyproject.toml`` and re-check them
with ``grimp`` (import-linter's own graph builder, ~20 ms), so a wave-3 agent sees the failure in
their own ``pytest`` run instead of at the gate.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import grimp
import pytest

pytestmark = pytest.mark.integration

#: Contracts re-checked here, by their ``name`` prefix in ``pyproject.toml``.
_FORBIDDEN_CONTRACT_PREFIXES = (
    "no-torch-in-core",
    "api-no-training",
    "nlu-no-db",
    "eval-no-ml",
)


def _contracts(repo_root: Path) -> list[dict[str, object]]:
    """Return the ``[[tool.importlinter.contracts]]`` tables from ``pyproject.toml``."""
    with (repo_root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    contracts = config["tool"]["importlinter"]["contracts"]
    assert isinstance(contracts, list)
    return contracts


@pytest.fixture(scope="module")
def graph() -> grimp.ImportGraph:
    """The static import graph of ``aicls``, including external packages."""
    return grimp.build_graph("aicls", include_external_packages=True)


def _forbidden_cases(repo_root: Path) -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    for contract in _contracts(repo_root):
        name = str(contract.get("name", ""))
        if contract.get("type") != "forbidden":
            continue
        if not name.startswith(_FORBIDDEN_CONTRACT_PREFIXES):
            continue
        sources = contract["source_modules"]
        forbidden = contract["forbidden_modules"]
        assert isinstance(sources, list)
        assert isinstance(forbidden, list)
        for source in sources:
            for target in forbidden:
                cases.append((name, str(source), str(target)))
    return cases


def test_forbidden_contracts_are_declared(repo_root: Path) -> None:
    """Guard the guard: the contracts this file re-checks must still exist."""
    names = {str(c.get("name", "")) for c in _contracts(repo_root)}
    for prefix in _FORBIDDEN_CONTRACT_PREFIXES:
        assert any(name.startswith(prefix) for name in names), (
            f"contract {prefix!r} disappeared from pyproject.toml"
        )


def test_no_static_import_path_to_forbidden_modules(
    repo_root: Path, graph: grimp.ImportGraph
) -> None:
    """Every ``type = "forbidden"`` contract holds, including transitive and lazy imports.

    A failure prints the shortest offending chain, which is what you need to fix it: either the
    dependent side should import from a purer module, or the dependency should split its
    heavy-stack helpers away from its value types (as ``aicls.asr.types`` does).
    """
    violations: list[str] = []
    for name, source, target in _forbidden_cases(repo_root):
        if target not in graph.modules:
            # The forbidden package is not installed here; nothing can import it.
            continue
        if not graph.chain_exists(importer=source, imported=target, as_packages=True):
            continue
        chain = None
        for module in (source, *sorted(graph.find_descendants(source))):
            chain = graph.find_shortest_chain(module, target)
            if chain is not None:
                break
        violations.append(f"{name}: {source} -> {target} via {' -> '.join(chain or (source,))}")
    assert not violations, "forbidden import contracts broken:\n" + "\n".join(violations)


def test_asr_value_types_stay_torch_free(graph: grimp.ImportGraph) -> None:
    """``aicls.asr.types`` is the module deterministic packages are allowed to depend on.

    ``aicls.asr.interfaces`` may reach ``torch`` (it owns ``get_transcriber`` and
    ``resolve_device``); ``aicls.asr.types`` may not, and neither may the pure helpers that
    ``aicls.collect`` calls.
    """
    for module in (
        "aicls.asr.types",
        "aicls.asr.audio_io",
        "aicls.asr.features",
        "aicls.asr.qc",
        "aicls.asr.augment",
        "aicls.asr.synth",
    ):
        for target in ("torch", "transformers"):
            chain = graph.find_shortest_chain(module, target)
            assert chain is None, f"{module} statically reaches {target}: {' -> '.join(chain)}"
