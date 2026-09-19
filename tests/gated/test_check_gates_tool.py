"""``tools/check_gates.py``: the static half of the approval gates (ADR-0012).

``require_gate`` stops a *deliberate* download. This tool stops an *accidental* one: a hub id in
a ``from_pretrained`` call, a torchvision ``weights=`` enum, a stray ``requests.get`` — code that
would fetch bytes the moment it runs, with nobody having approved anything. It is a grep, so it
is only as good as its rule set, and these tests pin the rules that matter.

The scanned trees are ``src``, ``training``, ``experiments``, ``scripts`` and ``tools``; the only
files allowed to carry download code are ``training/scripts/download_*.py`` and
``scripts/install_ml.*``. Nothing here touches the network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import types

import pytest

pytestmark = pytest.mark.gated

REPO_ROOT = Path(__file__).resolve().parents[2]

Loader = Callable[[str], types.ModuleType]

#: (source line, expected rule id) — each must be flagged outside a gated file.
FORBIDDEN_LINES: tuple[tuple[str, str], ...] = (
    ('model = Whisper.from_pretrained("openai/whisper-small")', "hub-id"),
    ('ds = load_dataset("google/mathwriting")', "hub-id"),
    ('path = hf_hub_download(repo_id=name, filename="config.json")', "hf-download"),
    ("snapshot_download(repo_id=name, local_dir=target)", "hf-download"),
    ('net = torch.hub.load("pytorch/vision", "resnet34")', "torch-hub"),
    ("encoder = resnet34(weights=ResNet34_Weights.DEFAULT)", "torchvision-weights"),
    ("encoder = resnet34(pretrained=True)", "torchvision-weights"),
    ('payload = requests.get("https://example.org/x").content', "url-fetch"),
    ("with urllib.request.urlopen(url) as fh:", "url-fetch"),
    (
        'URL = "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors"',
        "url-literal",
    ),
)


@pytest.fixture
def tool(load_script: Loader) -> types.ModuleType:
    """The imported ``tools/check_gates.py`` module."""
    return load_script("tools/check_gates.py")


def test_check_gates_tool_flags_hub_id_usage(tool: types.ModuleType, tmp_path: Path) -> None:
    """A hub id in ``from_pretrained`` is flagged and makes the tool exit 1.

    This is the pattern that would silently download half a gigabyte of weights on first import,
    which is exactly what ADR-0012 forbids: ``from_pretrained`` may only be given a local path
    under ``AICLS_MODELS_ROOT``.
    """
    offender = tmp_path / "sneaky_model.py"
    offender.write_text(
        "from transformers import WhisperForConditionalGeneration\n"
        "\n"
        "def load():\n"
        '    return WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")\n',
        encoding="utf-8",
    )

    hits = tool.scan_file(offender, root=REPO_ROOT)

    assert len(hits) == 1
    assert hits[0].rule == "hub-id"
    assert hits[0].line_no == 4
    assert "AICLS_MODELS_ROOT" in hits[0].message
    assert tool.main([str(offender)]) == 1


@pytest.mark.parametrize(("line", "rule"), FORBIDDEN_LINES)
def test_check_gates_tool_flags_every_implicit_download_pattern(
    tool: types.ModuleType, tmp_path: Path, line: str, rule: str
) -> None:
    """Each documented pattern is caught, with the rule id the ADR names."""
    hits = tool.scan_text(line + "\n", tmp_path / "candidate.py", suffix=".py")

    assert [hit.rule for hit in hits] == [rule], line


def test_check_gates_tool_allows_the_gated_download_scripts(tool: types.ModuleType) -> None:
    """The two download scripts are allow-listed; the same code elsewhere is not."""
    for name in ("download_mathwriting.py", "download_weights.py"):
        path = REPO_ROOT / "training" / "scripts" / name
        assert path.is_file(), name
        assert tool.scan_file(path, root=REPO_ROOT) == []

    # ... and the allow list is by path, not by content: a copy under another name is flagged.
    text = (REPO_ROOT / "training" / "scripts" / "download_weights.py").read_text(encoding="utf-8")
    hits = tool.scan_text(text, Path("training/scripts/fetch_weights.py"), suffix=".py")
    assert hits, "a renamed download script must not slip past the scanner"


def test_check_gates_tool_respects_the_reviewed_exception_marker(
    tool: types.ModuleType, tmp_path: Path
) -> None:
    """``# gate-ok: <reason>`` silences one reviewed line (the catalog url uses it)."""
    clean = 'URL = "https://huggingface.co/openai/whisper-small"  # gate-ok: descriptive only\n'

    assert tool.scan_text(clean, tmp_path / "catalog.py", suffix=".py") == []


def test_check_gates_tool_ignores_comments(tool: types.ModuleType, tmp_path: Path) -> None:
    """A pattern inside a comment is documentation, not a download."""
    commented = '# never write from_pretrained("openai/whisper-small") outside a gated script\n'

    assert tool.scan_text(commented, tmp_path / "notes.py", suffix=".py") == []


def test_check_gates_tool_passes_on_the_repository_tree(
    tool: types.ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The committed tree is clean — this is the check CI runs on every push."""
    hits = tool.scan_tree(REPO_ROOT)

    assert hits == [], "\n".join(hit.render(REPO_ROOT) for hit in hits)
    assert tool.main([]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_gates_tool_scans_the_training_tree(tool: types.ModuleType) -> None:
    """``training/`` is in the scanned set, so a config or script cannot hide a hub id."""
    assert "training" in tool.SCAN_DIRS
    assert ".yaml" in tool.SCAN_SUFFIXES
    assert tool.scan_tree(REPO_ROOT, dirs=("training",)) == []
