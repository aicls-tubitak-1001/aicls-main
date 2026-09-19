"""``aicls data``: manifests, verification, splits and shard preparation.

The verification test is the one that matters for reporting: a fingerprint quoted in a progress
report is only worth something if a later ``aicls data verify`` still recomputes it, so the test
tampers with a byte and asserts the command fails.

Nothing here downloads anything (ADR-0012); the InkML corpus is generated on the spot by
:func:`aicls.data.synth_inkml_dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny data root with two files."""
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\n", encoding="utf-8", newline="\n")
    (root / "sub" / "b.txt").write_text("beta\n", encoding="utf-8", newline="\n")
    return root


def _build_manifest(runner: CliRunner, app_module: object, corpus: Path) -> dict[str, object]:
    result = runner.invoke(
        app_module,
        [
            "data",
            "manifest",
            str(corpus),
            "--name",
            "smoke",
            "--license",
            "internal",
            "--source",
            "repo://",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    return payload_of(result.stdout)


def test_data_manifest_then_verify_round_trip(
    runner: CliRunner, app_module: object, corpus: Path
) -> None:
    """A freshly built manifest verifies clean and reports the fingerprint reports cite."""
    built = _build_manifest(runner, app_module, corpus)
    assert built["n_files"] == 2
    assert built["fingerprint"].startswith("sha256:")

    result = runner.invoke(app_module, ["data", "verify", "smoke", "--root", str(corpus), "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["ok"] is True
    assert payload["fingerprint"] == built["fingerprint"]
    assert payload["missing"] == [] and payload["changed"] == []


def test_data_verify_fails_on_a_changed_byte(
    runner: CliRunner, app_module: object, corpus: Path
) -> None:
    """A data root that drifted from its manifest must not pass silently."""
    _build_manifest(runner, app_module, corpus)
    (corpus / "a.txt").write_text("alpha!\n", encoding="utf-8", newline="\n")

    result = runner.invoke(app_module, ["data", "verify", "smoke", "--root", str(corpus), "--json"])
    assert result.exit_code == 1
    payload = payload_of(result.stdout)
    assert payload["ok"] is False
    assert payload["changed"] == ["a.txt"]
    assert payload["fingerprint_matches"] is False


def test_data_verify_reports_the_catalog_gate(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A catalogued dataset carries the approval gate its download sits behind (ADR-0012)."""
    root = tmp_path / "mathwriting"
    root.mkdir()
    (root / "note.txt").write_text("placeholder\n", encoding="utf-8", newline="\n")
    _build_manifest_for(runner, app_module, root, "mathwriting")
    result = runner.invoke(
        app_module, ["data", "verify", "mathwriting", "--root", str(root), "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    catalog = payload_of(result.stdout)["catalog"]
    assert catalog is not None
    assert catalog["gate"] == "DATASET_DOWNLOAD"
    # MathWriting is CC BY-NC-SA 4.0 (verified 2026-09-08 against the official README), not
    # Apache-2.0; the CLI must show the licence the corpus actually carries.
    assert catalog["license"] == "CC-BY-NC-SA-4.0"


def _build_manifest_for(runner: CliRunner, app_module: object, root: Path, name: str) -> None:
    result = runner.invoke(
        app_module,
        ["data", "manifest", str(root), "--name", name, "--license", "CC-BY-NC-SA-4.0", "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr


def test_data_split_is_group_disjoint_and_frozen(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """No speaker is ever split across train/valid/test, and a second run keeps the assignment."""
    manifest = tmp_path / "audio.jsonl"
    rows = [
        {
            "id": f"utt-{index:03d}",
            "speaker": f"P{index % 12:05d}",
            "university": ["mus", "firat"][index % 2],
            "gender": ["f", "m"][index % 2],
        }
        for index in range(120)
    ]
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    args = [
        "data",
        "split",
        str(manifest),
        "--group",
        "speaker",
        "--stratify",
        "university,gender",
        "--json",
    ]
    result = runner.invoke(app_module, args)
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["rows"] == 120
    assert payload["groups"] == 12
    assert sum(payload["sizes"].values()) == 120
    assert payload["fingerprint"].startswith("sha256:")

    from aicls.data import load_splits

    splits = load_splits(Path(payload["path"]))
    speaker_of = {row["id"]: row["speaker"] for row in rows}
    per_split = {
        name: {speaker_of[sample] for sample in splits[name]} for name in ("train", "valid", "test")
    }
    assert not per_split["train"] & per_split["test"]
    assert not per_split["train"] & per_split["valid"]
    assert not per_split["valid"] & per_split["test"]

    again = runner.invoke(app_module, args)
    assert again.exit_code == 0
    assert payload_of(again.stdout)["fingerprint"] == payload["fingerprint"]


def test_data_split_refuses_a_row_without_the_group_field(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A dropped row would change a reported corpus size, so a missing field is an error."""
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"id": "a"}\n', encoding="utf-8", newline="\n")
    result = runner.invoke(app_module, ["data", "split", str(manifest)])
    assert result.exit_code == 1
    assert "missing_group_field" in result.stderr


def test_data_prepare_mathwriting_writes_shards(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A generated InkML corpus becomes memory-mappable shards plus the official split file."""
    from aicls.data import synth_inkml_dataset

    root = tmp_path / "mathwriting"
    synth_inkml_dataset(root / "train", 6, seed=1234)
    out = tmp_path / "shards"
    result = runner.invoke(
        app_module,
        [
            "data",
            "prepare-mathwriting",
            str(root),
            "--out",
            str(out),
            "--split",
            "train",
            "--shard-size",
            "4",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["splits"][0]["split"] == "train"
    assert payload["splits"][0]["samples"] == 6
    assert payload["splits"][0]["shards"] == 2
    assert payload["official_splits"] is not None
    assert (out / "train").is_dir()
