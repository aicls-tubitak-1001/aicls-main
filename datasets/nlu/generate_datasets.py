"""Build ``commands_test.jsonl`` and ``asr_noisy_tr.jsonl`` from their two sources.

Run it from the repository root with the project venv::

    python datasets/nlu/generate_datasets.py

Sources (both authored by the project, see the provenance note in ``datasets/nlu/README.md``):

* the worked examples of ``src/aicls/nlu/inventory/commands.yaml`` -- these carry **gold slot
  values** and are what ``slot_f1`` is measured on;
* ``datasets/nlu/reading_scenarios.yaml`` -- the İP-3 reading-scenario library; these carry a
  gold intent only.

The noisy set applies :mod:`tests.unit.nlu.perturb` (loaded by path, it is test-support code)
with seed 1234, so regenerating the files reproduces them byte for byte.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "nlu"
PERTURB_PATH = REPO_ROOT / "tests" / "unit" / "nlu" / "perturb.py"
SEED = 1234
VARIANTS_PER_ROW = 2


def _load_perturb() -> types.ModuleType:
    """Import ``tests/unit/nlu/perturb.py`` by path (the tests tree is not a package)."""
    spec = importlib.util.spec_from_file_location("aicls_nlu_perturb", PERTURB_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {PERTURB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inventory_rows() -> Iterator[dict[str, Any]]:
    """Rows from the packaged inventory examples (gold intent *and* gold slots)."""
    from aicls.nlu import load_inventory

    inventory = load_inventory()
    counters: dict[str, int] = {}
    for intent_id, example in inventory.examples():
        counters[intent_id] = counters.get(intent_id, 0) + 1
        row: dict[str, Any] = {
            "id": f"inv-{intent_id}-{counters[intent_id]:02d}",
            "text": example.text,
            "intent": intent_id,
            "slots": dict(example.slots),
            "latex_expr": example.slots.get("expr") or example.slots.get("equation") or "",
            "source": "authored",
            "tags": ["inventory", *example.tags],
        }
        if example.context:
            row["context"] = dict(example.context)
        yield row


def _scenario_rows() -> Iterator[dict[str, Any]]:
    """Rows from the reading-scenario library (gold intent only)."""
    import yaml

    data = yaml.safe_load((DATASET_DIR / "reading_scenarios.yaml").read_text(encoding="utf-8"))
    counters: dict[str, int] = {}
    for scenario in data["scenarios"]:
        intent = scenario.get("intent")
        if intent is None:
            # `intent: null` marks pure expression-reading prompts: they are ASR material, not
            # commands, so they never enter the gold command set.
            continue
        # `intent: none` marks the out-of-scope prompts; their gold label is "no command at all".
        gold = "" if intent == "none" else str(intent)
        tag = scenario["tag"]
        extra = ["out_of_scope"] if gold == "" else []
        for sentence in scenario["sentences"]:
            counters[tag] = counters.get(tag, 0) + 1
            yield {
                "id": f"rs-{tag}-{counters[tag]:02d}",
                "text": sentence["text"],
                "intent": gold,
                "slots": {},
                "latex_expr": "",
                "source": "authored",
                "tags": [
                    "reading_scenarios",
                    tag,
                    f"difficulty{sentence.get('difficulty', 1)}",
                    *extra,
                ],
            }


def build_commands_test() -> list[dict[str, Any]]:
    """Merge both sources, dropping later duplicates of the same utterance."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in (*_inventory_rows(), *_scenario_rows()):
        key = (row["text"].casefold(), json.dumps(row.get("context", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def build_noisy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate ASR-like variants of every clean row."""
    perturb = _load_perturb()
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        variants = perturb.perturb_variants(row["text"], seed=SEED + index * 17, n=VARIANTS_PER_ROW)
        for position, (text, applied) in enumerate(variants):
            noisy = {
                "id": f"{row['id']}-n{position + 1}",
                "text": text,
                "intent": row["intent"],
                "slots": {},
                "latex_expr": "",
                "source": "authored",
                "origin_id": row["id"],
                "perturbations": applied,
                "tags": ["asr_noisy", *row["tags"]],
            }
            if "context" in row:
                noisy["context"] = row["context"]
            out.append(noisy)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` as UTF-8 JSONL with LF endings."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Regenerate both datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    rows = build_commands_test()
    write_jsonl(args.out / "commands_test.jsonl", rows)
    noisy = build_noisy(rows)
    write_jsonl(args.out / "asr_noisy_tr.jsonl", noisy)
    print(f"commands_test.jsonl: {len(rows)} rows")
    print(f"asr_noisy_tr.jsonl:  {len(noisy)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
