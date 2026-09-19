"""Deterministic, group-disjoint, stratified splits (ADR-0008, docs/ml/veri-yonetimi.md §3).

Two rules make a reported number defensible:

* **A group never crosses a split.** Every utterance of a speaker (and every sample of a
  MathWriting writer) lives in exactly one split, so a test score cannot be inflated by having
  heard the same voice during training.
* **The test set does not drift.** When new speakers arrive, :func:`make_splits` is called with
  ``frozen=<previous splits>``: every group that already had a split keeps it, and only the new
  groups are distributed. Comparing the 2027-05 and 2028-05 reports therefore compares the same
  test material plus additions, never a reshuffled one.

Assignment is a deterministic largest-deficit greedy over groups, run per stratum: groups are
sorted by size (ties broken by a seeded shuffle of the *sorted* group list, so the result does not
depend on the order the ids arrive in) and each group goes to the split whose remaining item
deficit inside its own stratum is largest. With enough groups per stratum this reproduces the
requested ratios closely and keeps every stratum's share inside a few percent in every split
(``test_speaker_disjoint_split_is_stratified_within_5pct``).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aicls.core.hashing import FINGERPRINT_PREFIX, sha256_json
from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path

_log = get_logger(__name__)

#: The three splits every dataset has.
CORE_SPLITS: tuple[str, str, str] = ("train", "valid", "test")

#: MathWriting's official split names (the two extra ones are not part of train/valid/test).
MATHWRITING_SPLITS: tuple[str, ...] = ("train", "valid", "test", "symbols", "synthetic")

#: Tie-break order when two splits have the same deficit: the test set is protected first.
_TIE_BREAK: tuple[str, str, str] = ("test", "valid", "train")

#: File extensions accepted as an official split listing.
_LIST_SUFFIXES = (".txt", ".lst", ".csv")

#: Default split file name for the audio-command corpus.
SPLITS_FILENAME = "audio_commands_splits.json"

#: Split-name aliases accepted from other modules and configs. ``aicls.collect`` names its middle
#: split ``dev`` while the training/eval side (and the MathWriting corpus) says ``valid``; both
#: mean the same thing and are normalised here so a collect export loads without a rename
#: (``tests/integration/test_data_contracts.py``).
SPLIT_ALIASES: dict[str, str] = {"dev": "valid", "val": "valid", "validation": "valid"}


def normalize_split_name(name: str) -> str:
    """Map a split alias (``dev``) onto its canonical name (``valid``)."""
    key = str(name).strip().lower()
    return SPLIT_ALIASES.get(key, key)


Stratum = tuple[Hashable, ...]


class Splits(BaseModel):
    """Named lists of sample ids.

    Attributes:
        train: Training ids (sorted).
        valid: Validation ids (sorted).
        test: Test ids (sorted).
        extra: Additional named lists (MathWriting's ``symbols`` and ``synthetic``).
        seed: Seed used by :func:`make_splits`; ``None`` for official splits.
        ratios: Requested ``(train, valid, test)`` ratios, when applicable.
        source: ``"make_splits"``, ``"official"`` or a free-form provenance label.
    """

    model_config = ConfigDict(extra="forbid")

    train: list[str] = Field(default_factory=list)
    valid: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    extra: dict[str, list[str]] = Field(default_factory=dict)
    seed: int | None = None
    ratios: tuple[float, float, float] | None = None
    source: str = "make_splits"
    group_key: str | None = None
    stratify: str | None = None

    def names(self) -> tuple[str, ...]:
        """All split names, core ones first."""
        return (*CORE_SPLITS, *self.extra)

    def __getitem__(self, name: str) -> list[str]:
        """Ids of split ``name``.

        Raises:
            InputValidationError: If ``name`` is not a known split.
        """
        if name in CORE_SPLITS:
            value: list[str] = getattr(self, name)
            return value
        if name in self.extra:
            return self.extra[name]
        raise invalid_data("unknown_split", split=name, available=list(self.names()))

    def get(self, name: str, default: list[str] | None = None) -> list[str] | None:
        """Ids of split ``name``, or ``default`` when it does not exist."""
        if name in CORE_SPLITS:
            value: list[str] = getattr(self, name)
            return value
        return self.extra.get(name, default)

    def sizes(self) -> dict[str, int]:
        """Number of ids per split."""
        return {name: len(self[name]) for name in self.names()}

    def all_ids(self) -> list[str]:
        """Every id in every split."""
        return [sample_id for name in self.names() for sample_id in self[name]]

    def split_of(self, sample_id: str) -> str | None:
        """Split holding ``sample_id``, or ``None`` when it is in none of them."""
        for name in self.names():
            if sample_id in set(self[name]):
                return name
        return None

    def index(self) -> dict[str, str]:
        """``sample id -> split name`` mapping (built once by the caller, then reused)."""
        return {sample_id: name for name in self.names() for sample_id in self[name]}

    @property
    def fingerprint(self) -> str:
        """``"sha256:<hex>"`` over the split contents — cited in every ASR/OCR report."""
        return FINGERPRINT_PREFIX + sha256_json({name: self[name] for name in self.names()})

    def to_json(self) -> str:
        """Serialise to pretty JSON (LF, UTF-8, trailing newline) with the fingerprint."""
        payload: dict[str, Any] = json.loads(self.model_dump_json())
        payload["fingerprint"] = self.fingerprint
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def make_splits(
    ids: Sequence[str],
    *,
    group_key: Callable[[str], str],
    stratify: Callable[[str], Stratum] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 1234,
    frozen: Splits | Mapping[str, Sequence[str]] | Path | None = None,
) -> Splits:
    """Split ``ids`` into train/valid/test without ever splitting a group.

    Args:
        ids: Sample ids (unique).
        group_key: Maps an id to its group — the speaker pseudonym or the writer id. The whole
            group travels together.
        stratify: Maps an id to a stratum tuple (``(university, gender)``); every id of a group
            must give the same stratum. ``None`` puts everything in one stratum.
        ratios: Requested ``(train, valid, test)`` item shares; must be positive and sum to 1.
        seed: Shuffle seed (repo-wide default 1234).
        frozen: Previous splits (object, mapping or path to a split file). Groups that already
            have a split keep it; only new groups are distributed.

    Returns:
        The splits, ids sorted inside each list.

    Raises:
        InputValidationError: On empty/duplicate ids, invalid ratios or a group whose ids
            disagree about their stratum.
        NotFoundError: If ``frozen`` is a path that does not exist.
    """
    sample_ids = list(ids)
    if not sample_ids:
        raise invalid_data("empty_ids")
    if len(set(sample_ids)) != len(sample_ids):
        duplicates = sorted({i for i in sample_ids if sample_ids.count(i) > 1})[:5]
        raise invalid_data("duplicate_ids", duplicates=duplicates)
    _check_ratios(ratios)

    groups, strata = _group(sample_ids, group_key, stratify)
    pinned = _pinned_groups(frozen, groups)

    targets = _targets(groups, strata, ratios, pinned)
    assignment: dict[str, str] = dict(pinned)
    rng = np.random.default_rng(seed)

    for stratum in sorted(set(strata.values()), key=repr):
        pending = sorted(
            (name for name, value in strata.items() if value == stratum and name not in pinned),
        )
        order = list(rng.permutation(len(pending)))
        shuffled = [pending[int(i)] for i in order]
        shuffled.sort(key=lambda name: -len(groups[name]))
        for name in shuffled:
            chosen = _best_split(targets, stratum)
            assignment[name] = chosen
            targets[chosen][stratum] -= len(groups[name])

    buckets: dict[str, list[str]] = {name: [] for name in CORE_SPLITS}
    for group_name, split_name in assignment.items():
        buckets[split_name].extend(groups.get(group_name, []))

    splits = Splits(
        train=sorted(buckets["train"]),
        valid=sorted(buckets["valid"]),
        test=sorted(buckets["test"]),
        seed=seed,
        ratios=ratios,
        source="make_splits",
        group_key=getattr(group_key, "__name__", None),
        stratify=getattr(stratify, "__name__", None) if stratify is not None else None,
    )
    _log.info(
        "data.splits.made",
        n_ids=len(sample_ids),
        n_groups=len(groups),
        sizes=splits.sizes(),
        frozen_groups=len(pinned),
        fingerprint=splits.fingerprint,
    )
    return splits


def mathwriting_official(root: Path) -> Splits:
    """Read MathWriting's official split assignment from a corpus root.

    Two layouts are accepted, in this order:

    * one directory per split (``<root>/train/*.inkml``) — the layout of the released archive;
    * one listing file per split (``<root>/train.txt``), one file name or id per line.

    Args:
        root: The extracted corpus root.

    Returns:
        Splits whose ids are InkML file stems; ``symbols`` and ``synthetic`` land in ``extra``.

    Raises:
        NotFoundError: If ``root`` does not exist or holds none of the five splits.
    """
    root = Path(root)
    if not root.is_dir():
        raise missing_path(root, "mathwriting_root_missing")

    found: dict[str, list[str]] = {}
    for name in MATHWRITING_SPLITS:
        folder = root / name
        if folder.is_dir():
            found[name] = sorted(path.stem for path in folder.rglob("*.inkml"))
            continue
        for suffix in _LIST_SUFFIXES:
            listing = root / f"{name}{suffix}"
            if listing.is_file():
                found[name] = [
                    Path(line.strip()).stem
                    for line in listing.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break
    if not any(found.values()):
        raise missing_path(root, "mathwriting_splits_missing", expected=list(MATHWRITING_SPLITS))

    splits = Splits(
        train=found.get("train", []),
        valid=found.get("valid", []),
        test=found.get("test", []),
        extra={name: found[name] for name in ("symbols", "synthetic") if name in found},
        seed=None,
        source="official",
        group_key="writer_id",
    )
    _log.info("data.splits.official", root=str(root), sizes=splits.sizes())
    return splits


def save_splits(splits: Splits, path: Path) -> Path:
    """Write ``splits`` as JSON (creating parent directories) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(splits.to_json())
    return path


def load_splits(path: Path) -> Splits:
    """Read a split file written by :func:`save_splits`.

    Raises:
        NotFoundError: If the file does not exist.
        InputValidationError: If it is not valid JSON or not a valid split document.
    """
    path = Path(path)
    if not path.is_file():
        raise missing_path(path, "splits_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise invalid_data("splits_not_json", path=str(path), error=str(exc)) from exc
    if isinstance(payload, dict):
        payload.pop("fingerprint", None)
        if "speakers" in payload:
            return _splits_from_speaker_map(payload)
    try:
        return Splits.model_validate(payload)
    except ValueError as exc:
        raise invalid_data("splits_invalid", path=str(path), error=str(exc)) from exc


# ------------------------------------------------------------------------------------ internals
def _splits_from_speaker_map(payload: Mapping[str, Any]) -> Splits:
    """Read ``aicls.collect``'s ``{"speakers": {"P00001": "dev", ...}}`` split file.

    The ids of the returned :class:`Splits` are **speaker pseudonyms**, not sample ids;
    :class:`~aicls.data.audio_commands.AudioCommandDataset` looks a row up by sample id first and
    by speaker second, so both files work.
    """
    speakers = payload.get("speakers") or {}
    if not isinstance(speakers, Mapping):
        raise invalid_data("speakers_map_invalid")
    buckets: dict[str, list[str]] = {name: [] for name in CORE_SPLITS}
    for pseudonym, split_name in speakers.items():
        canonical = normalize_split_name(str(split_name))
        if canonical not in buckets:
            raise invalid_data("unknown_split", split=str(split_name), available=list(CORE_SPLITS))
        buckets[canonical].append(str(pseudonym))
    ratios = payload.get("ratios") or {}
    shares = (
        (
            float(ratios.get("train", 0.8)),
            float(ratios.get("valid", ratios.get("dev", 0.1))),
            float(ratios.get("test", 0.1)),
        )
        if isinstance(ratios, Mapping)
        else None
    )
    return Splits(
        train=sorted(buckets["train"]),
        valid=sorted(buckets["valid"]),
        test=sorted(buckets["test"]),
        seed=int(payload.get("seed", 1234)),
        ratios=shares,
        source="collect.speakers",
        group_key="speaker_pseudonym",
    )


def _check_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise invalid_data("ratios_arity", ratios=list(ratios))
    if any(value <= 0.0 for value in ratios):
        raise invalid_data("ratios_not_positive", ratios=list(ratios))
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise invalid_data("ratios_sum", ratios=list(ratios), total=float(sum(ratios)))


def _group(
    sample_ids: Sequence[str],
    group_key: Callable[[str], str],
    stratify: Callable[[str], Stratum] | None,
) -> tuple[dict[str, list[str]], dict[str, Stratum]]:
    """Bucket ids by group and resolve one stratum per group."""
    groups: dict[str, list[str]] = {}
    strata: dict[str, Stratum] = {}
    for sample_id in sample_ids:
        name = str(group_key(sample_id))
        groups.setdefault(name, []).append(sample_id)
        stratum: Stratum = tuple(stratify(sample_id)) if stratify is not None else ()
        previous = strata.setdefault(name, stratum)
        if previous != stratum:
            raise invalid_data(
                "inconsistent_stratum",
                group=name,
                first=list(previous),
                second=list(stratum),
            )
    return groups, strata


def _pinned_groups(
    frozen: Splits | Mapping[str, Sequence[str]] | Path | None,
    groups: Mapping[str, list[str]],
) -> dict[str, str]:
    """Groups that keep a previously assigned split (test wins any conflict)."""
    if frozen is None:
        return {}
    if isinstance(frozen, str | Path):
        frozen = load_splits(Path(frozen))
    previous: dict[str, list[str]]
    if isinstance(frozen, Splits):
        previous = {name: frozen[name] for name in CORE_SPLITS}
    else:
        previous = {name: list(frozen.get(name, [])) for name in CORE_SPLITS}

    owner: dict[str, str] = {}
    for sample_id, group in (
        (sample_id, group) for group, members in groups.items() for sample_id in members
    ):
        owner[sample_id] = group

    pinned: dict[str, str] = {}
    for split_name in _TIE_BREAK:  # test first, so a conflicting group stays in test
        for sample_id in previous.get(split_name, []):
            owning_group = owner.get(sample_id)
            if owning_group is not None:
                pinned.setdefault(owning_group, split_name)
    return pinned


def _targets(
    groups: Mapping[str, list[str]],
    strata: Mapping[str, Stratum],
    ratios: tuple[float, float, float],
    pinned: Mapping[str, str],
) -> dict[str, dict[Stratum, float]]:
    """Remaining item budget per split and stratum, after subtracting the pinned groups."""
    counts: dict[Stratum, int] = {}
    for name, members in groups.items():
        counts[strata[name]] = counts.get(strata[name], 0) + len(members)
    # sort the strata so that float summation order (used as a tie-break) never depends on the
    # order the ids arrived in
    totals = {stratum: counts[stratum] for stratum in sorted(counts, key=repr)}
    targets = {
        split_name: {stratum: ratio * float(total) for stratum, total in totals.items()}
        for split_name, ratio in zip(CORE_SPLITS, ratios, strict=True)
    }
    for group_name, split_name in pinned.items():
        targets[split_name][strata[group_name]] -= len(groups[group_name])
    return targets


def _best_split(targets: Mapping[str, Mapping[Stratum, float]], stratum: Stratum) -> str:
    """Split with the largest remaining deficit for this stratum (ties: global, then test)."""
    return max(
        CORE_SPLITS,
        key=lambda name: (
            targets[name][stratum],
            sum(targets[name].values()),
            -_TIE_BREAK.index(name),
        ),
    )


__all__ = [
    "CORE_SPLITS",
    "MATHWRITING_SPLITS",
    "SPLITS_FILENAME",
    "SPLIT_ALIASES",
    "Splits",
    "load_splits",
    "make_splits",
    "mathwriting_official",
    "normalize_split_name",
    "save_splits",
]
