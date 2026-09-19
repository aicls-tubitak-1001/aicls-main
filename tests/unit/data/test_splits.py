"""Deterministic, group-disjoint and stratified splits (ADR-0008, docs/ml/veri-yonetimi.md §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import Splits, make_splits, mathwriting_official
from aicls.data.splits import load_splits, save_splits

UNIVERSITIES = ("mus", "mersin", "amasya")
GENDERS = ("f", "m")
UTTERANCES_PER_SPEAKER = 10


def speaker_of(sample_id: str) -> str:
    """``P00007-004`` -> ``P00007``."""
    return sample_id.split("-", 1)[0]


def stratum_of(sample_id: str) -> tuple[str, str]:
    """Stratum key ``(university, gender)`` derived from the speaker number."""
    index = int(speaker_of(sample_id)[1:]) - 1
    return (UNIVERSITIES[index % len(UNIVERSITIES)], GENDERS[(index // len(UNIVERSITIES)) % 2])


def make_ids(n_speakers: int = 60, per_speaker: int = UTTERANCES_PER_SPEAKER) -> list[str]:
    """Utterance ids for ``n_speakers`` speakers, ``per_speaker`` recordings each."""
    return [f"P{s + 1:05d}-{k:03d}" for s in range(n_speakers) for k in range(per_speaker)]


def test_speaker_disjoint_split_has_no_overlap() -> None:
    ids = make_ids()
    splits = make_splits(ids, group_key=speaker_of, stratify=stratum_of)

    assert sorted(splits.all_ids()) == sorted(ids)
    train, valid, test = (
        {speaker_of(i) for i in splits.train},
        {speaker_of(i) for i in splits.valid},
        {speaker_of(i) for i in splits.test},
    )
    assert train & valid == set()
    assert train & test == set()
    assert valid & test == set()
    assert len(train | valid | test) == 60
    # every utterance of a speaker travels with the speaker
    for split_ids in (splits.train, splits.valid, splits.test):
        for sample_id in split_ids:
            assert speaker_of(sample_id) in {speaker_of(i) for i in split_ids}
    assert 0.75 <= len(splits.train) / len(ids) <= 0.85


def test_speaker_disjoint_split_is_stratified_within_5pct() -> None:
    ids = make_ids()
    splits = make_splits(ids, group_key=speaker_of, stratify=stratum_of)

    def shares(sample_ids: list[str]) -> dict[tuple[str, str], float]:
        total = len(sample_ids)
        counts: dict[tuple[str, str], int] = {}
        for sample_id in sample_ids:
            counts[stratum_of(sample_id)] = counts.get(stratum_of(sample_id), 0) + 1
        return {key: value / total for key, value in counts.items()}

    overall = shares(ids)
    for name in ("train", "valid", "test"):
        split_shares = shares(splits[name])
        assert set(split_shares) == set(overall), name
        for stratum, share in overall.items():
            assert abs(split_shares[stratum] - share) <= 0.05, (name, stratum)


def test_split_freezes_previous_test_speakers() -> None:
    first = make_splits(make_ids(30), group_key=speaker_of, stratify=stratum_of)
    frozen_test_speakers = {speaker_of(i) for i in first.test}
    assert frozen_test_speakers

    # a second data-collection round adds 30 more speakers
    second = make_splits(make_ids(60), group_key=speaker_of, stratify=stratum_of, frozen=first)

    assert {speaker_of(i) for i in second.test} >= frozen_test_speakers
    assert {speaker_of(i) for i in second.valid} & frozen_test_speakers == set()
    assert {speaker_of(i) for i in second.train} & frozen_test_speakers == set()
    # every previously assigned speaker keeps its split
    for name in ("train", "valid", "test"):
        assert {speaker_of(i) for i in first[name]} <= {speaker_of(i) for i in second[name]}
    # and the new utterances of a frozen speaker land in the frozen split, not elsewhere
    assert sorted(second.all_ids()) == sorted(make_ids(60))


def test_make_splits_is_deterministic_and_seed_sensitive() -> None:
    ids = make_ids(30)
    a = make_splits(ids, group_key=speaker_of, stratify=stratum_of, seed=1234)
    b = make_splits(ids, group_key=speaker_of, stratify=stratum_of, seed=1234)
    c = make_splits(ids, group_key=speaker_of, stratify=stratum_of, seed=99)

    assert a.train == b.train and a.valid == b.valid and a.test == b.test
    assert a.fingerprint == b.fingerprint
    assert (a.test, a.valid) != (c.test, c.valid)
    # id order inside a split does not depend on the input order
    shuffled = list(reversed(ids))
    assert make_splits(shuffled, group_key=speaker_of, stratify=stratum_of).test == a.test


def test_make_splits_without_stratifier_still_disjoint() -> None:
    ids = make_ids(20)
    splits = make_splits(ids, group_key=speaker_of, ratios=(0.5, 0.25, 0.25))
    assert {speaker_of(i) for i in splits.train} & {speaker_of(i) for i in splits.test} == set()
    assert len(splits.train) == pytest.approx(len(ids) * 0.5, abs=len(ids) * 0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratios": (0.8, 0.1)},
        {"ratios": (0.8, 0.1, 0.2)},
        {"ratios": (1.0, 0.0, 0.0)},
        {"ratios": (-0.1, 0.6, 0.5)},
    ],
)
def test_make_splits_rejects_bad_ratios(kwargs: dict[str, object]) -> None:
    with pytest.raises(InputValidationError):
        make_splits(make_ids(6), group_key=speaker_of, **kwargs)  # type: ignore[arg-type]


def test_make_splits_rejects_empty_and_duplicate_ids() -> None:
    with pytest.raises(InputValidationError):
        make_splits([], group_key=speaker_of)
    with pytest.raises(InputValidationError):
        make_splits(["a-1", "a-1"], group_key=speaker_of)


def test_make_splits_rejects_inconsistent_stratum_inside_a_group() -> None:
    with pytest.raises(InputValidationError):
        make_splits(
            ["P1-a", "P1-b"],
            group_key=lambda i: i.split("-")[0],
            stratify=lambda i: (i.split("-")[1],),
        )


def test_splits_json_roundtrip(tmp_path: Path) -> None:
    splits = make_splits(make_ids(12), group_key=speaker_of, stratify=stratum_of)
    path = save_splits(splits, tmp_path / "audio_commands_splits.json")
    reloaded = load_splits(path)

    assert reloaded == splits
    assert reloaded.fingerprint == splits.fingerprint
    assert reloaded.seed == 1234
    assert reloaded.split_of("P00001-000") in {"train", "valid", "test"}
    with pytest.raises(NotFoundError):
        load_splits(tmp_path / "missing.json")


def test_mathwriting_official_reads_official_split_files(tmp_path: Path) -> None:
    root = tmp_path / "mathwriting"
    for split, n in (("train", 3), ("valid", 2), ("test", 2), ("symbols", 1), ("synthetic", 2)):
        folder = root / split
        folder.mkdir(parents=True)
        for k in range(n):
            (folder / f"{split}-{k:03d}.inkml").write_text("<ink/>\n", encoding="utf-8")

    splits = mathwriting_official(root)

    assert len(splits.train) == 3
    assert len(splits.valid) == 2
    assert len(splits.test) == 2
    assert len(splits["symbols"]) == 1
    assert len(splits["synthetic"]) == 2
    assert splits.train[0] == "train-000"
    assert splits.source == "official"
    assert set(splits.names()) == {"train", "valid", "test", "symbols", "synthetic"}


def test_mathwriting_official_accepts_split_list_files(tmp_path: Path) -> None:
    root = tmp_path / "mathwriting"
    root.mkdir()
    (root / "train.txt").write_text("a.inkml\nb.inkml\n", encoding="utf-8")
    (root / "test.txt").write_text("c.inkml\n", encoding="utf-8")

    splits = mathwriting_official(root)
    assert splits.train == ["a", "b"]
    assert splits.test == ["c"]
    assert splits.valid == []


def test_mathwriting_official_rejects_root_without_any_split(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(NotFoundError):
        mathwriting_official(tmp_path / "empty")


def test_splits_container_helpers() -> None:
    splits = Splits(train=["a"], valid=["b"], test=["c"])
    assert splits["train"] == ["a"]
    assert splits.split_of("c") == "test"
    assert splits.split_of("zzz") is None
    assert splits.sizes() == {"train": 1, "valid": 1, "test": 1}
    with pytest.raises(InputValidationError):
        _ = splits["nope"]
