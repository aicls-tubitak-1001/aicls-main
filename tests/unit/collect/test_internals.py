"""Edge cases of the collection helpers: resource parsing, paths, digests and report shapes."""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path

import pytest

from aicls.asr.qc import QcReport
from aicls.collect import (
    ConsentText,
    Demographics,
    ManifestReport,
    RecordingDecision,
    SplitFile,
    WithdrawReport,
    accept_consent,
    assign_splits,
    audio_raw_root,
    current_consent,
    export_manifest,
    load_splits,
    new_recording_path,
    save_splits,
    speaker_audio_dir,
    withdraw,
)
from aicls.collect.consent import _build, _coerce_date, consent_texts
from aicls.collect.service import _check_digest, _demographics
from aicls.collect.storage import check_pseudonym, delete_speaker_files, write_bytes_atomic
from aicls.core.errors import InputValidationError
from aicls.db import RecordingRepository, Speaker


class TestConsentCatalogue:
    """A malformed consent catalogue must fail loudly, never silently degrade the audit trail."""

    def test_as_dict_is_json_serialisable(self) -> None:
        payload = current_consent().as_dict()
        assert set(payload) == {"version", "text_tr", "sha256", "effective_from"}
        assert json.loads(json.dumps(payload))["version"] == current_consent().version

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (date(2026, 5, 1), date(2026, 5, 1)),
            (datetime(2026, 5, 1, 12, tzinfo=UTC), date(2026, 5, 1)),
            ("2026-05-01", date(2026, 5, 1)),
        ],
    )
    def test_coerce_date_accepts_yaml_shapes(self, value: object, expected: date) -> None:
        assert _coerce_date(value, version="1.0") == expected

    def test_coerce_date_rejects_nonsense(self) -> None:
        with pytest.raises(InputValidationError):
            _coerce_date("dün", version="1.0")

    @pytest.mark.parametrize(
        "entry",
        [
            "not-a-mapping",
            {"version": "", "text_tr": "metin", "effective_from": "2026-05-01"},
            {"version": "1.0", "text_tr": "", "effective_from": "2026-05-01"},
            {"version": "x" * 20, "text_tr": "metin", "effective_from": "2026-05-01"},
        ],
    )
    def test_build_rejects_a_broken_entry(self, entry: object) -> None:
        with pytest.raises(InputValidationError):
            _build(entry, index=0)

    def test_build_hashes_the_trimmed_text(self) -> None:
        text = _build(
            {"version": "9.9", "text_tr": "  Onam metni  ", "effective_from": "2026-05-01"},
            index=0,
        )
        assert isinstance(text, ConsentText)
        assert text.text_tr == "Onam metni"
        assert len(text.sha256) == 64

    def test_catalogue_is_cached(self) -> None:
        assert consent_texts() is consent_texts()


class TestDigestGuard:
    """Only hashes may be stored; a raw identifier is a programming error, not a warning."""

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_blank_values_become_none(self, value: str | None) -> None:
        assert _check_digest(value, field="ip_hash") is None

    def test_uppercase_digest_is_normalised(self) -> None:
        assert _check_digest("AB" * 32, field="ip_hash") == "ab" * 32

    @pytest.mark.parametrize("value", ["10.0.0.1", "::1", "not-a-hash", "abc"])
    def test_rejects_non_digests(self, value: str) -> None:
        with pytest.raises(InputValidationError):
            _check_digest(value, field="ip_hash")


class TestDemographicsCoercion:
    """The API layer may hand over a mapping instead of the model."""

    def test_mapping_is_validated(self) -> None:
        bands = _demographics({"age_band": "45+", "gender": "m", "university": "amasya"})
        assert isinstance(bands, Demographics)
        assert bands.age_band.value == "45+"

    def test_model_passes_through(self) -> None:
        bands = Demographics()
        assert _demographics(bands) is bands

    def test_mapping_with_an_identity_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match="email"):
            _demographics({"email": "ada@example.edu.tr"})

    def test_accept_consent_accepts_a_mapping(self, repo: RecordingRepository) -> None:
        speaker = accept_consent(
            repo,
            user_id=None,
            version=current_consent().version,
            demographics={"age_band": "35-44", "gender": "na", "university": "amasya"},
            ip_hash=None,
        )
        assert speaker.age_band.value == "35-44"


class TestStoragePaths:
    """Path building never trusts anything but a validated pseudonym."""

    @pytest.mark.parametrize(
        "pseudonym", ["", "P1", "p00001", "P0001", "../../etc", "P00001/../..", "P00001 "]
    )
    def test_check_pseudonym_rejects_anything_unexpected(self, pseudonym: str) -> None:
        with pytest.raises(InputValidationError):
            check_pseudonym(pseudonym)

    def test_paths_stay_under_the_audio_root(self, data_root: Path) -> None:
        path, relative = new_recording_path("P00007", data_root=data_root)
        assert audio_raw_root(data_root=data_root) in path.parents
        assert relative.startswith("audio/raw/P00007/")
        assert path == data_root / relative

    def test_write_is_atomic_and_leaves_no_temporary(self, tmp_path: Path) -> None:
        target = tmp_path / "a.wav"
        assert write_bytes_atomic(target, b"RIFFdata") == 8
        assert target.read_bytes() == b"RIFFdata"
        assert list(tmp_path.iterdir()) == [target]

    def test_a_failed_write_does_not_create_the_target(self, tmp_path: Path) -> None:
        target = tmp_path / "b.wav"
        (tmp_path / "b.wav.part").mkdir()  # the temporary name is taken by a directory
        with pytest.raises(OSError):
            write_bytes_atomic(target, b"RIFF")
        assert not target.exists()

    def test_delete_removes_nested_directories(self, data_root: Path) -> None:
        nested = speaker_audio_dir("P00001", data_root=data_root) / "chunk-1"
        nested.mkdir(parents=True)
        (nested / "a.wav").write_bytes(b"1234")
        files, size = delete_speaker_files("P00001", data_root=data_root)
        assert (files, size) == (1, 4)
        assert not nested.exists()
        assert not nested.parent.exists()

    def test_delete_of_an_unknown_speaker_is_a_no_op(self, data_root: Path) -> None:
        assert delete_speaker_files("P09999", data_root=data_root) == (0, 0)


class TestReportShapes:
    """The dataclasses the API and CLI serialise."""

    def test_recording_decision_of_a_rejected_upload(self) -> None:
        decision = RecordingDecision(accepted=False, qc=QcReport(passed=False, reasons_tr=["x"]))
        assert decision.reasons_tr == ["x"]
        assert decision.as_dict()["id"] is None
        assert decision.as_dict()["accepted"] is False

    def test_withdraw_report_as_dict(self) -> None:
        moment = datetime(2026, 9, 7, 12, 0, 0)
        payload = WithdrawReport("P00001", 2, 2, 100, True, moment).as_dict()
        assert payload["withdrawn_at"] == moment.isoformat()
        assert payload["consent_kept"] is True
        assert WithdrawReport("P00002").as_dict()["withdrawn_at"] is None

    def test_manifest_report_as_dict(self, tmp_path: Path) -> None:
        report = ManifestReport(
            path=tmp_path / "m.jsonl", splits_path=tmp_path / "s.json", rows=2, seconds=7200.0
        )
        payload = report.as_dict()
        assert payload["hours"] == pytest.approx(2.0)
        assert payload["path"].endswith("m.jsonl")
        assert json.loads(json.dumps(payload))["rows"] == 2

    def test_accent_region_none_stays_none(self) -> None:
        assert Demographics(accent_region=None).accent_region is None


class TestSplitFileHandling:
    """The split file is the frozen evidence behind the reported WER."""

    def test_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        save_splits(path, SplitFile({"P00002": "test", "P00001": "train"}, seed=7))
        stored = load_splits(path)
        assert stored.by_speaker == {"P00001": "train", "P00002": "test"}
        assert stored.seed == 7
        assert list(json.loads(path.read_text(encoding="utf-8"))["speakers"]) == [
            "P00001",
            "P00002",
        ]

    def test_missing_file_is_an_empty_map(self, tmp_path: Path) -> None:
        assert load_splits(tmp_path / "nothing.json").by_speaker == {}

    @pytest.mark.parametrize(
        "document",
        [{"speakers": []}, {"speakers": {"P00001": "holdout"}}, ["not", "a", "mapping"]],
    )
    def test_rejects_a_broken_document(self, tmp_path: Path, document: object) -> None:
        path = tmp_path / "splits.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(InputValidationError):
            load_splits(path)

    def test_assign_splits_is_deterministic_and_frozen(self) -> None:
        strata = {f"P{index:05d}": ("mus", "f", "18-24") for index in range(1, 11)}
        first = assign_splits(strata)
        assert first == assign_splits(strata)
        assert sorted(first.values()).count("train") == 8

        strata["P00011"] = ("mus", "f", "18-24")
        second = assign_splits(strata, frozen=first)
        assert {p: s for p, s in second.items() if p in first} == first

    def test_assign_splits_without_speakers(self) -> None:
        assert assign_splits({}) == {}


class TestExportArgumentValidation:
    """Bad arguments fail before a single row is written."""

    def test_unknown_source_label(self, repo: RecordingRepository, tmp_path: Path) -> None:
        with pytest.raises(InputValidationError):
            export_manifest(repo, out=tmp_path / "m.jsonl", source="scraped")

    @pytest.mark.parametrize("ratios", [(0.8, 0.2), (0.8, -0.1, 0.3), (0.0, 0.0, 0.0)])
    def test_broken_ratios(
        self, repo: RecordingRepository, tmp_path: Path, ratios: tuple[float, ...]
    ) -> None:
        with pytest.raises(InputValidationError):
            export_manifest(repo, out=tmp_path / "m.jsonl", ratios=ratios)

    def test_empty_database_produces_an_empty_manifest(
        self, repo: RecordingRepository, tmp_path: Path
    ) -> None:
        report = export_manifest(repo, out=tmp_path / "m.jsonl")
        assert report.rows == 0
        assert (tmp_path / "m.jsonl").read_text(encoding="utf-8") == ""
        assert report.splits_path.is_file()


class TestReConsent:
    """A withdrawn account may not silently re-appear under its old pseudonym."""

    def test_withdrawn_account_cannot_re_accept(
        self, repo: RecordingRepository, speaker: Speaker, data_root: Path
    ) -> None:
        from aicls.db import UserRepository

        user = UserRepository(repo.session).create("ada@example.edu.tr", "argon2id$h", "Ada")
        linked = accept_consent(
            repo,
            user_id=user.id,
            version=current_consent().version,
            demographics=Demographics(),
            ip_hash=None,
        )
        withdraw(repo, linked.id, data_root=data_root)
        with pytest.raises(InputValidationError) as excinfo:
            accept_consent(
                repo,
                user_id=user.id,
                version=current_consent().version,
                demographics=Demographics(),
                ip_hash=None,
            )
        assert excinfo.value.details["reason"] == "speaker_withdrawn"


class TestCatalogueGuards:
    """The two defensive branches of the catalogue loader (a broken resource ships nothing)."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> object:
        consent_texts.cache_clear()
        yield
        consent_texts.cache_clear()

    def _stub_resource(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
        path = tmp_path / "consent_texts.yaml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setattr("aicls.collect.consent.package_resource", lambda pkg, name: path)

    def test_empty_catalogue_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub_resource(monkeypatch, tmp_path, "consents: []\n")
        with pytest.raises(InputValidationError):
            consent_texts()

    def test_duplicate_versions_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = (
            "consents:\n"
            '  - {version: "1.0", effective_from: 2026-05-01, text_tr: "a"}\n'
            '  - {version: "1.0", effective_from: 2026-06-01, text_tr: "b"}\n'
        )
        self._stub_resource(monkeypatch, tmp_path, body)
        with pytest.raises(InputValidationError):
            consent_texts()
