"""Consent text versioning and acceptance (KVKK, docs/ml/veri-yonetimi.md §4)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from aicls.collect import (
    ContributionGateError,
    Demographics,
    accept_consent,
    consent_texts,
    contribution_enabled,
    current_consent,
    get_consent_text,
    hash_ip,
    require_contribution_enabled,
)
from aicls.collect.service import MAX_ETHICS_REF_LENGTH
from aicls.core.errors import ErrorCode, InputValidationError, NotFoundError
from aicls.core.hashing import sha256_bytes
from aicls.core.settings import Settings, settings_override
from aicls.db import ConsentVersion, RecordingRepository, Speaker


class TestConsentText:
    """The consent text is a versioned, hashed package resource."""

    def test_current_consent_is_versioned_and_hashed(self) -> None:
        consent = current_consent()
        assert consent.version
        assert consent.text_tr.strip()
        assert consent.sha256 == sha256_bytes(consent.text_tr.encode("utf-8"))
        assert len(consent.sha256) == 64
        assert isinstance(consent.effective_from, date)

    def test_consent_text_mentions_the_kvkk_rights(self) -> None:
        text = current_consent().text_tr.casefold()
        for needle in ("kvkk", "geri çek", "takma ad", "ses"):
            assert needle in text, needle

    def test_consent_texts_are_ordered_and_unique(self) -> None:
        texts = consent_texts()
        assert texts
        versions = [text.version for text in texts]
        assert len(set(versions)) == len(versions)
        dates = [text.effective_from for text in texts]
        assert dates == sorted(dates)
        assert current_consent() is texts[-1]

    def test_current_consent_respects_the_effective_date(self) -> None:
        first = consent_texts()[0]
        assert current_consent(on=first.effective_from) is first
        with pytest.raises(NotFoundError):
            current_consent(on=date(1970, 1, 1))

    def test_get_consent_text_rejects_an_unknown_version(self) -> None:
        assert get_consent_text(current_consent().version) is current_consent()
        with pytest.raises(NotFoundError):
            get_consent_text("v0-does-not-exist")


class TestContributionGate:
    """Without a recorded ethics approval reference the collection flow is closed (503)."""

    def test_contribution_is_enabled_in_the_test_profile(self, collect_settings: Settings) -> None:
        assert collect_settings.ethics_approval_ref == "TEST"
        assert contribution_enabled() is True
        require_contribution_enabled()

    def test_gate_blocks_when_the_ethics_reference_is_missing(
        self, collect_settings: Settings
    ) -> None:
        with settings_override(ethics_approval_ref=""):
            assert contribution_enabled() is False
            with pytest.raises(ContributionGateError) as excinfo:
                require_contribution_enabled()
        assert excinfo.value.code is ErrorCode.GATE_BLOCKED
        assert excinfo.value.http_status == 503

    def test_an_over_long_ethics_reference_is_a_configuration_error(
        self, collect_settings: Settings
    ) -> None:
        # consent_versions.ethics_approval_ref is VARCHAR(80): a longer value would be
        # truncated by SQLite and rejected by MySQL in strict mode.
        with (
            settings_override(ethics_approval_ref="E" * 81),
            pytest.raises(InputValidationError) as excinfo,
        ):
            require_contribution_enabled()
        assert excinfo.value.details["max_length"] == MAX_ETHICS_REF_LENGTH

    def test_accept_consent_is_blocked_by_the_gate(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        with (
            settings_override(ethics_approval_ref=""),
            pytest.raises(ContributionGateError),
        ):
            accept_consent(
                repo,
                user_id=None,
                version=current_consent().version,
                demographics=demographics,
                ip_hash="a" * 64,
            )


class TestAcceptConsent:
    """Acceptance creates a pseudonymous speaker and an immutable consent audit row."""

    def test_creates_pseudonymous_speaker(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        consent = current_consent()
        speaker = accept_consent(
            repo,
            user_id=None,
            version=consent.version,
            demographics=demographics,
            ip_hash="b" * 64,
        )
        assert speaker.pseudonym == "P00001"
        assert speaker.user_id is None
        assert speaker.ip_hash == "b" * 64
        assert speaker.withdrawn_at is None
        assert speaker.age_band.value == "18-24"
        assert speaker.gender.value == "f"
        assert speaker.university.value == "mus"

        stored = repo.get_consent_version(consent.version)
        assert stored is not None
        assert stored.sha256 == consent.sha256
        assert stored.text_tr == consent.text_tr
        assert stored.ethics_approval_ref == "TEST"
        assert speaker.consent_version_id == stored.id

    def test_pseudonyms_increase(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        version = current_consent().version
        pseudonyms = [
            accept_consent(
                repo, user_id=None, version=version, demographics=demographics, ip_hash=None
            ).pseudonym
            for _ in range(3)
        ]
        assert pseudonyms == ["P00001", "P00002", "P00003"]

    def test_speaker_row_stores_no_direct_identity(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        speaker = accept_consent(
            repo,
            user_id=None,
            version=current_consent().version,
            demographics=demographics,
            ip_hash=hash_ip("192.0.2.10"),
        )
        columns = {column.name for column in Speaker.__table__.columns}
        assert not columns & {"name", "display_name", "email", "birth_date", "phone"}
        assert speaker.ip_hash is not None
        assert "192.0.2.10" not in speaker.ip_hash

    def test_consent_row_is_reused_across_speakers(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        version = current_consent().version
        first = accept_consent(
            repo, user_id=None, version=version, demographics=demographics, ip_hash=None
        )
        second = accept_consent(
            repo, user_id=None, version=version, demographics=demographics, ip_hash=None
        )
        assert first.consent_version_id == second.consent_version_id
        rows = list(repo.session.scalars(select(ConsentVersion)))
        assert len(rows) == 1

    def test_rejects_an_outdated_or_unknown_version(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        with pytest.raises(InputValidationError) as excinfo:
            accept_consent(
                repo,
                user_id=None,
                version="v0-old",
                demographics=demographics,
                ip_hash=None,
            )
        assert excinfo.value.details["expected"] == current_consent().version

    def test_rejects_a_raw_ip_address(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        for raw in ("192.0.2.10", "2001:db8::1"):
            with pytest.raises(InputValidationError):
                accept_consent(
                    repo,
                    user_id=None,
                    version=current_consent().version,
                    demographics=demographics,
                    ip_hash=raw,
                )

    def test_links_a_signed_in_user_only_once(
        self, repo: RecordingRepository, demographics: Demographics
    ) -> None:
        from aicls.db import UserRepository

        user = UserRepository(repo.session).create("ada@example.edu.tr", "argon2id$h", "Ada")
        version = current_consent().version
        first = accept_consent(
            repo, user_id=user.id, version=version, demographics=demographics, ip_hash=None
        )
        again = accept_consent(
            repo, user_id=user.id, version=version, demographics=demographics, ip_hash=None
        )
        assert again.id == first.id
        assert repo.speaker_for_user(user.id) is first


class TestDemographics:
    """Only coarse bands may be collected (KVKK data minimisation)."""

    def test_defaults_are_not_answered(self) -> None:
        demo = Demographics()
        assert demo.gender.value == "na"
        assert demo.university.value == "other"

    def test_rejects_identity_fields(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Demographics(name="Ada Lovelace")  # type: ignore[call-arg]  # forbidden on purpose

    def test_rejects_an_unknown_band(self) -> None:
        with pytest.raises(ValueError, match="age_band"):
            Demographics(age_band="12-17")

    def test_accent_region_is_trimmed_and_bounded(self) -> None:
        assert Demographics(accent_region="  Doğu Anadolu  ").accent_region == "Doğu Anadolu"
        assert Demographics(accent_region="   ").accent_region is None
        with pytest.raises(ValueError, match="accent_region"):
            Demographics(accent_region="x" * 100)


class TestHashIp:
    """``hash_ip`` is the only way a network address may reach the database."""

    def test_is_deterministic_and_salted(self, collect_settings: Settings) -> None:
        first = hash_ip("192.0.2.10")
        assert first == hash_ip("192.0.2.10")
        assert first != hash_ip("192.0.2.11")
        assert first != hash_ip("192.0.2.10", salt="another-salt")
        assert len(first) == 64
        assert "192" not in first

    def test_empty_value_hashes_to_nothing(self, collect_settings: Settings) -> None:
        assert hash_ip("") is None
        assert hash_ip("   ") is None
