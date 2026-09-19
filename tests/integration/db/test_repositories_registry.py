"""Behaviour of the model registry, evaluation, telemetry and feedback repositories (§3.5)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.db import (
    EvalKind,
    EvalRunRepository,
    FeedbackCategory,
    FeedbackRepository,
    FeedbackStatus,
    LatencyRepository,
    Modality,
    ModelComponent,
    ModelVersionRepository,
    User,
)
from aicls.db.base import utcnow
from aicls.db.repositories.registry import percentile, sus_score

pytestmark = pytest.mark.integration

REPORT = {
    "name": "cas-curriculum",
    "kind": "cas",
    "git_sha": "a" * 40,
    "dirty": False,
    "citable": True,
    "dataset_fingerprint": "sha256:" + "b" * 64,
    "dataset_version": "2027-04",
    "model_version": None,
    "normalizer_versions": {"latex": "latex_v1", "tr": "tr_v1"},
    "n": 1000,
    "metrics": {"correct_rate": 0.994},
    "ci95": {"correct_rate": [0.989, 0.997]},
    "report_path": "reports/metrics/cas_2027-04.json",
}


class TestModelVersionRepository:
    """``model_versions`` is the single registry; ``models/registry.yaml`` is only a seed."""

    def test_register_is_idempotent_on_component_and_version(self, session: Session) -> None:
        repo = ModelVersionRepository(session)
        first = repo.register("ocr", "ocr-0.1.0", "ocr/0.1.0", "c" * 64, {"cer": 0.12})
        again = repo.register(
            ModelComponent.OCR, "ocr-0.1.0", "ocr/0.1.0b", "d" * 64, {"cer": 0.09}, run_id="r1"
        )
        assert again is first
        assert first.path == "ocr/0.1.0b"
        assert first.metrics_json["cer"] == 0.09
        assert first.run_id == "r1"
        assert len(repo.list_versions("ocr")) == 1

    def test_activate_promotes_exactly_one_version_per_component(
        self, session: Session, user: User
    ) -> None:
        repo = ModelVersionRepository(session)
        repo.register("ocr", "ocr-0.1.0", "p1", "c" * 64)
        newer = repo.register("ocr", "ocr-0.2.0", "p2", "e" * 64)
        asr = repo.register("asr", "asr-0.1.0", "p3", "f" * 64)
        assert repo.active("ocr") is None

        repo.activate("ocr", "ocr-0.1.0")
        promoted = repo.activate("ocr", "ocr-0.2.0", promoted_by=user.id)
        assert promoted is newer
        assert repo.active("ocr") is newer
        assert newer.promoted_at is not None and newer.promoted_by == user.id
        assert sum(1 for row in repo.list_versions("ocr") if row.is_active) == 1
        assert repo.active("asr") is None and asr.is_active is False

    def test_activate_rejects_an_unknown_version(self, session: Session) -> None:
        repo = ModelVersionRepository(session)
        with pytest.raises(NotFoundError):
            repo.activate("ocr", "ocr-9.9.9")
        with pytest.raises(InputValidationError):
            repo.active("nlu")


class TestEvalRunRepository:
    """``eval_runs`` mirrors ``reports/metrics/*.json`` so the admin pages read one source."""

    def test_add_maps_the_report_document(self, session: Session) -> None:
        repo = EvalRunRepository(session)
        row = repo.add(REPORT, is_reportable=True)
        assert row.kind is EvalKind.CAS
        assert row.n == 1000
        assert row.metrics_json == {"correct_rate": 0.994}
        assert row.ci95_json["correct_rate"] == [0.989, 0.997]
        assert row.normalizer_versions_json["tr"] == "tr_v1"
        assert row.dataset_fingerprint == "sha256:" + "b" * 64
        assert row.report_path.endswith(".json")
        assert row.is_reportable is True
        assert row.model_version is None

    def test_add_defaults_and_missing_keys(self, session: Session) -> None:
        row = EvalRunRepository(session).add({})
        assert row.kind is EvalKind.PARSER
        assert (row.n, row.dirty, row.is_reportable) == (0, False, False)

    def test_list_filters_by_kind_and_since_and_reportable(self, session: Session) -> None:
        repo = EvalRunRepository(session)
        repo.add(REPORT, is_reportable=True)
        repo.add({**REPORT, "kind": "asr"}, is_reportable=False)
        assert len(repo.list()) == 2
        assert [row.kind for row in repo.list("asr")] == [EvalKind.ASR]
        assert repo.list(since=utcnow() + timedelta(days=1)) == []
        assert [row.kind for row in repo.reportable()] == [EvalKind.CAS]
        assert repo.reportable("asr") == []


class TestLatencyRepository:
    """``latency_samples`` measures the contractual "< 3 s to rendered result"."""

    def test_stats_prefer_the_client_total_and_group_by_modality(self, session: Session) -> None:
        repo = LatencyRepository(session)
        for value in (100.0, 200.0, 300.0, 2000.0):
            repo.add(
                {
                    "request_id": "0" * 26,
                    "modality": "text",
                    "endpoint": "/compute/text",
                    "server_ms": value / 2,
                    "client_total_ms": value,
                    "render_ms": 12.0,
                    "stage": {"cas": 10.0},
                }
            )
        repo.add({"request_id": "1" * 26, "modality": "audio", "server_ms": 800.0})

        stats = repo.stats()
        assert stats.n == 5
        assert stats.p50_ms == 300.0
        assert stats.p95_ms == 2000.0
        assert stats.mean_ms == pytest.approx((100 + 200 + 300 + 2000 + 800) / 5)
        assert set(stats.by_modality) == {"text", "audio"}
        assert stats.by_modality["audio"] == 800.0

        text_only = repo.stats(modality=Modality.TEXT)
        assert text_only.n == 4
        assert repo.stats(since=utcnow() + timedelta(days=1)).n == 0

    def test_stats_on_an_empty_table(self, session: Session) -> None:
        stats = LatencyRepository(session).stats()
        assert (stats.n, stats.mean_ms, stats.p50_ms, stats.p95_ms) == (0, 0.0, 0.0, 0.0)
        assert stats.by_modality == {}


class TestFeedbackRepository:
    """``feedback`` and ``sus_responses`` (İP-9; SUS target > 75)."""

    def test_add_feedback_defaults_to_open_and_pages_by_filter(
        self, session: Session, user: User
    ) -> None:
        repo = FeedbackRepository(session)
        critical = repo.add_feedback(user.id, "critical", "Grafik açılmıyor", "/workspace")
        repo.add_feedback(None, FeedbackCategory.SUGGESTION, "Koyu tema", "/settings")
        assert critical.status is FeedbackStatus.OPEN
        assert len(repo.page().items) == 2
        assert [row.id for row in repo.page(category="critical").items] == [critical.id]
        assert len(repo.page(status="open").items) == 2
        repo.set_status(critical.id, "fixed")
        assert [row.id for row in repo.page(status=FeedbackStatus.FIXED).items] == [critical.id]
        with pytest.raises(NotFoundError):
            repo.set_status("0" * 26, "open")

    def test_add_sus_computes_the_score_and_averages_a_cohort(
        self, session: Session, user: User
    ) -> None:
        repo = FeedbackRepository(session)
        best = repo.add_sus(user.id, "pilot-2027", [5, 1, 5, 1, 5, 1, 5, 1, 5, 1])
        worst = repo.add_sus(None, "pilot-2027", [1, 5, 1, 5, 1, 5, 1, 5, 1, 5])
        assert best.score == 100.0
        assert worst.score == 0.0
        assert best.answers_json == {"answers": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]}
        assert repo.sus_average("pilot-2027") == 50.0
        assert repo.sus_average("baska") == 0.0
        assert repo.sus_average() == 50.0

    def test_add_sus_accepts_a_precomputed_score_and_rejects_bad_answers(
        self, session: Session
    ) -> None:
        repo = FeedbackRepository(session)
        row = repo.add_sus(None, "c", [3] * 10, score=77.5, collected_week=3)
        assert row.score == 77.5 and row.collected_week == 3
        with pytest.raises(InputValidationError):
            repo.add_sus(None, "c", [3] * 9)
        with pytest.raises(InputValidationError):
            repo.add_sus(None, "c", [9] * 10)


def test_percentile_and_sus_reference_values() -> None:
    """Hand-computed reference values for the two numeric helpers."""
    assert percentile([], 0.5) == 0.0
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    # All-neutral answers give the mid score of the scale.
    assert sus_score([3] * 10) == 50.0
