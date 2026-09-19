"""Prompt balancing: the least-recorded sentence comes first, scenarios stay mixed (İP-3)."""

from __future__ import annotations

import pytest

from aicls.collect import MAX_PROMPTS, next_prompts
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.db import PromptSentence, RecordingRepository, Speaker

PASSING_QC = {"passed": True, "snr_db": 22.0, "clipping_ratio": 0.0, "silence_ratio": 0.1}


def _record(repo: RecordingRepository, speaker: Speaker, prompt: PromptSentence) -> None:
    """Store a recording row for ``prompt`` (no file: only the counting matters here)."""
    repo.add(speaker.id, prompt.id, f"audio/raw/x/{prompt.id}.wav", PASSING_QC, 2.0, "a" * 64)


class TestNextPrompts:
    """``next_prompts`` drives the collection towards an even corpus."""

    def test_next_prompts_prefers_least_recorded(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        # Two prompts of the same scenario: record the first one twice, the second one once.
        turev = [p for p in prompts if p.scenario_tag == "turev"]
        assert len(turev) == 2
        other = repo.create_speaker(speaker.consent_version_id)
        for _ in range(2):
            _record(repo, other, turev[0])
        _record(repo, other, turev[1])

        ranked = next_prompts(repo, speaker.id, n=MAX_PROMPTS)
        order = [p.id for p in ranked]
        # Never-recorded prompts first, then the once-recorded one, then the twice-recorded one.
        assert order.index(turev[1].id) < order.index(turev[0].id)
        assert order[-1] == turev[0].id
        assert order[0] not in (turev[0].id, turev[1].id)

    def test_excludes_prompts_the_speaker_already_read(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        first = next_prompts(repo, speaker.id, n=1)[0]
        _record(repo, speaker, first)
        assert first.id not in {p.id for p in next_prompts(repo, speaker.id, n=MAX_PROMPTS)}

    def test_spreads_over_reading_scenarios(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        # Four scenarios ("turev" twice); the first four suggestions must not repeat a scenario.
        tags = [p.scenario_tag for p in next_prompts(repo, speaker.id, n=4)]
        assert len(set(tags)) == 4

    def test_accepts_the_pseudonym_as_the_speaker_id(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        by_pseudonym = next_prompts(repo, speaker.pseudonym, n=3)
        by_ulid = next_prompts(repo, speaker.id, n=3)
        assert [p.id for p in by_pseudonym] == [p.id for p in by_ulid]

    def test_unknown_speaker_is_not_found(
        self, repo: RecordingRepository, prompts: list[PromptSentence]
    ) -> None:
        with pytest.raises(NotFoundError):
            next_prompts(repo, "P99999", n=3)

    def test_withdrawn_speaker_gets_no_prompts(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        from aicls.collect import ConsentRequiredError, withdraw

        withdraw(repo, speaker.id)
        with pytest.raises(ConsentRequiredError):
            next_prompts(repo, speaker.id, n=3)

    @pytest.mark.parametrize("n", [0, -1, MAX_PROMPTS + 1])
    def test_rejects_an_out_of_range_count(
        self, repo: RecordingRepository, speaker: Speaker, n: int
    ) -> None:
        with pytest.raises(InputValidationError):
            next_prompts(repo, speaker.id, n=n)

    def test_empty_pool_returns_an_empty_list(
        self, repo: RecordingRepository, speaker: Speaker
    ) -> None:
        assert next_prompts(repo, speaker.id, n=5) == []
