"""The in-process pipeline client the latency runner times."""

from __future__ import annotations

from aicls.eval.pipeline import LocalPipelineClient
from aicls.mathengine import InProcessExecutor, MathEngine


def _client(**kwargs: object) -> LocalPipelineClient:
    return LocalPipelineClient(MathEngine(InProcessExecutor()), **kwargs)  # type: ignore[arg-type]


def test_client_times_nlu_and_cas_for_a_computation() -> None:
    """A real command pays for both stages, and ``total`` covers them."""
    client = _client()
    timings = client.run("x karenin türevini al")
    assert set(timings) >= {"nlu", "cas", "total"}
    assert timings["total"] >= timings["nlu"] + timings["cas"] - 1e-6
    assert client.errors == 0
    assert client.no_command == 0


def test_client_counts_a_text_that_produces_no_cas_command() -> None:
    """A book lookup or a refusal costs the NLU stage only; it is still a measured response."""
    client = _client()
    timings = client.run("türev tanımını getir")
    assert "cas" not in timings
    assert timings["total"] > 0.0
    assert client.no_command == 1


def test_client_counts_an_engine_refusal_without_raising() -> None:
    """A corpus run must not stop at the first command the CAS cannot answer."""
    client = _client()
    client.run("qwerty asdf zxcv")
    assert client.no_command == 1


def test_client_clears_the_cache_between_commands_by_default() -> None:
    """A latency figure that mostly measured LRU lookups would be meaningless.

    Asserted on the cache counters, not on the clock: the same command twice must cost two
    cache misses by default, and one miss plus one hit when caching is deliberately enabled.
    """
    engine = MathEngine(InProcessExecutor())
    client = LocalPipelineClient(engine)
    client.run("x karenin türevini al")
    client.run("x karenin türevini al")
    assert engine.cache.hits == 0

    warm_engine = MathEngine(InProcessExecutor())
    warm_client = LocalPipelineClient(warm_engine, cached=True)
    warm_client.run("x karenin türevini al")
    warm_client.run("x karenin türevini al")
    assert warm_engine.cache.hits == 1


def test_client_builds_its_own_engine_when_none_is_given() -> None:
    """The default construction is what the harness uses."""
    assert LocalPipelineClient().run("x karenin türevini al")["total"] > 0.0


def test_client_counts_an_engine_error_and_keeps_going() -> None:
    """A CAS refusal is measured like any other response; the corpus run continues."""
    from aicls.mathengine import UnsupportedProblem
    from aicls.mathengine.models import CasCommand, Result

    class RefusingEngine(MathEngine):
        """An engine that refuses every command."""

        def execute(self, cmd: CasCommand | object) -> Result:  # type: ignore[override]
            raise UnsupportedProblem

    client = LocalPipelineClient(RefusingEngine(InProcessExecutor()))
    timings = client.run("x karenin türevini al")
    assert client.errors == 1
    assert "cas" in timings and timings["total"] > 0.0
