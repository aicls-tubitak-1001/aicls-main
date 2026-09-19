"""The dataset/weights catalog is descriptive only — no download code lives here (ADR-0012)."""

from __future__ import annotations

import inspect

import pytest

from aicls.core.errors import NotFoundError
from aicls.core.gates import GATE_NAMES
from aicls.data import catalog as catalog_module
from aicls.data.catalog import CatalogEntry, catalog_entry, load_catalog


def test_catalog_describes_mathwriting_with_url_size_license_and_gate() -> None:
    entry = catalog_entry("mathwriting")

    assert isinstance(entry, CatalogEntry)
    assert entry.version == "mathwriting-2024.09"
    assert entry.url is not None and entry.url.startswith("https://")
    # verified 2026-09-08 against the archive itself: the .zip url answers 403, the .tgz
    # answers 200 with this exact content-length.
    assert entry.url.endswith("mathwriting-2024.tgz")
    assert entry.expected_bytes == 3_096_141_721
    assert entry.gate == "DATASET_DOWNLOAD"
    # Recorded on 2026-09-08 from the archive the approved download actually produced. Until then
    # it was null; from now on it is the thing later verifications compare against, so what has to
    # hold is that it is a real digest and not a placeholder somebody typed.
    assert entry.sha256 is not None
    assert len(entry.sha256) == 64 and set(entry.sha256) <= set("0123456789abcdef")
    assert entry.kind == "dataset"
    assert "Gervais" in entry.citation


def test_catalog_records_the_noncommercial_sharealike_licence_of_mathwriting() -> None:
    """MathWriting is CC BY-NC-SA 4.0, not Apache-2.0, and the note has to say why that matters.

    The project has discussed a patent / commercial phase; NonCommercial and ShareAlike are the
    two words that decide whether weights trained on this corpus may cross into it. Recording
    the wrong licence would hide that decision, so it is asserted here and not only in a doc.
    """
    entry = catalog_entry("mathwriting")

    assert entry.license == "CC-BY-NC-SA-4.0"
    assert "NonCommercial" in entry.notes_tr
    assert "ShareAlike" in entry.notes_tr
    assert "ticari" in entry.notes_tr.lower()
    assert catalog_entry("mathwriting_excerpt").license == entry.license


def test_catalog_offers_the_official_excerpt_next_to_the_full_archive() -> None:
    """The 1.5 MB official excerpt proves the whole pipeline before 2.9 GiB is spent."""
    excerpt = catalog_entry("mathwriting_excerpt")
    full = catalog_entry("mathwriting")

    assert excerpt.url is not None and excerpt.url.endswith("mathwriting-2024-excerpt.tgz")
    assert excerpt.expected_bytes == 1_620_634
    assert excerpt.expected_bytes < full.expected_bytes / 1000
    assert excerpt.gate == full.gate == "DATASET_DOWNLOAD"
    assert excerpt.splits == full.splits
    # Both archives have been fetched through the gate, so both carry a digest, and they are not
    # the same archive.
    assert excerpt.sha256 is not None and excerpt.sha256 != full.sha256


def test_catalog_records_a_digest_only_for_what_has_actually_been_downloaded() -> None:
    """A digest is a record of a fetch, never a promise about one that has not happened.

    whisper-large-v3-turbo was fetched through the gate on 2026-09-08 and carries its digest.
    The other two have not been fetched and must not carry one: a digest that nobody produced by
    downloading is invented, and every later "verification" against it would be theatre.
    """
    from aicls.data.catalog import catalog_entry

    fetched = catalog_entry("whisper-large-v3-turbo")
    assert fetched.sha256 is not None
    assert len(fetched.sha256) == 64 and set(fetched.sha256) <= set("0123456789abcdef")
    assert "MIT" in fetched.license

    for name in ("whisper-small", "wav2vec2-xls-r-300m"):
        assert catalog_entry(name).sha256 is None, f"{name} has a digest but was never downloaded"


def test_catalog_entry_archive_name_comes_from_the_url_not_from_a_guessed_suffix() -> None:
    """The download script writes this name and the prepare script streams it; one source."""
    assert catalog_entry("mathwriting").archive_name == "mathwriting-2024.tgz"
    assert catalog_entry("mathwriting_excerpt").archive_name == "mathwriting-2024-excerpt.tgz"
    # an entry without a url (internal data, weights) falls back to its version label
    assert catalog_entry("audio_commands").archive_name == "aicls-tr-speech-2027.02"
    assert (
        CatalogEntry(version="v1", license="x", url="https://h/a/b.tgz?x=1").archive_name == "b.tgz"
    )
    assert CatalogEntry(version="v1", license="x", url="https://h/dir/").archive_name == "dir"


def test_catalog_describes_the_internal_audio_command_dataset_without_a_gate() -> None:
    entry = catalog_entry("audio_commands")

    assert entry.gate is None
    assert entry.url is None
    assert entry.source == "internal"
    assert entry.kind == "dataset"


def test_catalog_gate_names_are_real_gates_and_weights_are_listed() -> None:
    catalog = load_catalog()

    assert set(catalog.datasets) >= {"mathwriting", "audio_commands"}
    assert "whisper-small" in catalog.weights
    for entry in catalog.entries():
        assert entry.gate is None or entry.gate in GATE_NAMES
        assert entry.license
        assert entry.name == entry.name.strip()
    assert catalog.get("whisper-small").gate == "WEIGHTS_DOWNLOAD"
    assert load_catalog() is catalog  # cached, parsed once per process


def test_catalog_entry_rejects_an_unknown_name() -> None:
    with pytest.raises(NotFoundError):
        catalog_entry("does-not-exist")


def test_catalog_module_contains_no_download_helper() -> None:
    source = inspect.getsource(catalog_module)
    for forbidden in ("urllib", "requests", "hf_hub_download", "snapshot_download", "urlopen"):
        assert forbidden not in source
    assert not [name for name in dir(catalog_module) if name.startswith("download")]
