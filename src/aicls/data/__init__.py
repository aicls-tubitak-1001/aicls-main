"""Dataset layer: manifests, catalog, splits, shards, datasets, synthetic generators, collate.

This package is the single place that knows *what data exists, where it lives and how it becomes
tensors* (docs/ml/veri-yonetimi.md, docs/ml/egitim-altyapisi.md §2, ADR-0008):

* :class:`DataManifest` — SHA-256 identity of a large data root, with licence, source and the
  fingerprint every report cites;
* :mod:`aicls.data.catalog` — ``catalog.yaml``: what each dataset/checkpoint is, how big it is,
  which licence it carries and which approval gate its download sits behind (**descriptive only,
  this package downloads nothing**, ADR-0012);
* :func:`make_splits` / :func:`mathwriting_official` — deterministic writer- and speaker-disjoint,
  stratified splits that freeze the previous test groups;
* :func:`prepare_shards` / :func:`prepare_shards_from_source` — InkML to memory-mapped ``npz``
  shards, once, with a manifest; the source is either an extracted directory or the released
  ``.tgz`` streamed member by member (:mod:`aicls.data.sources`);
* :class:`MathWritingDataset` / :class:`AudioCommandDataset` and their synthetic siblings;
* :func:`synth_ink` / :func:`synth_image` / :func:`synth_audio` / :func:`synth_inkml_dataset` —
  the seeded generators every CPU test and smoke config runs on;
* :func:`collate_ink` / :func:`collate_image` / :func:`collate_audio` — padded batches.

Importing this package pulls in numpy, Pillow, matplotlib (lazily) and :mod:`aicls.ocr`/
:mod:`aicls.asr`, but **never torch**: ``Dataset.to_torch()`` and ``collate_*(to_torch=True)``
import it lazily, and every numpy path works without it.
"""

from __future__ import annotations

from aicls.data.audio_commands import (
    AudioCommandDataset,
    AudioRow,
    AudioSample,
    SyntheticAudioDataset,
)
from aicls.data.catalog import Catalog, CatalogEntry, catalog_entry, load_catalog
from aicls.data.collate import collate_audio, collate_image, collate_ink, pad_sequences
from aicls.data.curriculum import Problem, ProblemExpected, ProblemInput, load_curriculum
from aicls.data.manifest import (
    MANIFEST_FILENAME,
    DataManifest,
    FileEntry,
    VerifyReport,
    load_manifest,
)
from aicls.data.mathwriting import InkSample, MathWritingDataset, SyntheticInkDataset
from aicls.data.shards import (
    SHARD_MANIFEST_FILENAME,
    ShardEntry,
    ShardManifest,
    ShardReader,
    ShardSample,
    build_vocab_from_shards,
    discover_shard_splits,
    load_shard_manifest,
    normalize_label,
    prepare_shards,
    prepare_shards_from_source,
    shard_fingerprints,
    shards_root_fingerprint,
)
from aicls.data.sources import (
    DirectoryInkmlSource,
    InkmlRecord,
    InkmlSource,
    SplitCount,
    TarInkmlSource,
    open_inkml_source,
)
from aicls.data.splits import (
    CORE_SPLITS,
    MATHWRITING_SPLITS,
    SPLITS_FILENAME,
    Splits,
    load_splits,
    make_splits,
    mathwriting_official,
    save_splits,
)
from aicls.data.synthetic import (
    synth_audio,
    synth_image,
    synth_image_dataset,
    synth_ink,
    synth_inkml_dataset,
)

__all__ = [
    "CORE_SPLITS",
    "MANIFEST_FILENAME",
    "MATHWRITING_SPLITS",
    "SHARD_MANIFEST_FILENAME",
    "SPLITS_FILENAME",
    "AudioCommandDataset",
    "AudioRow",
    "AudioSample",
    "Catalog",
    "CatalogEntry",
    "DataManifest",
    "DirectoryInkmlSource",
    "FileEntry",
    "InkSample",
    "InkmlRecord",
    "InkmlSource",
    "MathWritingDataset",
    "Problem",
    "ProblemExpected",
    "ProblemInput",
    "ShardEntry",
    "ShardManifest",
    "ShardReader",
    "ShardSample",
    "SplitCount",
    "Splits",
    "SyntheticAudioDataset",
    "SyntheticInkDataset",
    "TarInkmlSource",
    "VerifyReport",
    "build_vocab_from_shards",
    "catalog_entry",
    "collate_audio",
    "collate_image",
    "collate_ink",
    "discover_shard_splits",
    "load_catalog",
    "load_curriculum",
    "load_manifest",
    "load_shard_manifest",
    "load_splits",
    "make_splits",
    "mathwriting_official",
    "normalize_label",
    "open_inkml_source",
    "pad_sequences",
    "prepare_shards",
    "prepare_shards_from_source",
    "save_splits",
    "shard_fingerprints",
    "shards_root_fingerprint",
    "synth_audio",
    "synth_image",
    "synth_image_dataset",
    "synth_ink",
    "synth_inkml_dataset",
]
