"""Deterministic ASR-like corruption of Turkish command sentences.

``datasets/nlu/asr_noisy_tr.jsonl`` is generated from ``commands_test.jsonl`` with these
functions, and ``test_nlu_asr_noisy_set_intent_accuracy_above_0_90`` measures how much intent
accuracy survives them. They imitate what a Whisper-class decoder actually does to Turkish maths
dictation (docs/ml/degerlendirme.md, İP-3 error analysis):

* it drops or mangles case endings (``türevini`` → ``türevi``, ``türev``);
* it writes a slowly dictated ending as its own word (``karenin`` → ``kare nin``);
* it loses the apostrophe of ``x'in``;
* it mis-hears near-homophones (``türev`` → ``türef``, ``integral`` → ``entegral``);
* it inserts hesitation words (``yani``, ``şey``);
* it joins two words (``bir bölü`` → ``birbölü``);
* it confuses ``I``/``İ`` casing.

Every function is pure and every choice is seeded, so the generated dataset is reproducible
byte-for-byte (seed 1234 by default, ADR-0010).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import random

#: Near-homophone substitutions an ASR decoder is likely to make on Turkish maths vocabulary.
MISSPELLINGS: dict[str, str] = {
    "türev": "türef",
    "integral": "entegral",
    "sinüs": "sinus",
    "kosinüs": "kosinus",
    "tanjant": "tanjand",
    "eşittir": "esittir",
    "çarpı": "carpi",
    "bölü": "bolu",
    "karekök": "kare kök",
    "sıfır": "sifir",
    "grafik": "grafig",
    "sadeleştir": "sadelestir",
    "çarpanlarına": "carpanlarina",
}

#: Case endings the decoder may shorten (long form -> short form, "" drops the ending). Every
#: replacement must yield a *pronounceable* Turkish form: a decoder mishears an ending, it does
#: not produce letter salad, so "ifadesinin" -> "ifadesi" is in and "ifadesiin" is not.
SUFFIX_SHORTENINGS: tuple[tuple[str, str], ...] = (
    ("ini", "i"),
    ("ını", "ı"),
    ("unu", "u"),
    ("ünü", "ü"),
    ("sinin", "sini"),
    ("sının", "sını"),
    ("unun", "unu"),
    ("ünün", "ünü"),
    ("nin", ""),
    ("nın", ""),
    ("nun", ""),
    ("nün", ""),
)

#: Content stems whose ending a slow dictation may transcribe as a separate word. Splitting at a
#: real morpheme boundary is what an ASR decoder does; splitting mid-morpheme ("ifades ini") is
#: not, so the stems are listed explicitly instead of chopping a fixed number of characters.
DETACHABLE_STEMS = (
    "türev",
    "integral",
    "limit",
    "denklem",
    "ifade",
    "fonksiyon",
    "grafik",
    "değer",
    "açılım",
    "tanım",
    "teorem",
    "çarpan",
    "kare",
    "küp",
    "kök",
    "tahta",
    "ekran",
    "komut",
    "işlem",
    "sonuç",
    "çalışma",
)

#: Hesitation words an ASR decoder happily transcribes.
FILLERS = ("yani", "şey", "işte", "hani")

#: Turkish capitalisation of the two ``i`` letters (Python's ``str.upper`` gets these wrong).
TURKISH_UPPER = {"i": "İ", "ı": "I"}


def drop_apostrophe(text: str, rng: random.Random) -> str | None:
    """``x'in`` → ``x in``."""
    if "'" not in text:
        return None
    return text.replace("'", " ")


def detach_suffix(text: str, rng: random.Random) -> str | None:
    """``karenin`` → ``kare nin`` (the ending becomes its own word, at a real boundary)."""
    words = text.split()
    options = [
        (index, stem)
        for index, word in enumerate(words)
        for stem in DETACHABLE_STEMS
        if word.startswith(stem) and len(word) > len(stem) + 1
    ]
    if not options:
        return None
    index, stem = options[rng.randrange(len(options))]
    word = words[index]
    words[index] = f"{stem} {word[len(stem) :]}"
    return " ".join(words)


def shorten_suffix(text: str, rng: random.Random) -> str | None:
    """``türevini`` → ``türevi`` (the decoder loses part of the ending)."""
    words = text.split()
    options = [
        (index, long, short)
        for index, word in enumerate(words)
        for long, short in SUFFIX_SHORTENINGS
        if len(word) > len(long) + 2 and word.endswith(long)
    ]
    if not options:
        return None
    index, long, short = options[rng.randrange(len(options))]
    word = words[index]
    words[index] = word[: -len(long)] + short
    return " ".join(words)


def misspell(text: str, rng: random.Random) -> str | None:
    """Replace one maths word by its near-homophone."""
    options = [word for word in MISSPELLINGS if word in text]
    if not options:
        return None
    word = options[rng.randrange(len(options))]
    return text.replace(word, MISSPELLINGS[word], 1)


def insert_filler(text: str, rng: random.Random) -> str | None:
    """Insert a hesitation word at a word boundary."""
    words = text.split()
    if len(words) < 2:
        return None
    position = rng.randrange(1, len(words))
    filler = FILLERS[rng.randrange(len(FILLERS))]
    return " ".join([*words[:position], filler, *words[position:]])


def join_words(text: str, rng: random.Random) -> str | None:
    """Join two adjacent words, as a fast dictation is often transcribed."""
    words = text.split()
    if len(words) < 3:
        return None
    position = rng.randrange(len(words) - 1)
    joined = words[position] + words[position + 1]
    return " ".join([*words[:position], joined, *words[position + 2 :]])


def capitalise(text: str, rng: random.Random) -> str | None:
    """Capitalise one word the *Turkish* way (``i`` → ``İ``, ``ı`` → ``I``).

    A Turkish ASR decoder capitalises sentence starts and proper nouns with the dotted/dotless
    pair, which is exactly the case ``aicls.core.turkish.tr_casefold`` exists for: the default
    Unicode mapping would turn ``İ`` into ``i`` + combining dot and break every lexicon lookup.
    """
    words = text.split()
    if not words:
        return None
    position = rng.randrange(len(words))
    word = words[position]
    if not word:
        return None
    first = TURKISH_UPPER.get(word[0], word[0].upper())
    words[position] = first + word[1:]
    swapped = " ".join(words)
    return swapped if swapped != text else None


#: Every perturbation with the relative frequency assumed for a Turkish maths ASR decoder.
#:
#: The weights are an **assumption**, not a measurement: no fine-tuned decoder exists yet, so
#: they encode the error mix reported for Whisper-class models on inflected languages (endings
#: first, homophones second, word-boundary errors rarely). They must be revisited against real
#: decoder output before any number from ``asr_noisy_tr.jsonl`` is quoted in a report.
PERTURBATIONS: tuple[tuple[str, Callable[[str, random.Random], str | None], int], ...] = (
    ("drop_apostrophe", drop_apostrophe, 20),
    ("detach_suffix", detach_suffix, 20),
    ("shorten_suffix", shorten_suffix, 20),
    ("misspell", misspell, 20),
    ("insert_filler", insert_filler, 10),
    ("capitalise", capitalise, 5),
    ("join_words", join_words, 5),
)


def perturb(text: str, *, seed: int = 1234, count: int = 1) -> tuple[str, list[str]]:
    """Apply ``count`` perturbations to ``text``.

    Args:
        text: The clean sentence.
        seed: Seed for the choice of perturbations (deterministic).
        count: How many perturbations to apply, at most.

    Returns:
        ``(perturbed_text, applied_names)``; ``applied_names`` is empty when nothing applied.
    """
    rng = random.Random(seed)
    names = [name for name, _, _ in PERTURBATIONS]
    weights = [weight for _, _, weight in PERTURBATIONS]
    functions = {name: function for name, function, _ in PERTURBATIONS}
    order: list[str] = []
    pool, pool_weights = list(names), list(weights)
    while pool:
        picked = rng.choices(pool, weights=pool_weights, k=1)[0]
        index = pool.index(picked)
        pool.pop(index)
        pool_weights.pop(index)
        order.append(picked)
    applied: list[str] = []
    current = text
    for name in order:
        if len(applied) >= count:
            break
        result = functions[name](current, rng)
        if result is None or result == current:
            continue
        current = result
        applied.append(name)
    return current, applied


def perturb_variants(text: str, *, seed: int = 1234, n: int = 3) -> list[tuple[str, list[str]]]:
    """Return ``n`` distinct perturbed variants of ``text`` (fewer if it cannot vary that much)."""
    seen: dict[str, list[str]] = {}
    for offset in range(n * 4):
        variant, applied = perturb(text, seed=seed + offset, count=1)
        if variant != text and variant not in seen:
            seen[variant] = applied
        if len(seen) >= n:
            break
    return [(variant, applied) for variant, applied in seen.items()]


__all__: Sequence[str] = [
    "DETACHABLE_STEMS",
    "FILLERS",
    "MISSPELLINGS",
    "PERTURBATIONS",
    "SUFFIX_SHORTENINGS",
    "TURKISH_UPPER",
    "capitalise",
    "detach_suffix",
    "drop_apostrophe",
    "insert_filler",
    "join_words",
    "misspell",
    "perturb",
    "perturb_variants",
    "shorten_suffix",
]
