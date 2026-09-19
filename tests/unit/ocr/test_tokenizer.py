"""LaTeX tokenizer tests (vocabulary, special tokens, encode/decode roundtrip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.ocr import LatexTokenizer, OcrInputError
from aicls.ocr.curriculum import CURRICULUM_LATEX
from aicls.ocr.tokenizer import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, default_tokenizer


@pytest.fixture(scope="module")
def tok() -> LatexTokenizer:
    return default_tokenizer()


class TestVocabulary:
    def test_special_tokens_come_first(self, tok: LatexTokenizer) -> None:
        assert SPECIAL_TOKENS == ("<pad>", "<s>", "</s>", "<unk>")
        assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
        assert [tok.id_to_token(i) for i in range(4)] == list(SPECIAL_TOKENS)

    def test_vocab_is_built_from_curriculum_tokens(self) -> None:
        built = LatexTokenizer.build(CURRICULUM_LATEX)
        assert built.vocab_size > len(SPECIAL_TOKENS)
        for command in ("\\frac", "\\int", "\\lim", "\\sum", "\\sqrt"):
            assert command in built.vocab
        assert built.vocab_size == len(built.vocab)

    def test_vocab_is_sorted_and_deterministic(self) -> None:
        first = LatexTokenizer.build(CURRICULUM_LATEX)
        second = LatexTokenizer.build(reversed(list(CURRICULUM_LATEX)))
        assert first.vocab == second.vocab

    def test_min_freq_filters_rare_tokens(self) -> None:
        tokenizer = LatexTokenizer.build(["\\alpha \\alpha", "\\beta"], min_freq=2)
        assert "\\alpha" in tokenizer.vocab
        assert "\\beta" not in tokenizer.vocab

    def test_empty_vocabulary_is_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            LatexTokenizer(tokens=[])


class TestTokenize:
    def test_commands_digits_and_symbols_split(self, tok: LatexTokenizer) -> None:
        assert tok.tokenize("\\frac{12}{x}") == ["\\frac", "{", "1", "2", "}", "{", "x", "}"]

    def test_control_symbols_are_single_tokens(self, tok: LatexTokenizer) -> None:
        assert tok.tokenize("a\\,b") == ["a", "\\,", "b"]
        assert tok.tokenize("\\{x\\}") == ["\\{", "x", "\\}"]

    def test_whitespace_is_not_a_token(self, tok: LatexTokenizer) -> None:
        assert tok.tokenize("  \\sin   x \n") == ["\\sin", "x"]

    def test_input_length_is_capped(self, tok: LatexTokenizer) -> None:
        with pytest.raises(OcrInputError):
            tok.encode("x" * 5000)


class TestEncodeDecode:
    def test_tokenizer_encode_decode_roundtrip_on_curriculum_latex(
        self, tok: LatexTokenizer
    ) -> None:
        for latex in CURRICULUM_LATEX:
            ids = tok.encode(latex)
            assert UNK_ID not in ids, f"curriculum token missing from the vocabulary: {latex}"
            assert tok.decode(ids) == latex
            assert tok.encode(tok.decode(ids)) == ids

    def test_decode_normalises_redundant_whitespace(self, tok: LatexTokenizer) -> None:
        assert tok.decode(tok.encode("\\frac { a } { b }")) == "\\frac{a}{b}"
        assert tok.normalize("  x  ^ { 2 }  ") == "x^{2}"

    def test_normalize_is_idempotent(self, tok: LatexTokenizer) -> None:
        for latex in CURRICULUM_LATEX:
            assert tok.normalize(tok.normalize(latex)) == tok.normalize(latex)

    def test_special_tokens_can_wrap_a_sequence(self, tok: LatexTokenizer) -> None:
        ids = tok.encode("x^{2}", add_special=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID
        assert tok.decode(ids) == "x^{2}"

    def test_decode_stops_at_eos(self, tok: LatexTokenizer) -> None:
        ids = tok.encode("x^{2}", add_special=True) + tok.encode("y")
        assert tok.decode(ids) == "x^{2}"

    def test_decode_can_keep_special_tokens(self, tok: LatexTokenizer) -> None:
        out = tok.decode([BOS_ID, *tok.encode("x"), EOS_ID], skip_special=False)
        assert out.startswith("<s>")
        assert out.endswith("</s>")

    def test_unknown_tokens_map_to_unk(self, tok: LatexTokenizer) -> None:
        ids = tok.encode("\\thisisnotacommand")
        assert ids == [UNK_ID]
        assert tok.decode(ids) == ""

    def test_out_of_range_ids_are_rejected(self, tok: LatexTokenizer) -> None:
        with pytest.raises(OcrInputError):
            tok.decode([tok.vocab_size + 5])

    def test_max_len_truncation(self, tok: LatexTokenizer) -> None:
        ids = tok.encode("x^{2}+y^{2}", max_len=4)
        assert len(ids) == 4


class TestPersistence:
    def test_save_and_from_file_roundtrip(self, tmp_path: Path, tok: LatexTokenizer) -> None:
        path = tmp_path / "vocab.json"
        tok.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["specials"] == list(SPECIAL_TOKENS)

        restored = LatexTokenizer.from_file(path)
        assert restored.vocab == tok.vocab
        assert restored.encode("\\frac{1}{2}") == tok.encode("\\frac{1}{2}")

    def test_from_file_accepts_a_plain_token_list(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.json"
        path.write_text(json.dumps(["x", "^", "{", "2", "}"]), encoding="utf-8")
        restored = LatexTokenizer.from_file(path)
        assert restored.decode(restored.encode("x^{2}")) == "x^{2}"

    def test_from_file_rejects_a_missing_file(self, tmp_path: Path) -> None:
        from aicls.core.errors import NotFoundError

        with pytest.raises(NotFoundError):
            LatexTokenizer.from_file(tmp_path / "nope.json")

    def test_from_file_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(OcrInputError):
            LatexTokenizer.from_file(path)

    def test_default_tokenizer_is_cached(self) -> None:
        assert default_tokenizer() is default_tokenizer()
