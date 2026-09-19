"""Curriculum LaTeX seed table used for the tokenizer vocabulary and the stub recogniser.

These are canonical *Temel Analiz* (Calculus) expressions taken from the topics of the İP-4
curriculum set (limits, derivatives, integrals, series). They are deliberately written in the
tokenizer's canonical spacing, so ``LatexTokenizer.decode(encode(s)) == s`` holds for every entry
(``test_tokenizer_encode_decode_roundtrip_on_curriculum_latex``).

The real vocabulary for a trained model is built from the MathWriting labels
(``aicls.data.prepare_shards`` -> ``vocab.json``); this table only keeps the torch-free path —
the stub recogniser, the API smoke tests and the evaluation harness — running offline.
"""

from __future__ import annotations

#: Canonical curriculum expressions (order is stable; new entries are appended, never inserted).
CURRICULUM_LATEX: tuple[str, ...] = (
    r"\lim_{x\to0}\frac{\sin x}{x}=1",
    r"\int x^{2}\,dx=\frac{x^{3}}{3}+C",
    r"\frac{d}{dx}\left(x^{2}\right)=2x",
    r"\sum_{n=1}^{\infty}\frac{1}{n^{2}}=\frac{\pi^{2}}{6}",
    r"\lim_{h\to0}\frac{f(x+h)-f(x)}{h}",
    r"\int_{0}^{1}x^{2}dx=\frac{1}{3}",
    r"\frac{\partial f}{\partial x}",
    r"y=\sqrt{x^{2}+1}",
    r"\ln(x)+\log_{10}(x)",
    r"e^{x}\cos(x)",
    r"\tan\theta=\frac{\sin\theta}{\cos\theta}",
    r"f'(x)=2x+3",
    r"\int\frac{1}{x}dx=\ln|x|+C",
    r"\lim_{x\to\infty}\frac{1}{x}=0",
    r"\left[a,b\right]",
    r"\frac{dy}{dx}=3x^{2}-2x",
    r"\sin^{2}(x)+\cos^{2}(x)=1",
    r"\int_{a}^{b}f(x)dx=F(b)-F(a)",
    r"\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n}=e",
    r"\frac{d}{dx}e^{x}=e^{x}",
    r"\int e^{x}dx=e^{x}+C",
    r"\sqrt[3]{x}",
    r"\arctan(x)",
    r"|x-a|<\delta",
    r"\forall\varepsilon>0",
    r"\binom{n}{k}",
    r"\frac{x^{2}-1}{x-1}",
    r"2\pi r",
    r"\int_{0}^{\pi}\sin x\,dx=2",
    r"\frac{d}{dx}\ln x=\frac{1}{x}",
    r"a_{n}=\frac{1}{n^{2}}",
    r"\lim_{x\to0^{+}}\ln x=-\infty",
)

#: Extra single-character tokens always kept in the vocabulary so a model can spell rare labels.
DEFAULT_EXTRA_TOKENS: tuple[str, ...] = tuple(
    list("0123456789")
    + [chr(code) for code in range(ord("a"), ord("z") + 1)]
    + [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    + list("+-*/=<>(),.;:!|'[]{}^_")
)

__all__ = ["CURRICULUM_LATEX", "DEFAULT_EXTRA_TOKENS"]
