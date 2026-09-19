/**
 * The worked examples of the usage guide.
 *
 * These are real Temel Analiz problems, not placeholders: a reader can copy the Turkish command
 * into the command bar and get the printed answer back. The Turkish sentences are the ones the
 * command inventory actually resolves (`datasets/nlu/commands_test.jsonl`), the CAS command column
 * is what the engine receives, and the LaTeX is what the result panel prints — so the example and
 * the product cannot drift apart without a test noticing.
 *
 * LaTeX and CAS commands stay here rather than in `tr.json`: they are content, identical in every
 * locale, and putting a backslash-heavy string through the message catalogue would only make both
 * harder to read. The Turkish prose around them lives in `tr.json` as usual.
 */
import type { MessageKey } from "@/i18n";

/** One end-to-end example: what the lecturer says, what the engine gets, what the class sees. */
export interface WorkedExample {
  readonly id: string;
  /** "Yazarak" / "Söyleyerek" / "El yazısıyla". */
  readonly modeKey: MessageKey;
  /** What the lecturer physically does. */
  readonly howKey: MessageKey;
  /** The Turkish command, copyable. */
  readonly commandKey: MessageKey;
  /** The expression the system understood, as LaTeX. */
  readonly inputLatex: string;
  /** The typed command handed to the computer algebra system. */
  readonly casCommand: string;
  /** The answer, as LaTeX. */
  readonly resultLatex: string;
  /** The same answer in the accessible plain form the API returns next to the LaTeX. */
  readonly resultPlain: string;
  /** A short remark about this modality. */
  readonly noteKey: MessageKey;
}

/** The three input modalities, one worked example each. */
export const WORKED_EXAMPLES: readonly WorkedExample[] = [
  {
    id: "yazarak",
    modeKey: "guide.usage.inputs.typing.mode",
    howKey: "guide.usage.inputs.typing.how",
    commandKey: "guide.example.derivative.text",
    inputLatex: "x^{2}",
    casCommand: "diff(x**2, x)",
    resultLatex: "2 x",
    resultPlain: "2*x",
    noteKey: "guide.usage.inputs.typing.note",
  },
  {
    id: "soyleyerek",
    modeKey: "guide.usage.inputs.speech.mode",
    howKey: "guide.usage.inputs.speech.how",
    commandKey: "guide.example.integral.text",
    inputLatex: "\\int_{0}^{1} x^{2}\\,dx",
    casCommand: "integrate(x**2, (x, 0, 1))",
    resultLatex: "\\frac{1}{3}",
    resultPlain: "1/3",
    noteKey: "guide.usage.inputs.speech.note",
  },
  {
    id: "el-yazisiyla",
    modeKey: "guide.usage.inputs.ink.mode",
    howKey: "guide.usage.inputs.ink.how",
    commandKey: "guide.example.limit.text",
    inputLatex: "\\lim_{x \\to 0} \\frac{\\sin x}{x}",
    casCommand: "limit(sin(x)/x, x, 0)",
    resultLatex: "1",
    resultPlain: "1",
    noteKey: "guide.usage.inputs.ink.note",
  },
];

/** The clarification example: one command, two readings, no guessing. */
export const CLARIFICATION_EXAMPLE = {
  commandKey: "guide.usage.clarify.text",
  questionKey: "guide.usage.clarify.question",
  inputLatex: "x^{2} + y^{2}",
  candidates: [
    { labelKey: "guide.usage.clarify.candidateX", latex: "2 x", command: "diff(x**2 + y**2, x)" },
    { labelKey: "guide.usage.clarify.candidateY", latex: "2 y", command: "diff(x**2 + y**2, y)" },
  ],
} as const satisfies {
  commandKey: MessageKey;
  questionKey: MessageKey;
  inputLatex: string;
  candidates: readonly { labelKey: MessageKey; latex: string; command: string }[];
};
