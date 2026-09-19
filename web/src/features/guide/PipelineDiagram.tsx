/**
 * The contractual pipeline, drawn.
 *
 * Inline SVG rather than an image file for three reasons: the labels are Turkish text that has to
 * be searchable, selectable and readable by a screen reader; the colours come from the design
 * tokens, so the drawing follows the light/dark theme and the more-contrast preference without a
 * second asset; and it prints as vector, which matters for a handout.
 *
 * Accessibility: the figure is an `img` with a title and a description, and the same information
 * is repeated underneath as an ordered list of stages. The list is not a fallback — it is the
 * canonical text form, and it is what a reader gets on paper if the drawing is cropped.
 *
 * Geometry is written out rather than computed: a hand-placed diagram of nine boxes stays legible,
 * and a layout engine here would be more code than the picture.
 */
import { t, type MessageKey } from "@/i18n";
import styles from "./PipelineDiagram.module.css";

interface DiagramNode {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly titleKey: MessageKey;
  readonly subKey: MessageKey;
  /** Deterministic core boxes are drawn in the accent colour. */
  readonly core?: boolean;
}

const BOX_H = 62;

const NODES: readonly DiagramNode[] = [
  { x: 10, y: 26, w: 130, titleKey: "guide.diagram.ink", subKey: "guide.diagram.inkSub" },
  { x: 10, y: 118, w: 130, titleKey: "guide.diagram.voice", subKey: "guide.diagram.voiceSub" },
  { x: 10, y: 210, w: 130, titleKey: "guide.diagram.typing", subKey: "guide.diagram.typingSub" },
  { x: 190, y: 26, w: 150, titleKey: "guide.diagram.ocr", subKey: "guide.diagram.ocrSub" },
  { x: 190, y: 118, w: 150, titleKey: "guide.diagram.asr", subKey: "guide.diagram.asrSub" },
  {
    x: 430,
    y: 118,
    w: 170,
    titleKey: "guide.diagram.nlu",
    subKey: "guide.diagram.nluSub",
    core: true,
  },
  {
    x: 650,
    y: 118,
    w: 170,
    titleKey: "guide.diagram.parser",
    subKey: "guide.diagram.parserSub",
    core: true,
  },
  {
    x: 870,
    y: 118,
    w: 150,
    titleKey: "guide.diagram.cas",
    subKey: "guide.diagram.casSub",
    core: true,
  },
  { x: 1070, y: 118, w: 170, titleKey: "guide.diagram.render", subKey: "guide.diagram.renderSub" },
];

const STEP_KEYS = [
  "guide.diagram.step1",
  "guide.diagram.step2",
  "guide.diagram.step3",
  "guide.diagram.step4",
  "guide.diagram.step5",
  "guide.diagram.step6",
  "guide.diagram.step7",
] as const;

/** The pipeline figure: SVG plus its text equivalent. */
export function PipelineDiagram(): JSX.Element {
  const titleId = "aicls-pipeline-baslik";
  const descId = "aicls-pipeline-aciklama";

  return (
    <figure className={styles.figure}>
      <div
        className={`${styles.canvas} scroll-x`}
        role="group"
        aria-label={t("guide.diagram.title")}
        tabIndex={0}
      >
        <svg
          className={styles.svg}
          viewBox="0 0 1260 300"
          role="img"
          aria-labelledby={`${titleId} ${descId}`}
          preserveAspectRatio="xMinYMid meet"
        >
          <title id={titleId}>{t("guide.diagram.title")}</title>
          <desc id={descId}>{t("guide.diagram.desc")}</desc>

          <defs>
            <marker
              id="aicls-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" className={styles.arrowHead} />
            </marker>
            <marker
              id="aicls-arrow-core"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" className={styles.arrowHeadCore} />
            </marker>
          </defs>

          {/* Handwriting lane: ink to OCR, OCR to the command layer as LaTeX. */}
          <path className={styles.edge} markerEnd="url(#aicls-arrow)" d="M 140 57 H 184" />
          <path
            className={styles.edge}
            markerEnd="url(#aicls-arrow)"
            d="M 340 57 H 400 V 149 H 424"
          />
          <text className={styles.edgeLabel} x="370" y="44" textAnchor="middle">
            {t("guide.diagram.edge.latex")}
          </text>

          {/* Voice lane: microphone to ASR, ASR to the command layer as Turkish text. */}
          <path className={styles.edge} markerEnd="url(#aicls-arrow)" d="M 140 149 H 184" />
          <path className={styles.edge} markerEnd="url(#aicls-arrow)" d="M 340 149 H 424" />
          <text className={styles.edgeLabel} x="384" y="140" textAnchor="middle">
            {t("guide.diagram.edge.text")}
          </text>

          {/* Typed lane: straight into the command layer. */}
          <path
            className={styles.edge}
            markerEnd="url(#aicls-arrow)"
            d="M 140 241 H 400 V 149 H 424"
          />
          <text className={styles.edgeLabel} x="268" y="232" textAnchor="middle">
            {t("guide.diagram.edge.text")}
          </text>

          {/* Deterministic core. */}
          <path
            className={styles.edgeCore}
            markerEnd="url(#aicls-arrow-core)"
            d="M 600 149 H 644"
          />
          <text className={styles.edgeLabel} x="625" y="106" textAnchor="middle">
            {t("guide.diagram.edge.command")}
          </text>
          <path
            className={styles.edgeCore}
            markerEnd="url(#aicls-arrow-core)"
            d="M 820 149 H 864"
          />
          <text className={styles.edgeLabel} x="845" y="106" textAnchor="middle">
            {t("guide.diagram.edge.sympy")}
          </text>
          <path
            className={styles.edgeCore}
            markerEnd="url(#aicls-arrow-core)"
            d="M 1020 149 H 1064"
          />
          <text className={styles.edgeLabel} x="1044" y="106" textAnchor="middle">
            {t("guide.diagram.edge.result")}
          </text>

          {NODES.map((node) => (
            <g key={node.titleKey}>
              <rect
                className={node.core === true ? styles.boxCore : styles.box}
                x={node.x}
                y={node.y}
                width={node.w}
                height={BOX_H}
                rx="10"
              />
              <text
                className={node.core === true ? styles.boxTitleCore : styles.boxTitle}
                x={node.x + node.w / 2}
                y={node.y + 27}
                textAnchor="middle"
              >
                {t(node.titleKey)}
              </text>
              <text
                className={styles.boxSub}
                x={node.x + node.w / 2}
                y={node.y + 46}
                textAnchor="middle"
              >
                {t(node.subKey)}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <figcaption className={styles.caption}>{t("guide.tech.pipeline.figCaption")}</figcaption>

      <div className={styles.textAlt}>
        <h3 className={styles.textAltTitle}>{t("guide.tech.pipeline.textAlt")}</h3>
        <ol className={styles.steps}>
          {STEP_KEYS.map((key) => (
            <li key={key}>{t(key)}</li>
          ))}
        </ol>
      </div>
    </figure>
  );
}
