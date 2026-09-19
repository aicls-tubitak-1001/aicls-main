/**
 * One tagged textbook unit — a definition, a theorem, an example (İP-8).
 *
 * The body arrives as server-rendered HTML built from the tagged JSON-LD content (ADR-0019), never
 * from user input, which is why it can be injected; the LaTeX blocks that travel beside it go
 * through `MathView`, so the sanitiser and the KaTeX policy apply to book content exactly as they
 * do to a CAS answer. Figures carry the Turkish `alt_tr` the loader stored — a figure without alt
 * text is not shown as a decorative blank, it is shown with its caption.
 *
 * The same component renders inside the book rail and pinned beside the result, so a definition the
 * lecturer pins looks identical to the one they searched for.
 */
import type { ContentUnitOut } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Chip";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { MathView } from "@/features/results/MathView";
import styles from "./BookUnitView.module.css";

/** Turkish label of the unit types the interface knows (`aicls.book.schema.UnitType`). */
const TYPE_LABELS: Record<string, MessageKey> = {
  definition: "book.type.definition",
  theorem: "book.type.theorem",
  example: "book.type.example",
  figure: "book.type.figure",
};

/**
 * Turkish name of a unit type.
 *
 * `ContentUnitOut.type` is an open string on the wire — the book loader may tag a kind before the
 * client knows about it — so an unknown type is shown as it came rather than dropped.
 */
function typeLabel(type: string): string {
  const key = TYPE_LABELS[type];
  return key === undefined ? type : t(key);
}

export interface BookUnitViewProps {
  /** The unit to render. */
  unit: ContentUnitOut;
  /** Heading level to use, so the panel and the board keep a sane outline. */
  headingLevel?: 3 | 4;
  /** Pin the unit beside the result. */
  onPin?: (unit: ContentUnitOut) => void;
  /** Remove the pin (shown instead of "sabitle" when the unit is the pinned one). */
  onUnpin?: () => void;
  /** Load the units one hop away in the graph. */
  onShowRelated?: (unit: ContentUnitOut) => void;
  /** Compact presentation for a dense list. */
  compact?: boolean;
}

/** Render one textbook unit with its LaTeX blocks and figures. */
export function BookUnitView({
  unit,
  headingLevel = 3,
  onPin,
  onUnpin,
  onShowRelated,
  compact = false,
}: BookUnitViewProps): JSX.Element {
  const Heading = headingLevel === 3 ? "h3" : "h4";
  const rendered = unit.rendered;
  const assets = rendered.assets ?? [];
  const blocks = rendered.latex_blocks ?? [];

  return (
    <article className={cx(styles.unit, compact && styles.compact)} data-testid="book-unit">
      <div className={styles.head}>
        <Chip tone="accent">{typeLabel(unit.type)}</Chip>
        <Heading className={styles.title}>{unit.title_tr}</Heading>
      </div>

      <p className={cx(styles.meta, "chrome-secondary")}>
        {unit.number === undefined || unit.number === ""
          ? null
          : t("book.number", { number: unit.number })}
        {unit.chapter === undefined || unit.chapter === "" ? null : (
          <span className={styles.chapter}>{t("book.chapter", { chapter: unit.chapter })}</span>
        )}
      </p>

      {rendered.body_html === "" ? null : (
        <div
          className={styles.body}
          // Book HTML is produced server-side from tagged content, never from user input.
          dangerouslySetInnerHTML={{ __html: rendered.body_html }}
        />
      )}

      {blocks.map((block, index) => (
        <MathView key={`${unit.id}-${String(index)}`} latex={block} display />
      ))}

      {assets.map((asset) => (
        <figure key={asset.id} className={styles.figure}>
          <img src={asset.url} alt={asset.alt_tr === "" ? t("book.figure") : asset.alt_tr} />
          {asset.caption_tr === "" ? null : <figcaption>{asset.caption_tr}</figcaption>}
        </figure>
      ))}

      {onPin === undefined && onUnpin === undefined && onShowRelated === undefined ? null : (
        <div className={styles.actions}>
          {onUnpin === undefined ? null : (
            <Button variant="ghost" onClick={onUnpin}>
              {t("book.unpin")}
            </Button>
          )}
          {onPin === undefined ? null : (
            <Button
              variant="ghost"
              onClick={() => {
                onPin(unit);
              }}
            >
              {t("book.pin")}
            </Button>
          )}
          {onShowRelated === undefined ? null : (
            <Button
              variant="ghost"
              onClick={() => {
                onShowRelated(unit);
              }}
            >
              {t("book.relatedLoad")}
            </Button>
          )}
        </div>
      )}
    </article>
  );
}
