/**
 * The Matplotlib plot a `plot` command produced.
 *
 * The PNG comes from `GET /compute/plots/{asset_id}.png`, which is owner-scoped on the server; the
 * URL always arrives inside the result envelope, so this component never builds one itself. It is a
 * plain `<img>` because the plot is server-rendered and deterministic — there is nothing for the
 * client to draw.
 *
 * Accessibility is the reason for the extra props: a chart with `alt="grafik"` is invisible to a
 * student using a screen reader, so the caller passes the result's `plain` form and the alt text
 * becomes "…ifadesinin grafiği". A load failure says so in Turkish instead of leaving a broken
 * image icon on the board.
 */
import { useState } from "react";

import { t } from "@/i18n";
import styles from "./PlotView.module.css";

export interface PlotViewProps {
  /** Absolute or app-relative URL of the PNG; `null` when the result has no plot. */
  url: string | null;
  /** Turkish alt text; defaults to the generic plot description. */
  alt?: string;
}

/** Render the plot, a placeholder, or a load error. */
export function PlotView({ url, alt }: PlotViewProps): JSX.Element {
  const [failed, setFailed] = useState(false);

  if (url === null) {
    return <p className={styles.empty}>{t("plot.empty")}</p>;
  }
  if (failed) {
    return (
      <p className={styles.empty} role="status">
        {t("errors.network")}
      </p>
    );
  }
  return (
    <figure className={styles.figure}>
      <img
        className={styles.image}
        src={url}
        alt={alt ?? t("plot.alt")}
        loading="lazy"
        decoding="async"
        onError={() => {
          setFailed(true);
        }}
      />
      <figcaption className={styles.caption}>
        <span>{t("plot.title")}</span>
        <a className={styles.link} href={url} target="_blank" rel="noreferrer">
          {t("plot.open")}
        </a>
      </figcaption>
    </figure>
  );
}
