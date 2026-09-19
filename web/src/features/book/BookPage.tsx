/**
 * Standalone book route: free search over the tagged textbook content (İP-8).
 *
 * The same panel appears inside the workspace when a command resolves to book content; here it is
 * the whole page so a student can browse definitions and theorems without running a computation.
 * The route owns the page's single `<h1>` — the panel's own heading is an `<h2>`, and inside the
 * workspace the `<h1>` belongs to the board — so the document keeps one top-level heading either
 * way. It is visually hidden because the panel's title already names the surface on screen.
 */
import { t } from "@/i18n";
import { BookPanel } from "./BookPanel";

/** The book route. */
export function BookPage(): JSX.Element {
  return (
    <>
      <h1 className="visually-hidden">{t("book.pageHeading")}</h1>
      <BookPanel />
    </>
  );
}
