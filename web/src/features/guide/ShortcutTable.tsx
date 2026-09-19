/**
 * The documented keyboard set, rendered from the binding table itself.
 *
 * The rows come from `app/shortcuts.ts`, not from a copy: a shortcut added to the application
 * appears here without anyone remembering to update the guide, and a shortcut removed from the
 * application cannot linger in the documentation. This is the same list the in-app help dialog
 * shows — one table, one source.
 */
import { shortcutSummary } from "@/app/shortcuts";
import { t } from "@/i18n";
import tables from "./tables.module.css";

/** The keyboard shortcut reference. */
export function ShortcutTable(): JSX.Element {
  const rows = shortcutSummary();
  return (
    <div className={`${tables.wrap} scroll-x`}>
      <table className={tables.table} data-testid="guide-shortcut-table">
        <caption className={tables.caption}>{t("shortcuts.title")}</caption>
        <thead>
          <tr>
            <th scope="col">{t("guide.usage.shortcuts.colKey")}</th>
            <th scope="col">{t("guide.usage.shortcuts.colAction")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <th scope="row">
                <kbd className={tables.kbd}>{row.display}</kbd>
              </th>
              <td>{t(row.labelKey)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
