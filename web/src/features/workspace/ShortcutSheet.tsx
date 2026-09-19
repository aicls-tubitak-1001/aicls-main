/**
 * The discoverable half of the keyboard set.
 *
 * A shortcut nobody can find is not a feature, so the table lives in one place (`app/shortcuts.ts`)
 * and is rendered here verbatim — the help sheet and the handlers can never drift apart. The three
 * bindings the workspace owns itself (push-to-talk, submit, cancel) are appended, because from the
 * lecturer's side they are part of the same set.
 *
 * This is the one modal in the workspace, and it is deliberate: the lecturer asked for it with `?`,
 * it traps focus only while it is open, Escape closes it, and focus returns to where it was.
 */
import { Dialog } from "@/components/ui/Dialog";
import { shortcutSummary } from "@/app/shortcuts";
import { t, type MessageKey } from "@/i18n";
import styles from "./ShortcutSheet.module.css";

/** Bindings handled by the workspace itself rather than by the global table. */
const LOCAL_BINDINGS: { display: string; labelKey: MessageKey }[] = [
  { display: "Space", labelKey: "shortcuts.pushToTalk" },
  { display: "Ctrl + Enter", labelKey: "shortcuts.submit" },
  { display: "Esc", labelKey: "shortcuts.cancel" },
];

export interface ShortcutSheetProps {
  open: boolean;
  onClose: () => void;
}

/** The keyboard reference sheet. */
export function ShortcutSheet({ open, onClose }: ShortcutSheetProps): JSX.Element {
  const rows = [
    ...shortcutSummary().map((item) => ({ display: item.display, label: t(item.labelKey) })),
    ...LOCAL_BINDINGS.map((item) => ({ display: item.display, label: t(item.labelKey) })),
  ];

  return (
    <Dialog
      open={open}
      title={t("shortcuts.title")}
      description={t("shortcuts.hint")}
      onClose={onClose}
    >
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">{t("shortcuts.keyColumn")}</th>
            <th scope="col">{t("shortcuts.actionColumn")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.display}-${row.label}`}>
              <th scope="row">
                <kbd className={styles.key}>{row.display}</kbd>
              </th>
              <td>{row.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Dialog>
  );
}
