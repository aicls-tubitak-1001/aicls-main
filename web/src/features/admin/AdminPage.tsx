/**
 * The administration route (`/yonetim`, api.md §3.11).
 *
 * It is deliberately plain: seven panels behind a tab strip, each one a thin, honest view of one
 * endpoint. Two rules shape it.
 *
 * * **The role gate is part of the page, not of the route table.** {@link RequireRole} wraps the
 *   content here, so mounting `AdminPage` anywhere — including a unit test — refuses a student
 *   with the same Turkish explanation the router would show, and the admin-only tabs
 *   (registry, evaluations, users, book) are not even offered to an instructor.
 * * **Switching a tab is not navigation.** The URL does not change and nothing else on screen
 *   moves, because an administrator is usually reading a queue while something else is running.
 *
 * The tab strip follows the ARIA tabs pattern: arrow keys move between tabs, Home/End jump to the
 * ends, and each panel is labelled by its tab.
 */
import { useMemo, useState, type KeyboardEvent } from "react";

import { useAuth } from "@/app/auth";
import { RequireRole } from "@/app/guards";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { BookLoadPanel } from "./BookLoadPanel";
import { ContribStatsPanel } from "./ContribStatsPanel";
import { EvalRunsPanel } from "./EvalRunsPanel";
import { LatencyPanel } from "./LatencyPanel";
import { ModelRegistryPanel } from "./ModelRegistryPanel";
import { QcQueuePanel } from "./QcQueuePanel";
import { ADMIN_ROLES } from "./roles";
import { UsersPanel } from "./UsersPanel";
import styles from "./Admin.module.css";

/** One tab of the administration area. */
interface AdminTab {
  readonly id: string;
  readonly label: MessageKey;
  /** `true` when only an administrator may see it. */
  readonly adminOnly: boolean;
  readonly render: () => JSX.Element;
}

const TABS: readonly AdminTab[] = [
  { id: "qc", label: "admin.tab.qc", adminOnly: false, render: () => <QcQueuePanel /> },
  { id: "stats", label: "admin.tab.stats", adminOnly: false, render: () => <ContribStatsPanel /> },
  { id: "latency", label: "admin.tab.latency", adminOnly: false, render: () => <LatencyPanel /> },
  {
    id: "models",
    label: "admin.tab.models",
    adminOnly: true,
    render: () => <ModelRegistryPanel />,
  },
  { id: "evals", label: "admin.tab.evals", adminOnly: true, render: () => <EvalRunsPanel /> },
  { id: "users", label: "admin.tab.users", adminOnly: true, render: () => <UsersPanel /> },
  { id: "book", label: "admin.tab.book", adminOnly: true, render: () => <BookLoadPanel /> },
];

/** The tabbed administration surface (already inside the role gate). */
function AdminTabs(): JSX.Element {
  const { can } = useAuth();
  const isAdmin = can("admin");
  const tabs = useMemo(() => TABS.filter((tab) => !tab.adminOnly || isAdmin), [isAdmin]);
  const [active, setActive] = useState<string>(tabs[0]?.id ?? "qc");

  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const index = tabs.findIndex((tab) => tab.id === active);
    if (index === -1) {
      return;
    }
    let next = index;
    if (event.key === "ArrowRight") {
      next = (index + 1) % tabs.length;
    } else if (event.key === "ArrowLeft") {
      next = (index - 1 + tabs.length) % tabs.length;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = tabs.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    const target = tabs[next];
    if (target !== undefined) {
      setActive(target.id);
      document.getElementById(`admin-tab-${target.id}`)?.focus();
    }
  };

  return (
    <>
      <div
        className={styles.tabs}
        role="tablist"
        aria-label={t("admin.sections")}
        onKeyDown={onKeyDown}
      >
        {tabs.map((tab) => {
          const selected = current?.id === tab.id;
          return (
            <button
              key={tab.id}
              id={`admin-tab-${tab.id}`}
              className={cx(styles.tab, selected && styles.tabActive)}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`admin-panel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => {
                setActive(tab.id);
              }}
            >
              {t(tab.label)}
            </button>
          );
        })}
      </div>

      {current === undefined ? null : (
        <div
          className={styles.panel}
          id={`admin-panel-${current.id}`}
          role="tabpanel"
          aria-labelledby={`admin-tab-${current.id}`}
          tabIndex={0}
        >
          {current.render()}
        </div>
      )}

      {isAdmin ? null : <p className={styles.muted}>{t("admin.adminOnly")}</p>}
    </>
  );
}

/** The `/yonetim` route. */
export function AdminPage(): JSX.Element {
  return (
    <section className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>{t("admin.title")}</h1>
        <p className={styles.lead}>{t("admin.intro")}</p>
      </header>

      <RequireRole roles={ADMIN_ROLES}>
        <AdminTabs />
      </RequireRole>
    </section>
  );
}
