/**
 * The campaign counter: how much has been collected so far, all volunteers together.
 *
 * `GET /contribute/stats` is a **staff** endpoint (`StaffUser` in
 * `aicls.api.routers.contribute.contribution_stats`), so this panel is only mounted for an
 * instructor or an administrator. A volunteer is shown nothing rather than a 403 they can do
 * nothing about — and the one line explaining why is in the panel for the staff member reading it.
 */
import { useCallback, useEffect, useState } from "react";

import type { Schema } from "@/api/schema";
import { Card, Chip } from "@/components/ui";
import { t } from "@/i18n";
import { fetchContributionStats } from "./api";
import styles from "./CommunityCounter.module.css";

/** Round to one decimal without dragging in a formatting library. */
function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

/** Community totals, for staff. */
export function CommunityCounter(): JSX.Element | null {
  const [stats, setStats] = useState<Schema<"ContribStatsOut"> | null>(null);

  const load = useCallback(async (): Promise<void> => {
    const { data } = await fetchContributionStats();
    setStats(data);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (stats === null) {
    return null;
  }

  return (
    <Card title={t("contribute.communityTitle")} muted>
      <ul className={styles.list}>
        <li>
          <Chip tone="accent" large>
            {t("contribute.communityHours", { hours: round1(stats.hours_accepted) })}
          </Chip>
        </li>
        <li>
          <Chip tone="neutral">
            {t("contribute.communityUtterances", { count: stats.utterances })}
          </Chip>
        </li>
        <li>
          <Chip tone="neutral">{t("contribute.communitySpeakers", { count: stats.speakers })}</Chip>
        </li>
      </ul>
      <p className={styles.hint}>{t("contribute.communityHint")}</p>
    </Card>
  );
}
