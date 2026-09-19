/**
 * Accounts and roles (`GET /admin/users`, `PATCH /admin/users/{id}/role`).
 *
 * A role change is a security decision, so it is made with a labelled `<select>` per account
 * rather than with a row of icon buttons, it is confirmed in words afterwards, and the audited
 * result is read back from the API (the list is refetched) instead of being assumed. The four
 * roles are exactly the ones api.md §1 defines.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { Role, UserOut } from "@/api/types";
import { ROLE_LABEL_KEYS, ROLES } from "@/app/auth";
import { Button, Card, EmptyState, ErrorState, Select, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { fetchUsers, setUserRole } from "./api";
import styles from "./Admin.module.css";

/** The account list. */
export function UsersPanel(): JSX.Element {
  const [users, setUsers] = useState<UserOut[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async (next?: string): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchUsers(next);
    setLoading(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setProblem(null);
    setUsers((current) => (next === undefined ? data.items : [...current, ...data.items]));
    setCursor(data.next_cursor ?? null);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const changeRole = async (user: UserOut, role: Role): Promise<void> => {
    setBusyId(user.id);
    setProblem(null);
    const { problem: failure } = await setUserRole(user.id, role);
    setBusyId(null);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    setNotice(
      t("admin.usersRoleChanged", {
        name: user.display_name,
        role: t(ROLE_LABEL_KEYS[role]),
      }),
    );
    await load();
  };

  return (
    <Card
      title={t("admin.usersTitle")}
      actions={
        <Button
          variant="ghost"
          onClick={() => {
            void load();
          }}
        >
          {t("admin.refresh")}
        </Button>
      }
    >
      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}
      {notice === null ? null : (
        <p className={styles.notice} role="status">
          {notice}
        </p>
      )}
      {loading && users.length === 0 ? <Spinner label={t("common.loading")} /> : null}
      {!loading && users.length === 0 ? <EmptyState title={t("admin.usersEmpty")} /> : null}

      {users.length === 0 ? null : (
        <div className="scroll-x">
          <table className={styles.table}>
            <caption className="visually-hidden">
              {t("admin.tableCaption", { title: t("admin.usersTitle") })}
            </caption>
            <thead>
              <tr>
                <th scope="col">{t("admin.usersName")}</th>
                <th scope="col">{t("admin.usersEmail")}</th>
                <th scope="col">{t("admin.usersRole")}</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id}>
                  <th scope="row">{user.display_name}</th>
                  <td>{user.email}</td>
                  <td>
                    <label className="visually-hidden" htmlFor={`role-${user.id}`}>
                      {t("admin.usersRoleChange", { name: user.display_name })}
                    </label>
                    <Select
                      id={`role-${user.id}`}
                      value={user.role}
                      disabled={busyId === user.id}
                      onChange={(event) => {
                        void changeRole(user, event.target.value as Role);
                      }}
                    >
                      {ROLES.map((role) => (
                        <option key={role} value={role}>
                          {t(ROLE_LABEL_KEYS[role])}
                        </option>
                      ))}
                    </Select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {cursor === null ? null : (
        <Button
          variant="secondary"
          loading={loading}
          onClick={() => {
            void load(cursor);
          }}
        >
          {t("admin.usersMore")}
        </Button>
      )}
    </Card>
  );
}
