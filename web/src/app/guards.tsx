/**
 * Route guards.
 *
 * A guard here never navigates on its own. Surprise navigation in the middle of a lecture is
 * worse than a refused panel: the lecturer would lose the board they were pointing at. So the
 * default behaviour is to render an explanation in place, with a way forward, and `redirectTo` is
 * opt-in for the rare screen that genuinely must not be reachable (the admin area).
 *
 * While the session is still `"unknown"` — the first `POST /auth/refresh` has not answered yet —
 * the guard shows a quiet spinner instead of deciding, so a reload never flashes a login prompt at
 * a signed-in lecturer.
 */
import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";

import type { Role } from "@/api/types";
import { EmptyState } from "@/components/ui/States";
import { Spinner } from "@/components/ui/Spinner";
import { t } from "@/i18n";
import { useAuth } from "./auth";

export interface RequireAuthProps {
  /** What to render for a signed-in user. */
  children: ReactNode;
  /** Roles allowed through; empty means "any signed-in user". */
  roles?: readonly Role[];
  /** Rendered while the session is being restored. */
  pending?: ReactNode;
  /** Rendered when the visitor may not pass; defaults to a Turkish explanation. */
  fallback?: ReactNode;
  /** Opt-in redirect for screens that must not be reachable at all. */
  redirectTo?: string;
}

/**
 * Render `children` only for a signed-in user (optionally with one of `roles`).
 *
 * @example
 * ```tsx
 * <RequireAuth roles={["instructor", "admin"]}>
 *   <AdminPanel />
 * </RequireAuth>
 * ```
 */
export function RequireAuth({
  children,
  roles = [],
  pending,
  fallback,
  redirectTo,
}: RequireAuthProps): JSX.Element {
  const { status, isAuthenticated, can } = useAuth();

  if (status === "unknown") {
    return <>{pending ?? <Spinner size="lg" label={t("auth.restoring")} />}</>;
  }
  const allowed = isAuthenticated && can(...roles);
  if (allowed) {
    return <>{children}</>;
  }
  if (redirectTo !== undefined) {
    return <Navigate to={redirectTo} replace />;
  }
  return (
    <>
      {fallback ?? (
        <EmptyState
          title={isAuthenticated ? t("auth.forbiddenTitle") : t("auth.requiredTitle")}
          description={
            isAuthenticated ? t("auth.forbiddenDescription") : t("auth.requiredDescription")
          }
        />
      )}
    </>
  );
}

export interface RequireRoleProps extends Omit<RequireAuthProps, "roles"> {
  /** The roles allowed through. */
  roles: readonly Role[];
}

/** {@link RequireAuth} with the roles spelled out at the call site. */
export function RequireRole({ roles, ...rest }: RequireRoleProps): JSX.Element {
  return <RequireAuth roles={roles} {...rest} />;
}
