/**
 * Route table.
 *
 * Paths are Turkish because the interface is Turkish-only; they are the URLs that appear in the
 * usability study and in the report screenshots, so they are treated as part of the UI text even
 * though they live outside `i18n/tr.json`.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import { t } from "@/i18n";
import { AdminPage } from "@/features/admin/AdminPage";
import { AccountPage } from "@/features/auth/AccountPage";
import { LoginPage } from "@/features/auth/LoginPage";
import { AUTH_ROUTES } from "@/features/auth/paths";
import { RegisterPage } from "@/features/auth/RegisterPage";
import { BookPage } from "@/features/book/BookPage";
import { ContributePage } from "@/features/contribute/ContributePage";
import { FeedbackPage } from "@/features/feedback/FeedbackPage";
import { GuidePage } from "@/features/guide/GuidePage";
import { GUIDE_ROUTES } from "@/features/guide/routes";
import { WorkspacePage } from "@/features/workspace/WorkspacePage";

/** Canonical paths, referenced by the navigation and the Playwright specs. */
export const ROUTES = {
  workspace: "/calisma-alani",
  book: "/kitap",
  contribute: "/katkida-bulun",
  feedback: "/geri-bildirim",
  guide: GUIDE_ROUTES.root,
  admin: "/yonetim",
  login: AUTH_ROUTES.login,
  register: AUTH_ROUTES.register,
  account: AUTH_ROUTES.account,
} as const;

/** Shown for unknown paths. */
function NotFoundPage(): JSX.Element {
  return (
    <section>
      <h1>404</h1>
      <p>{t("nav.notFound")}</p>
      <a href={ROUTES.workspace}>{t("nav.backHome")}</a>
    </section>
  );
}

/** The application's routes. */
export function AppRoutes(): JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Navigate to={ROUTES.workspace} replace />} />
      <Route path={ROUTES.workspace} element={<WorkspacePage />} />
      <Route path={ROUTES.book} element={<BookPage />} />
      <Route path={ROUTES.contribute} element={<ContributePage />} />
      <Route path={ROUTES.feedback} element={<FeedbackPage />} />
      {/* The guide is three audiences on three URLs; the bare path opens the lecturer's one. */}
      <Route path={GUIDE_ROUTES.root} element={<Navigate to={GUIDE_ROUTES.usage} replace />} />
      <Route path={GUIDE_ROUTES.usage} element={<GuidePage page="usage" />} />
      <Route path={GUIDE_ROUTES.tech} element={<GuidePage page="tech" />} />
      <Route path={GUIDE_ROUTES.status} element={<GuidePage page="status" />} />
      {/* The administration area gates itself (see `AdminPage`), so a refused visitor gets the
          Turkish explanation in place instead of being redirected mid-lecture. */}
      <Route path={ROUTES.admin} element={<AdminPage />} />
      <Route path={ROUTES.login} element={<LoginPage />} />
      <Route path={ROUTES.register} element={<RegisterPage />} />
      <Route path={ROUTES.account} element={<AccountPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
