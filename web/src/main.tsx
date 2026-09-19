/**
 * Browser entry point.
 *
 * Loads the design tokens, the global stylesheet and KaTeX's own CSS (bundled from
 * `node_modules`, never a CDN — the CSP allows `style-src 'self' 'unsafe-inline'` and
 * `font-src 'self'` only, guvenlik-gizlilik.md §5), restores the appearance and density the
 * lecturer last chose, then mounts the router inside the toast provider.
 *
 * The session is restored in the background: `POST /auth/refresh` runs while the shell paints, so
 * a reload in front of a class never shows a login prompt to someone who is still signed in.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import "katex/dist/katex.min.css";
import "./styles/tokens.css";
import "./styles/global.css";

import { App } from "./App";
import { applyDensity, readDensity } from "./app/density";
import { restoreSession } from "./app/auth";
import { applyTheme, readThemePreference } from "./app/theme";
import { ToastProvider } from "./components/ui/Toast";

applyTheme(readThemePreference());
applyDensity(readDensity());

const container = document.getElementById("root");
if (container === null) {
  throw new Error("Mount point #root is missing from index.html.");
}

// Fire and forget: the shell renders immediately and the session store updates when the refresh
// answers. A failure simply leaves the visitor anonymous.
void restoreSession();

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <App />
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
);
