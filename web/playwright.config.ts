import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration — the three browsers the proposal names (A.1.2: Chrome, Firefox,
 * Safari) map to chromium, firefox and webkit.
 *
 * These specs are **not** part of the pull-request gate: `web.yml` runs chromium on every PR and
 * `nightly.yml` runs all three (gelistirme/is-akisi.md §CI). They need a running API
 * (`aicls serve --profile full`) plus the Vite dev server; `webServer` starts the front end and
 * reuses one that is already listening.
 *
 * The browser binaries are a download, so they are installed once with `npx playwright install`
 * outside the approval-gated environment — the specs here are written but never executed by the
 * scaffolding agent.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 7_000 },
  fullyParallel: true,
  forbidOnly: Boolean(process.env["CI"]),
  retries: process.env["CI"] === undefined ? 0 : 1,
  reporter: process.env["CI"] === undefined ? [["list"]] : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env["AICLS_WEB_BASE_URL"] ?? "http://127.0.0.1:5173",
    locale: "tr-TR",
    timezoneId: "Europe/Istanbul",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    // The voice tab needs the microphone; fake devices keep the run headless and deterministic.
    permissions: ["microphone"],
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
