/**
 * `e2e/contribute.spec.ts` — the consent gate must come before any recording UI, and an accepted
 * take must be acknowledged (api.md §3.8, ADR-0017).
 *
 * Recording itself is driven with Chromium's fake media device, so the spec runs headless. The
 * WAV bytes the browser produces are validated by `wavEncoder.test.ts` and, end to end, by the
 * server-side `test_api_asr_transcribe_wav_only`.
 */
import { expect, test, type Page } from "@playwright/test";

const CONSENT = {
  version: "2026-09-01",
  text_tr: "Bu çalışmada ses kayıtlarınız takma adla saklanacaktır…",
  sha256: "sha256:0000",
};

const PROMPTS = [
  { id: "p-1", text_tr: "x karenin türevini al", scenario_tag: "turev", difficulty: 1 },
];

async function stubApi(page: Page): Promise<void> {
  await page.route("**/api/v1/contribute/consent/current", async (route) => {
    await route.fulfill({ json: CONSENT });
  });
  await page.route("**/api/v1/contribute/consent", async (route) => {
    await route.fulfill({ json: { speaker_pseudonym: "P00012", token: "spk-token" } });
  });
  await page.route("**/api/v1/contribute/prompts/next**", async (route) => {
    await route.fulfill({ json: PROMPTS });
  });
  await page.route("**/api/v1/contribute/me/progress", async (route) => {
    await route.fulfill({ json: { n_accepted: 3, seconds_accepted: 42, target_seconds: 180000 } });
  });
  await page.route("**/api/v1/contribute/recordings", async (route) => {
    await route.fulfill({
      json: {
        id: "rec-1",
        accepted: true,
        duration_s: 2.4,
        qc: {
          passed: true,
          checks: [{ name: "snr_db", value: 24.5, threshold: 15, passed: true }],
          reasons_tr: [],
        },
      },
    });
  });
}

test.describe("ses verisi toplama", () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page);
  });

  test("onam verilmeden kayıt arayüzü görünmez", async ({ page }) => {
    await page.goto("/katkida-bulun");

    await expect(page.getByTestId("consent-text")).toBeVisible();
    await expect(page.getByRole("button", { name: "Kaydı başlat" })).toHaveCount(0);
    // The accept button stays disabled until the checkbox is ticked.
    await expect(page.getByRole("button", { name: "Onayla ve başla" })).toBeDisabled();
  });

  test("onam sonrası cümle gösterilir ve kayıt kabul edilir", async ({ page, browserName }) => {
    test.skip(browserName !== "chromium", "sahte medya aygıtı yalnızca chromium'da kuruluyor");

    await page.goto("/katkida-bulun");
    await page.getByLabel("Onam metnini okudum ve kabul ediyorum.").check();
    await page.getByRole("button", { name: "Onayla ve başla" }).click();

    await expect(page.getByTestId("prompt-sentence")).toHaveText("x karenin türevini al");
    await expect(page.getByRole("progressbar", { name: "İlerlemeniz" })).toBeVisible();

    await page.getByRole("button", { name: "Kaydı başlat" }).click();
    await page.waitForTimeout(1200);
    await page.getByRole("button", { name: "Kaydı durdur" }).click();

    await expect(page.getByTestId("qc-verdict")).toContainText("Kayıt kabul edildi");
  });
});
