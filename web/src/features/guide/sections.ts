/**
 * The guide's table of contents, as data.
 *
 * One list per page, shared by three consumers that must never disagree: the article renders a
 * `<Section>` per entry, the sidebar renders a link per entry, and the scroll-spy observes exactly
 * these ids. A section that exists in the article but not here would be unreachable from the table
 * of contents, so the unit test walks these lists and asserts that every id resolves to a real
 * heading in the rendered page.
 *
 * Ids are Turkish slugs because they appear in the URL (`/kilavuz/teknik#boru-hatti`), which is a
 * deep link a lecturer may paste into a lesson plan or a reviewer into a comment. They are frozen:
 * renaming one breaks somebody else's bookmark.
 *
 * Each list is `as const satisfies` its element type, and {@link byId} turns it into a record keyed
 * by those literal ids — so an article can write `USAGE.sonucPaneli` and get a compile error rather
 * than a runtime `undefined` if the section is ever removed.
 */
import type { MessageKey } from "@/i18n";

/** One addressable section of a guide page. */
export interface GuideSection {
  /** DOM id and URL fragment; stable, because links to it are pasted into other documents. */
  readonly id: string;
  /** i18n key of the `<h2>`. */
  readonly titleKey: MessageKey;
}

/** Which of the three guide pages a section belongs to. */
export type GuidePageId = "usage" | "tech" | "status";

/** Index a section list by its literal ids. */
function byId<const T extends readonly GuideSection[]>(
  list: T,
): { readonly [K in T[number]["id"]]: Extract<T[number], { id: K }> } {
  return Object.fromEntries(list.map((section) => [section.id, section])) as {
    readonly [K in T[number]["id"]]: Extract<T[number], { id: K }>;
  };
}

/** Audience A — the lecturer and the student. */
export const USAGE_SECTIONS = [
  { id: "ai-cls-nedir", titleKey: "guide.usage.what.heading" },
  { id: "komut-vermenin-uc-yolu", titleKey: "guide.usage.inputs.heading" },
  { id: "aciklama-istegi", titleKey: "guide.usage.clarify.heading" },
  { id: "kapsam-disi-problem", titleKey: "guide.usage.unsupported.heading" },
  { id: "sonuc-paneli", titleKey: "guide.usage.result.heading" },
  { id: "kitap-paneli", titleKey: "guide.usage.book.heading" },
  { id: "gecmis-ve-disa-aktarma", titleKey: "guide.usage.history.heading" },
  { id: "geri-bildirim", titleKey: "guide.usage.feedback.heading" },
  { id: "sunum-modu", titleKey: "guide.usage.presentation.heading" },
  { id: "klavye-kisayollari", titleKey: "guide.usage.shortcuts.heading" },
] as const satisfies readonly GuideSection[];

/** Audience B — the technical reader. */
export const TECH_SECTIONS = [
  { id: "boru-hatti", titleKey: "guide.tech.pipeline.heading" },
  { id: "neden-cebir-motoru", titleKey: "guide.tech.why.heading" },
  { id: "on-yuz-yigini", titleKey: "guide.tech.frontend.heading" },
  { id: "arka-uc-yigini", titleKey: "guide.tech.backend.heading" },
  { id: "api-yuzeyi", titleKey: "guide.tech.api.heading" },
  { id: "websocket-akislari", titleKey: "guide.tech.ws.heading" },
  { id: "veri-modeli", titleKey: "guide.tech.data.heading" },
  { id: "kitap-bilgi-grafigi", titleKey: "guide.tech.book.heading" },
  { id: "guvenlik-ve-gizlilik", titleKey: "guide.tech.security.heading" },
  { id: "onay-kapilari", titleKey: "guide.tech.gates.heading" },
  { id: "degerlendirme", titleKey: "guide.tech.eval.heading" },
] as const satisfies readonly GuideSection[];

/** Audience C — the honest status report. */
export const STATUS_SECTIONS = [
  { id: "nasil-okunmali", titleKey: "guide.status.how.heading" },
  { id: "olculen-degerler", titleKey: "guide.status.runs.heading" },
  { id: "sozlesme-olcutleri", titleKey: "guide.status.contract.heading" },
  { id: "kanita-giden-yol", titleKey: "guide.status.next.heading" },
] as const satisfies readonly GuideSection[];

/** Usage sections keyed by id. */
export const USAGE = byId(USAGE_SECTIONS);
/** Technical sections keyed by id. */
export const TECH = byId(TECH_SECTIONS);
/** Status sections keyed by id. */
export const STATUS = byId(STATUS_SECTIONS);

/** Every page's sections, addressed by page id. */
export const GUIDE_SECTIONS: Readonly<Record<GuidePageId, readonly GuideSection[]>> = {
  usage: USAGE_SECTIONS,
  tech: TECH_SECTIONS,
  status: STATUS_SECTIONS,
};
