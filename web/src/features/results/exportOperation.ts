/**
 * "Save this board" — one operation as LaTeX source or a one-page PDF.
 *
 * `POST /workspaces/{ws_id}/operations/{op_id}/export?format=tex|pdf` answers with the file itself
 * (`text/x-tex` or `application/pdf`, both with a `Content-Disposition` filename), not with JSON, so
 * the response is read as a blob and handed to the browser's own download path. Nothing is
 * generated client-side: what the lecturer saves is byte-for-byte what the backend produced, which
 * is what makes an exported derivation quotable in a report.
 */
import { api, call } from "@/api/client";
import type { AiclsProblem } from "@/api/problem";
import { t } from "@/i18n";

/** The two formats the endpoint accepts. */
export type ExportFormat = "tex" | "pdf";

/** File extension and MIME type per format (mirrors `aicls.api.routers.workspaces`). */
const MEDIA: Record<ExportFormat, { extension: string; mime: string }> = {
  tex: { extension: "tex", mime: "text/x-tex" },
  pdf: { extension: "pdf", mime: "application/pdf" },
};

/**
 * Hand a blob to the browser as a download.
 *
 * @returns `false` when the environment has no object-URL support (jsdom, a locked-down webview),
 *   so the caller can report the failure instead of pretending the file was saved.
 */
export function downloadBlob(blob: Blob, filename: string): boolean {
  const factory = globalThis.URL;
  if (typeof factory?.createObjectURL !== "function") {
    return false;
  }
  const url = factory.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  factory.revokeObjectURL(url);
  return true;
}

/**
 * Export one operation and save it.
 *
 * @param workspaceId - Workspace that owns the operation.
 * @param operationId - Operation to export (`ComputeOut.operation_id`).
 * @param format - `"tex"` or `"pdf"`.
 * @returns `null` on success, or the Turkish problem to show.
 */
export async function exportOperation(
  workspaceId: string,
  operationId: string,
  format: ExportFormat,
): Promise<AiclsProblem | null> {
  const { data, problem } = await call(
    api.POST("/workspaces/{ws_id}/operations/{op_id}/export", {
      params: { path: { ws_id: workspaceId, op_id: operationId }, query: { format } },
      parseAs: "blob",
    }),
    { fallbackDetailTr: t("results.exportFailed") },
  );
  if (data === null) {
    return problem;
  }
  const media = MEDIA[format];
  const blob = data instanceof Blob ? data : new Blob([String(data)], { type: media.mime });
  const saved = downloadBlob(blob, `aicls-${operationId}.${media.extension}`);
  if (!saved) {
    return {
      type: "https://ai-cls.example/errors/CLIENT_TRANSPORT",
      title: t("results.exportFailed"),
      status: 0,
      code: "CLIENT_TRANSPORT",
      detail_tr: t("results.exportFailed"),
      instance: null,
      extra: {},
    };
  }
  return null;
}
