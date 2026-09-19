/**
 * Feature panels that talk to the API: command bar, book, history and the SUS questionnaire.
 *
 * The `fetch` router in `apiMock.ts` stands in for the backend, so these tests assert the two things
 * the front end is responsible for: the request it builds and the Turkish surface it renders.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ComputeOut, HitOut, OperationOut } from "@/api/types";
import { BookPanel } from "@/features/book/BookPanel";
import { CommandBar } from "@/features/command/CommandBar";
import { SusForm } from "@/features/feedback/SusForm";
import { HistoryPanel } from "@/features/history/HistoryPanel";
import { installApiMock, problemResponse, requestBody } from "./apiMock";

const RESULT: ComputeOut = {
  kind: "result",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}", var: "x" },
  result: {
    kind: "expr",
    latex: "2 x",
    plain: "2*x",
    srepr: "Mul(Integer(2), Symbol('x'))",
    steps: [],
    plot_url: null,
    cache_hit: false,
  },
  clarification: null,
  book_units: [],
  nlu: { intent: "differentiate", score: 0.93, trace: [], source: "rules" },
  timings: { total: 58.2 },
  model_versions: {},
};

const HIT: HitOut = {
  unit: {
    id: "tanim-2.4",
    type: "definition",
    title_tr: "Türev tanımı",
    chapter: "2",
    number: "2.4",
    rendered: {
      body_html: "<p>Fark oranının limiti.</p>",
      latex_blocks: ["f'(x)"],
      assets: [],
    },
  },
  score: 0.9,
  stage: "alias",
  matched_alias: "türev",
};

const OPERATION: OperationOut = {
  id: "01J8R2ZQ7G4V0X9K3M2N5P6Q80",
  input_kind: "text",
  input_summary: "x karenin türevini al",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}" },
  result_latex: null,
  result_kind: "expr",
  plot_url: null,
  error_code: null,
  timings: { total: 40 },
  created_at: "2027-03-01T09:15:00Z",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CommandBar", () => {
  it("posts the typed Turkish command and reports the outcome", async () => {
    const mock = installApiMock({ "POST /compute/text": RESULT });
    const onOutcome = vi.fn();
    render(<CommandBar workspaceId="ws-1" lastLatex="x^{2}" onOutcome={onOutcome} />);

    fireEvent.change(screen.getByLabelText("Komut"), {
      target: { value: "x karenin türevini al" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    await waitFor(() => {
      expect(onOutcome).toHaveBeenCalledWith(RESULT);
    });
    expect(await requestBody(mock)).toEqual({
      text: "x karenin türevini al",
      workspace_id: "ws-1",
      context: { last_latex: "x^{2}" },
    });
  });

  it("fills the input from an example chip", () => {
    installApiMock({});
    render(<CommandBar onOutcome={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "sin x'in integralini al" }));

    expect(screen.getByLabelText("Komut")).toHaveValue("sin x'in integralini al");
  });

  it("shows the Turkish problem detail when the API refuses", async () => {
    installApiMock({
      "POST /compute/text": () =>
        problemResponse(
          "UNSUPPORTED_PROBLEM",
          422,
          "Bu problem tipi şu an için desteklenmemektedir.",
        ),
    });
    const onError = vi.fn();
    render(<CommandBar onOutcome={vi.fn()} onError={onError} />);

    fireEvent.change(screen.getByLabelText("Komut"), { target: { value: "uzay-zamanı çöz" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Bu problem tipi şu an için desteklenmemektedir.",
    );
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("does not submit an empty command", () => {
    const mock = installApiMock({ "POST /compute/text": RESULT });
    render(<CommandBar onOutcome={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(mock).not.toHaveBeenCalled();
  });

  it("clears the input", () => {
    installApiMock({});
    render(<CommandBar onOutcome={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Komut"), { target: { value: "abc" } });

    fireEvent.click(screen.getByRole("button", { name: "Temizle" }));

    expect(screen.getByLabelText("Komut")).toHaveValue("");
  });
});

describe("BookPanel", () => {
  it("searches the textbook and shows the tagged unit", async () => {
    installApiMock({ "GET /book/search": [HIT] });
    render(<BookPanel />);

    fireEvent.change(screen.getByLabelText("Tanım, teorem veya örnek arayın"), {
      target: { value: "türev" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ara" }));

    expect(await screen.findByText("Türev tanımı")).toBeInTheDocument();
    expect(screen.getByText("Tanım")).toBeInTheDocument();
    expect(screen.getByText("Fark oranının limiti.")).toBeInTheDocument();
  });

  it("reports an empty search result in Turkish", async () => {
    installApiMock({ "GET /book/search": [] });
    render(<BookPanel />);

    fireEvent.change(screen.getByLabelText("Tanım, teorem veya örnek arayın"), {
      target: { value: "yok" },
    });
    fireEvent.keyDown(screen.getByLabelText("Tanım, teorem veya örnek arayın"), { key: "Enter" });

    expect(await screen.findByText("Eşleşen içerik bulunamadı.")).toBeInTheDocument();
  });

  it("renders units handed in by a book command without a search box", () => {
    installApiMock({});
    render(<BookPanel hits={[HIT]} searchable={false} />);

    expect(screen.getByText("Türev tanımı")).toBeInTheDocument();
    expect(screen.queryByRole("searchbox")).toBeNull();
  });
});

describe("HistoryPanel", () => {
  it("lists the operations of a workspace", async () => {
    installApiMock({
      "GET /workspaces/ws-1/operations": { items: [OPERATION], next_cursor: null },
    });
    const onReuse = vi.fn();
    render(<HistoryPanel workspaceId="ws-1" onReuse={onReuse} />);

    expect(await screen.findByText("x karenin türevini al")).toBeInTheDocument();
    expect(screen.getByText("Metin")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Yeniden çalıştır" }));
    expect(onReuse).toHaveBeenCalledWith(OPERATION);
  });

  it("pages with the cursor the API returned", async () => {
    const mock = installApiMock({
      "GET /workspaces/ws-1/operations": (request: Request) =>
        new Response(
          JSON.stringify(
            new URL(request.url).searchParams.get("cursor") === null
              ? { items: [OPERATION], next_cursor: "c2" }
              : {
                  items: [{ ...OPERATION, id: "op-2", input_summary: "ikinci işlem" }],
                  next_cursor: null,
                },
          ),
          { headers: { "content-type": "application/json" } },
        ),
    });
    render(<HistoryPanel workspaceId="ws-1" />);

    fireEvent.click(await screen.findByRole("button", { name: "Daha fazla yükle" }));

    expect(await screen.findByText("ikinci işlem")).toBeInTheDocument();
    expect(mock).toHaveBeenCalledTimes(2);
  });

  it("shows the empty state without a workspace", () => {
    installApiMock({});
    render(<HistoryPanel workspaceId={null} />);

    expect(screen.getByText("Bu çalışma alanında henüz işlem yok.")).toBeInTheDocument();
  });
});

describe("SusForm", () => {
  it("requires every statement before it submits", () => {
    installApiMock({ "POST /feedback/sus": { score: 82.5 } });
    render(<SusForm cohort="pilot" />);

    expect(screen.getByRole("button", { name: "Anketi gönder" })).toBeDisabled();
  });

  it("posts the ten answers and shows the score the server computed", async () => {
    const mock = installApiMock({ "POST /feedback/sus": { score: 82.5 } });
    const onScored = vi.fn();
    render(<SusForm cohort="pilot-2027" onScored={onScored} />);

    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(50);
    for (let question = 0; question < 10; question += 1) {
      fireEvent.click(radios[question * 5 + 3] as HTMLElement); // answer 4 everywhere
    }

    fireEvent.click(screen.getByRole("button", { name: "Anketi gönder" }));

    expect(await screen.findByText("SUS puanınız: 82.5")).toBeInTheDocument();
    expect(onScored).toHaveBeenCalledWith(82.5);
    expect(await requestBody(mock)).toEqual({
      answers: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      cohort: "pilot-2027",
    });
  });
});
