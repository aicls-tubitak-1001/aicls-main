/**
 * UI primitives and the presentational feature components.
 *
 * These are the pieces the usability study is judged on, so the assertions are about what a
 * participant can see and reach: Turkish labels, accessible roles, the confirmation step, and the
 * fact that a rejected recording explains itself.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ComputeOut, PromptOut, StepDto } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { Dialog } from "@/components/ui/Dialog";
import { CommandPreview } from "@/features/command/CommandPreview";
import { ProgressBar } from "@/features/contribute/ProgressBar";
import { PromptCard } from "@/features/contribute/PromptCard";
import { formatCommand } from "@/lib/commandFormat";
import { markSubmit, resetLatencyMarks } from "@/lib/latency";
import { PlotView } from "@/features/results/PlotView";
import { ResultCard } from "@/features/results/ResultCard";
import { StepsView } from "@/features/results/StepsView";

const OUTCOME: ComputeOut = {
  kind: "result",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
  result: {
    kind: "expr",
    latex: "2 x",
    plain: "2*x",
    srepr: "Mul(Integer(2), Symbol('x'))",
    steps: [],
    plot_url: null,
    cache_hit: true,
  },
  clarification: null,
  book_units: [],
  nlu: { intent: "differentiate", score: 0.93, trace: [], source: "rules" },
  timings: { nlu: 12.3, parse: 1.1, cas: 41, render: 0.4, total: 58.2 },
  model_versions: { ocr: null, asr: null },
};

beforeEach(() => {
  // ResultCard posts a latency sample on mount; keep it off the network.
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetLatencyMarks();
});

describe("ui primitives", () => {
  it("renders a button that does not submit by default", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Çalıştır</Button>);
    const button = screen.getByRole("button", { name: "Çalıştır" });

    expect(button).toHaveAttribute("type", "button");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("disables a button and blocks the click", () => {
    const onClick = vi.fn();
    render(
      <Button disabled variant="primary" size="small" block onClick={onClick}>
        Gönder
      </Button>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));
    expect(onClick).not.toHaveBeenCalled();
  });

  it("labels a titled card as a section", () => {
    render(
      <Card title="Sonuç" actions={<Chip tone="success">Önbellekten</Chip>}>
        <p>gövde</p>
      </Card>,
    );

    expect(screen.getByRole("region", { name: "Sonuç" })).toBeInTheDocument();
    expect(screen.getByText("Önbellekten")).toBeInTheDocument();
  });

  it("renders an untitled card as a plain container", () => {
    render(<Card>içerik</Card>);

    expect(screen.queryByRole("region")).toBeNull();
    expect(screen.getByText("içerik")).toBeInTheDocument();
  });

  it("opens a modal dialog and closes it on Escape", () => {
    const onClose = vi.fn();
    render(
      <Dialog open title="Geri Bildirim" onClose={onClose} footer={<Button>Gönder</Button>}>
        <p>metin</p>
      </Dialog>,
    );

    expect(screen.getByRole("dialog", { name: "Geri Bildirim" })).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders nothing when the dialog is closed", () => {
    render(
      <Dialog open={false} title="Kapalı" onClose={() => undefined}>
        <p>metin</p>
      </Dialog>,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("CommandPreview", () => {
  it("shows the parsed command and confirms it", () => {
    const onConfirm = vi.fn();
    const onEdit = vi.fn();
    render(
      <CommandPreview
        latex="x^{2}"
        command={{ op: "diff", expr: "x^{2}", var: "x", order: 1 }}
        confidence={0.93}
        onConfirm={onConfirm}
        onEdit={onEdit}
      />,
    );

    expect(screen.getByTestId("command-preview-command")).toHaveTextContent(
      "diff(expr=x^{2}, var=x, order=1)",
    );
    expect(screen.getByText("%93")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /onayla/i }));
    expect(onConfirm).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /düzenle/i }));
    expect(onEdit).toHaveBeenCalledTimes(1);
  });

  it("warns when the confidence is below the confirmation threshold", () => {
    render(
      <CommandPreview
        latex="x"
        command={{ op: "simplify", expr: "x" }}
        confidence={0.42}
        onConfirm={() => undefined}
        onEdit={() => undefined}
        voice
      />,
    );

    expect(screen.getByText("Güven düşük; lütfen komutu kontrol edin.")).toBeInTheDocument();
    expect(
      screen.getByText("Ses girdisinde sonuç her zaman onayınızdan sonra hesaplanır."),
    ).toBeInTheDocument();
  });

  it("formats a command without optional slots", () => {
    expect(formatCommand({ op: "simplify", expr: "x+x" })).toBe("simplify(expr=x+x)");
  });
});

describe("ResultCard", () => {
  it("shows the empty state before the first command", () => {
    render(<ResultCard outcome={null} />);

    expect(
      screen.getByText("Henüz bir sonuç yok. Bir komut girerek başlayın."),
    ).toBeInTheDocument();
  });

  it("renders the answer, the plain form and the cache badge", () => {
    render(<ResultCard outcome={OUTCOME} />);

    expect(screen.getByTestId("result-latex")).toBeInTheDocument();
    expect(screen.getByText("2*x")).toBeInTheDocument();
    expect(screen.getByText("Önbellekten")).toBeInTheDocument();
    expect(screen.getByText("Toplam: 58 ms")).toBeInTheDocument();
  });

  it("posts the latency sample for an operation that was marked on submit", async () => {
    markSubmit(OUTCOME.operation_id as string, "text");
    render(<ResultCard outcome={OUTCOME} />);

    await vi.waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalled();
    });
    const request = vi.mocked(globalThis.fetch).mock.calls[0]?.[0] as Request;
    expect(request.url).toContain("/api/v1/telemetry/latency");
  });

  it("sends no telemetry when the operation was never marked (nothing to measure)", async () => {
    render(<ResultCard outcome={OUTCOME} />);

    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

describe("StepsView and PlotView", () => {
  it("numbers the derivation steps", () => {
    const steps: StepDto[] = [
      { title_tr: "Türev kuralını uygula", latex: "2x", rule: "power_rule", detail_tr: "" },
      { title_tr: "Sadeleştir", latex: "", rule: "", detail_tr: "" },
    ];
    render(<StepsView steps={steps} />);

    expect(screen.getByText("Türev kuralını uygula")).toBeInTheDocument();
    expect(screen.getByText("Sadeleştir")).toBeInTheDocument();
  });

  it("shows a placeholder when there are no steps", () => {
    render(<StepsView steps={[]} />);

    expect(screen.getByText("Bu sonuç için adım gösterimi yok.")).toBeInTheDocument();
  });

  it("renders the plot image with Turkish alternative text", () => {
    render(<PlotView url="/api/v1/compute/plots/abc.png" />);

    expect(screen.getByAltText("Hesaplanan fonksiyonun grafiği")).toBeInTheDocument();
  });

  it("shows a placeholder when the result has no plot", () => {
    render(<PlotView url={null} />);

    expect(screen.getByText("Bu sonuç için grafik yok.")).toBeInTheDocument();
  });
});

describe("contribute widgets", () => {
  it("exposes the session goal as an accessible progressbar", () => {
    render(
      <ProgressBar
        sessionAccepted={3}
        progress={{ n_accepted: 12, seconds_accepted: 90, target_seconds: 1800 }}
      />,
    );
    const bar = screen.getByRole("progressbar", { name: "İlerlemeniz" });

    expect(bar).toHaveAttribute("aria-valuenow", "3");
    expect(bar).toHaveAttribute("aria-valuemax", "10");
    expect(screen.getByText("3 / 10 kabul edilen kayıt")).toBeInTheDocument();
    expect(screen.getByText("Kabul edilen kayıt: 12")).toBeInTheDocument();
  });

  it("says that the sitting is complete once the goal is reached", () => {
    render(<ProgressBar sessionAccepted={10} progress={null} />);

    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "10");
    expect(
      screen.getByText(
        "Oturum hedefine ulaştınız. Teşekkürler — dilediğiniz zaman durabilirsiniz.",
      ),
    ).toBeInTheDocument();
  });

  it("shows the sentence to read aloud", () => {
    const prompt: PromptOut = {
      id: "p-1",
      text_tr: "x karenin türevini al",
      scenario_tag: "turev",
      difficulty: 1,
    };
    render(<PromptCard prompt={prompt} index={2} total={5} />);

    expect(screen.getByTestId("prompt-sentence")).toHaveTextContent("x karenin türevini al");
    expect(screen.getByText("2 / 5")).toBeInTheDocument();
  });
});
