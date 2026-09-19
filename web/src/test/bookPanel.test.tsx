/**
 * The textbook panel (İP-8): search, command resolution, related units and pinning.
 *
 * The panel's promise is that one box does both readings of what the lecturer typed — a search
 * phrase and a Turkish fetch command — because during a lecture nobody wants to learn which is
 * which. These tests pin that behaviour to the two endpoints it is built from.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ContentUnitOut, HitOut } from "@/api/types";
import { BookPage } from "@/features/book/BookPage";
import { BookPanel } from "@/features/book/BookPanel";
import { installApiMock, jsonResponse } from "./apiMock";

const UNIT: ContentUnitOut = {
  id: "tanim-2.4",
  type: "definition",
  title_tr: "Türev tanımı",
  chapter: "2",
  number: "2.4",
  aliases: ["türev"],
  jsonld: {},
  rendered: {
    body_html: "<p>Fark oranının limiti.</p>",
    latex_blocks: ["f'(x)"],
    assets: [],
  },
};

const RELATED: ContentUnitOut = {
  ...UNIT,
  id: "teorem-2.9",
  type: "theorem",
  title_tr: "Türevlenebilirlik süreklilik gerektirir",
  number: "2.9",
};

const HIT: HitOut = { unit: UNIT, score: 0.9, stage: "alias", matched_alias: "türev" };

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BookPanel", () => {
  it("searches the textbook and renders the tagged unit", async () => {
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

  it("falls back to command resolution when the search finds nothing", async () => {
    const mock = installApiMock({
      "GET /book/search": [],
      "POST /book/resolve": [HIT],
    });
    render(<BookPanel />);

    fireEvent.change(screen.getByLabelText("Tanım, teorem veya örnek arayın"), {
      target: { value: "türev tanımını getir" },
    });
    fireEvent.keyDown(screen.getByLabelText("Tanım, teorem veya örnek arayın"), { key: "Enter" });

    expect(await screen.findByText("Türev tanımı")).toBeInTheDocument();
    const resolve = mock.mock.calls
      .map((call) => call[0] as Request)
      .find((request) => request.url.includes("/book/resolve"));
    expect(await resolve?.clone().json()).toEqual({ text: "türev tanımını getir" });
  });

  it("says so in Turkish when neither reading matches", async () => {
    installApiMock({ "GET /book/search": [], "POST /book/resolve": [] });
    render(<BookPanel />);

    fireEvent.change(screen.getByLabelText("Tanım, teorem veya örnek arayın"), {
      target: { value: "olmayan" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ara" }));

    expect(await screen.findByText("Eşleşen içerik bulunamadı.")).toBeInTheDocument();
  });

  it("walks one hop of the graph with the related units", async () => {
    installApiMock({
      "GET /book/units/tanim-2.4/related": () => jsonResponse([RELATED]),
    });
    render(<BookPanel hits={[HIT]} searchable={false} />);

    fireEvent.click(screen.getByRole("button", { name: "İlgili içerikleri getir" }));

    await waitFor(() => {
      expect(screen.getByText("Türevlenebilirlik süreklilik gerektirir")).toBeInTheDocument();
    });
  });

  it("pins a unit beside the result", () => {
    installApiMock({});
    const onPin = vi.fn();
    render(<BookPanel hits={[HIT]} searchable={false} onPin={onPin} />);

    fireEvent.click(screen.getByRole("button", { name: "Sonucun yanına sabitle" }));

    expect(onPin).toHaveBeenCalledWith(UNIT);
  });

  it("hides the pin action for the unit that is already pinned", () => {
    installApiMock({});
    render(<BookPanel hits={[HIT]} searchable={false} onPin={vi.fn()} pinnedId={UNIT.id} />);

    expect(screen.queryByRole("button", { name: "Sonucun yanına sabitle" })).toBeNull();
  });

  it("renders units handed in by a command without a search box", () => {
    installApiMock({});
    render(<BookPanel hits={[HIT]} searchable={false} />);

    expect(screen.getByText("Türev tanımı")).toBeInTheDocument();
    expect(screen.queryByRole("searchbox")).toBeNull();
  });
});

describe("BookPage", () => {
  it("gives the /kitap route its own single h1 above the panel's h2", () => {
    // The panel's heading is an `<h2>` because inside the workspace the `<h1>` belongs to the
    // board. On its own route nothing supplied that `<h1>`, which left the page starting at
    // level two — this is the assertion that keeps one top-level heading per view.
    installApiMock({});
    render(<BookPage />);

    const headings = screen.getAllByRole("heading");
    expect(headings.filter((node) => node.tagName === "H1")).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Kitap");
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Kitap İçeriği");
  });
});
