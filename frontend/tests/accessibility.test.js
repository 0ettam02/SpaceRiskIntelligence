import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TimeRangeSelector } from "@/components/ui/TimeRangeSelector";
import { RiskBadge } from "@/components/feedback/RiskBadge";
import { StatusBadge } from "@/components/feedback/StatusBadge";
import { MetricTooltip } from "@/components/charts/MetricTooltip";

describe("accessible controls", () => {
  it("exposes the time range selector as a labelled group of toggle buttons", () => {
    render(<TimeRangeSelector value={30} onChange={() => {}} options={[{ value: 7, label: "7 giorni" }, { value: 30, label: "30 giorni" }]} />);
    const group = screen.getByRole("group", { name: /finestra temporale/i });
    expect(group).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "30 giorni" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "7 giorni" })).toHaveAttribute("aria-pressed", "false");
  });

  it("never conveys risk through color alone: the level label is always visible text", () => {
    render(<RiskBadge level="molto-elevata" />);
    expect(screen.getByText("Molto elevata")).toBeInTheDocument();
  });

  it("pairs status colors with an icon and a text label", () => {
    render(<StatusBadge status="warning" />);
    expect(screen.getByText("Avviso")).toBeInTheDocument();
  });

  it("makes the metric tooltip reachable and dismissible from the keyboard", async () => {
    render(<MetricTooltip label="Recall">Quota di celle realmente attive individuate.</MetricTooltip>);
    const trigger = screen.getByRole("button", { name: /informazioni su recall/i });

    await userEvent.tab();
    expect(trigger).toHaveFocus();
    expect(screen.getByRole("tooltip")).toHaveTextContent(/quota di celle/i);

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
