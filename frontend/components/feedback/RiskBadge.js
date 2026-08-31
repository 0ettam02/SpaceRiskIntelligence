import { AlertOctagon, AlertTriangle, Flame, Info } from "lucide-react";
import { getRiskLevelById } from "@/lib/risk-utils";

const ICONS = {
  bassa: Info,
  moderata: AlertTriangle,
  elevata: AlertOctagon,
  "molto-elevata": Flame,
};

// Il livello di rischio non è mai comunicato solo dal colore: icona e testo
// accompagnano sempre il chip colorato (WCAG 1.4.1).
export function RiskBadge({ level, size = "md" }) {
  const risk = getRiskLevelById(level);
  if (!risk) {
    return <span className="text-xs text-muted">Rischio non disponibile</span>;
  }
  const Icon = ICONS[risk.id] || Info;
  const isCompact = size === "sm";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border border-black/10 font-semibold dark:border-white/10 ${
        isCompact ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs"
      }`}
      style={{ backgroundColor: risk.color, color: risk.onColorText }}
    >
      <Icon aria-hidden="true" size={isCompact ? 12 : 14} />
      {risk.label}
    </span>
  );
}
