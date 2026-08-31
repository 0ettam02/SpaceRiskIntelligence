import { CheckCircle2, CircleHelp, TriangleAlert, XCircle } from "lucide-react";
import { getStatusMeta } from "@/lib/risk-utils";

const ICONS = {
  passed: CheckCircle2,
  warning: TriangleAlert,
  failed: XCircle,
  not_available: CircleHelp,
};

// Stato passed / warning / failed / not_available: colore fisso, mai
// riutilizzato per identificare serie o categorie, sempre con icona + testo.
export function StatusBadge({ status, label }) {
  const meta = getStatusMeta(status);
  const Icon = ICONS[status] || CircleHelp;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border bg-elevated px-2.5 py-1 text-xs font-semibold" style={{ color: meta.color }}>
      <Icon aria-hidden="true" size={14} />
      {label || meta.label}
    </span>
  );
}
