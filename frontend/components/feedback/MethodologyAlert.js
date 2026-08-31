import { TriangleAlert } from "lucide-react";
import { RESEARCH_DISCLAIMER } from "@/lib/constants";

export function MethodologyAlert({ title = "Avvertenza metodologica", items = [], showDisclaimer = true }) {
  return (
    <aside className="rounded-2xl border border-amber-400/25 bg-amber-400/[0.06] p-5 shadow-panel" aria-labelledby="methodology-alert-title">
      <div className="flex gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-400/10 text-amber-300">
          <TriangleAlert aria-hidden="true" size={20} />
        </span>
        <div className="min-w-0">
          <h2 id="methodology-alert-title" className="font-semibold">
            {title}
          </h2>
          {items.length > 0 ? (
            <ul className="mt-3 list-disc space-y-2 pl-4 text-sm leading-6 text-muted">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
      {showDisclaimer ? (
        <div className="mt-5 border-t border-amber-400/20 pt-5">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-amber-300">Uso responsabile</p>
          <p className="mt-2 text-sm leading-6 text-muted">{RESEARCH_DISCLAIMER}</p>
        </div>
      ) : null}
    </aside>
  );
}
