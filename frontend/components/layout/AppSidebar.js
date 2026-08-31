"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronsLeft, ChevronsRight, Orbit } from "lucide-react";
import { NAV_ITEMS, isNavItemActive } from "@/lib/constants";

export function AppSidebar({ collapsed, onToggleCollapse }) {
  const pathname = usePathname();

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-30 hidden h-screen overflow-hidden border-r bg-surface transition-[width] duration-200 lg:flex lg:flex-col ${
        collapsed ? "lg:w-[4.75rem]" : "lg:w-[17rem]"
      }`}
      aria-label="Navigazione principale"
    >
      <div className="flex h-20 shrink-0 items-center gap-3 border-b px-4">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-brand-400/30 bg-brand-400/10 text-brand-300">
          <Orbit aria-hidden="true" size={22} />
        </span>
        {!collapsed ? (
          <div className="min-w-0">
            <p className="truncate text-sm font-bold leading-tight tracking-tight">SpaceRisk</p>
            <p className="truncate text-xs text-muted">Intelligence</p>
          </div>
        ) : null}
      </div>
      <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto p-3" aria-label="Sezioni dell'applicazione">
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const active = isNavItemActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              title={collapsed ? label : undefined}
              className={`flex min-h-11 items-center gap-3 rounded-xl px-3 text-sm font-medium transition-colors ${
                active ? "bg-brand-400/12 text-brand-300 ring-1 ring-inset ring-brand-400/20" : "text-muted hover:bg-elevated hover:text-ink"
              } ${collapsed ? "justify-center px-0" : ""}`}
            >
              <Icon aria-hidden="true" size={18} />
              {!collapsed ? label : <span className="sr-only">{label}</span>}
            </Link>
          );
        })}
      </nav>
      {!collapsed ? (
        <div className="m-3 shrink-0 rounded-xl border bg-elevated p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-300">Research prototype</p>
          <p className="mt-2 text-xs leading-5 text-muted">Stime sperimentali, non operative.</p>
        </div>
      ) : null}
      <button
        type="button"
        onClick={onToggleCollapse}
        aria-pressed={collapsed}
        className="m-3 mt-0 flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-xl border bg-elevated text-xs font-semibold text-muted hover:text-ink"
      >
        {collapsed ? <ChevronsRight aria-hidden="true" size={16} /> : <ChevronsLeft aria-hidden="true" size={16} />}
        {!collapsed ? "Comprimi" : <span className="sr-only">Espandi il menu</span>}
      </button>
    </aside>
  );
}
