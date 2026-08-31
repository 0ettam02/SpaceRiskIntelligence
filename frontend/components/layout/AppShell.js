"use client";

import { useState } from "react";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { AppHeader } from "@/components/layout/AppHeader";
import { MobileDrawer } from "@/components/layout/MobileDrawer";

export function AppShell({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <div className="min-h-screen">
      <AppSidebar collapsed={collapsed} onToggleCollapse={() => setCollapsed((prev) => !prev)} />
      <MobileDrawer isOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />
      <div className={`min-w-0 transition-[padding] duration-200 ${collapsed ? "lg:pl-[4.75rem]" : "lg:pl-[17rem]"}`}>
        <AppHeader onOpenMobileNav={() => setMobileNavOpen(true)} />
        <main id="contenuto-principale" className="mx-auto w-full max-w-[1680px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}
