"use client";

import { useEffect } from "react";

// Bus di eventi minimale per propagare il pulsante "Aggiorna dati" dell'header
// a qualunque pagina stia consumando un service tramite useAsyncData.
const REFRESH_EVENT = "sri:refresh";
const bus = typeof window !== "undefined" ? window : null;

export function emitRefresh() {
  bus?.dispatchEvent(new Event(REFRESH_EVENT));
}

export function useRefreshListener(callback) {
  useEffect(() => {
    if (!bus) return undefined;
    bus.addEventListener(REFRESH_EVENT, callback);
    return () => bus.removeEventListener(REFRESH_EVENT, callback);
  }, [callback]);
}
