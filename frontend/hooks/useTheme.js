"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "sri-theme-preference";

function getPreferredTheme() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

// Gestisce il tema chiaro/scuro con preferenza persistente in localStorage,
// applicando l'attributo data-theme sull'elemento <html>. Lo stato iniziale è
// sempre "dark" (come l'attributo impostato lato server in RootLayout): la
// vera preferenza dell'utente viene letta in un useEffect, dopo l'hydration,
// per evitare un mismatch fra HTML del server e primo render del client.
export function useTheme() {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    setTheme(getPreferredTheme());
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }, []);

  return { theme, setTheme, toggleTheme };
}
