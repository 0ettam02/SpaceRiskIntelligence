"use client";

import { useCallback, useState } from "react";

// Stato aperto/chiuso riutilizzabile per drawer di navigazione, drawer dei
// filtri e bottom sheet del dettaglio cella.
export function useDisclosure(initialOpen = false) {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);
  return { isOpen, open, close, toggle };
}
