"use client";

import { useEffect, useState } from "react";

// Restituisce true/false in base a una media query, aggiornandosi ai
// cambi di viewport (usato per passare da sidebar a drawer, pannello a
// bottom sheet, tabella a card impilate).
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mediaQueryList = window.matchMedia(query);
    const update = () => setMatches(mediaQueryList.matches);
    update();
    mediaQueryList.addEventListener("change", update);
    return () => mediaQueryList.removeEventListener("change", update);
  }, [query]);

  return matches;
}
