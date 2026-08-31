"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Esegue una funzione asincrona (tipicamente un service in services/) e
// normalizza gli stati loading / success / empty / error per i componenti
// client che consumano i dati mock o, in futuro, un'API reale.
export function useAsyncData(asyncFn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const requestId = useRef(0);

  const load = useCallback(() => {
    const currentRequest = (requestId.current += 1);
    setState((prev) => ({ ...prev, loading: true, error: null }));
    asyncFn()
      .then((data) => {
        if (currentRequest === requestId.current) {
          setState({ data, loading: false, error: null });
        }
      })
      .catch((error) => {
        if (currentRequest === requestId.current) {
          setState({ data: null, loading: false, error });
        }
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    load();
  }, [load]);

  return { ...state, reload: load };
}
