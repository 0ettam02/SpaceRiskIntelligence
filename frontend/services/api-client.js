import { API_BASE_URL, DATA_SOURCE, SIMULATED_LATENCY_MS } from "@/lib/constants";

export const isMockMode = DATA_SOURCE !== "api";

// Introduce una latenza artificiale breve così gli stati di caricamento sono
// visibili anche con dati mock generati in locale.
export function withSimulatedLatency(value, ms = SIMULATED_LATENCY_MS) {
  return new Promise((resolve) => {
    setTimeout(() => resolve(value), ms);
  });
}

// Punto di ingresso unico per le chiamate a un backend reale. Quando
// NEXT_PUBLIC_DATA_SOURCE=api, i service in questa cartella devono chiamare
// questa funzione al posto dei moduli in data/*, mantenendo invariata la
// firma (stessa Promise, stesso contratto JSON) usata dai componenti.
export async function fetchFromApi(path, options = {}) {
  if (!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL non configurato: impossibile contattare l'API reale.");
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Richiesta API fallita (${response.status}) per ${path}`);
  }
  return response.json();
}
