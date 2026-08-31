import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
  // Il progetto usa file .js (non .jsx) per i componenti React, come richiesto
  // dalle linee guida del frontend: esbuild deve trattarli come JSX.
  esbuild: {
    loader: "jsx",
    include: /.*\.js$/,
    exclude: /node_modules/,
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: { ".js": "jsx" },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.js"],
    css: true,
  },
});
