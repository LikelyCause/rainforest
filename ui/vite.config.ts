import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies API calls to a locally running FastAPI (default :8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // xfwd passes the browser's host on, which the API's same-origin check compares with Origin.
    proxy: { "/api": { target: process.env.API_URL ?? "http://localhost:8000", changeOrigin: true, xfwd: true } },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
