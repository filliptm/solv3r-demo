import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const api = process.env.SOLV3R_API ?? "http://127.0.0.1:7860";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": { target: api, changeOrigin: true } } },
  build: { chunkSizeWarningLimit: 1200 },
});
