import { defineConfig } from "vite";

// GitHub Pages serves the site from /<repo>/; set SITE_BASE=/ for local or custom-domain builds.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/rl-warehouse-fulfillment/",
  build: {
    target: "es2022",
    outDir: "dist",
    rollupOptions: { output: { manualChunks: { three: ["three"] } } },
    chunkSizeWarningLimit: 700,
  },
});
