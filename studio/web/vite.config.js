import { defineConfig } from "vite";

export default defineConfig({
  esbuild: { jsx: "automatic" },
  server: {
    host: "0.0.0.0",
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:7861" },
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1100,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/three/")) return "three";
          if (id.includes("/node_modules/@google/model-viewer/"))
            return "model-viewer";
        },
      },
    },
  },
});
