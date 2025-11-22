import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // Use root base for local development; the CI/deploy can set
  // the repo subpath when publishing (e.g. GitHub Pages).
  base: '/',
  plugins: [
    tailwindcss(),
  ],
  build: {
    outDir: 'dist'
  }
})
