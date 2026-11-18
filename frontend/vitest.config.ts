import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  // Vitest 2 bundles Vite 5 typings while the app uses Vite 6; the plugin works with both at runtime.
  plugins: [vue() as never],
  test: {
    environment: 'jsdom',
  },
})
