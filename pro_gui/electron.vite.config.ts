import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'fs'

function copySummaryPopupPlugin() {
  return {
    name: 'copy-summary-popup',
    closeBundle() {
      const src = resolve('src/renderer/summary-popup.html')
      const outDir = resolve('out/renderer')
      const content = readFileSync(src, 'utf-8')
      mkdirSync(outDir, { recursive: true })
      writeFileSync(resolve(outDir, 'summary-popup.html'), content, 'utf-8')
    }
  }
}

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()]
  },
  preload: {
    plugins: [externalizeDepsPlugin()]
  },
  renderer: {
    resolve: {
      alias: {
        '@': resolve('src/renderer/src')
      }
    },
    plugins: [react(), copySummaryPopupPlugin()]
  }
})

