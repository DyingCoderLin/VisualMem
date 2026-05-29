#!/usr/bin/env node

const { spawn } = require('child_process')

const command = process.platform === 'win32' ? 'electron-vite.cmd' : 'electron-vite'
const child = spawn(command, ['dev'], {
  env: {
    ...process.env,
    OPEN_DEVTOOLS: 'false'
  },
  shell: process.platform === 'win32',
  stdio: 'inherit',
  windowsHide: true
})

child.on('error', (error) => {
  console.error(error)
  process.exit(1)
})

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal)
    return
  }

  process.exit(code || 0)
})
