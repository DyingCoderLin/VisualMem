#!/usr/bin/env node

const fs = require('fs')
const os = require('os')
const path = require('path')
const { spawnSync } = require('child_process')

const LOG_PREFIX = '[ensure-electron]'
const DEFAULT_ELECTRON_MIRROR = 'https://npmmirror.com/mirrors/electron/'

function log(message) {
  console.log(`${LOG_PREFIX} ${message}`)
}

function warn(message) {
  console.warn(`${LOG_PREFIX} ${message}`)
}

function fail(message, error) {
  console.error(`${LOG_PREFIX} ${message}`)
  if (error) {
    console.error(error.stack || error.message || error)
  }
  process.exit(1)
}

function getElectronPackageDir() {
  try {
    return path.dirname(require.resolve('electron/package.json'))
  } catch (error) {
    fail('Electron dependency is missing. Run npm install in pro_gui first.', error)
  }
}

function getPlatform() {
  return process.env.npm_config_platform || os.platform()
}

function getArch() {
  return process.env.npm_config_arch || os.arch()
}

function getElectronMirror() {
  return (
    process.env.npm_config_electron_mirror ||
    process.env.NPM_CONFIG_ELECTRON_MIRROR ||
    process.env.ELECTRON_MIRROR ||
    process.env.npm_package_config_electron_mirror ||
    DEFAULT_ELECTRON_MIRROR
  )
}

function withElectronDownloadDefaults(env) {
  const nextEnv = { ...env }
  delete nextEnv.ELECTRON_SKIP_BINARY_DOWNLOAD

  if (
    !nextEnv.npm_config_electron_mirror &&
    !nextEnv.NPM_CONFIG_ELECTRON_MIRROR &&
    !nextEnv.ELECTRON_MIRROR
  ) {
    nextEnv.ELECTRON_MIRROR = DEFAULT_ELECTRON_MIRROR
  }

  return nextEnv
}

function getPlatformPath(platform) {
  switch (platform) {
    case 'mas':
    case 'darwin':
      return 'Electron.app/Contents/MacOS/Electron'
    case 'freebsd':
    case 'openbsd':
    case 'linux':
      return 'electron'
    case 'win32':
      return 'electron.exe'
    default:
      throw new Error(`Electron builds are not available on platform: ${platform}`)
  }
}

function getInstallState(electronDir, version, platformPath) {
  const distDir = process.env.ELECTRON_OVERRIDE_DIST_PATH || path.join(electronDir, 'dist')
  const pathFile = path.join(electronDir, 'path.txt')
  const versionFile = path.join(distDir, 'version')

  try {
    const installedVersion = fs.readFileSync(versionFile, 'utf8').trim().replace(/^v/, '')
    const installedPath = fs.readFileSync(pathFile, 'utf8')
    const executablePath = path.join(distDir, installedPath)

    return {
      ok: installedVersion === version && installedPath === platformPath && fs.existsSync(executablePath),
      distDir,
      executablePath,
      installedVersion,
      installedPath
    }
  } catch {
    return {
      ok: false,
      distDir,
      executablePath: path.join(distDir, platformPath)
    }
  }
}

function runElectronPostinstall(electronDir) {
  const installScript = path.join(electronDir, 'install.js')
  if (!fs.existsSync(installScript)) {
    return
  }

  log('running Electron postinstall check')
  const result = spawnSync(process.execPath, [installScript], {
    cwd: electronDir,
    env: withElectronDownloadDefaults(process.env),
    stdio: 'inherit'
  })

  if (result.error) {
    warn(`Electron postinstall could not run: ${result.error.message}`)
  } else if (result.status !== 0) {
    warn(`Electron postinstall exited with code ${result.status}`)
  }
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    windowsHide: true
  })

  return !result.error && result.status === 0
}

function powershellQuote(value) {
  return `'${value.replace(/'/g, "''")}'`
}

async function tryExtractZipPackage(zipPath, distDir) {
  let extract
  try {
    extract = require('extract-zip')
  } catch {
    return false
  }

  try {
    await Promise.race([
      extract(zipPath, { dir: distDir }),
      new Promise((_, reject) => {
        setTimeout(() => reject(new Error('extract-zip timed out')), 120000)
      })
    ])
    return true
  } catch (error) {
    warn(`extract-zip fallback failed: ${error.message}`)
    return false
  }
}

async function extractZip(zipPath, distDir) {
  fs.mkdirSync(distDir, { recursive: true })

  const attempts = []
  if (process.platform === 'win32') {
    const systemTar = process.env.SystemRoot
      ? path.join(process.env.SystemRoot, 'System32', 'tar.exe')
      : 'tar.exe'
    attempts.push({
      name: 'Windows tar',
      command: systemTar,
      args: ['-xf', zipPath, '-C', distDir]
    })
    attempts.push({
      name: 'PowerShell Expand-Archive',
      command: 'powershell.exe',
      args: [
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        `Expand-Archive -LiteralPath ${powershellQuote(zipPath)} -DestinationPath ${powershellQuote(distDir)} -Force`
      ]
    })
  } else {
    attempts.push({
      name: 'unzip',
      command: 'unzip',
      args: ['-oq', zipPath, '-d', distDir]
    })
    attempts.push({
      name: 'bsdtar',
      command: 'bsdtar',
      args: ['-xf', zipPath, '-C', distDir]
    })
  }

  for (const attempt of attempts) {
    log(`extracting Electron with ${attempt.name}`)
    if (runCommand(attempt.command, attempt.args)) {
      return
    }
  }

  log('extracting Electron with extract-zip fallback')
  if (await tryExtractZipPackage(zipPath, distDir)) {
    return
  }

  throw new Error(`Unable to extract Electron archive: ${zipPath}`)
}

async function downloadAndInstall(electronDir, version, platform, arch, platformPath) {
  const { downloadArtifact } = require('@electron/get')
  const checksumsPath = path.join(electronDir, 'checksums.json')
  const checksums = fs.existsSync(checksumsPath) ? require(checksumsPath) : undefined
  const distDir = process.env.ELECTRON_OVERRIDE_DIST_PATH || path.join(electronDir, 'dist')

  log(`downloading Electron ${version} for ${platform}-${arch}`)
  const zipPath = await downloadArtifact({
    version,
    artifactName: 'electron',
    platform,
    arch,
    checksums,
    mirrorOptions: {
      mirror: getElectronMirror()
    }
  })

  log(`using Electron archive ${zipPath}`)
  await extractZip(zipPath, distDir)

  const extractedTypeDefinitions = path.join(distDir, 'electron.d.ts')
  const packageTypeDefinitions = path.join(electronDir, 'electron.d.ts')
  if (fs.existsSync(extractedTypeDefinitions)) {
    fs.copyFileSync(extractedTypeDefinitions, packageTypeDefinitions)
    fs.rmSync(extractedTypeDefinitions, { force: true })
  }

  fs.writeFileSync(path.join(electronDir, 'path.txt'), platformPath)
}

async function main() {
  const electronDir = getElectronPackageDir()
  const { version } = require(path.join(electronDir, 'package.json'))
  const platform = getPlatform()
  const arch = getArch()
  const platformPath = getPlatformPath(platform)

  let state = getInstallState(electronDir, version, platformPath)
  if (state.ok) {
    log(`Electron is ready: ${state.executablePath}`)
    return
  }

  runElectronPostinstall(electronDir)

  state = getInstallState(electronDir, version, platformPath)
  if (state.ok) {
    log(`Electron is ready: ${state.executablePath}`)
    return
  }

  await downloadAndInstall(electronDir, version, platform, arch, platformPath)

  state = getInstallState(electronDir, version, platformPath)
  if (!state.ok) {
    fail(`Electron install repair did not produce ${state.executablePath}`)
  }

  log(`Electron is ready: ${state.executablePath}`)
}

main().catch((error) => {
  fail('Electron install check failed.', error)
})
