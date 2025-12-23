import { app, BrowserWindow, ipcMain, desktopCapturer, globalShortcut } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { join, dirname, resolve } from 'path'
import { fileURLToPath } from 'url'
import { existsSync } from 'fs'
import * as http from 'http'

const __dirname = dirname(fileURLToPath(import.meta.url))

// 开发环境检测
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
const BACKEND_PORT = 8080

// 设置 IPC 处理器
function setupIPC(): void {
  // 处理桌面截屏请求
  ipcMain.handle('desktop-capturer-get-sources', async (_event, options) => {
    try {
      const sources = await desktopCapturer.getSources(options)
      return sources
    } catch (error) {
      console.error('Failed to get desktop sources:', error)
      return []
    }
  })

  // 获取项目根目录
  ipcMain.handle('get-project-root', () => {
    return findProjectRoot()
  })
}

// 初始化 IPC 处理器
setupIPC()

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#121212',
    titleBarStyle: 'hiddenInset',
    frame: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false, // 允许截屏需要关闭 webSecurity
    }
  })

  if (isDev) {
    // 开发模式：连接到 Vite 开发服务器
    mainWindow.loadURL('http://localhost:5173')
    // 开发模式下默认打开开发者工具，除非明确禁用
    // 使用 OPEN_DEVTOOLS=false npm run dev 来禁用
    if (process.env.OPEN_DEVTOOLS !== 'false') {
      mainWindow.webContents.openDevTools()
    }
  } else {
    // 生产模式：加载构建后的文件
    mainWindow.loadFile(join(__dirname, '../../renderer/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
  
  // 在窗口准备好后注册快捷键（开发模式）
  mainWindow.webContents.once('did-finish-load', () => {
    if (isDev) {
      // 注册快捷键切换开发者工具
      globalShortcut.register('CommandOrControl+Shift+I', () => {
        if (mainWindow) {
          if (mainWindow.webContents.isDevToolsOpened()) {
            mainWindow.webContents.closeDevTools()
          } else {
            mainWindow.webContents.openDevTools()
          }
        }
      })
      console.log('Developer tools shortcut registered: Cmd+Shift+I (or Ctrl+Shift+I)')
    }
  })
}

function findProjectRoot(): string {
  // 从当前目录向上查找，直到找到包含 gui_backend_server.py 的目录
  let currentDir = resolve(__dirname)
  const maxDepth = 10
  let depth = 0
  
  while (depth < maxDepth) {
    const candidate = join(currentDir, 'gui_backend_server.py')
    if (existsSync(candidate)) {
      return currentDir
    }
    const parent = resolve(currentDir, '..')
    if (parent === currentDir) {
      // 到达文件系统根目录
      break
    }
    currentDir = parent
    depth++
  }
  
  // 如果没找到，回退到从 __dirname 向上两级（开发模式）或三级（生产模式）
  console.warn('Could not find gui_backend_server.py by searching, using fallback path')
  return isDev ? join(__dirname, '../../..') : join(__dirname, '../../../..')
}

function checkBackendHealth(): Promise<boolean> {
  return new Promise((resolve) => {
    // 使用 127.0.0.1 而不是 localhost，避免 IPv6 连接问题
    const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/health`, (res: http.IncomingMessage) => {
      // 检查状态码
      if (res.statusCode !== 200) {
        console.log(`Health check returned status code: ${res.statusCode}`)
        resolve(false)
        return
      }
      
      let data = ''
      res.on('data', (chunk: string) => {
        data += chunk
      })
      res.on('end', () => {
        try {
          const json = JSON.parse(data)
          const isOk = json.status === 'ok'
          if (!isOk) {
            console.log('Health check response:', json)
          }
          resolve(isOk)
        } catch (e) {
          console.error('Failed to parse health check response:', data, e)
          resolve(false)
        }
      })
      
      res.on('error', (err) => {
        console.error('Error reading health check response:', err)
        resolve(false)
      })
    })
    
    req.on('error', (err) => {
      const errCode = (err as NodeJS.ErrnoException).code
      // 连接错误（ECONNREFUSED）是正常的，表示后端还没启动
      // 只在非连接错误时打印日志
      if (errCode !== 'ECONNREFUSED') {
        console.error('Health check request error:', errCode, err)
      }
      resolve(false)
    })
    
    req.setTimeout(2000, () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function waitForBackend(maxRetries: number = 60, interval: number = 1000): Promise<boolean> {
  console.log('Waiting for backend to be ready...')
  
  for (let i = 0; i < maxRetries; i++) {
    const isReady = await checkBackendHealth()
    if (isReady) {
      console.log('✅ Backend is ready!')
      return true
    }
    
    if (i < maxRetries - 1) {
      // console.log(`⏳ Backend not ready yet, retrying in ${interval}ms... (${i + 1}/${maxRetries})`)
      await new Promise(resolve => setTimeout(resolve, interval))
    }
  }
  
  console.error('❌ Backend failed to start within timeout')
  return false
}

function startPythonBackend(): Promise<void> {
  // 检测 Python 后端是否已经在运行
  // 如果后端已经在运行（例如通过外部启动），则不需要启动
  // gui_backend_server.py 位于项目根目录，不在 pro_gui 目录下
  
  const rootDir = findProjectRoot()
  const pythonScript = join(rootDir, 'gui_backend_server.py')
  
  // console.log('Starting Python backend...')
  // console.log(`Python script path: ${pythonScript}`)
  // console.log(`Working directory: ${rootDir}`)
  
  if (!existsSync(pythonScript)) {
    console.error(`Python backend script not found at: ${pythonScript}`)
    return Promise.reject(new Error('Backend script not found'))
  }
  
  // 在开发模式下，可以启动 Python 后端
  // 生产模式下，假设后端已经通过其他方式启动
  if (isDev) {
    return new Promise((resolve, reject) => {
      pythonProcess = spawn('python', [pythonScript], {
        cwd: rootDir,
        stdio: 'inherit',
        shell: process.platform === 'win32'
      })

      pythonProcess.on('error', (error) => {
        console.error('Failed to start Python backend:', error)
        reject(error)
      })

      pythonProcess.on('exit', (code) => {
        console.log(`Python backend exited with code ${code}`)
        pythonProcess = null
        // 如果后端退出，前端也退出
        app.quit()
      })
      
      // 不等待进程退出，而是等待健康检查通过
      // 给后端一些时间启动
      setTimeout(() => {
        resolve()
      }, 2000)
    })
  } else {
    // 生产模式，假设后端已启动，直接返回
    return Promise.resolve()
  }
}

function stopPythonBackend(): void {
  if (pythonProcess) {
    pythonProcess.kill()
    pythonProcess = null
  }
}

app.whenReady().then(async () => {
  // 先启动后端，等待后端就绪后再创建窗口
  try {
    await startPythonBackend()
    const backendReady = await waitForBackend()
    
    if (backendReady) {
      // 后端就绪后再创建窗口
      createWindow()
    } else {
      console.error('Failed to start backend, exiting...')
      app.quit()
    }
  } catch (error) {
    console.error('Error starting backend:', error)
    app.quit()
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  // 注销所有全局快捷键
  globalShortcut.unregisterAll()
  stopPythonBackend()
  // 无论什么平台，关闭所有窗口后都退出应用
  app.quit()
})

app.on('before-quit', () => {
  stopPythonBackend()
})

