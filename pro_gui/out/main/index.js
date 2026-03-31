"use strict";
const electron = require("electron");
const child_process = require("child_process");
const path = require("path");
const url = require("url");
const fs = require("fs");
const http = require("http");
function _interopNamespaceDefault(e) {
  const n = Object.create(null, { [Symbol.toStringTag]: { value: "Module" } });
  if (e) {
    for (const k in e) {
      if (k !== "default") {
        const d = Object.getOwnPropertyDescriptor(e, k);
        Object.defineProperty(n, k, d.get ? d : {
          enumerable: true,
          get: () => e[k]
        });
      }
    }
  }
  n.default = e;
  return Object.freeze(n);
}
const http__namespace = /* @__PURE__ */ _interopNamespaceDefault(http);
const __dirname$1 = path.dirname(url.fileURLToPath(require("url").pathToFileURL(__filename).href));
const isDev = process.env.NODE_ENV === "development" || !electron.app.isPackaged;
let mainWindow = null;
let pythonProcess = null;
let isDownloading = false;
let isLoadingModel = false;
let isStartingUp = true;
const BACKEND_PORT = 8080;
electron.app.commandLine.appendSwitch("disable-features", [
  "SafeBrowsing",
  "Translate",
  "TranslateUI",
  "OptimizationHints",
  "OptimizationHintsFetching",
  "BackupSignedInSyncTransport",
  "SyncService"
].join(","));
electron.app.commandLine.appendSwitch("disable-background-networking");
electron.app.commandLine.appendSwitch("disable-component-update");
electron.app.commandLine.appendSwitch("no-first-run");
electron.app.commandLine.appendSwitch("no-default-browser-check");
function setupIPC() {
  electron.ipcMain.handle("desktop-capturer-get-sources", async (_event, options) => {
    try {
      const sources = await electron.desktopCapturer.getSources(options);
      return sources;
    } catch (error) {
      console.error("Failed to get desktop sources:", error);
      return [];
    }
  });
  electron.ipcMain.handle("get-project-root", () => {
    return findProjectRoot();
  });
}
setupIPC();
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 1600,
    height: 1e3,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: "#121212",
    titleBarStyle: "hiddenInset",
    frame: true,
    webPreferences: {
      preload: path.join(__dirname$1, "../preload/index.js"),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false
      // 允许截屏需要关闭 webSecurity
    }
  });
  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    if (process.env.OPEN_DEVTOOLS !== "false") {
      mainWindow.webContents.openDevTools();
    }
  } else {
    mainWindow.loadFile(path.join(__dirname$1, "../../renderer/index.html"));
  }
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.webContents.once("did-finish-load", () => {
    if (isDev) {
      electron.globalShortcut.register("CommandOrControl+Shift+I", () => {
        if (mainWindow) {
          if (mainWindow.webContents.isDevToolsOpened()) {
            mainWindow.webContents.closeDevTools();
          } else {
            mainWindow.webContents.openDevTools();
          }
        }
      });
      console.log("Developer tools shortcut registered: Cmd+Shift+I (or Ctrl+Shift+I)");
    }
  });
}
function findProjectRoot() {
  let currentDir = path.resolve(__dirname$1);
  const maxDepth = 10;
  let depth = 0;
  while (depth < maxDepth) {
    const candidate = path.join(currentDir, "gui_backend_server.py");
    if (fs.existsSync(candidate)) {
      return currentDir;
    }
    const parent = path.resolve(currentDir, "..");
    if (parent === currentDir) {
      break;
    }
    currentDir = parent;
    depth++;
  }
  console.warn("Could not find gui_backend_server.py by searching, using fallback path");
  return isDev ? path.join(__dirname$1, "../../..") : path.join(__dirname$1, "../../../..");
}
function checkBackendHealth() {
  return new Promise((resolve2) => {
    const req = http__namespace.get(`http://127.0.0.1:${BACKEND_PORT}/health`, (res) => {
      if (res.statusCode !== 200) {
        console.log(`Health check returned status code: ${res.statusCode}`);
        resolve2(false);
        return;
      }
      let data = "";
      res.on("data", (chunk) => {
        data += chunk;
      });
      res.on("end", () => {
        try {
          const json = JSON.parse(data);
          const isOk = json.status === "ok";
          if (!isOk) {
            console.log("Health check response:", json);
          }
          resolve2(isOk);
        } catch (e) {
          console.error("Failed to parse health check response:", data, e);
          resolve2(false);
        }
      });
      res.on("error", (err) => {
        console.error("Error reading health check response:", err);
        resolve2(false);
      });
    });
    req.on("error", (err) => {
      const errCode = err.code;
      if (errCode !== "ECONNREFUSED" && errCode !== "ECONNRESET") {
        console.error("Health check request error:", errCode, err);
      }
      resolve2(false);
    });
    req.setTimeout(2e3, () => {
      req.destroy();
      resolve2(false);
    });
  });
}
async function waitForBackend(maxRetries = 180, interval = 1e3) {
  console.log("\nWaiting for backend to be ready...");
  let retries = 0;
  while (retries < maxRetries || isDownloading || isLoadingModel) {
    const isReady = await checkBackendHealth();
    if (isReady) {
      console.log("✅ Backend is ready!");
      return true;
    }
    if (isDownloading || isLoadingModel) {
      if (retries % 30 === 0 && retries > 0) {
        if (isDownloading) {
          console.log("⏳ Still downloading model...");
        } else if (isLoadingModel) {
          console.log("⏳ Still loading model into memory (this may take 1-2 minutes on first run)...");
        }
      }
    } else {
      retries++;
    }
    if (pythonProcess && pythonProcess.exitCode !== null) {
      console.error(`❌ Backend process exited with code ${pythonProcess.exitCode}. Check logs/backend_server.log for details.`);
      return false;
    }
    await new Promise((resolve2) => setTimeout(resolve2, interval));
  }
  console.error("❌ Backend failed to start within timeout");
  return false;
}
function startPythonBackend() {
  const rootDir = findProjectRoot();
  const pythonScript = path.join(rootDir, "gui_backend_server.py");
  if (!fs.existsSync(pythonScript)) {
    console.error(`Python backend script not found at: ${pythonScript}`);
    return Promise.reject(new Error("Backend script not found"));
  }
  if (isDev) {
    return new Promise((resolve2, reject) => {
      const logDir = path.join(rootDir, "logs");
      if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
      }
      const logFile = path.join(logDir, "backend_server.log");
      const logStream = fs.createWriteStream(logFile, { flags: "a" });
      console.log("Starting Python backend...");
      console.log(`Backend logs are being redirected to: ${logFile}`);
      pythonProcess = child_process.spawn("python", [pythonScript], {
        cwd: rootDir,
        stdio: ["ignore", "pipe", "pipe"],
        shell: process.platform !== "win32",
        env: {
          ...process.env,
          PYTHONUTF8: process.env.PYTHONUTF8 || "1",
          PYTHONIOENCODING: process.env.PYTHONIOENCODING || "utf-8"
        }
      });
      const handleOutput = (data, isStderr) => {
        const str = data.toString();
        if (isStartingUp) {
          if (isStderr) {
            process.stderr.write(data);
          } else {
            process.stdout.write(data);
          }
        }
        if (str.includes("Starting download")) {
          if (!isDownloading) {
            isDownloading = true;
            console.log("⏳ Detected model download, waiting for it to complete...");
          }
        }
        if (str.includes("download complete!")) {
          console.log("✅ A model download has finished!");
        }
        if (str.includes("Loading encoder") && str.includes("[1/")) {
          console.log("🚀 All pre-flight downloads finished. Backend is now loading models into memory...");
          console.log("⏳ This may take 1-2 minutes on first run (loading 2B+ parameter model)...");
          isDownloading = false;
          isLoadingModel = true;
        }
        if (str.includes("All backend components initialized successfully!")) {
          console.log("✅ All models loaded successfully!");
          isLoadingModel = false;
          isStartingUp = false;
        }
      };
      if (pythonProcess.stdout) {
        pythonProcess.stdout.on("data", (data) => handleOutput(data, false));
        pythonProcess.stdout.pipe(logStream);
      }
      if (pythonProcess.stderr) {
        pythonProcess.stderr.on("data", (data) => handleOutput(data, true));
        pythonProcess.stderr.pipe(logStream);
      }
      pythonProcess.on("error", (error) => {
        console.error("Failed to start Python backend:", error);
        reject(error);
      });
      pythonProcess.on("exit", (code) => {
        console.log(`Python backend exited with code ${code}`);
        pythonProcess = null;
        electron.app.quit();
      });
      setTimeout(() => {
        resolve2();
      }, 2e3);
    });
  } else {
    return Promise.resolve();
  }
}
function stopPythonBackend() {
  if (!pythonProcess?.pid) return;
  const pid = pythonProcess.pid;
  console.log(`Stopping Python backend (PID: ${pid})...`);
  try {
    if (process.platform === "win32") {
      child_process.execSync(`taskkill /pid ${pid} /T /F`, { stdio: "ignore" });
    } else {
      try {
        process.kill(-pid, "SIGTERM");
      } catch (e) {
        pythonProcess.kill("SIGTERM");
      }
      const processToKill = pythonProcess;
      setTimeout(() => {
        try {
          if (processToKill && processToKill.exitCode === null) {
            console.log("Python backend did not exit in time, force killing...");
            try {
              process.kill(-pid, "SIGKILL");
            } catch (e) {
              processToKill.kill("SIGKILL");
            }
          }
        } catch (e) {
        }
      }, 3e3);
    }
  } catch (e) {
    console.log("Process already terminated or error during cleanup:", e);
  }
  pythonProcess = null;
}
electron.app.whenReady().then(async () => {
  try {
    await startPythonBackend();
    const backendReady = await waitForBackend();
    if (backendReady) {
      createWindow();
    } else {
      console.error("Failed to start backend (timeout or backend exited). Check logs/backend_server.log, then try: python gui_backend_server.py");
      electron.app.quit();
    }
  } catch (error) {
    console.error("Error starting backend:", error);
    electron.app.quit();
  }
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});
electron.app.on("window-all-closed", () => {
  electron.globalShortcut.unregisterAll();
  stopPythonBackend();
  electron.app.quit();
});
electron.app.on("before-quit", () => {
  stopPythonBackend();
});
