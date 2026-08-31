import { contextBridge, ipcRenderer } from 'electron'

export interface SummaryData {
  summary: string
  time_range: string
  error?: string
}

// 暴露受保护的方法给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  desktopCapturer: {
    getSources: async (options: Electron.SourcesOptions) => {
      return await ipcRenderer.invoke('desktop-capturer-get-sources', options)
    }
  },
  getProjectRoot: async () => {
    return await ipcRenderer.invoke('get-project-root')
  },
  showSummaryPopup: (data: any) => {
    ipcRenderer.send('show-summary-popup', data)
  },
  closeSummaryPopup: () => {
    ipcRenderer.send('close-summary-popup')
  },
  onSummaryData: (callback: (data: any) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: any) => {
      callback(data)
    }
    ipcRenderer.on('summary-data', listener)
    return () => {
      ipcRenderer.removeListener('summary-data', listener)
    }
  },
  requestAdvice: (summary: string) => {
    ipcRenderer.send('request-advice', summary)
  },
  onAdviceRequested: (callback: (summary: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, summary: string) => {
      callback(summary)
    }
    ipcRenderer.on('advice-requested', listener)
    return () => {
      ipcRenderer.removeListener('advice-requested', listener)
    }
  },
  onPopupConfig: (callback: (config: Record<string, any>) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, config: Record<string, any>) => {
      callback(config)
    }
    ipcRenderer.on('popup-config', listener)
    return () => {
      ipcRenderer.removeListener('popup-config', listener)
    }
  }
})
