import { contextBridge, ipcRenderer } from 'electron'

// 暴露受保护的方法给渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
  desktopCapturer: {
    getSources: async (options: Electron.SourcesOptions) => {
      return await ipcRenderer.invoke('desktop-capturer-get-sources', options)
    }
  },
  getProjectRoot: async () => {
    return await ipcRenderer.invoke('get-project-root')
  }
})
