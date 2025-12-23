export {}

declare global {
  interface Window {
    electronAPI: {
      desktopCapturer: {
        getSources: (options: any) => Promise<any[]>
      }
      getProjectRoot: () => Promise<string>
    }
  }
}
