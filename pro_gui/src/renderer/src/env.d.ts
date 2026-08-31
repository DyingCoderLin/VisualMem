export {}

export interface SummaryData {
  summary: string
  time_range: string
  error?: string
}

declare global {
  interface Window {
    electronAPI: {
      desktopCapturer: {
        getSources: (options: any) => Promise<any[]>
      }
      getProjectRoot: () => Promise<string>
      showSummaryPopup: (data: any) => void
      closeSummaryPopup: () => void
      onSummaryData: (callback: (data: any) => void) => (() => void)
      requestAdvice: (summary: string) => void
      onAdviceRequested: (callback: (summary: string) => void) => (() => void)
      onPopupConfig: (callback: (config: Record<string, any>) => void) => (() => void)
    }
  }
}
