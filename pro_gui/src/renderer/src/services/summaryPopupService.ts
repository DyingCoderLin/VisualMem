import { apiClient } from './api'

const SUMMARY_INTERVAL_MS = 5 * 60 * 1000

export interface SummaryPopupData {
  type: 'summary' | 'advice'
  summary?: string
  advice?: string
  time_range?: string
  error?: string
}

class SummaryPopupService {
  private intervalId: number | null = null
  private isRunning = false
  private currentSummary = ''

  start(): void {
    if (this.isRunning) {
      console.log('[SummaryPopup] Already running, ignoring start')
      return
    }
    this.isRunning = true
    console.log('[SummaryPopup] Started')

    this.runSummaryCycle()

    this.intervalId = window.setInterval(() => {
      if (this.isRunning) {
        this.runSummaryCycle()
      }
    }, SUMMARY_INTERVAL_MS)

    if (window.electronAPI && window.electronAPI.onAdviceRequested) {
      window.electronAPI.onAdviceRequested((summary: string) => {
        this.handleAdviceRequest(summary)
      })
    }
  }

  stop(): void {
    if (!this.isRunning) return
    this.isRunning = false

    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }

    if (window.electronAPI && window.electronAPI.closeSummaryPopup) {
      window.electronAPI.closeSummaryPopup()
    }
    console.log('[SummaryPopup] Stopped')
  }

  private async runSummaryCycle(): Promise<void> {
    if (!this.isRunning) return
    console.log('[SummaryPopup] Running summary cycle...')
    try {
      const result = await apiClient.summarizeRecent(5)
      if (this.isRunning) {
        this.currentSummary = result.summary
        this.sendToPopup({
          type: 'summary',
          summary: result.summary,
          time_range: result.time_range,
        })
        console.log('[SummaryPopup] Summary sent to popup')
      }
    } catch (error) {
      console.error('[SummaryPopup] Failed to generate summary:', error)
      if (this.isRunning) {
        this.sendToPopup({
          type: 'summary',
          summary: '',
          time_range: '',
          error: `Summary generation failed: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    }
  }

  private async handleAdviceRequest(summary: string): Promise<void> {
    if (!this.isRunning) return
    try {
      const result = await apiClient.suggestAdvice(summary)
      if (this.isRunning) {
        this.sendToPopup({
          type: 'advice',
          advice: result.advice,
          summary: summary,
          time_range: undefined,
        })
        console.log('[SummaryPopup] Advice sent to popup')
      }
    } catch (error) {
      console.error('[SummaryPopup] Failed to generate advice:', error)
      if (this.isRunning) {
        this.sendToPopup({
          type: 'advice',
          error: `Advice generation failed: ${error instanceof Error ? error.message : String(error)}`,
        })
      }
    }
  }

  private sendToPopup(data: SummaryPopupData): void {
    if (window.electronAPI && window.electronAPI.showSummaryPopup) {
      window.electronAPI.showSummaryPopup(data as any)
    }
  }
}

export const summaryPopupService = new SummaryPopupService()
