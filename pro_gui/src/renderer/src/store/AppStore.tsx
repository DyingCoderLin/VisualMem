import React, { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react'
import { apiClient } from '../services/api'
import type { FrontendTheme } from '../services/api'
import { recordingService, RecordingMode, RecordingStatus } from '../services/recording'
import { summaryPopupService } from '../services/summaryPopupService'

export type ViewType = 'timeline' | 'realtime' | 'tags' | 'settings' | 'daily'

export interface SearchResult {
  answer: string
  frames: Array<{
    frame_id: string
    timestamp: string
    image_base64?: string
    image_path?: string
    relevance?: number
  }>
}

interface DateRange {
  earliest_date: string | null
  latest_date: string | null
}

interface AppStoreContextType {
  // Frontend theme
  themeMode: FrontendTheme

  // Date range state
  dateRange: DateRange
  refreshDateRange: () => Promise<void>

  // Recording state
  isRecording: boolean
  /** 首次截图 + store_frame drain，尚未进入稳态 interval */
  isWarmingUp: boolean
  isModelLoading: boolean
  ensureModelsReady: () => Promise<void>
  recordingMode: RecordingMode
  startRecording: () => Promise<void>
  stopRecording: () => Promise<void>
  setRecordingMode: (mode: RecordingMode) => void

  // Refresh timeline
  refreshTimeline: () => void
  timelineRefreshTrigger: number

  // View state
  currentView: ViewType
  setCurrentView: (view: ViewType) => void

  // Search state
  realtimeSearchResult: SearchResult | null
  setRealtimeSearchResult: (result: SearchResult | null) => void
}

const AppStoreContext = createContext<AppStoreContextType | undefined>(undefined)

function normalizeFrontendTheme(theme: unknown): FrontendTheme {
  const normalized = typeof theme === 'string' ? theme.trim().toLowerCase() : ''
  return normalized === 'light' ? 'light' : 'dark'
}

export const useAppStore = () => {
  const context = useContext(AppStoreContext)
  if (!context) {
    throw new Error('useAppStore must be used within AppStoreProvider')
  }
  return context
}

interface AppStoreProviderProps {
  children: ReactNode
}

export const AppStoreProvider: React.FC<AppStoreProviderProps> = ({ children }) => {
  const [themeMode, setThemeMode] = useState<FrontendTheme>(() =>
    normalizeFrontendTheme(import.meta.env.VITE_VISUALMEM_THEME)
  )
  const [dateRange, setDateRange] = useState<DateRange>({
    earliest_date: null,
    latest_date: null
  })
  const [isRecording, setIsRecording] = useState(false)
  const [isWarmingUp, setIsWarmingUp] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [recordingMode, setRecordingModeState] = useState<RecordingMode>(recordingService.getMode())
  const [timelineRefreshTrigger, setTimelineRefreshTrigger] = useState(0)
  const [currentView, setCurrentView] = useState<ViewType>('timeline')
  const [realtimeSearchResult, setRealtimeSearchResult] = useState<SearchResult | null>(null)
  const modelLoadPromiseRef = useRef<Promise<void> | null>(null)

  useEffect(() => {
    const root = document.documentElement
    root.dataset.theme = themeMode
    root.style.colorScheme = themeMode
  }, [themeMode])

  useEffect(() => {
    let cancelled = false
    apiClient.getFrontendConfig()
      .then((frontendConfig) => {
        if (!cancelled) {
          setThemeMode(normalizeFrontendTheme(frontendConfig.theme))
        }
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        console.warn('Failed to fetch frontend config, using bundled theme fallback:', error)
      })

    return () => {
      cancelled = true
    }
  }, [])

  // 设置录制模式
  const setRecordingMode = useCallback((mode: RecordingMode) => {
    recordingService.setMode(mode)
    setRecordingModeState(mode)
  }, [])

  // 刷新日期范围
  const refreshDateRange = useCallback(async () => {
    try {
      const range = await apiClient.getDateRange()
      setDateRange({
        earliest_date: range.earliest_date,
        latest_date: range.latest_date
      })
      // console.log('Date range updated:', range)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      console.error('Failed to fetch date range:', error)
    }
  }, [])

  // 刷新时间轴
  const refreshTimeline = useCallback(() => {
    setTimelineRefreshTrigger(prev => prev + 1)
  }, [])

  const waitForModelsReady = useCallback(async (timeoutMs: number = 300000) => {
    const startedAt = Date.now()
    while (Date.now() - startedAt < timeoutMs) {
      const status = await apiClient.getModelsStatus()
      if (status.loaded) {
        return
      }
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
    throw new Error('Timed out waiting for models to finish loading')
  }, [])

  const ensureModelsReady = useCallback(async () => {
    if (modelLoadPromiseRef.current) {
      return modelLoadPromiseRef.current
    }

    modelLoadPromiseRef.current = (async () => {
      let showLoading = false
      try {
        const modelsStatus = await apiClient.getModelsStatus()
        if (modelsStatus.loaded) {
          return
        }

        showLoading = true
        setIsModelLoading(true)
        if (modelsStatus.loading) {
          await waitForModelsReady()
        } else {
          const result = await apiClient.loadModels()
          if (result.status === 'loading') {
            await waitForModelsReady()
          }
        }
      } finally {
        if (showLoading) {
          setIsModelLoading(false)
        }
        modelLoadPromiseRef.current = null
      }
    })()

    return modelLoadPromiseRef.current
  }, [waitForModelsReady])

  const applyRecordingStatus = useCallback((status: RecordingStatus) => {
    setIsWarmingUp(status.isWarmup)
    setIsRecording(status.isLiveRecording)
    if (status.sessionActive) {
      const now = new Date()
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
      setDateRange((prev) => ({
        ...prev,
        latest_date: today,
      }))
      refreshTimeline()
    } else {
      refreshDateRange()
    }
  }, [refreshDateRange, refreshTimeline])

  // 监听录制服务的状态变化
  useEffect(() => {
    const unsubscribe = recordingService.subscribeStatus(applyRecordingStatus)
    return unsubscribe
  }, [applyRecordingStatus])

  // 开始录制（先确保模型已加载）
  const startRecording = useCallback(async () => {
    try {
      await ensureModelsReady()
      await recordingService.start()
      // setIsRecording 将通过 subscribeStatus 自动更新
    } catch (error) {
      setIsModelLoading(false)
      console.error('Failed to start recording:', error)
    }
  }, [ensureModelsReady])

  // 停止录制
  const stopRecording = useCallback(async () => {
    try {
      await recordingService.stop()
      // setIsRecording 将通过 subscribeStatus 自动更新
    } catch (error) {
      console.error('Failed to stop recording:', error)
    }
  }, [])

  // 初始化：获取日期范围
  useEffect(() => {
    refreshDateRange()
    
    // 每30秒刷新一次日期范围
    const interval = setInterval(refreshDateRange, 30000)
    return () => clearInterval(interval)
  }, [refreshDateRange])

  // 监听录制服务的新帧事件（如果 recordingService 支持）
  useEffect(() => {
    if (isRecording || isWarmingUp) {
      const refreshInterval = setInterval(() => {
        refreshTimeline()
      }, 5000)
      return () => clearInterval(refreshInterval)
    }
  }, [isRecording, isWarmingUp, refreshTimeline])

  // 录制开始时启动摘要弹窗定时任务，录制停止时终止
  useEffect(() => {
    if (isRecording) {
      summaryPopupService.start()
    } else {
      summaryPopupService.stop()
    }
  }, [isRecording])

  const value: AppStoreContextType = {
    themeMode,
    dateRange,
    refreshDateRange,
    isRecording,
    isWarmingUp,
    isModelLoading,
    ensureModelsReady,
    recordingMode,
    startRecording,
    stopRecording,
    setRecordingMode,
    refreshTimeline,
    timelineRefreshTrigger,
    currentView,
    setCurrentView,
    realtimeSearchResult,
    setRealtimeSearchResult
  }

  return (
    <AppStoreContext.Provider value={value}>
      {children}
    </AppStoreContext.Provider>
  )
}
