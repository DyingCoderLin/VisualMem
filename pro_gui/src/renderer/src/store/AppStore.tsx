import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { apiClient } from '../services/api'
import type { FrontendTheme, TaskMemoryAskResponse } from '../services/api'
import { recordingService, RecordingMode, RecordingStatus } from '../services/recording'

export type ViewType = 'timeline' | 'realtime' | 'rewind' | 'tags' | 'settings' | 'daily'

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

export interface RewindAskContext {
  taskMemoryId: string
  title: string
  markdown: string
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

  // Rewind Task Memory ask state
  rewindAskContext: RewindAskContext | null
  setRewindAskContext: (context: RewindAskContext | null) => void
  rewindAskResult: TaskMemoryAskResponse | null
  rewindAskError: string | null
  isRewindAsking: boolean
  askRewindMemory: (question: string) => Promise<void>
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
  const [rewindAskContextState, setRewindAskContextState] = useState<RewindAskContext | null>(null)
  const [rewindAskResult, setRewindAskResult] = useState<TaskMemoryAskResponse | null>(null)
  const [rewindAskError, setRewindAskError] = useState<string | null>(null)
  const [isRewindAsking, setIsRewindAsking] = useState(false)

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

  const setRewindAskContext = useCallback((context: RewindAskContext | null) => {
    setRewindAskContextState((prev) => {
      if ((prev?.taskMemoryId || null) !== (context?.taskMemoryId || null)) {
        setRewindAskResult(null)
        setRewindAskError(null)
      }
      return context
    })
  }, [])

  const askRewindMemory = useCallback(async (question: string) => {
    const trimmed = question.trim()
    if (!trimmed) return
    if (!rewindAskContextState) {
      setRewindAskResult(null)
      setRewindAskError('Build or select a Task Memory first.')
      return
    }

    setIsRewindAsking(true)
    setRewindAskResult(null)
    setRewindAskError(null)
    try {
      const response = await apiClient.askTaskMemory(rewindAskContextState.taskMemoryId, {
        question: trimmed,
        markdown: rewindAskContextState.markdown
      })
      setRewindAskResult(response)
    } catch (error) {
      const message =
        error instanceof DOMException && error.name === 'AbortError'
          ? 'Ask timed out. The AI service did not return in time; try again with a narrower Task Memory.'
          : error instanceof Error
            ? error.message
            : 'Ask failed.'
      setRewindAskError(message)
    } finally {
      setIsRewindAsking(false)
    }
  }, [rewindAskContextState])

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
      // Check if models are loaded, if not, load them first
      const modelsStatus = await apiClient.getModelsStatus()
      if (!modelsStatus.loaded) {
        setIsModelLoading(true)
        try {
          const result = await apiClient.loadModels()
          if (result.status === 'loading') {
            await waitForModelsReady()
          }
        } finally {
          setIsModelLoading(false)
        }
      }
      await recordingService.start()
      // setIsRecording 将通过 subscribeStatus 自动更新
    } catch (error) {
      setIsModelLoading(false)
      console.error('Failed to start recording:', error)
    }
  }, [waitForModelsReady])

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

  const value: AppStoreContextType = {
    themeMode,
    dateRange,
    refreshDateRange,
    isRecording,
    isWarmingUp,
    isModelLoading,
    recordingMode,
    startRecording,
    stopRecording,
    setRecordingMode,
    refreshTimeline,
    timelineRefreshTrigger,
    currentView,
    setCurrentView,
    realtimeSearchResult,
    setRealtimeSearchResult,
    rewindAskContext: rewindAskContextState,
    setRewindAskContext,
    rewindAskResult,
    rewindAskError,
    isRewindAsking,
    askRewindMemory
  }

  return (
    <AppStoreContext.Provider value={value}>
      {children}
    </AppStoreContext.Provider>
  )
}
