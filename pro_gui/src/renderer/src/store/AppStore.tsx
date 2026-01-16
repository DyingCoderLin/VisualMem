import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { apiClient } from '../services/api'
import { recordingService, RecordingMode } from '../services/recording'

export type ViewType = 'timeline' | 'realtime' | 'tags' | 'settings'

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
  // Date range state
  dateRange: DateRange
  refreshDateRange: () => Promise<void>
  
  // Recording state
  isRecording: boolean
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
  const [dateRange, setDateRange] = useState<DateRange>({
    earliest_date: null,
    latest_date: null
  })
  const [isRecording, setIsRecording] = useState(false)
  const [recordingMode, setRecordingModeState] = useState<RecordingMode>(recordingService.getMode())
  const [timelineRefreshTrigger, setTimelineRefreshTrigger] = useState(0)
  const [currentView, setCurrentView] = useState<ViewType>('timeline')
  const [realtimeSearchResult, setRealtimeSearchResult] = useState<SearchResult | null>(null)

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
      console.error('Failed to fetch date range:', error)
    }
  }, [])

  // 刷新时间轴
  const refreshTimeline = useCallback(() => {
    setTimelineRefreshTrigger(prev => prev + 1)
  }, [])

  // 监听录制服务的状态变化
  useEffect(() => {
    const unsubscribe = recordingService.subscribeStatus((status) => {
      setIsRecording(status)
      if (status) {
        // 如果录制开启，确保日期范围包含今天
        const now = new Date()
        const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
        setDateRange(prev => ({
          ...prev,
          latest_date: today
        }))
        refreshTimeline()
      } else {
        refreshDateRange()
      }
    })
    return unsubscribe
  }, [refreshDateRange, refreshTimeline])

  // 开始录制
  const startRecording = useCallback(async () => {
    try {
      await recordingService.start()
      // setIsRecording 将通过 subscribeStatus 自动更新
    } catch (error) {
      console.error('Failed to start recording:', error)
    }
  }, [])

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
    // 如果录制中，定期刷新时间轴以显示新录制的帧
    if (isRecording) {
      const refreshInterval = setInterval(() => {
        refreshTimeline()
      }, 5000) // 每5秒刷新一次
      
      return () => clearInterval(refreshInterval)
    }
  }, [isRecording, refreshTimeline])

  const value: AppStoreContextType = {
    dateRange,
    refreshDateRange,
    isRecording,
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

