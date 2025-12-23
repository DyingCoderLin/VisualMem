import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { apiClient } from '../services/api'
import { recordingService } from '../services/recording'

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
  startRecording: () => Promise<void>
  stopRecording: () => Promise<void>
  
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
  const [timelineRefreshTrigger, setTimelineRefreshTrigger] = useState(0)
  const [currentView, setCurrentView] = useState<ViewType>('timeline')
  const [realtimeSearchResult, setRealtimeSearchResult] = useState<SearchResult | null>(null)

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

  // 开始录制
  const startRecording = useCallback(async () => {
    try {
      await recordingService.start()
      setIsRecording(true)
      
      // 更新日期范围：将 latest_date 设置为今天（如果录制开始）
      const now = new Date()
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
      setDateRange(prev => ({
        ...prev,
        latest_date: today
      }))
      
      // 刷新时间轴，让新录制的帧显示出来
      refreshTimeline()
      
      console.log('Recording started')
    } catch (error) {
      console.error('Failed to start recording:', error)
      setIsRecording(false)
    }
  }, [refreshTimeline])

  // 停止录制
  const stopRecording = useCallback(async () => {
    recordingService.stop()
    setIsRecording(false)
    console.log('Recording stopped')
    
    // 停止后刷新日期范围和时间轴
    await refreshDateRange()
    refreshTimeline()
  }, [refreshDateRange, refreshTimeline])

  // 初始化：获取日期范围
  useEffect(() => {
    refreshDateRange()
    
    // 检查录制状态：不仅检查本地 service，还要检查后端
    const checkStatus = async () => {
      const status = recordingService.getStatus()
      setIsRecording(status)
      
      // 如果本地认为没在录制，但实际上后端可能还在接收数据（或者前端刚刷新）
      // 我们需要确保 recordingService 的状态与 UI 同步
      // 注意：由于录制是在前端 RecordingService 维护的定时器，
      // 如果前端刷新导致 RecordingService 实例重建，我们需要恢复它的定时器
      if (status && !isRecording) {
        console.log('Restoring recording state after UI refresh')
        setIsRecording(true)
      }
    }
    
    checkStatus()
    
    // 每30秒刷新一次日期范围
    const interval = setInterval(refreshDateRange, 30000)
    return () => clearInterval(interval)
  }, [refreshDateRange, isRecording])

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
    startRecording,
    stopRecording,
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

