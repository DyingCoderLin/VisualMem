import React, { useState, useEffect, useRef } from 'react'
import { apiClient } from '../services/api'
import { useAppStore } from '../store/AppStore'

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

interface SearchBarProps {
  onSearchResult: (result: SearchResult | null) => void
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearchResult }) => {
  const [queryByView, setQueryByView] = useState<Record<string, string>>({})
  const [isSearching, setIsSearching] = useState(false)
  const searchRequestRef = useRef<AbortController | null>(null)

  const {
    isRecording,
    isWarmingUp,
    isModelLoading,
    startRecording,
    stopRecording,
    recordingMode,
    setRecordingMode,
    currentView,
    setRealtimeSearchResult,
    rewindAskContext,
    isRewindAsking,
    askRewindMemory
  } = useAppStore()
  const query = queryByView[currentView] || ''
  const isRewindMode = currentView === 'rewind'
  const isAskEnabledView = currentView === 'timeline' || currentView === 'realtime' || currentView === 'rewind'
  const isBusy = isRewindMode ? isRewindAsking : isSearching
  const searchPlaceholder = isRewindMode
    ? rewindAskContext
      ? 'Ask with this memory...'
      : 'Build or select a Task Memory first'
    : isAskEnabledView
      ? 'Ask VisualMem...'
      : 'Ask is available in Timeline, Real-time, or Rewind'

  const updateQuery = (value: string) => {
    setQueryByView((prev) => ({
      ...prev,
      [currentView]: value
    }))
  }

  const handleSearch = async () => {
    if (!query.trim()) return
    if (!isAskEnabledView) return

    if (isBusy) {
      console.log('Search already in progress, ignoring duplicate request')
      return
    }

    if (isRewindMode) {
      await askRewindMemory(query.trim())
      return
    }

    if (searchRequestRef.current) {
      searchRequestRef.current.abort()
    }

    const abortController = new AbortController()
    searchRequestRef.current = abortController

    setIsSearching(true)
    try {
      let startTime = undefined as string | undefined
      let endTime = undefined as string | undefined

      if (currentView === 'realtime') {
        const now = new Date()
        const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000)
        startTime = fiveMinutesAgo.toISOString()
        endTime = now.toISOString()
      }

      const result = await apiClient.queryRagWithTime(
        {
          query: query.trim(),
          start_time: startTime,
          end_time: endTime,
          search_type: 'image'
        },
        abortController.signal
      )

      if (abortController.signal.aborted) {
        return
      }

      if (currentView === 'realtime') {
        setRealtimeSearchResult(result)
      } else {
        onSearchResult(result)
      }
    } catch (error: any) {
      if (error.name === 'AbortError' || abortController.signal.aborted) {
        return
      }
      console.error('Search failed:', error)
      onSearchResult({
        answer: '搜索失败，请稍后重试。',
        frames: []
      })
    } finally {
      if (!abortController.signal.aborted) {
        setIsSearching(false)
        searchRequestRef.current = null
      }
    }
  }

  const handleToggleRecording = async () => {
    try {
      if (isRecording || isWarmingUp) {
        stopRecording()
      } else {
        await startRecording()
      }
    } catch (error) {
      console.error('Recording toggle failed:', error)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isBusy) {
      e.preventDefault()
      handleSearch()
    }
  }

  useEffect(() => {
    return () => {
      if (searchRequestRef.current) {
        searchRequestRef.current.abort()
      }
    }
  }, [])

  return (
    <>
      <div className="search-container">
        <div className="search-input-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder={searchPlaceholder}
            value={query}
            onChange={(e) => updateQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={!isAskEnabledView || (isRewindMode && !rewindAskContext)}
          />
        </div>

        <div className="toggle-group">
          <button
            className={`toggle-btn ${recordingMode === 'primary' ? 'active' : ''}`}
            onClick={() => setRecordingMode('primary')}
            title="仅录制主屏幕"
          >
            主屏幕
          </button>
          <button
            className={`toggle-btn ${recordingMode === 'all' ? 'active' : ''}`}
            onClick={() => setRecordingMode('all')}
            title="录制所有扩展屏幕"
          >
            所有屏幕
          </button>
        </div>

        {isModelLoading ? (
          <div className="record-btn-loading">
            <svg className="loading-spinner" width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="50 20" />
            </svg>
            <span className="loading-text">Loading model</span>
          </div>
        ) : isWarmingUp ? (
          <div className="record-btn-loading" style={{ gap: '10px' }}>
            <svg className="loading-spinner" width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="50 20" />
            </svg>
            <span className="loading-text">Warming up</span>
            <button
              type="button"
              className="record-btn recording"
              onClick={() => stopRecording()}
              title="Stop warming up"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" />
              </svg>
            </button>
          </div>
        ) : (
          <button
            className={`record-btn ${isRecording ? 'recording' : ''}`}
            onClick={handleToggleRecording}
            title={isRecording ? '停止录制' : '开始录制'}
          >
            {isRecording ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style={{ marginLeft: '2px' }}>
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
        )}

        <button
          className="btn btn-primary"
          onClick={handleSearch}
          disabled={isBusy || !isAskEnabledView || (isRewindMode && !rewindAskContext)}
        >
          {isBusy ? (isRewindMode ? 'Asking...' : 'Searching...') : (isRewindMode ? 'Ask' : 'Search')}
        </button>
      </div>
    </>
  )
}

export default SearchBar
