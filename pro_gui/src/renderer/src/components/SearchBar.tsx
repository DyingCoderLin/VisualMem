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
  const [query, setQuery] = useState('')
  const [searchType, setSearchType] = useState<'image' | 'text'>('image')
  // 日期选择器已注释，保留状态以便后续使用
  // const [startDate, setStartDate] = useState('2025-10-27')
  // const [endDate, setEndDate] = useState('2025-12-18')
  const [isSearching, setIsSearching] = useState(false)
  const searchRequestRef = useRef<AbortController | null>(null)
  
  // 使用全局状态
  const { isRecording, startRecording, stopRecording, currentView, setRealtimeSearchResult } = useAppStore()

  const handleSearch = async () => {
    if (!query.trim()) return
    
    // 如果正在搜索，忽略新的请求
    if (isSearching) {
      console.log('Search already in progress, ignoring duplicate request')
      return
    }

    // 取消之前的请求（如果有）
    if (searchRequestRef.current) {
      searchRequestRef.current.abort()
    }

    // 创建新的 AbortController
    const abortController = new AbortController()
    searchRequestRef.current = abortController

    setIsSearching(true)
    try {
      // 如果是实时追踪视图，时间写死为最近 5 分钟
      let startTime = undefined
      let endTime = undefined
      
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
          search_type: searchType
        },
        abortController.signal // 传递 AbortSignal
      )
      
      // 检查请求是否被取消
      if (abortController.signal.aborted) {
        return
      }
      
      if (currentView === 'realtime') {
        setRealtimeSearchResult(result)
      } else {
        onSearchResult(result)
      }
    } catch (error: any) {
      // 如果请求被取消，不显示错误
      if (error.name === 'AbortError' || abortController.signal.aborted) {
        return
      }
      console.error('Search failed:', error)
      onSearchResult({
        answer: '搜索失败，请稍后重试。',
        frames: []
      })
    } finally {
      // 只有在当前请求还没被取消时才更新状态
      if (!abortController.signal.aborted) {
        setIsSearching(false)
        searchRequestRef.current = null
      }
    }
  }

  const handleToggleRecording = async () => {
    try {
      if (isRecording) {
        stopRecording()
      } else {
        await startRecording()
      }
    } catch (error) {
      console.error('Recording toggle failed:', error)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isSearching) {
      e.preventDefault() // 防止表单提交（如果有表单）
      handleSearch()
    }
  }
  
  // 组件卸载时取消正在进行的请求
  useEffect(() => {
    return () => {
      if (searchRequestRef.current) {
        searchRequestRef.current.abort()
      }
    }
  }, [])

  // 组件卸载时停止录制（如果需要）
  useEffect(() => {
    return () => {
      // 注意：这里不自动停止录制，让用户手动控制
      // 如果需要在组件卸载时停止，可以取消下面的注释
      // if (isRecording) {
      //   stopRecording()
      // }
    }
  }, [])

  return (
    <>
      <div className="search-container">
        <div className="search-input-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder="Ask VisualMem..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
          />
        </div>
        
        <div className="toggle-group">
          <button
            className={`toggle-btn ${searchType === 'image' ? 'active' : ''}`}
            onClick={() => setSearchType('image')}
            title="基于图片视觉特征搜索"
          >
            Image
          </button>
          <button
            className={`toggle-btn ${searchType === 'text' ? 'active' : ''}`}
            onClick={() => setSearchType('text')}
            title="基于 OCR 文本语义搜索"
          >
            Text
          </button>
        </div>

        {/* 日期选择器已注释，保留以便后续使用 */}
        {/* <div className="date-range-picker">
          <input
            type="date"
            className="date-input"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
          <span>→</span>
          <input
            type="date"
            className="date-input"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div> */}

        {/* 录屏按钮 */}
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

        <button
          className="btn btn-primary"
          onClick={handleSearch}
          disabled={isSearching}
        >
          {isSearching ? 'Searching...' : 'Search'}
        </button>
      </div>
    </>
  )
}

export default SearchBar

