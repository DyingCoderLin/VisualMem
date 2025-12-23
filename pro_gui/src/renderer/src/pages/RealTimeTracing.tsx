import React, { useState, useEffect } from 'react'
import { apiClient, FrameResult } from '../services/api'
import { useAppStore } from '../store/AppStore'
import ImagePreview from '../components/ImagePreview'
import MarkdownRenderer from '../components/MarkdownRenderer'

const RealTimeTracing: React.FC = () => {
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [projectRoot, setProjectRoot] = useState<string | null>(null)
  const [previewImage, setPreviewImage] = useState<{ url: string; timestamp: string } | null>(null)
  
  const { realtimeSearchResult } = useAppStore()

  useEffect(() => {
    if (window.electronAPI && window.electronAPI.getProjectRoot) {
      window.electronAPI.getProjectRoot().then(setProjectRoot)
    }

    fetchRecentFrames()
    const interval = setInterval(fetchRecentFrames, 30000)
    
    const handleRecordingRefreshed = () => {
      fetchRecentFrames()
    }
    window.addEventListener('recording-data-refreshed', handleRecordingRefreshed)

    return () => {
      clearInterval(interval)
      window.removeEventListener('recording-data-refreshed', handleRecordingRefreshed)
    }
  }, [])

  const fetchRecentFrames = async () => {
    try {
      const data = await apiClient.getRecentFrames(5)
      const sortedFrames = (data.frames || []).sort((a: any, b: any) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      setFrames(sortedFrames)
    } catch (error) {
      console.error('Failed to fetch recent frames:', error)
    }
  }

  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp)
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const day = String(date.getDate()).padStart(2, '0')
      const hours = String(date.getHours()).padStart(2, '0')
      const minutes = String(date.getMinutes()).padStart(2, '0')
      const seconds = String(date.getSeconds()).padStart(2, '0')
      return `${month}-${day} ${hours}:${minutes}:${seconds}`
    } catch {
      return timestamp
    }
  }

  const getImageUrl = (path?: string) => {
    if (!path) return ''
    if (path.startsWith('/') || /^[a-zA-Z]:\\/.test(path)) {
      return `file://${path}`
    }
    if (projectRoot) {
      const absolutePath = `${projectRoot}/${path}`.replace(/\/+/g, '/')
      return `file://${absolutePath}`
    }
    return apiClient.getImageUrl(path)
  }

  return (
    <div className="real-time-tracing" style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 提问结果区域 - 始终存在，样式参考 SearchResults */}
      <div className="rag-results-container" style={{ height: '45vh', minHeight: '350px', marginBottom: '20px', flexShrink: 0 }}>
        <div className="rag-header-bar">
          <span className="rag-header-title">Real-time Q&A Results (Last 5 Minutes)</span>
        </div>
        
        <div className="rag-content-wrapper">
          {realtimeSearchResult ? (
            <>
              <div className="rag-images-panel">
                <div className="rag-section-title">Extracted Evidence by RAG</div>
                <div className="rag-images-scroll">
                  {realtimeSearchResult.frames.map((frame) => {
                    const imageUrl = frame.image_path ? getImageUrl(frame.image_path) : ''
                    return (
                      <div key={frame.frame_id} className="rag-image-item">
                        <img
                          src={imageUrl}
                          alt={`Frame ${frame.frame_id}`}
                          onClick={() => setPreviewImage({ url: imageUrl, timestamp: formatTimestamp(frame.timestamp) })}
                          style={{ cursor: 'pointer' }}
                        />
                        <div className="timestamp-label">
                          {formatTimestamp(frame.timestamp)}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
              <div className="rag-answer-panel">
                <div className="rag-section-title">AI Answer</div>
                <div className="rag-answer-scroll">
                  <div className="rag-answer-text">
                    <MarkdownRenderer content={realtimeSearchResult.answer} />
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
             Ask a question in the search bar above, to obtain analysis results from the last 5 minutes' computer usage
            </div>
          )}
        </div>
      </div>

      {/* 5分钟屏幕截图 - 使用 timelineview 里的展示方法 */}
      <div className="timeline-section" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ marginBottom: '15px' }}>Recent 5-Minute Screen Records ({frames.length})</h3>
        <div className="timeline-images" style={{ display: 'flex', gap: '10px', overflowX: 'auto', paddingBottom: '10px' }}>
          {frames.map(frame => {
            const imageUrl = getImageUrl(frame.image_path)
            return (
              <div key={frame.frame_id} className="timeline-image-item" style={{ flexShrink: 0, width: '200px' }}>
                <img 
                  src={imageUrl} 
                  alt={frame.timestamp}
                  style={{ width: '100%', borderRadius: '4px', cursor: 'pointer', border: '1px solid var(--border-dim)' }}
                  onClick={() => setPreviewImage({ url: imageUrl, timestamp: formatTimestamp(frame.timestamp) })}
                />
                <div className="frame-time" style={{ textAlign: 'center', fontSize: '12px', marginTop: '4px', color: 'var(--text-secondary)' }}>
                  {new Date(frame.timestamp).toLocaleTimeString()}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {previewImage && (
        <ImagePreview 
          imageUrl={previewImage.url} 
          timestamp={previewImage.timestamp} 
          onClose={() => setPreviewImage(null)} 
        />
      )}
    </div>
  )
}

export default RealTimeTracing
