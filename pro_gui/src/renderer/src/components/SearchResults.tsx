import React, { useState } from 'react'
import { apiClient } from '../services/api'
import ImagePreview from './ImagePreview'
import MarkdownRenderer from './MarkdownRenderer'

interface SearchResultsProps {
  result: {
    answer: string
    frames: Array<{
      frame_id: string
      timestamp: string
      image_base64?: string
      image_path?: string
      relevance?: number
    }>
  }
}

const SearchResults: React.FC<SearchResultsProps> = ({ result }) => {
  const [previewImage, setPreviewImage] = useState<{ url: string; timestamp: string } | null>(null)
  const [isMinimized, setIsMinimized] = useState(false)

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

  const getImageUrl = (frame: typeof result.frames[0]): string => {
    if (frame.image_base64) {
      return `data:image/jpeg;base64,${frame.image_base64}`
    }
    if (frame.image_path) {
      return apiClient.getImageUrl(frame.image_path)
    }
    return ''
  }

  const validFrames = result.frames.filter(frame => frame.image_path || frame.image_base64)
  
  // 按时间戳排序：从早到晚
  const sortedFrames = [...validFrames].sort((a, b) => {
    const timeA = new Date(a.timestamp).getTime()
    const timeB = new Date(b.timestamp).getTime()
    return timeA - timeB
  })

  return (
    <div className={`rag-results-container ${isMinimized ? 'rag-results-minimized' : ''}`}>
      {/* 最小化时的标题栏 */}
      {isMinimized && (
        <div className="rag-minimized-header">
          <span className="rag-minimized-title">RAG Search Results</span>
          <button 
            className="rag-toggle-button"
            onClick={() => setIsMinimized(false)}
            title="展开"
          >
            ▲
          </button>
        </div>
      )}
      
      {/* 展开时的内容 */}
      {!isMinimized && (
        <div className="rag-content-wrapper">
          {/* 左侧：图片滚动列表（2/3 宽度，网格布局，从左到右、从上到下） */}
          <div className="rag-images-panel">
            <div className="rag-section-title">RAG 提取证据</div>
              <div className="rag-images-scroll">
                {sortedFrames.map((frame) => {
                  const imageUrl = getImageUrl(frame)
                  if (!imageUrl) return null
                  
                  return (
                    <div key={frame.frame_id} className="rag-image-item">
                      <img
                        src={imageUrl}
                        alt={`Frame ${frame.frame_id}`}
                        loading="lazy"
                        onClick={() => setPreviewImage({ url: imageUrl, timestamp: formatTimestamp(frame.timestamp) })}
                        style={{ cursor: 'pointer' }}
                        onError={(e) => {
                          const item = (e.target as HTMLImageElement).closest('.rag-image-item') as HTMLElement
                          if (item) item.style.display = 'none'
                        }}
                      />
                      <div className="timestamp-label">
                        {formatTimestamp(frame.timestamp)}
                        {frame.relevance !== undefined && (
                          <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                            相关度: {Math.round(frame.relevance * 100)}%
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

          {/* 右侧：AI 回答（1/3 宽度，上下滚动） */}
          <div className="rag-answer-panel">
            <div className="rag-section-title-with-button">
              <span>AI Answer</span>
              <button 
                className="rag-toggle-button-small"
                onClick={() => setIsMinimized(true)}
                title="Minimize"
              >
                ▼
              </button>
            </div>
              <div className="rag-answer-scroll">
                <div className="rag-answer-text">
                  <MarkdownRenderer content={result.answer} />
                </div>
              </div>
          </div>
        </div>
      )}
      
      {/* 图片预览模态框 */}
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

export default SearchResults
