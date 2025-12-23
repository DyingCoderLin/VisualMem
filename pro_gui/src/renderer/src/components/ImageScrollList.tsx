import React, { useRef, useEffect } from 'react'
import { apiClient } from '../services/api'

interface Frame {
  frame_id: string
  timestamp: string
  image_base64?: string
  image_path?: string
  relevance?: number
}

interface ImageScrollListProps {
  frames: Frame[]
  selectedFrameId: string | null
  onSelectFrame: (frameId: string) => void
}

const ImageScrollList: React.FC<ImageScrollListProps> = ({
  frames,
  selectedFrameId,
  onSelectFrame
}) => {
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

  const getImageUrl = (frame: Frame): string => {
    if (frame.image_base64) {
      return `data:image/jpeg;base64,${frame.image_base64}`
    }
    if (frame.image_path) {
      // 使用 apiClient 统一处理图片 URL
      return apiClient.getImageUrl(frame.image_path)
    }
    return ''
  }

  return (
    <div className="horizontal-scroll-list">
      {frames
        .filter(frame => frame.image_path || frame.image_base64) // 只显示有图片的帧
        .map((frame) => {
          const imageUrl = getImageUrl(frame)
          // 如果图片 URL 为空，不渲染该项
          if (!imageUrl) return null
          
          return (
            <div
              key={frame.frame_id}
              className={`scroll-item ${selectedFrameId === frame.frame_id ? 'active' : ''}`}
              onClick={() => onSelectFrame(frame.frame_id)}
            >
              <img
                src={imageUrl}
                alt={`Frame ${frame.frame_id}`}
                loading="lazy"
                onError={(e) => {
                  // 图片加载失败时，隐藏整个 scroll-item
                  const scrollItem = (e.target as HTMLImageElement).closest('.scroll-item') as HTMLElement
                  if (scrollItem) {
                    scrollItem.style.display = 'none'
                  }
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
  )
}

export default ImageScrollList

