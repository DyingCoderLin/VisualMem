import React, { useState, useEffect } from 'react'
import { apiClient, FrameResult } from '../services/api'
import { useAppStore } from '../store/AppStore'
import ImagePreview from '../components/ImagePreview'
import MarkdownRenderer from '../components/MarkdownRenderer'

const ITEM_WIDTH = 200

function groupFramesByTime(frames: FrameResult[]): FrameResult[][] {
  if (!frames || frames.length === 0) return [];
  const groups: FrameResult[][] = [];
  let currentGroup: FrameResult[] = [frames[0]];
  
  for (let i = 1; i < frames.length; i++) {
    const f1 = currentGroup[0];
    const f2 = frames[i];
    const t1 = new Date(f1.timestamp).getTime();
    const t2 = new Date(f2.timestamp).getTime();
    
    // Group if within 1000ms
    if (Math.abs(t2 - t1) < 1000) {
      currentGroup.push(f2);
    } else {
      groups.push(currentGroup);
      currentGroup = [f2];
    }
  }
  groups.push(currentGroup);
  return groups;
}

const FrameGroupItem = ({ 
  frames, 
  onPreview, 
  getImageUrl, 
  formatTimestamp 
}: { 
  frames: FrameResult[], 
  onPreview: (url: string, ts: string) => void, 
  getImageUrl: (path?: string) => string, 
  formatTimestamp: (ts: string) => string 
}) => {
  const [mainIndex, setMainIndex] = useState(0);
  const safeIndex = mainIndex < frames.length ? mainIndex : 0;
  const mainFrame = frames[safeIndex];

  return (
    <div className="timeline-image-item" style={{ flexShrink: 0, width: `${ITEM_WIDTH}px`, position: 'relative' }}>
      {/* Main Image */}
      {mainFrame && (
        <img
          src={getImageUrl(mainFrame.image_path)}
          alt={`Frame ${mainFrame.frame_id}`}
          loading="lazy"
          style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '4px', cursor: 'pointer', backgroundColor: '#1a1a1a', border: '1px solid var(--border-dim)' }}
          onClick={() => onPreview(getImageUrl(mainFrame.image_path), formatTimestamp(mainFrame.timestamp))}
        />
      )}
      
      {/* Thumbnails for other screens */}
      {frames.length > 1 && (
        <div style={{ 
          position: 'absolute', 
          bottom: '28px', // right above the timestamp label
          right: '4px', 
          display: 'flex', 
          gap: '4px',
          padding: '4px',
          background: 'rgba(0,0,0,0.6)',
          borderRadius: '4px',
          backdropFilter: 'blur(4px)',
          maxWidth: '90%',
          overflowX: 'auto'
        }}>
          {frames.map((f, idx) => {
            if (idx === safeIndex) return null;
            return (
              <img
                key={f.frame_id}
                src={getImageUrl(f.image_path)}
                alt={`Screen ${idx}`}
                style={{ 
                  width: '40px', 
                  height: '24px', 
                  objectFit: 'cover', 
                  borderRadius: '2px', 
                  cursor: 'pointer', 
                  border: '1px solid rgba(255,255,255,0.8)',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.5)',
                  flexShrink: 0
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setMainIndex(idx);
                }}
                title={`切换到屏幕 ${idx + 1}`}
              />
            )
          })}
        </div>
      )}
      
      <div className="frame-time" style={{ textAlign: 'center', fontSize: '12px', marginTop: '4px', color: 'var(--text-secondary)' }}>
        {mainFrame ? new Date(mainFrame.timestamp).toLocaleTimeString() : ''}
      </div>
    </div>
  );
};

const RealTimeTracing: React.FC = () => {
  const [frames, setFrames] = useState<FrameResult[]>([])
  const [projectRoot, setProjectRoot] = useState<string | null>(null)
  const [previewImage, setPreviewImage] = useState<{ url: string; timestamp: string } | null>(null)
  
  const { realtimeSearchResult, currentView } = useAppStore()

  const fetchRecentFrames = async () => {
    try {
      const data = await apiClient.getRecentFrames(5)
      const sortedFrames = (data.frames || []).sort((a: any, b: any) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      setFrames(sortedFrames)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
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

  // Refresh immediately when this tab becomes visible
  useEffect(() => {
    if (currentView === 'realtime') {
      fetchRecentFrames()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentView])

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const getImageUrl = (path?: string) => {
    if (!path) return ''
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
          {groupFramesByTime(frames).map(gFrames => (
            <FrameGroupItem 
              key={gFrames[0].frame_id} 
              frames={gFrames} 
              onPreview={(url, ts) => setPreviewImage({ url, timestamp: ts })}
              getImageUrl={getImageUrl}
              formatTimestamp={formatTimestamp}
            />
          ))}
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
