import React, { useEffect, useState } from 'react'
import { apiClient } from '../services/api'

interface SystemStatusData {
  frames: number
  disk: string
  storage: string
  vlm: string
}

const SystemStatus: React.FC = () => {
  const [status, setStatus] = useState<SystemStatusData>({
    frames: 0,
    disk: '—',
    storage: '—',
    vlm: '—'
  })

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const stats = await apiClient.getStats()
        setStatus({
          frames: stats.total_frames || 0,
          disk: stats.disk_usage || '—',
          storage: stats.storage || '—',
          vlm: stats.vlm_model || '—'
        })
      } catch (error) {
        console.error('Failed to fetch system status:', error)
      }
    }

    fetchStatus()
    
    // 每30秒更新一次状态
    const interval = setInterval(fetchStatus, 30000)
    
    // 监听录制服务的数据刷新事件（每 10 帧后触发）
    const handleRecordingDataRefreshed = (event: CustomEvent) => {
      const { stats: refreshedStats } = event.detail
      if (refreshedStats) {
        setStatus({
          frames: refreshedStats.total_frames || 0,
          disk: refreshedStats.disk_usage || '—',
          storage: refreshedStats.storage || '—',
          vlm: refreshedStats.vlm_model || '—'
        })
        // console.log('System status updated from recording service')
      }
    }
    
    window.addEventListener('recording-data-refreshed', handleRecordingDataRefreshed as EventListener)
    
    return () => {
      clearInterval(interval)
      window.removeEventListener('recording-data-refreshed', handleRecordingDataRefreshed as EventListener)
    }
  }, [])

  const formatFrames = (count: number): string => {
    return count.toLocaleString()
  }

  return (
    <div className="system-status">
      <div className="status-item">
        <span className="status-label">FRAMES</span>
        <span className="status-value">{formatFrames(status.frames)}</span>
      </div>
      <div className="status-item">
        <span className="status-label">DISK USAGE</span>
        <span className="status-value">{status.disk}</span>
      </div>
      <div className="status-item">
        <span className="status-label">STORAGE</span>
        <span className="status-value">{status.storage}</span>
      </div>
      <div className="status-item">
        <span className="status-label">VLM MODEL</span>
        <span className="status-value">{status.vlm}</span>
      </div>
    </div>
  )
}

export default SystemStatus

