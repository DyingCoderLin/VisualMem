import type {
  DailyReportListResponse,
  DailyReportPayload,
  GenerateDailyReportResponse
} from '../types/dailyReport'

const API_BASE_URL = 'http://localhost:18080'

interface QueryRagRequest {
  query: string
  start_time?: string
  end_time?: string
  search_type?: 'image' | 'text'
  ocr_mode?: boolean
}

export interface SubFrameResult {
  sub_frame_id: string
  timestamp: string
  app_name: string
  window_name: string
  image_path: string | null
}

export interface FrameResult {
  frame_id: string
  timestamp: string
  image_base64?: string
  image_path?: string
  ocr_text?: string
  relevance?: number
  sub_frames?: SubFrameResult[]
}

interface QueryRagResponse {
  answer: string
  frames: FrameResult[]
}

interface StatsResponse {
  total_frames: number
  disk_usage?: string
  storage?: string
  vlm_model?: string
  diff_threshold?: number  // 帧差阈值配置
  capture_interval_seconds?: number  // 截屏间隔（秒）
  max_image_width?: number  // 最大图片宽度
  image_quality?: number  // 图片质量（1-100）
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeoutMs: number = 30000  // 默认 30 秒超时
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    // 为所有请求设置超时，防止请求无限挂起
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(url, {
        ...options,
        signal: options.signal || controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      })

      if (!response.ok) {
        const text = await response.text()
        let message = response.statusText || 'Request failed'
        try {
          const errBody = JSON.parse(text) as { detail?: unknown }
          const d = errBody?.detail
          if (typeof d === 'string') {
            message = d
          } else if (Array.isArray(d) && d.length > 0) {
            message = d
              .map((x: { msg?: string }) => x?.msg || '')
              .filter(Boolean)
              .join('; ')
          }
        } catch {
          if (text) message = text.slice(0, 500)
        }
        throw new Error(message)
      }

      return response.json()
    } finally {
      clearTimeout(timeoutId)
    }
  }

  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/api/stats')
  }

  async loadModels(): Promise<{ status: string; message: string }> {
    // Load heavy ML models on demand. Uses long timeout since model loading takes time.
    return this.request<{ status: string; message: string }>('/api/load_models', {
      method: 'POST'
    }, 300000)  // 5 min timeout for model loading
  }

  async getModelsStatus(): Promise<{ loaded: boolean; loading: boolean }> {
    return this.request<{ loaded: boolean; loading: boolean }>('/api/models_status')
  }

  async getRecentFrames(minutes: number = 5): Promise<{ frames: FrameResult[] }> {
    return this.request<{ frames: FrameResult[] }>(`/api/recent_frames?minutes=${minutes}`)
  }

  async queryRagWithTime(
    req: QueryRagRequest,
    signal?: AbortSignal
  ): Promise<QueryRagResponse> {
    // 统一使用 query_rag_with_time 端点，通过 search_type 区分搜索模式
    const endpoint = '/api/query_rag_with_time'

    return this.request<QueryRagResponse>(endpoint, {
      method: 'POST',
      signal, // 传递 AbortSignal
      body: JSON.stringify({
        query: req.query,
        start_time: req.start_time,
        end_time: req.end_time,
        search_type: req.search_type || 'image',
        ocr_mode: req.ocr_mode || false
      }),
    })
  }

  getImageUrl(imagePath: string): string {
    // 通过后端 API 获取图片
    // 如果路径已经是完整的 URL，直接返回
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
      return imagePath
    }
    return `${this.baseUrl}/api/image?path=${encodeURIComponent(imagePath)}`
  }

  async getFramesByDateRange(
    startDate: string,
    endDate: string,
    offset: number = 0,
    limit: number = 50
  ): Promise<FrameResult[]> {
    // 使用 POST 请求获取时间范围内的帧列表
    return this.request<FrameResult[]>('/api/frames', {
      method: 'POST',
      body: JSON.stringify({
        start_date: startDate,
        end_date: endDate,
        offset,
        limit
      })
    })
  }

  async getDateRange(): Promise<{ earliest_date: string | null; latest_date: string | null }> {
    // 获取数据库中最早和最新的照片日期
    return this.request<{ earliest_date: string | null; latest_date: string | null }>('/api/date-range', {
      method: 'GET'
    })
  }

  async getFramesCountByDate(date: string): Promise<{ date: string; total_count: number }> {
    // 获取某一天的照片总数
    return this.request<{ date: string; total_count: number }>('/api/frames/date/count', {
      method: 'POST',
      body: JSON.stringify({ date })
    })
  }

  async getFramesByDate(
    date: string,
    offset: number = 0,
    limit: number = 50  // 可调整参数：每次加载的照片数量
  ): Promise<FrameResult[]> {
    // 获取某一天的照片（支持分页）
    return this.request<FrameResult[]>('/api/frames/date', {
      method: 'POST',
      body: JSON.stringify({
        date,
        offset,
        limit
      })
    })
  }

  async stopRecording(): Promise<{ status: string }> {
    // 刷新视频缓冲 + BatchWriteBuffer 可能很慢（积压帧多时），必须长于默认 30s，否则会 AbortError
    return this.request<{ status: string }>(
      '/api/recording/stop',
      { method: 'POST' },
      300000
    )
  }

  async storeFrame(req: {
    frame_id: string
    timestamp: string
    image_base64: string
    monitor_id?: number
    metadata?: Record<string, any>
    windows?: Array<{
      app_name: string
      window_name: string
      image_base64: string
    }>
  }): Promise<{
    status: string
    frame_id?: string
    sub_frame_count?: number
    today_count?: number
    frame_summary?: FrameResult
  }> {
    // 存储帧到后端（支持窗口信息）
    // 使用更长的超时（120秒），因为后端需要做 embedding + OCR
    return this.request<{
      status: string; frame_id?: string; sub_frame_count?: number
      today_count?: number; frame_summary?: FrameResult
    }>(
      '/api/store_frame',
      {
        method: 'POST',
        body: JSON.stringify({
          ...req,
          monitor_id: req.monitor_id ?? 0
        })
      },
      120000  // 120 秒超时
    )
  }

  async extractVideoFrame(videoPath: string, frameIndex: number = 0, fps: number = 1.0): Promise<{ image_base64: string; width: number; height: number }> {
    // 从视频中提取单帧
    return this.request<{ image_base64: string; width: number; height: number }>(
      `/api/video/extract_frame?video_path=${encodeURIComponent(videoPath)}&frame_index=${frameIndex}&fps=${fps}`
    )
  }

  getVideoFrameImageUrl(videoPath: string, frameIndex: number = 0, fps: number = 1.0): string {
    // 获取视频帧图片URL（用于img标签）
    return `${this.baseUrl}/api/video/frame_image?video_path=${encodeURIComponent(videoPath)}&frame_index=${frameIndex}&fps=${fps}`
  }

  /** List ISO dates (YYYY-MM-DD) that have a saved daily report, newest first. */
  async listDailyReports(): Promise<DailyReportListResponse> {
    return this.request<DailyReportListResponse>('/api/daily-reports')
  }

  /** Full daily report JSON for one calendar day. */
  async getDailyReport(date: string): Promise<DailyReportPayload> {
    return this.request<DailyReportPayload>(`/api/daily-reports/${encodeURIComponent(date)}`)
  }

  /**
   * Run the same pipeline as `python scripts/daily_report.py --date <date>`.
   * Long-running (minutes); uses an extended timeout.
   */
  async generateDailyReport(date: string): Promise<GenerateDailyReportResponse> {
    return this.request<GenerateDailyReportResponse>(
      '/api/daily-reports/generate',
      {
        method: 'POST',
        body: JSON.stringify({ date })
      },
      900000
    )
  }
}

export const apiClient = new ApiClient()

