const API_BASE_URL = 'http://localhost:8080'

interface QueryRagRequest {
  query: string
  start_time?: string
  end_time?: string
  search_type?: 'image' | 'text'
  ocr_mode?: boolean
}

export interface FrameResult {
  frame_id: string
  timestamp: string
  image_base64?: string
  image_path?: string
  ocr_text?: string
  relevance?: number
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
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    const response = await fetch(url, {
      ...options,
      signal: options.signal, // 支持 AbortSignal
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    return response.json()
  }

  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/api/stats')
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

  async startRecording(): Promise<{ status: string }> {
    // 开始录制
    return this.request<{ status: string }>('/api/recording/start', {
      method: 'POST'
    })
  }

  async stopRecording(): Promise<{ status: string }> {
    // 停止录制
    return this.request<{ status: string }>('/api/recording/stop', {
      method: 'POST'
    })
  }

  async getRecordingStatus(): Promise<{ is_recording: boolean }> {
    // 获取录制状态
    return this.request<{ is_recording: boolean }>('/api/recording/status', {
      method: 'GET'
    })
  }

  async storeFrame(req: {
    frame_id: string
    timestamp: string
    image_base64: string
    metadata?: Record<string, any>
  }): Promise<{ status: string }> {
    // 存储帧到后端
    return this.request<{ status: string }>('/api/store_frame', {
      method: 'POST',
      body: JSON.stringify(req)
    })
  }
}

export const apiClient = new ApiClient()

