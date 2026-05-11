import { apiClient } from './api'

/**
 * 录屏服务
 * 在 Electron 中完成截屏和帧差过滤，然后发送到后端进行 embedding 和 OCR
 */

export type RecordingMode = 'primary' | 'all'

/** sessionActive: capture+send allowed; isWarmup: first-tick drain; isLiveRecording: steady interval (UI red button) */
export interface RecordingStatus {
  sessionActive: boolean
  isWarmup: boolean
  isLiveRecording: boolean
}

interface RecordingOptions {
  interval?: number // 截屏间隔（毫秒），默认 3000ms
  diffThreshold?: number // 帧差阈值，默认 0.006
  mode?: RecordingMode // 录制模式，默认 'primary'
}

class RecordingService {
  private intervalId: number | null = null
  private lastImageDataArray: (ImageData | null)[] = [] // 存储用于对比的低分辨率图像数据
  private canvas: HTMLCanvasElement | null = null
  private ctx: CanvasRenderingContext2D | null = null
  private diffCanvas: HTMLCanvasElement | null = null
  private diffCtx: CanvasRenderingContext2D | null = null
  private options: Required<RecordingOptions>
  /** 会话进行中（含 warmup 与稳态录屏） */
  private sessionActive: boolean = false
  /** 首次截图 + store_frame drain 阶段 */
  private warmupPhase: boolean = false
  /** 稳态录屏（setInterval 已启动），对应 UI「正在录制」 */
  private liveRecording: boolean = false
  private frameCounter: number = 0 // 计数器：0-10，每成功发送 10 帧后刷新数据
  private statusListeners: ((status: RecordingStatus) => void)[] = []
  private pendingRequests: Set<AbortController> = new Set() // 跟踪正在进行的请求

  // 发送队列：每个显示器独立一条队列 + 一个并行 in-flight HTTP 请求。
  //
  // 历史：原来所有显示器共用一条 ``sendQueue`` + 一个 ``isSending`` 锁，
  // 在 ``mode='all'`` + 多屏场景下等价于把 N 路独立流强行串行化——
  // 后端单帧处理 10~30s 时，第 N 个显示器的帧要等 N-1 个前序帧依次串完
  // 才能发出，队列轻易冲到 20 上限并开始丢旧帧，前端看起来就是「卡在
  // 12:37」。改造后：每屏一条 queue + 一个 in-flight，背压阈值按屏数放大。
  private sendQueues: Map<number, Array<{ base64Data: string; frameId: string; timestamp: string; width: number; height: number; monitorId: number; captureMs: number }>> = new Map()
  private sendingMonitors: Set<number> = new Set()
  /** 当前活跃屏幕数（最近一次 captureScreen 的结果数），用于放大背压阈值 */
  private screenCount: number = 1
  /** 单屏基准阈值，实际值会按 screenCount 放大 */
  private readonly perMonitorMaxQueueSize: number = 20
  private readonly perMonitorBackpressureThreshold: number = 12
  private readonly perMonitorBackpressureResumeThreshold: number = 8
  /** setInterval 不等待 async，防止上一 tick 仍在 await captureScreen 时又开一轮截屏 */
  private captureTickInFlight: boolean = false
  /** 背压锁存：进入后备压一直生效直到队列充分下降 */
  private backpressureLatched: boolean = false
  private lastBackpressureLogMs: number = 0
  private lastQueueDropLogMs: number = 0
  /**
   * 后端 FrameEnrichmentWorker 积压：queue_depth + inflight（与 /api/stats、
   * store_frame 响应字段一致）。用于在 HTTP 发送队列尚不深时仍暂停截屏。
   */
  private backendPipelineDepth: number = 0
  private enrichBpHigh: number = 14
  private enrichBpLow: number = 6
  /**
   * 进入背压后不再发 store_frame，`enrichment_pipeline_depth` 会停在旧值。
   * 节流拉 /api/stats 刷新真实积压，避免永远卡在 pipeline==high。
   */
  private lastEnrichmentStatsPollMs: number = 0
  private static readonly ENRICHMENT_STATS_POLL_MS = 2000

  private maxImageWidth: number = 1920  // 最大图片宽度，从后端获取（默认 1920）
  private imageQuality: number = 0.85  // 图片质量（0-1），从后端获取（默认 0.85，对应 85%）

  constructor(options: RecordingOptions = {}) {
    // 默认值（如果后端配置加载失败时使用）
    // 注意：interval 默认值应该是 CAPTURE_INTERVAL_SECONDS * 1000（毫秒）
    // 但这里先设为 3000ms（3秒），等从后端加载后再更新
    this.options = {
      interval: options.interval || 3000,  // 默认 3 秒（与 CAPTURE_INTERVAL_SECONDS 默认值一致）
      diffThreshold: options.diffThreshold || 0.006,
      mode: options.mode || 'primary'
    }
    
    // 恢复状态：检查 sessionStorage (仅在页面刷新时保留，应用关闭后自动清除)
    const savedState = sessionStorage.getItem('vlm_is_recording')
    if (savedState === 'true') {
      console.log('[RecordingService] Restoring recording state from sessionStorage after refresh')
      this.sessionActive = true
      this.warmupPhase = true
      this.liveRecording = false
      queueMicrotask(() => this.notifyStatusListeners())
      setTimeout(() => {
        if (!this.sessionActive) return
        this.resumeAfterPageReload().catch((err) => {
          console.error('Failed to auto-resume recording:', err)
          this.sessionActive = false
          this.warmupPhase = false
          this.liveRecording = false
          sessionStorage.removeItem('vlm_is_recording')
          this.notifyStatusListeners()
        })
      }, 1000)
    }

    // 异步从后端获取配置（不阻塞初始化）
    this.loadConfigFromBackend()
  }

  /**
   * 状态监听
   */
  subscribeStatus(listener: (status: RecordingStatus) => void): () => void {
    this.statusListeners.push(listener)
    listener(this.getRecordingStatus())
    return () => {
      this.statusListeners = this.statusListeners.filter(l => l !== listener)
    }
  }

  private getRecordingStatus(): RecordingStatus {
    return {
      sessionActive: this.sessionActive,
      isWarmup: this.warmupPhase,
      isLiveRecording: this.liveRecording,
    }
  }

  private notifyStatusListeners(): void {
    const s = this.getRecordingStatus()
    this.statusListeners.forEach((listener) => listener(s))
  }

  /** 解析后端返回的 enrichment 背压信号（/api/stats 与 store_frame） */
  private applyEnrichmentSignals(s: {
    enrichment_pipeline_depth?: number
    enrichment_queue_depth?: number
    enrichment_inflight?: number
    enrichment_backpressure_high?: number
    enrichment_backpressure_low?: number
  }): void {
    if (typeof s.enrichment_pipeline_depth === 'number') {
      this.backendPipelineDepth = s.enrichment_pipeline_depth
    } else if (
      typeof s.enrichment_queue_depth === 'number' ||
      typeof s.enrichment_inflight === 'number'
    ) {
      const q = s.enrichment_queue_depth ?? 0
      const inf = s.enrichment_inflight ?? 0
      this.backendPipelineDepth = q + inf
    }
    if (typeof s.enrichment_backpressure_high === 'number') {
      this.enrichBpHigh = s.enrichment_backpressure_high
    }
    if (typeof s.enrichment_backpressure_low === 'number') {
      this.enrichBpLow = s.enrichment_backpressure_low
    }
  }

  /**
   * 从后端获取所有配置（diff_threshold, capture_interval, max_image_width, image_quality）
   */
  private async loadConfigFromBackend(): Promise<void> {
    try {
      const stats = await apiClient.getStats()
      // console.log('[RecordingService] Stats from backend:', stats)
      
      // 更新帧差阈值
      if (stats.diff_threshold !== undefined && stats.diff_threshold !== null) {
        this.options.diffThreshold = stats.diff_threshold
        console.log(`[RecordingService] Loaded diff_threshold from backend: ${this.options.diffThreshold}`)
      }
      
      // 更新截屏间隔（从秒转换为毫秒）
      if (stats.capture_interval_seconds !== undefined && stats.capture_interval_seconds !== null) {
        const newInterval = stats.capture_interval_seconds * 1000
        if (this.options.interval !== newInterval) {
          console.log(`[RecordingService] Updating capture_interval from ${this.options.interval}ms to ${newInterval}ms`)
          this.options.interval = newInterval
          
          // 如果正在录制，重启定时器以应用新间隔
          if (this.liveRecording && this.intervalId !== null) {
            clearInterval(this.intervalId)
            this.intervalId = window.setInterval(() => this.captureAndProcessLoop(), this.options.interval)
          }
        }
      }
      
      // 更新最大图片宽度
      if (stats.max_image_width !== undefined && stats.max_image_width !== null) {
        this.maxImageWidth = stats.max_image_width
        console.log(`[RecordingService] Loaded max_image_width from backend: ${this.maxImageWidth}`)
      }
      
      // 更新图片质量（后端返回的是 1-100，需要转换为 0-1）
      if (stats.image_quality !== undefined && stats.image_quality !== null) {
        this.imageQuality = stats.image_quality / 100.0
        console.log(`[RecordingService] Loaded image_quality from backend: ${stats.image_quality}% (${this.imageQuality})`)
      }

      this.applyEnrichmentSignals(stats)
    } catch (error) {
      console.warn('[RecordingService] Failed to load config from backend, using defaults:', error)
      // 使用默认值，不阻塞
    }
  }

  /**
   * 计算两张图片的归一化均方根差异
   */
  private calculateNormalizedRMSDiff(imgData1: ImageData, imgData2: ImageData): number {
    if (imgData1.width !== imgData2.width || imgData1.height !== imgData2.height) {
      return 1.0 // 尺寸不同，认为完全不同
    }

    const data1 = imgData1.data
    const data2 = imgData2.data
    let sumSquaredDiff = 0
    const pixelCount = imgData1.width * imgData1.height

    for (let i = 0; i < data1.length; i += 4) {
      // 只比较 RGB，忽略 Alpha
      const r1 = data1[i]
      const g1 = data1[i + 1]
      const b1 = data1[i + 2]
      const r2 = data2[i]
      const g2 = data2[i + 1]
      const b2 = data2[i + 2]

      const rDiff = r1 - r2
      const gDiff = g1 - g2
      const bDiff = b1 - b2

      sumSquaredDiff += rDiff * rDiff + gDiff * gDiff + bDiff * bDiff
    }

    const mse = sumSquaredDiff / (pixelCount * 3) // 3 个通道
    const rms = Math.sqrt(mse)
    return rms / 255.0 // 归一化到 0-1
  }

  /**
   * Check if a frame is solid-color (black screen, white screen, etc.)
   * Uses the standard deviation of grayscale pixel values from the low-res diff image.
   */
  private isSolidColorFrame(diffData: ImageData): boolean {
    const data = diffData.data
    const pixelCount = diffData.width * diffData.height
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < data.length; i += 4) {
      // Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      sum += gray
      sumSq += gray * gray
    }
    const mean = sum / pixelCount
    const std = Math.sqrt(sumSq / pixelCount - mean * mean)
    return std < 5.0
  }

  /**
   * 使用 Electron desktopCapturer API 和 WebRTC 截屏
   */
  private async captureScreen(): Promise<{ base64Data: string; diffData: ImageData; index: number; width: number; height: number; captureMs: number }[]> {
    try {
      // 检查 electronAPI 是否可用
      const electronAPI = (window as any).electronAPI
      if (!electronAPI || !electronAPI.desktopCapturer) {
        console.error('desktopCapturer API not available', { electronAPI })
        return []
      }

      // 获取所有屏幕源
      const sources = await electronAPI.desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: { width: 1, height: 1 } // 我们不再需要缩略图，设为最小以节省开销
      })

      if (!sources || sources.length === 0) {
        console.error('No screen source found')
        return []
      }

      // 根据模式选择源
      const sourcesToCapture = this.options.mode === 'primary' ? [sources[0]] : sources
      
      const results: { base64Data: string; diffData: ImageData; index: number; width: number; height: number; captureMs: number }[] = []

      for (let i = 0; i < sourcesToCapture.length; i++) {
        const source = sourcesToCapture[i]
        try {
          // 使用 WebRTC 获取真实的屏幕流
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              mandatory: {
                chromeMediaSource: 'desktop',
                chromeMediaSourceId: source.id,
                minWidth: 1280,
                maxWidth: 4096,
                minHeight: 720,
                maxHeight: 2304
              }
            } as any
          })

          // 将流转换为图片数据
          const tCapture0 = performance.now()
          const captureResult = await new Promise<{ base64Data: string; diffData: ImageData; width: number; height: number; captureMs: number } | null>((resolve) => {
            const video = document.createElement('video')
            video.style.display = 'none'
            document.body.appendChild(video)
            video.srcObject = stream
            
            video.onloadedmetadata = async () => {
              try {
                await video.play()
                
                // 1. 首先绘制到小画布用于帧差检测 (160x120)
                if (!this.diffCanvas) {
                  this.diffCanvas = document.createElement('canvas')
                  this.diffCanvas.width = 160
                  this.diffCanvas.height = 120
                  this.diffCtx = this.diffCanvas.getContext('2d', { willReadFrequently: true })
                }

                if (!this.diffCtx) {
                  resolve(null)
                  return
                }

                this.diffCtx.drawImage(video, 0, 0, 160, 120)
                const diffData = this.diffCtx.getImageData(0, 0, 160, 120)

                // 2. 检查是否需要捕获全图
                let shouldCaptureFull = true
                if (this.lastImageDataArray[i]) {
                  const diff = this.calculateNormalizedRMSDiff(this.lastImageDataArray[i]!, diffData)
                  if (diff < this.options.diffThreshold) {
                    shouldCaptureFull = false
                  }
                }

                if (!shouldCaptureFull) {
                  resolve({ base64Data: '', diffData, width: video.videoWidth, height: video.videoHeight, captureMs: performance.now() - tCapture0 })
                  return
                }

                // 3. 需要捕获全图
                if (!this.canvas) {
                  this.canvas = document.createElement('canvas')
                  this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })
                }

                if (!this.canvas || !this.ctx) {
                  resolve(null)
                  return
                }

                // 根据配置限制最大宽度
                const maxWidth = this.maxImageWidth || 1920
                let targetWidth = video.videoWidth
                let targetHeight = video.videoHeight
                
                if (targetWidth > maxWidth) {
                  const ratio = maxWidth / targetWidth
                  targetWidth = maxWidth
                  targetHeight = Math.round(video.videoHeight * ratio)
                }

                this.canvas.width = targetWidth
                this.canvas.height = targetHeight
                this.ctx.drawImage(video, 0, 0, targetWidth, targetHeight)
                
                const base64 = this.canvas.toDataURL('image/jpeg', this.imageQuality || 0.85)
                resolve({ 
                  base64Data: base64.split(',')[1], 
                  diffData, 
                  width: video.videoWidth, 
                  height: video.videoHeight,
                  captureMs: performance.now() - tCapture0,
                })
              } catch (e) {
                console.error('Failed to capture frame from video:', e)
                resolve(null)
              } finally {
                // 停止流并清理元素
                stream.getTracks().forEach(track => track.stop())
                video.remove()
              }
            }
            
            video.onerror = (err) => {
              console.error('Video error:', err)
              stream.getTracks().forEach(track => track.stop())
              video.remove()
              resolve(null)
            }
          })

          if (captureResult) {
            results.push({ 
              base64Data: captureResult.base64Data, 
              diffData: captureResult.diffData, 
              index: i,
              width: captureResult.width,
              height: captureResult.height,
              captureMs: captureResult.captureMs,
            })
          }
        } catch (err) {
          console.error(`Failed to capture screen ${source.name} via WebRTC:`, err)
        }
      }

      return results
    } catch (error) {
      console.error('Capture screen error:', error)
      return []
    }
  }

  /**
   * 轻量级 stats 刷新（每 10 帧后调用）
   * 仅获取 stats 用于 SystemStatus 组件，不再请求 count/frames
   * （帧数据已通过 store_frame 响应直接推送给 TimelineView）
   */
  private async refreshStatsOnly(): Promise<void> {
    try {
      const statsResult = await apiClient.getStats()
      this.applyEnrichmentSignals(statsResult)
      if (typeof window !== 'undefined' && statsResult) {
        window.dispatchEvent(new CustomEvent('recording-data-refreshed', {
          detail: { stats: statsResult }
        }))
      }
    } catch (error) {
      console.error('Error refreshing stats:', error)
    }
  }

  /**
   * 开始录制：warmup（第一次截图 + 全部 store_frame 返回）→ 稳态 interval
   */
  async start(): Promise<void> {
    if (this.sessionActive) {
      console.warn('Recording session already active')
      return
    }

    this.sessionActive = true
    this.warmupPhase = true
    this.liveRecording = false
    this.lastImageDataArray = []
    this.frameCounter = 0
    this.sendQueues.clear()
    this.sendingMonitors.clear()
    this.backpressureLatched = false
    this.backendPipelineDepth = 0
    this.lastEnrichmentStatsPollMs = 0
    sessionStorage.setItem('vlm_is_recording', 'true')
    this.notifyStatusListeners()

    try {
      await this.runFirstScreenshotWarmup()
    } catch (err) {
      this.sessionActive = false
      this.warmupPhase = false
      this.liveRecording = false
      sessionStorage.removeItem('vlm_is_recording')
      this.notifyStatusListeners()
      throw err
    }

    if (!this.sessionActive) {
      return
    }

    this.warmupPhase = false
    this.liveRecording = true
    this.notifyStatusListeners()

    this.startLoopSteady()
  }

  /** 页面刷新后恢复：与手动开始相同，先 warmup 再 interval */
  private async resumeAfterPageReload(): Promise<void> {
    this.lastImageDataArray = []
    this.frameCounter = 0
    this.sendQueues.clear()
    this.sendingMonitors.clear()
    this.backpressureLatched = false
    await this.runFirstScreenshotWarmup()
    if (!this.sessionActive) return
    this.warmupPhase = false
    this.liveRecording = true
    this.notifyStatusListeners()
    this.startLoopSteady()
  }

  /**
   * 第一次截图产生的全部 store_frame 请求完成，并给后端 worker 一个有上限的启动缓冲。
   */
  private async runFirstScreenshotWarmup(): Promise<void> {
    console.log('[RecordingService] Warmup: first capture tick + bounded backend settle')
    await this.runSingleCaptureTick()
    await this.drainSendQueue()
    await this.waitForBackendWarmupSettle()
    console.log('[RecordingService] Warmup complete')
  }

  private totalQueueDepth(): number {
    let total = 0
    for (const q of this.sendQueues.values()) total += q.length
    return total
  }

  private anySending(): boolean {
    return this.sendingMonitors.size > 0
  }

  private async drainSendQueue(): Promise<void> {
    const maxWaitMs = 30 * 60 * 1000
    const t0 = Date.now()
    while (this.totalQueueDepth() > 0 || this.anySending()) {
      if (!this.sessionActive) {
        break
      }
      if (Date.now() - t0 > maxWaitMs) {
        console.warn('[RecordingService] drainSendQueue: timeout waiting for queue (30m)')
        break
      }
      await new Promise((r) => setTimeout(r, 50))
    }
  }

  private async waitForBackendWarmupSettle(): Promise<void> {
    const maxWaitMs = Math.min(15000, Math.max(5000, this.options.interval * 2))
    const pollMs = 1000
    const statsTimeoutMs = 3000
    const logEveryMs = 3000
    const t0 = Date.now()
    let lastLogMs = 0

    while (this.sessionActive) {
      let pipelineDepth = this.backendPipelineDepth
      if (pipelineDepth <= 0) {
        break
      }

      try {
        const stats = await apiClient.getStats(statsTimeoutMs)
        this.applyEnrichmentSignals(stats)
        pipelineDepth = this.backendPipelineDepth
      } catch (error) {
        console.warn('[RecordingService] Warmup: backend stats unavailable, entering steady loop with backpressure:', error)
        break
      }

      if (pipelineDepth <= 0) {
        break
      }

      const now = Date.now()
      if (now - t0 > maxWaitMs) {
        console.warn(
          `[RecordingService] Warmup: backend enrichment still running ` +
            `(depth=${pipelineDepth}) after ${Math.round(maxWaitMs / 1000)}s; ` +
            `entering steady loop with backpressure.`
        )
        break
      }

      if (now - lastLogMs > logEveryMs) {
        console.log(
          `[RecordingService] Warmup: backend enrichment depth=${pipelineDepth}; ` +
            `waiting up to ${Math.round(maxWaitMs / 1000)}s before steady loop`
        )
        lastLogMs = now
      }

      await new Promise((r) => setTimeout(r, pollMs))
    }
  }

  /**
   * 仅启动定时间隔，不立即再截一帧（首帧已在 warmup 完成）
   */
  private startLoopSteady(): void {
    console.log(
      `[RecordingService] Steady capture loop interval: ${this.options.interval}ms, mode: ${this.options.mode}`
    )

    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
    }
    this.intervalId = window.setInterval(() => this.captureAndProcessLoop(), this.options.interval)
  }

  /** 单次截屏 + 入队（与 captureAndProcessLoop 主体一致） */
  private async runSingleCaptureTick(): Promise<void> {
    if (this.captureTickInFlight) {
      return
    }
    this.captureTickInFlight = true
    try {
      const captureResults = await this.captureScreen()
      if (!this.sessionActive || captureResults.length === 0) {
        return
      }

      const now = new Date()
      const timestamp = now.toISOString()
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hours = String(now.getHours()).padStart(2, '0')
      const minutes = String(now.getMinutes()).padStart(2, '0')
      const seconds = String(now.getSeconds()).padStart(2, '0')
      const frameIdPrefix = `${year}${month}${day}_${hours}${minutes}${seconds}_`

      if (captureResults.length > 0) {
        this.screenCount = captureResults.length
      }

      for (const { base64Data, diffData, index, width, height, captureMs } of captureResults) {
        if (!this.sessionActive) {
          break
        }
        this.lastImageDataArray[index] = diffData
        if (!base64Data) {
          continue
        }
        if (this.isSolidColorFrame(diffData)) {
          continue
        }
        const microSeconds = String(index).padStart(6, '0')
        const frameId = `${frameIdPrefix}${microSeconds}`
        this.enqueueFrame(base64Data, frameId, timestamp, width, height, index, captureMs)
      }
    } catch (error) {
      if (!this.sessionActive) {
        return
      }
      console.error('Error in warmup capture tick:', error)
    } finally {
      this.captureTickInFlight = false
    }
  }

  /**
   * 核心捕获和处理逻辑
   */
  private async captureAndProcessLoop(): Promise<void> {
    if (!this.sessionActive || !this.liveRecording) {
      return
    }

    // Warmup 阶段不背压，保证第一次截图全部入队并可被 drain
    // 背压（带滞回）：队列冲高后暂停截屏，降到 resume 以下才恢复，避免 12/13 边界来回跳。
    // 多屏场景下阈值按 ``screenCount`` 放大——每屏独立一条 inflight，
    // 整体积压上限自然应线性放大。
    if (!this.warmupPhase) {
      const pollNow = Date.now()
      if (
        pollNow - this.lastEnrichmentStatsPollMs >= RecordingService.ENRICHMENT_STATS_POLL_MS &&
        (this.backpressureLatched ||
          this.backendPipelineDepth >= Math.max(0, this.enrichBpHigh - 1))
      ) {
        this.lastEnrichmentStatsPollMs = pollNow
        try {
          const st = await apiClient.getStats()
          this.applyEnrichmentSignals(st)
        } catch {
          /* 仍用上次 pipeline；下一轮再试 */
        }
      }

      const screens = Math.max(1, this.screenCount)
      const depth = this.totalQueueDepth()
      const trigger = this.perMonitorBackpressureThreshold * screens
      const resume = this.perMonitorBackpressureResumeThreshold * screens
      const maxTotal = this.perMonitorMaxQueueSize * screens
      const pipeline = this.backendPipelineDepth
      if (depth >= trigger || pipeline >= this.enrichBpHigh) {
        this.backpressureLatched = true
      }
      if (depth <= resume && pipeline <= this.enrichBpLow) {
        this.backpressureLatched = false
      }
      if (this.backpressureLatched) {
        const now = Date.now()
        if (now - this.lastBackpressureLogMs > 8000) {
          console.warn(
            `[RecordingService] Backpressure: send_queue=${depth}/${maxTotal} (screens=${screens}, resume≤${resume}); ` +
              `enrich_pipeline=${pipeline} (resume≤${this.enrichBpLow}, high≥${this.enrichBpHigh}). Skipping capture tick.`
          )
          this.lastBackpressureLogMs = now
        }
        return
      }
    }

    if (this.captureTickInFlight) {
      return
    }
    this.captureTickInFlight = true
    try {
      // 截屏（可能包含多个屏幕）
      const captureResults = await this.captureScreen()

      // 再次检查录制状态（可能在截屏过程中停止了）
      if (!this.sessionActive || !this.liveRecording || captureResults.length === 0) {
        return
      }

      // 更新屏幕数（背压阈值随之自动伸缩）
      if (captureResults.length > 0) {
        this.screenCount = captureResults.length
      }

      const now = new Date()
      const timestamp = now.toISOString()

      // 生成时间戳格式的 frame_id 前缀：YYYYMMDD_HHMMSS_
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hours = String(now.getHours()).padStart(2, '0')
      const minutes = String(now.getMinutes()).padStart(2, '0')
      const seconds = String(now.getSeconds()).padStart(2, '0')
      const frameIdPrefix = `${year}${month}${day}_${hours}${minutes}${seconds}_`

      for (const { base64Data, diffData, index, width, height, captureMs } of captureResults) {
        // 再次检查录制状态
        if (!this.sessionActive || !this.liveRecording) {
          break
        }

        // 更新上一帧（用于下一次对比）
        this.lastImageDataArray[index] = diffData

        // 如果 base64Data 为空，说明帧差过滤未通过，跳过发送
        if (!base64Data) {
          continue
        }

        // Skip solid-color / black-screen frames (std of grayscale pixels < 5)
        if (this.isSolidColorFrame(diffData)) {
          continue
        }

        // 生成 frame_id：YYYYMMDD_HHMMSS_00000X
        // 微秒部分使用 index 区分不同屏幕
        const microSeconds = String(index).padStart(6, '0')
        const frameId = `${frameIdPrefix}${microSeconds}`

        // 加入发送队列（截屏不等发送，发送逐个排队避免 HTTP 堆积）
        this.enqueueFrame(base64Data, frameId, timestamp, width, height, index, captureMs)
      }
    } catch (error) {
      // 如果已经停止录制，忽略错误
      if (!this.sessionActive) {
        return
      }
      console.error('Error in capture loop:', error)
    } finally {
      this.captureTickInFlight = false
    }
  }

  /**
   * 将帧加入目标显示器的发送队列。各屏独立积压、独立 inflight；
   * 一个慢屏不会阻塞其他屏的发送。
   */
  private enqueueFrame(base64Data: string, frameId: string, timestamp: string, width: number, height: number, monitorId: number, captureMs: number): void {
    let queue = this.sendQueues.get(monitorId)
    if (!queue) {
      queue = []
      this.sendQueues.set(monitorId, queue)
    }

    // 单屏队列满时丢弃最旧的帧（保留最新的截屏）
    if (queue.length >= this.perMonitorMaxQueueSize) {
      const dropped = queue.shift()
      const now = Date.now()
      if (now - this.lastQueueDropLogMs > 3000) {
        console.warn(
          `[RecordingService] Send queue full for monitor ${monitorId} ` +
            `(${this.perMonitorMaxQueueSize}), dropped oldest frame ${dropped?.frameId}`
        )
        this.lastQueueDropLogMs = now
      }
    }
    queue.push({ base64Data, frameId, timestamp, width, height, monitorId, captureMs })

    // 启动该屏的 queue 处理（如果没在运行）
    this.processSendQueue(monitorId)
  }

  /**
   * 每个显示器的发送循环。一屏一个 in-flight HTTP 请求，屏与屏之间并行。
   *
   * 浏览器对单一 origin 有 6 个并发 HTTP 连接上限，4 屏并发也远未触及该上限；
   * 后端 ``FrameEnrichmentWorker`` 也会并发消费这些请求（见
   * ``FRAME_ENRICHMENT_WORKERS``），因此不必串行。
   */
  private async processSendQueue(monitorId: number): Promise<void> {
    if (this.sendingMonitors.has(monitorId)) return
    this.sendingMonitors.add(monitorId)

    try {
      while (this.sessionActive) {
        const queue = this.sendQueues.get(monitorId)
        if (!queue || queue.length === 0) break
        const frame = queue.shift()!
        try {
          await this.sendFrameToBackendDirectly(
            frame.base64Data, frame.frameId, frame.timestamp,
            frame.width, frame.height, frame.monitorId, frame.captureMs
          )
        } catch (err) {
          console.error(`Error sending frame ${frame.frameId} (monitor ${monitorId}):`, err)
        }
      }
    } finally {
      this.sendingMonitors.delete(monitorId)
    }
  }

  /**
   * 直接发送 Base64 帧到后端
   */
  private async sendFrameToBackendDirectly(base64Data: string, frameId: string, timestamp: string, width: number, height: number, monitorId: number = 0, captureMs: number = 0): Promise<void> {
    // 再次检查录制状态
    if (!this.sessionActive) {
      return
    }
    
    try {
      const result = await apiClient.storeFrame({
        frame_id: frameId,
        timestamp: timestamp,
        image_base64: base64Data,
        monitor_id: monitorId,
        metadata: {
          width: width,
          height: height,
          monitor_id: monitorId
        },
        client_capture_ms: captureMs,
      })

      this.applyEnrichmentSignals(result)

      // Dispatch frame data directly to TimelineView (no extra HTTP round-trips)
      if (result.status === 'ok' && result.frame_summary && typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('recording-frame-stored', {
          detail: {
            frame: result.frame_summary,
            todayCount: result.today_count || 0,
          }
        }))
      }

      // 递增计数器（0-10）
      this.frameCounter = (this.frameCounter + 1) % 10

      // 每 10 次成功发送后刷新 stats（仅用于 SystemStatus 组件）
      if (this.frameCounter === 0) {
        this.refreshStatsOnly().catch(err => {
          console.error('Error in refreshStatsOnly:', err)
        })
      }
    } catch (error) {
      // 忽略因停止录制导致的请求取消错误
      if (error instanceof DOMException && error.name === 'AbortError') {
        // 请求被取消，这是正常的（停止录制时）
        return
      }
      // 只在录制状态下打印错误
      if (this.sessionActive) {
        console.error('Failed to send frame to backend:', error)
      }
    }
  }

  /**
   * 停止录制
   *
   * 顺序要点：
   *   1. 翻 ``sessionActive=false``，清队列 → 让所有 ``processSendQueue(monitorId)``
   *      自然退出，``sendingMonitors`` 会在各自 ``finally`` 里自行释放。
   *   2. **先** ``await apiClient.stopRecording()`` 让后端先 drain enrichment
   *      再 flush video/batch buffer——这段期间后端对其它 API 是忙的，但
   *      前端静默 AbortError 即可（见 ``api.ts`` 超时调优 + ``DailyReportView``
   *      等页面）。
   *   3. ``await`` 返回后才释放 ``sendingMonitors``——不过实际上此时它已经
   *      空了，再显式 clear() 是兜底。
   */
  async stop(): Promise<void> {
    this.sessionActive = false
    this.warmupPhase = false
    this.liveRecording = false
    sessionStorage.removeItem('vlm_is_recording')
    this.notifyStatusListeners()

    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }

    this.lastImageDataArray = []
    this.sendQueues.clear()
    this.backpressureLatched = false
    this.captureTickInFlight = false
    console.log('Recording stopped; flushing backend...')

    try {
      await apiClient.stopRecording()
      console.log('Backend buffer flushed on stop')
    } catch (error) {
      const aborted =
        error instanceof DOMException &&
        (error.name === 'AbortError' || error.message?.includes('aborted'))
      if (aborted) {
        console.warn(
          '[RecordingService] stopRecording timed out or was aborted before response; ' +
            'backend may still be flushing. Check logs/backend_server.log if data looks incomplete.'
        )
      } else {
        console.warn('Failed to notify backend to flush buffer:', error)
      }
    } finally {
      // 兜底：即使后端响应了 error，依然把 inflight 标记清掉，避免下次录制
      // 开始时 ``processSendQueue`` 误以为该屏仍有在途请求。
      this.sendingMonitors.clear()
    }
  }

  /**
   * 获取录制状态
   */
  getStatus(): boolean {
    return this.liveRecording
  }

  /**
   * 设置录制模式
   */
  setMode(mode: RecordingMode): void {
    if (this.options.mode !== mode) {
      console.log(`[RecordingService] Switching mode from ${this.options.mode} to ${mode}`)
      this.options.mode = mode
      // 切换模式时清空上一帧缓存，确保新模式下的第一帧能被捕获
      this.lastImageDataArray = []
    }
  }

  /**
   * 获取当前录制模式
   */
  getMode(): RecordingMode {
    return this.options.mode
  }
}

export const recordingService = new RecordingService()
