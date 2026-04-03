import { apiClient } from './api'

/**
 * 录屏服务
 * 在 Electron 中完成截屏和帧差过滤，然后发送到后端进行 embedding 和 OCR
 */

export type RecordingMode = 'primary' | 'all'

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
  private isRecording: boolean = false
  private frameCounter: number = 0 // 计数器：0-10，每成功发送 10 帧后刷新数据
  private statusListeners: ((status: boolean) => void)[] = []
  private pendingRequests: Set<AbortController> = new Set() // 跟踪正在进行的请求

  // 发送队列：截屏照常进行，发送排队执行，避免 HTTP 连接堆积
  private sendQueue: Array<{ base64Data: string; frameId: string; timestamp: string; width: number; height: number; monitorId: number }> = []
  private isSending: boolean = false
  private readonly maxQueueSize: number = 20  // 队列超过此大小时丢弃最旧的帧

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
      // 立即设置内部状态
      this.isRecording = true
      // 延迟启动录制循环，确保环境已就绪
      setTimeout(() => {
        if (this.isRecording) {
          this.startLoop().catch(err => {
            console.error('Failed to auto-resume recording loop:', err)
            this.isRecording = false
            this.notifyStatusListeners()
          })
        }
      }, 1000)
    }

    // 异步从后端获取配置（不阻塞初始化）
    this.loadConfigFromBackend()
  }

  /**
   * 状态监听
   */
  subscribeStatus(listener: (status: boolean) => void): () => void {
    this.statusListeners.push(listener)
    // 立即通知当前状态
    listener(this.isRecording)
    return () => {
      this.statusListeners = this.statusListeners.filter(l => l !== listener)
    }
  }

  private notifyStatusListeners(): void {
    this.statusListeners.forEach(listener => listener(this.isRecording))
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
          if (this.isRecording && this.intervalId !== null) {
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
  private async captureScreen(): Promise<{ base64Data: string; diffData: ImageData; index: number; width: number; height: number }[]> {
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
      
      const results: { base64Data: string; diffData: ImageData; index: number; width: number; height: number }[] = []

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
          const captureResult = await new Promise<{ base64Data: string; diffData: ImageData; width: number; height: number } | null>((resolve) => {
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
                  resolve({ base64Data: '', diffData, width: video.videoWidth, height: video.videoHeight })
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
                  height: video.videoHeight 
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
              height: captureResult.height
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
   * 刷新数据（每 10 帧后调用）
   * - 获取今天的图片 count（轻量级，只更新数量，不加载实际数据）
   * - 获取 stats 更新左下角的数据
   */
  private async refreshData(): Promise<void> {
    try {
      // 获取今天的本地日期字符串，避免 UTC 导致的时间差问题
      const now = new Date()
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
      
      // 并行获取今天的图片 count 和 stats
      let countFailed = false
      let statsFailed = false
      const [countResult, statsResult] = await Promise.all([
        apiClient.getFramesCountByDate(today).catch(err => {
          countFailed = true
          console.error('Failed to get today\'s frame count:', err)
          return { date: today, total_count: 0 }
        }),
        apiClient.getStats().catch(err => {
          statsFailed = true
          console.error('Failed to get stats:', err)
          return null
        })
      ])

      console.log(`Refreshed data after 10 frames: today's count=${countFailed ? 'FAILED' : countResult.total_count}, stats=`, statsFailed ? 'FAILED' : statsResult)

      // 触发全局刷新事件，通知 SystemStatus 和 TimelineView 更新
      if (typeof window !== 'undefined') {
        // 事件1：通知 SystemStatus 更新 stats（仅在成功时）
        if (!statsFailed && statsResult) {
          window.dispatchEvent(new CustomEvent('recording-data-refreshed', {
            detail: {
              todayCount: countResult.total_count,
              stats: statsResult
            }
          }))
        }

        // 事件2：通知 TimelineView 只更新今天的 totalCount（仅在 count API 成功时，防止用 0 覆盖真实值）
        if (!countFailed) {
          window.dispatchEvent(new CustomEvent('recording-timeline-refresh', {
            detail: {
              date: today,
              totalCount: countResult.total_count  // 只传递总数量
            }
          }))
        }
      }
    } catch (error) {
      console.error('Error refreshing data:', error)
    }
  }

  /**
   * 开始录制
   */
  async start(): Promise<void> {
    if (this.isRecording && this.intervalId !== null) {
      console.warn('Recording is already in progress')
      return
    }

    this.isRecording = true
    sessionStorage.setItem('vlm_is_recording', 'true')
    this.notifyStatusListeners()
    
    await this.startLoop()
  }

  /**
   * 启动录制循环
   */
  private async startLoop(): Promise<void> {
    this.lastImageDataArray = []
    this.frameCounter = 0 // 重置计数器

    console.log(`[RecordingService] Starting capture loop with interval: ${this.options.interval}ms, mode: ${this.options.mode}`)

    // 立即执行一次
    this.captureAndProcessLoop()

    // 设置定时器
    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
    }
    this.intervalId = window.setInterval(() => this.captureAndProcessLoop(), this.options.interval)
  }

  /**
   * 核心捕获和处理逻辑
   */
  private async captureAndProcessLoop(): Promise<void> {
    // 在函数开始处检查录制状态
    if (!this.isRecording) {
      return
    }

    try {
      // 截屏（可能包含多个屏幕）
      const captureResults = await this.captureScreen()

      // 再次检查录制状态（可能在截屏过程中停止了）
      if (!this.isRecording || captureResults.length === 0) {
        return
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

      for (const { base64Data, diffData, index, width, height } of captureResults) {
        // 再次检查录制状态
        if (!this.isRecording) {
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
        this.enqueueFrame(base64Data, frameId, timestamp, width, height, index)
      }
    } catch (error) {
      // 如果已经停止录制，忽略错误
      if (!this.isRecording) {
        return
      }
      console.error('Error in capture loop:', error)
    }
  }

  /**
   * 将帧加入发送队列。截屏不受影响，发送逐个执行避免 HTTP 连接堆积。
   */
  private enqueueFrame(base64Data: string, frameId: string, timestamp: string, width: number, height: number, monitorId: number): void {
    // 队列满时丢弃最旧的帧（保留最新的截屏）
    if (this.sendQueue.length >= this.maxQueueSize) {
      const dropped = this.sendQueue.shift()
      console.warn(`[RecordingService] Send queue full (${this.maxQueueSize}), dropped oldest frame ${dropped?.frameId}`)
    }
    this.sendQueue.push({ base64Data, frameId, timestamp, width, height, monitorId })
    // 启动队列处理（如果没在运行）
    this.processSendQueue()
  }

  /**
   * 逐个发送队列中的帧，一次只有一个 HTTP 请求在飞。
   * 这样浏览器的 6 个连接中只有 1 个被 store_frame 占用，
   * 剩余 5 个可以服务 getStats/getFramesByDate 等轻量请求。
   */
  private async processSendQueue(): Promise<void> {
    if (this.isSending) return  // 已经在处理中
    this.isSending = true

    try {
      while (this.sendQueue.length > 0 && this.isRecording) {
        const frame = this.sendQueue.shift()!
        try {
          await this.sendFrameToBackendDirectly(
            frame.base64Data, frame.frameId, frame.timestamp,
            frame.width, frame.height, frame.monitorId
          )
        } catch (err) {
          console.error(`Error sending frame ${frame.frameId}:`, err)
        }
      }
    } finally {
      this.isSending = false
    }
  }

  /**
   * 直接发送 Base64 帧到后端
   */
  private async sendFrameToBackendDirectly(base64Data: string, frameId: string, timestamp: string, width: number, height: number, monitorId: number = 0): Promise<void> {
    // 再次检查录制状态
    if (!this.isRecording) {
      return
    }
    
    try {
      await apiClient.storeFrame({
        frame_id: frameId,
        timestamp: timestamp,
        image_base64: base64Data,
        monitor_id: monitorId,
        metadata: {
          width: width,
          height: height,
          monitor_id: monitorId
        }
      })
      
      // 递增计数器（0-10）
      this.frameCounter = (this.frameCounter + 1) % 10
      
      // 每 10 次成功发送后刷新数据
      if (this.frameCounter === 0) {
        this.refreshData().catch(err => {
          console.error('Error in refreshData:', err)
        })
      }
    } catch (error) {
      // 忽略因停止录制导致的请求取消错误
      if (error instanceof DOMException && error.name === 'AbortError') {
        // 请求被取消，这是正常的（停止录制时）
        return
      }
      // 只在录制状态下打印错误
      if (this.isRecording) {
        console.error('Failed to send frame to backend:', error)
      }
    }
  }

  /**
   * 停止录制
   */
  async stop(): Promise<void> {
    // 立即设置停止标志
    this.isRecording = false
    sessionStorage.removeItem('vlm_is_recording')
    this.notifyStatusListeners()
    
    // 清除定时器
    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }

    // 重置状态
    this.lastImageDataArray = []
    this.sendQueue = []
    console.log('Recording stopped')

    // 通知后端刷新缓冲区
    try {
      await apiClient.stopRecording()
      console.log('Backend buffer flushed on stop')
    } catch (error) {
      console.warn('Failed to notify backend to flush buffer:', error)
    }
  }

  /**
   * 获取录制状态
   */
  getStatus(): boolean {
    return this.isRecording
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

