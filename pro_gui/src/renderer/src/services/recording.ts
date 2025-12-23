import { apiClient } from './api'

/**
 * 录屏服务
 * 在 Electron 中完成截屏和帧差过滤，然后发送到后端进行 embedding 和 OCR
 */

interface RecordingOptions {
  interval?: number // 截屏间隔（毫秒），默认 3000ms
  diffThreshold?: number // 帧差阈值，默认 0.006
}

class RecordingService {
  private intervalId: number | null = null
  private lastImageData: ImageData | null = null
  private canvas: HTMLCanvasElement | null = null
  private ctx: CanvasRenderingContext2D | null = null
  private options: Required<RecordingOptions>
  private isRecording: boolean = false
  private frameCounter: number = 0 // 计数器：0-10，每成功发送 10 帧后刷新数据

  private maxImageWidth: number = 1920  // 最大图片宽度，从后端获取（默认 1920）
  private imageQuality: number = 0.85  // 图片质量（0-1），从后端获取（默认 0.85，对应 85%）

  constructor(options: RecordingOptions = {}) {
    // 默认值（如果后端配置加载失败时使用）
    // 注意：interval 默认值应该是 CAPTURE_INTERVAL_SECONDS * 1000（毫秒）
    // 但这里先设为 3000ms（3秒），等从后端加载后再更新
    this.options = {
      interval: options.interval || 3000,  // 默认 3 秒（与 CAPTURE_INTERVAL_SECONDS 默认值一致）
      diffThreshold: options.diffThreshold || 0.006
    }
    
    // 恢复状态：检查 localStorage
    const savedState = localStorage.getItem('vlm_is_recording')
    if (savedState === 'true') {
      console.log('[RecordingService] Restoring recording state from localStorage')
      // 延迟启动，确保环境已就绪
      setTimeout(() => {
        this.start().catch(err => console.error('Failed to auto-resume recording:', err))
      }, 1000)
    }

    // 异步从后端获取配置（不阻塞初始化）
    this.loadConfigFromBackend()
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
   * 压缩图片到指定最大宽度
   */
  private compressImage(canvas: HTMLCanvasElement): string {
    const maxWidth = this.maxImageWidth
    const quality = this.imageQuality
    
    if (canvas.width <= maxWidth) {
      return canvas.toDataURL('image/jpeg', quality)
    }

    const ratio = maxWidth / canvas.width
    const newHeight = canvas.height * ratio

    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = maxWidth
    tempCanvas.height = newHeight
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) {
      return canvas.toDataURL('image/jpeg', quality)
    }

    tempCtx.drawImage(canvas, 0, 0, maxWidth, newHeight)
    return tempCanvas.toDataURL('image/jpeg', quality)
  }

  /**
   * 使用 Electron desktopCapturer API 截屏
   */
  private async captureScreen(): Promise<ImageData | null> {
    try {
      // 检查 electronAPI 是否可用
      const electronAPI = (window as any).electronAPI
      if (!electronAPI || !electronAPI.desktopCapturer) {
        console.error('desktopCapturer API not available', { electronAPI })
        return null
      }

      // 使用 Electron 的 API 截屏
      // 使用配置的最大图片宽度（如果还没加载，使用默认值 1920）
      const thumbnailWidth = this.maxImageWidth || 1920
      const thumbnailHeight = Math.round(thumbnailWidth * 9 / 16)  // 16:9 比例
      
      const sources = await electronAPI.desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: { width: thumbnailWidth, height: thumbnailHeight }
      })

      if (!sources || sources.length === 0) {
        console.error('No screen source found')
        return null
      }

      // 使用主屏幕
      const primarySource = sources[0]
      // console.log('Captured screen source:', primarySource.name, 'thumbnail size:', primarySource.thumbnail.getSize())
      
      // 创建一个 image 元素来加载缩略图
      const img = new Image()
      img.src = primarySource.thumbnail.toDataURL()

      return new Promise((resolve) => {
        img.onload = () => {
          if (!this.canvas) {
            this.canvas = document.createElement('canvas')
            this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })
          }

          if (!this.canvas || !this.ctx) {
            console.error('Failed to create canvas or context')
            resolve(null)
            return
          }

          this.canvas.width = img.width
          this.canvas.height = img.height
          this.ctx.drawImage(img, 0, 0)
          
          try {
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height)
            // console.log('Captured image data:', imageData.width, 'x', imageData.height)
            resolve(imageData)
          } catch (e) {
            console.error('Failed to get image data:', e)
            resolve(null)
          }
        }
        img.onerror = (err) => {
          console.error('Image load error:', err)
          resolve(null)
        }
      })
    } catch (error) {
      console.error('Capture screen error:', error)
      return null
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
      const [countResult, statsResult] = await Promise.all([
        apiClient.getFramesCountByDate(today).catch(err => {
          console.error('Failed to get today\'s frame count:', err)
          return { date: today, total_count: 0 }
        }),
        apiClient.getStats().catch(err => {
          console.error('Failed to get stats:', err)
          return null
        })
      ])
      
      console.log(`Refreshed data after 10 frames: today's count=${countResult.total_count}, stats=`, statsResult)
      
      // 触发全局刷新事件，通知 SystemStatus 和 TimelineView 更新
      if (typeof window !== 'undefined') {
        // 事件1：通知 SystemStatus 更新 stats
        window.dispatchEvent(new CustomEvent('recording-data-refreshed', {
          detail: {
            todayCount: countResult.total_count,
            stats: statsResult
          }
        }))
        
        // 事件2：通知 TimelineView 只更新今天的 totalCount（轻量级，不加载实际数据）
        window.dispatchEvent(new CustomEvent('recording-timeline-refresh', {
          detail: {
            date: today,
            totalCount: countResult.total_count  // 只传递总数量
          }
        }))
      }
    } catch (error) {
      console.error('Error refreshing data:', error)
    }
  }

  /**
   * 发送帧到后端
   */
  private async sendFrameToBackend(imageData: ImageData, frameId: string, timestamp: string): Promise<void> {
    if (!this.canvas) {
      return
    }

    // 将 ImageData 绘制到 canvas
    if (!this.ctx) {
      return
    }

    this.canvas.width = imageData.width
    this.canvas.height = imageData.height
    this.ctx.putImageData(imageData, 0, 0)

    // 压缩图片（使用从后端获取的配置）
    const compressedDataUrl = this.compressImage(this.canvas)
    const base64Data = compressedDataUrl.split(',')[1]

    // 发送到后端
    try {
      // console.log('Sending frame to backend:', frameId, 'size:', base64Data.length)
      const result = await apiClient.storeFrame({
        frame_id: frameId,
        timestamp: timestamp,
        image_base64: base64Data,
        metadata: {
          width: imageData.width,
          height: imageData.height
        }
      })
      // console.log('Frame stored successfully:', result)
      
      // 递增计数器（0-10）
      this.frameCounter = (this.frameCounter + 1) % 10
      
      // 每 10 次成功发送后刷新数据
      if (this.frameCounter === 0) {
        console.log('Reached 10 frames, refreshing data...')
        // 异步刷新数据，不阻塞后续的帧发送
        this.refreshData().catch(err => {
          console.error('Error in refreshData:', err)
        })
      }
    } catch (error) {
      console.error('Failed to send frame to backend:', error)
      // 发送失败不计入计数器
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
    localStorage.setItem('vlm_is_recording', 'true')
    this.lastImageData = null
    this.frameCounter = 0 // 重置计数器

    console.log(`[RecordingService] Starting capture loop with interval: ${this.options.interval}ms`)

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

    const startTime = Date.now()
    // console.log(`[RecordingService] Loop started at ${new Date(startTime).toLocaleTimeString()}, interval: ${this.options.interval}ms`)

    try {
      // 截屏
      const imageData = await this.captureScreen()
      
      // 再次检查录制状态（可能在截屏过程中停止了）
      if (!this.isRecording || !imageData) {
        return
      }

      // 帧差过滤
      if (this.lastImageData) {
        const diff = this.calculateNormalizedRMSDiff(this.lastImageData, imageData)
        if (diff < this.options.diffThreshold) {
          // 差异太小，跳过这一帧
          return
        }
      }

      // 再次检查录制状态
      if (!this.isRecording) {
        return
      }

      // 更新上一帧
      this.lastImageData = imageData

      // 生成 frame_id 和 timestamp（使用时间戳格式：YYYYMMDD_HHMMSS_ffffff）
      const now = new Date()
      const timestamp = now.toISOString()
      
      // 生成时间戳格式的 frame_id：YYYYMMDD_HHMMSS_000000
      // 微秒部分设为 000000，因为每秒只截屏一次
      const year = now.getFullYear()
      const month = String(now.getMonth() + 1).padStart(2, '0')
      const day = String(now.getDate()).padStart(2, '0')
      const hours = String(now.getHours()).padStart(2, '0')
      const minutes = String(now.getMinutes()).padStart(2, '0')
      const seconds = String(now.getSeconds()).padStart(2, '0')
      
      const frameId = `${year}${month}${day}_${hours}${minutes}${seconds}_000000`

      // 最后检查一次录制状态
      if (!this.isRecording) {
        return
      }

      // 发送到后端（异步，不阻塞）
      this.sendFrameToBackend(imageData, frameId, timestamp).catch(err => {
        console.error('Error sending frame:', err)
      })
    } catch (error) {
      // 如果已经停止录制，忽略错误
      if (!this.isRecording) {
        return
      }
      console.error('Error in capture loop:', error)
    }
  }

  /**
   * 停止录制
   */
  async stop(): Promise<void> {
    // 立即设置停止标志
    this.isRecording = false
    localStorage.removeItem('vlm_is_recording')
    
    // 清除定时器
    if (this.intervalId !== null) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }

    // 重置状态
    this.lastImageData = null
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
}

export const recordingService = new RecordingService()

