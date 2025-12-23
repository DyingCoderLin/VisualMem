import React, { useState, useEffect, useRef, useCallback } from 'react'
// 【步骤 1：复制到本地后，请取消下面这行的注释，并删除下方的模拟 apiClient 代码块】
import { apiClient } from '../services/api'
import ImagePreview from '../components/ImagePreview'
import { useAppStore } from '../store/AppStore'

interface Frame {
  frame_id: string
  timestamp: string
  image_base64?: string
  image_path?: string
}

interface DateGroup {
  date: string
  frames: Frame[]
  totalCount: number
  loadedCount: number
  isLoading: boolean
}

// ========== 可调整参数 ==========
const INITIAL_LOAD_BATCH_SIZE = 50
const LOAD_MORE_BATCH_SIZE = 50
const LOAD_MORE_THRESHOLD = 10  // 降低阈值：距离末尾 10 个 item 宽度时触发加载（约 2080px）
const DAYS_TO_LOAD_PER_BATCH = 4  // 每次希望能找出多少天的数据（初始加载更多，让用户能看到更多内容）
const MAX_EMPTY_CHECKS = 30       // 关键参数：如果连着查了30天都没数据，先暂停，防止瞬间请求过多

// 样式常量
const ITEM_WIDTH = 200
const GAP = 8
// =================================

const TimelineView: React.FC = () => {
  const [dateGroups, setDateGroups] = useState<DateGroup[]>([])
  const [loading, setLoading] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [previewImage, setPreviewImage] = useState<{ url: string; timestamp: string } | null>(null)
  const [projectRoot, setProjectRoot] = useState<string>('')
  
  // 使用全局状态
  const { dateRange, timelineRefreshTrigger, refreshTimeline } = useAppStore()
  const earliestDate = dateRange.earliest_date
  
  const containerRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLElement | null>(null) // 指向真正的滚动容器
  const dateGroupRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const dateGroupsRef = useRef<DateGroup[]>([])
  
  // 使用 ref 存储最新的 loading 和 hasMore 状态，避免闭包问题
  const loadingRef = useRef(false)
  const hasMoreRef = useRef(true)
  
  // 【关键修改】cursorDateRef: 记录时间轴目前探索到的最后日期
  // 无论那天有没有数据，只要检查过，这个指针就往前推
  const cursorDateRef = useRef<Date | null>(null)
  
  // 获取项目根目录
  useEffect(() => {
    const fetchProjectRoot = async () => {
      if (window.electronAPI && window.electronAPI.getProjectRoot) {
        const root = await window.electronAPI.getProjectRoot()
        // console.log('Project root:', root)
        setProjectRoot(root)
      }
    }
    fetchProjectRoot()
  }, [])

  // 同步 ref 和 state
  useEffect(() => {
    loadingRef.current = loading
  }, [loading])
  
  useEffect(() => {
    hasMoreRef.current = hasMore
  }, [hasMore])

  useEffect(() => {
    dateGroupsRef.current = dateGroups
  }, [dateGroups])

  // 当 timelineRefreshTrigger 变化时，刷新时间轴数据
  useEffect(() => {
    if (timelineRefreshTrigger > 0) {
      // console.log('Timeline refresh triggered')
      
      // 使用本地日期，与 AppStore 保持一致
      const now = new Date()
      const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
      
      const refreshTodayIncremental = async () => {
        try {
          // 1. 获取今天最新的总数
          const countResponse = await apiClient.getFramesCountByDate(today)
          const totalCount = countResponse.total_count
          
          if (totalCount === 0) return

          // 2. 检查今天是否已经在列表中
          const currentGroups = dateGroupsRef.current
          const todayGroup = currentGroups.find(g => g.date === today)
          
          if (todayGroup) {
            // 如果已存在，且有新数据，则更新总数并加载新帧
            if (totalCount > todayGroup.loadedCount) {
              // console.log(`[TimelineView] Incremental refresh for ${today}: ${todayGroup.loadedCount} -> ${totalCount}`)
              setDateGroups(prev => prev.map(g => 
                g.date === today ? { ...g, totalCount: totalCount } : g
              ))
              // 触发加载（使用 setTimeout 确保 state 已更新到 ref）
              setTimeout(() => {
                if (loadMoreFramesForDateRef.current) {
                  loadMoreFramesForDateRef.current(today)
                }
              }, 100)
            }
          } else {
            // 如果今天不在列表中，且有数据，则只加载今天的第一批数据并插入到最前面
            // 而不是调用 loadMoreDates() 加载多天
            // console.log(`[TimelineView] Today's group not found during refresh, fetching first batch for ${today}`)
            const frames = await apiClient.getFramesByDate(today, 0, INITIAL_LOAD_BATCH_SIZE)
            const validFrames = frames.filter(f => f.image_path || f.image_base64)
            const sortedFrames = validFrames.sort((a, b) => {
              const timeA = new Date(a.timestamp).getTime()
              const timeB = new Date(b.timestamp).getTime()
              return timeA - timeB
            })
            
            if (sortedFrames.length > 0) {
              const newGroup: DateGroup = {
                date: today,
                frames: sortedFrames,
                totalCount: totalCount,
                loadedCount: sortedFrames.length,
                isLoading: false
              }
              setDateGroups(prev => {
                if (prev.some(g => g.date === today)) return prev
                return [newGroup, ...prev]
              })
            }
          }
        } catch (error) {
          console.error('Failed to perform incremental refresh:', error)
        }
      }
      
      refreshTodayIncremental()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timelineRefreshTrigger])

  // 监听录制服务的数据刷新事件（每 10 帧后触发）
  useEffect(() => {
    const handleRecordingTimelineRefresh = (event: CustomEvent) => {
      const { date, totalCount } = event.detail
      // console.log('Recording timeline refresh event received:', date, 'totalCount:', totalCount)
      
      // 更新该日期的 totalCount，并检查是否需要自动加载新数据
      setDateGroups(prev => {
        const existingGroup = prev.find(g => g.date === date)
        
        if (existingGroup) {
          // 如果已存在，更新 totalCount
          const updatedGroups = prev.map(group => {
            if (group.date === date) {
              return {
                ...group,
                totalCount: totalCount
              }
            }
            return group
          })
          
          // 如果有新数据未加载，自动触发加载（不再检查滚动位置，实现完全增量）
          if (existingGroup.loadedCount < totalCount) {
            setTimeout(() => {
              if (loadMoreFramesForDateRef.current) {
                loadMoreFramesForDateRef.current(date)
              }
            }, 100)
          }
          
          return updatedGroups
        } else if (totalCount > 0) {
          // 如果该日期组还不存在，且有数据，则在下次 refreshTimeline 时会处理它
          // 或者我们也可以在这里直接触发一次 refreshTimeline
          return prev
        }
        return prev
      })
    }
    
    window.addEventListener('recording-timeline-refresh', handleRecordingTimelineRefresh as EventListener)
    
    return () => {
      window.removeEventListener('recording-timeline-refresh', handleRecordingTimelineRefresh as EventListener)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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

  // 存储 loadMoreDates 的 ref，用于 scroll handler
  const loadMoreDatesRef = useRef<() => Promise<void>>()
  
  // 【核心逻辑重构】加载更多日期
  const loadMoreDates = useCallback(async () => {
    // 使用 ref 获取最新状态，防止重入
    if (loadingRef.current || !hasMoreRef.current) return

    setLoading(true)
    
    try {
      // 1. 确定搜索起始点
      let currentDate: Date
      if (cursorDateRef.current) {
        // 从指针的下一天（前一天）开始
        currentDate = new Date(cursorDateRef.current)
        currentDate.setDate(currentDate.getDate() - 1)
      } else {
        // 第一次加载：取 (今天) 和 (后端最新日期) 的较大值，确保不漏掉今天
        const today = new Date()
        if (dateRange.latest_date) {
          const latest = new Date(dateRange.latest_date)
          // 比较日期部分，确保 currentDate 是两者中较晚的一个
          currentDate = latest > today ? latest : today
        } else {
          currentDate = today
        }
        // console.log(`[TimelineView] Initial load starting from: ${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')} (latest_date: ${dateRange.latest_date})`)
      }

      const newGroups: DateGroup[] = []
      let checksCount = 0 // 安全计数器：防止 while 循环无限运行
      
      // 2. 循环查找，直到找到足够的日期组，或者查了太多天空数据，或者到底了
      // 这里的逻辑是："一直往前翻日历，直到找到 DAYS_TO_LOAD_PER_BATCH 个有照片的日子"
      while (newGroups.length < DAYS_TO_LOAD_PER_BATCH && checksCount < MAX_EMPTY_CHECKS) {
        // 使用本地日期字符串，避免 UTC 导致的时间差问题（例如北京时间凌晨看到的是昨天）
        const dateStr = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')}`
        
        // 检查边界：是否早于数据库最早日期
        if (earliestDate && dateStr < earliestDate) {
          // console.log('Reached earliest date boundary')
          setHasMore(false)
          hasMoreRef.current = false
          break
        }

        // 更新 cursor，这一步非常关键！
        // 只要我们决定检查这一天，无论它有没有数据，指针都已经移过去了。
        // 下次调用 loadMoreDates 会从这一天的前一天开始。
        cursorDateRef.current = new Date(currentDate)

        try {
          const countResponse = await apiClient.getFramesCountByDate(dateStr)
          
          if (countResponse.total_count > 0) {
            // 只有 count > 0 才去加载详情并加入列表
            const frames = await apiClient.getFramesByDate(dateStr, 0, INITIAL_LOAD_BATCH_SIZE)
            const validFrames = frames.filter(f => f.image_path || f.image_base64)
            
            // 确保按时间戳升序排序（从最早到最晚）
            const sortedFrames = validFrames.sort((a, b) => {
              const timeA = new Date(a.timestamp).getTime()
              const timeB = new Date(b.timestamp).getTime()
              return timeA - timeB
            })
            
            if (sortedFrames.length > 0) {
              newGroups.push({
                date: dateStr,
                frames: sortedFrames,
                totalCount: countResponse.total_count,
                loadedCount: sortedFrames.length,
                isLoading: false
              })
            }
          } else {
            // count === 0，静默跳过，但 while 循环会继续处理前一天
          }
        } catch (error) {
          console.error(`Error checking date ${dateStr}`, error)
        }

        // 准备检查前一天
        currentDate.setDate(currentDate.getDate() - 1)
        checksCount++
      }

      // 3. 更新状态（【修复】增加去重逻辑，防止添加重复的日期组）
      if (newGroups.length > 0) {
        setDateGroups(prev => {
          const existingDates = new Set(prev.map(g => g.date))
          const uniqueNewGroups = newGroups.filter(g => !existingDates.has(g.date))
          return [...prev, ...uniqueNewGroups]
        })
      }

      // 4. 特殊情况处理：
      // 如果循环结束了（查了 MAX_EMPTY_CHECKS 天），但还没找到足够的组，且没到底部
      // 我们需要再次自动触发 loadMoreDates，否则用户看到的就是一片空白，必须手动滚动才能触发
      if (newGroups.length === 0 && hasMoreRef.current && checksCount >= MAX_EMPTY_CHECKS) {
        //  console.log('Checked batch was empty, auto-retrying next batch...')
         // 使用 setTimeout 让出主线程，避免 UI 卡死，然后继续递归查找
         setTimeout(() => {
             setLoading(false) // 先释放锁
             loadingRef.current = false
             if (loadMoreDatesRef.current) {
               loadMoreDatesRef.current()   // 再次触发
             }
         }, 10)
         return // 直接返回，不要在下面 setLoading(false) 了，因为我们已经重置并调度了
      }

    } catch (error) {
      console.error('Fatal error in loadMoreDates', error)
    } finally {
      setLoading(false)
      loadingRef.current = false
    }
  }, [earliestDate, dateRange.latest_date]) // 当 latest_date 更新时也重新加载
  
  // 更新 loadMoreDatesRef
  useEffect(() => {
    loadMoreDatesRef.current = loadMoreDates
  }, [loadMoreDates])

  // 【修复】加载更多照片（增加帧级别的去重逻辑）
  const loadMoreFramesForDate = useCallback(async (date: string) => {
    // 使用 ref 获取最新的 dateGroups，避免闭包问题
    const currentGroups = dateGroupsRef.current
    const dateGroup = currentGroups.find(g => g.date === date)
    if (!dateGroup) {
      console.warn(`[TimelineView] Date group not found for ${date}`)
      return
    }
    
    if (dateGroup.isLoading) {
      // console.log(`[TimelineView] Already loading frames for ${date}`)
      return
    }
    
    if (dateGroup.loadedCount >= dateGroup.totalCount) {
      // console.log(`[TimelineView] All frames loaded for ${date} (${dateGroup.loadedCount}/${dateGroup.totalCount})`)
      return
    }

    // console.log(`[TimelineView] Loading more frames for ${date}: offset=${dateGroup.loadedCount}, limit=${LOAD_MORE_BATCH_SIZE}, total=${dateGroup.totalCount}`)
    setDateGroups(prev => prev.map(g => 
      g.date === date ? { ...g, isLoading: true } : g
    ))

    try {
      const offset = dateGroup.loadedCount
      const frames = await apiClient.getFramesByDate(date, offset, LOAD_MORE_BATCH_SIZE)
      const validFrames = frames.filter(f => f.image_path || f.image_base64)
      
      // console.log(`[TimelineView] Received ${validFrames.length} valid frames for ${date} (offset=${offset})`)

      setDateGroups(prev => prev.map(g => {
        if (g.date !== date) return g

        // 【关键修复】去重逻辑：确保新加载的 frame 不在已有列表中
        // 这解决了 "Encountered two children with the same key" 错误
        const existingIds = new Set(g.frames.map(f => f.frame_id))
        const uniqueNewFrames = validFrames.filter(f => !existingIds.has(f.frame_id))
        
        // console.log(`[TimelineView] Adding ${uniqueNewFrames.length} new frames to ${date} (had ${g.frames.length} frames, ${validFrames.length - uniqueNewFrames.length} duplicates filtered)`)
        
        // 合并并确保按时间戳升序排序（从最早到最晚）
        const mergedFrames = [...g.frames, ...uniqueNewFrames]
        const sortedFrames = mergedFrames.sort((a, b) => {
          const timeA = new Date(a.timestamp).getTime()
          const timeB = new Date(b.timestamp).getTime()
          return timeA - timeB
        })

        return {
          ...g,
          frames: sortedFrames,
          loadedCount: sortedFrames.length, // 使用实际长度更新 count，防止偏差
          isLoading: false
        }
      }))
    } catch (error) {
      console.error(`[TimelineView] Failed to load more frames for date ${date}:`, error)
      setDateGroups(prev => prev.map(g => 
        g.date === date ? { ...g, isLoading: false } : g
      ))
    }
  }, [])
  
  // 存储 loadMoreFramesForDate 的 ref，避免闭包问题
  const loadMoreFramesForDateRef = useRef<((date: string) => Promise<void>) | null>(null)
  
  // 更新 loadMoreFramesForDateRef
  useEffect(() => {
    loadMoreFramesForDateRef.current = loadMoreFramesForDate
  }, [loadMoreFramesForDate])
  
  // 使用 useRef 存储最新的回调，避免依赖变化导致事件监听器重新绑定
  const handleVerticalScrollRef = useRef<() => void>()
  
  const handleVerticalScroll = useCallback(() => {
    // 使用 scrollContainerRef（真正的滚动容器）而不是 containerRef
    const scrollContainer = scrollContainerRef.current
    if (!scrollContainer) return
    
    // 使用 ref 获取最新的状态值，避免闭包问题
    if (loadingRef.current || !hasMoreRef.current) return

    const { scrollTop, scrollHeight, clientHeight } = scrollContainer
    const distanceToBottom = scrollHeight - scrollTop - clientHeight

    if (distanceToBottom < 500) {
      // 使用 ref 调用，确保总是调用最新的 loadMoreDates
      if (loadMoreDatesRef.current) {
        loadMoreDatesRef.current()
      }
    }
  }, [])
  
  // 更新 ref，确保 scrollHandler 总是调用最新的 handleVerticalScroll
  useEffect(() => {
    handleVerticalScrollRef.current = handleVerticalScroll
  }, [handleVerticalScroll])

  const handleHorizontalScroll = useCallback((date: string, scrollElement: HTMLElement) => {
    const currentGroups = dateGroupsRef.current
    const group = currentGroups.find(g => g.date === date)
    
    if (!group) {
      console.warn(`[TimelineView] Group not found for date: ${date}`)
      return
    }
    
    if (group.isLoading) {
      return
    }
    
    // 检查是否还有更多数据需要加载
    if (group.loadedCount >= group.totalCount) {
      return  // 已加载全部数据
    }

    const scrollLeft = scrollElement.scrollLeft
    const clientWidth = scrollElement.clientWidth
    
    // 计算距离已加载内容末尾的距离
    const loadedWidth = group.loadedCount * (ITEM_WIDTH + GAP)
    const distanceToLoadedEnd = loadedWidth - scrollLeft - clientWidth
    const itemWidth = ITEM_WIDTH + GAP
    const threshold = itemWidth * LOAD_MORE_THRESHOLD
    
    // 调试日志：每次滚动都记录（但只在接近时触发）
    if (distanceToLoadedEnd < threshold * 2) {  // 只在接近时记录日志，避免过多输出
      // console.log(`[TimelineView] Scroll event for ${date}: scrollLeft=${scrollLeft.toFixed(0)}, loadedWidth=${loadedWidth.toFixed(0)}, distanceToLoadedEnd=${distanceToLoadedEnd.toFixed(0)}px, threshold=${threshold.toFixed(0)}px, loaded=${group.loadedCount}/${group.totalCount}`)
    }
    
    // 如果距离已加载内容末尾很近（小于阈值），触发加载
    if (distanceToLoadedEnd < threshold) {
      // console.log(`[TimelineView] ✅ Triggering load more for ${date}, distanceToLoadedEnd: ${distanceToLoadedEnd.toFixed(0)}px < threshold: ${threshold.toFixed(0)}px`)
      if (loadMoreFramesForDateRef.current) {
        loadMoreFramesForDateRef.current(date).catch(err => {
          console.error(`Failed to load more frames for ${date}:`, err)
        })
      } else {
        console.error(`[TimelineView] loadMoreFramesForDateRef.current is null!`)
      }
    }
  }, [])

  // 查找真正的滚动容器（.timeline-view-container）并绑定滚动事件
  useEffect(() => {
    // 使用稳定的包装函数，避免因 handleVerticalScroll 变化导致重复绑定
    const scrollHandler = () => {
      if (handleVerticalScrollRef.current) {
        handleVerticalScrollRef.current()
      }
    }

    // 查找父元素中的 .timeline-view-container
    const findScrollContainer = (): HTMLElement | null => {
      // 方法1：从 containerRef 向上查找
      if (containerRef.current) {
        let parent = containerRef.current.parentElement
        while (parent) {
          if (parent.classList.contains('timeline-view-container')) {
            return parent as HTMLElement
          }
          parent = parent.parentElement
        }
      }
      
      // 方法2：如果方法1失败，使用 querySelector（更可靠）
      const found = document.querySelector('.timeline-view-container') as HTMLElement
      return found || null
    }

    let timeoutId: NodeJS.Timeout | null = null
    let retryCount = 0
    const MAX_RETRIES = 5
    
    const bindScroll = () => {
      const scrollContainer = findScrollContainer()
      if (scrollContainer) {
        // 如果已经绑定过，先解绑
        if (scrollContainerRef.current && scrollContainerRef.current !== scrollContainer) {
          scrollContainerRef.current.removeEventListener('scroll', scrollHandler)
        }
        
        scrollContainerRef.current = scrollContainer
        scrollContainer.addEventListener('scroll', scrollHandler, { passive: true })
        // console.log('Scroll container bound successfully')
        retryCount = 0 // 重置重试计数
      } else {
        retryCount++
        if (retryCount < MAX_RETRIES) {
          console.warn(`Could not find timeline-view-container, retrying (${retryCount}/${MAX_RETRIES})...`)
          // 如果没找到，100ms 后重试
          timeoutId = setTimeout(bindScroll, 100)
        } else {
          console.error('Failed to find timeline-view-container after multiple retries')
        }
      }
    }

    // 初始延迟，确保 DOM 已渲染（特别是 SearchResults 渲染后）
    timeoutId = setTimeout(bindScroll, 100)

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      if (scrollContainerRef.current) {
        scrollContainerRef.current.removeEventListener('scroll', scrollHandler)
        scrollContainerRef.current = null
      }
    }
  }, []) // 空依赖数组，只绑定一次
  
  // 监听 DOM 变化，当 SearchResults 渲染后重新绑定滚动容器
  useEffect(() => {
    let scrollHandler: (() => void) | null = null
    
    const createHandler = () => {
      return () => {
        if (handleVerticalScrollRef.current) {
          handleVerticalScrollRef.current()
        }
      }
    }
    
    const bindScrollContainer = () => {
      // 先解绑旧的
      if (scrollContainerRef.current && scrollHandler) {
        scrollContainerRef.current.removeEventListener('scroll', scrollHandler)
        scrollHandler = null
      }
      
      // 查找新的滚动容器
      const found = document.querySelector('.timeline-view-container') as HTMLElement
      if (found) {
        scrollHandler = createHandler()
        scrollContainerRef.current = found
        found.addEventListener('scroll', scrollHandler, { passive: true })
        // console.log('Scroll container re-bound')
        return true
      }
      return false
    }
    
    // 使用 MutationObserver 监听 DOM 变化（当 SearchResults 渲染时）
    const observer = new MutationObserver(() => {
      setTimeout(() => {
        if (!scrollContainerRef.current || !document.contains(scrollContainerRef.current)) {
          bindScrollContainer()
        }
      }, 100)
    })
    
    // 观察 content-area 的变化（SearchResults 在这里渲染）
    const contentArea = document.querySelector('.content-area')
    if (contentArea) {
      observer.observe(contentArea, {
        childList: true,
        subtree: false // 只监听直接子元素变化
      })
    }
    
    return () => {
      observer.disconnect()
      if (scrollContainerRef.current && scrollHandler) {
        scrollContainerRef.current.removeEventListener('scroll', scrollHandler)
      }
    }
  }, [])

  useEffect(() => {
    // 1. 如果还没有数据，触发初始加载
    if (dateGroups.length === 0 && !loadingRef.current) {
      loadMoreDates()
    } 
    // 2. 如果有了更晚的日期（比如后端刚更新），触发增量刷新
    else if (dateRange.latest_date && dateGroups.length > 0) {
      const newestLoadedDate = dateGroups[0].date
      if (dateRange.latest_date > newestLoadedDate) {
        // console.log(`[TimelineView] New latest date detected: ${dateRange.latest_date} > ${newestLoadedDate}, triggering refresh`)
        refreshTimeline()
      }
    }
  }, [dateRange.latest_date, loadMoreDates, dateGroups.length, refreshTimeline]) 

  useEffect(() => {
    const scrollHandlers: Map<string, () => void> = new Map()
    
    // console.log(`[TimelineView] Binding scroll handlers for ${dateGroups.length} date groups`)
    
    dateGroups.forEach(group => {
      const groupElement = dateGroupRefs.current.get(group.date)
      if (!groupElement) {
        console.warn(`[TimelineView] Group element not found for ${group.date}, will retry on next render`)
        // 如果元素还没有挂载，稍后重试
        return
      }
      
      const scrollContainer = groupElement.querySelector('.timeline-images') as HTMLElement
      if (!scrollContainer) {
        console.warn(`[TimelineView] Scroll container (.timeline-images) not found for ${group.date}`)
        return
      }
      
      // 如果已经绑定过，先移除旧的监听器
      if (scrollHandlers.has(group.date)) {
        const oldHandler = scrollHandlers.get(group.date)!
        scrollContainer.removeEventListener('scroll', oldHandler)
      }
      
      // 创建新的滚动处理函数
      const handler = () => {
        handleHorizontalScroll(group.date, scrollContainer)
      }
      
      scrollContainer.addEventListener('scroll', handler, { passive: true })
      scrollHandlers.set(group.date, handler)
      // console.log(`[TimelineView] ✅ Bound scroll handler for ${group.date} (loaded: ${group.loadedCount}/${group.totalCount})`)
      
      // 初始检查：如果已经滚动到已加载内容末尾附近，立即触发加载
      const { scrollLeft, clientWidth } = scrollContainer
      const loadedWidth = group.loadedCount * (ITEM_WIDTH + GAP)
      const distanceToLoadedEnd = loadedWidth - scrollLeft - clientWidth
      const itemWidth = ITEM_WIDTH + GAP
      const threshold = itemWidth * LOAD_MORE_THRESHOLD
      
      // console.log(`[TimelineView] Initial check for ${group.date}: loadedWidth=${loadedWidth.toFixed(0)}, distanceToLoadedEnd=${distanceToLoadedEnd.toFixed(0)}px, threshold=${threshold.toFixed(0)}px`)
      
      if (distanceToLoadedEnd < threshold && group.loadedCount < group.totalCount && !group.isLoading) {
        // console.log(`[TimelineView] Initial check: ${group.date} is near end, triggering load`)
        if (loadMoreFramesForDateRef.current) {
          loadMoreFramesForDateRef.current(group.date).catch(err => {
            console.error(`Failed to load initial frames for ${group.date}:`, err)
          })
        }
      }
    })
    
    return () => {
      // console.log(`[TimelineView] Cleaning up scroll handlers`)
      scrollHandlers.forEach((handler, date) => {
        const groupElement = dateGroupRefs.current.get(date)
        if (groupElement) {
          const scrollContainer = groupElement.querySelector('.timeline-images') as HTMLElement
          if (scrollContainer) {
            scrollContainer.removeEventListener('scroll', handler)
          }
        }
      })
    }
  }, [dateGroups, handleHorizontalScroll])

  const getImageUrl = (frame: Frame): string => {
    if (frame.image_base64) {
      return `data:image/jpeg;base64,${frame.image_base64}`
    }
    if (frame.image_path) {
      let fullPath = frame.image_path
      const isAbsolute = fullPath.startsWith('/') || fullPath.includes(':')
      
      if (isAbsolute) {
        // 绝对路径直接使用 file://
        return `file://${fullPath}`
      } else {
        // 相对路径处理
        if (projectRoot) {
          // 如果拿到了项目根目录，拼接成绝对路径使用 file://
          const cleanPath = fullPath.startsWith('./') ? fullPath.substring(2) : fullPath
          return `file://${projectRoot}/${cleanPath}`
        } else {
          // 如果还没拿到 projectRoot，使用后端 API 兜底
          // 这样可以保证图片在任何情况下都能显示，只是速度稍慢
          return apiClient.getImageUrl(fullPath)
        }
      }
    }
    return ''
  }

  const LoadingSpinner = () => (
    <div className="loading-spinner">
      <div className="spinner-icon" />
      <style>{`
        .loading-spinner { display: flex; justify-content: center; padding: 20px; }
        .spinner-icon { 
          width: 30px; height: 30px; 
          border: 3px solid #f3f3f3; 
          border-top: 3px solid #3498db; 
          border-radius: 50%; 
          animation: spin 1s linear infinite; 
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  )

  return (
    <div 
      ref={containerRef}
      className="timeline-container"
    >
      {dateGroups.map((group) => {
        // 计算剩余未加载的照片数量，用于撑开滚动条
        // 使用 loadedCount 而不是 frames.length，因为可能有无效的帧被过滤掉了
        const remainingCount = Math.max(0, group.totalCount - group.loadedCount);
        const spacerWidth = remainingCount * (ITEM_WIDTH + GAP);
        
        // 调试日志：只在有剩余时才输出
        if (remainingCount > 0) {
          // console.log(`[TimelineView] ${group.date}: loaded=${group.loadedCount}, total=${group.totalCount}, remaining=${remainingCount}, spacerWidth=${spacerWidth}px`)
        }

        return (
          <div 
            key={group.date} 
            className="timeline-date-group"
            ref={(el) => {
              if (el) dateGroupRefs.current.set(group.date, el)
              else dateGroupRefs.current.delete(group.date)
            }}
          >
            <div className="timeline-date-header">
              {group.date} ({group.loadedCount} / {group.totalCount}) 
            </div>
            <div 
              className="timeline-images" 
              style={{ display: 'flex', overflowX: 'auto', gap: `${GAP}px`, paddingBottom: '0px' }}
              data-date={group.date}
              data-loaded={group.loadedCount}
              data-total={group.totalCount}
            >
              {group.frames.map((frame) => (
                <div key={frame.frame_id} className="scroll-item" style={{ flex: '0 0 auto', width: `${ITEM_WIDTH}px` }}>
                  <img
                    src={getImageUrl(frame)}
                    alt={`Frame ${frame.frame_id}`}
                    loading="lazy"
                    style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '4px', cursor: 'pointer' }}
                    onClick={() => setPreviewImage({ url: getImageUrl(frame), timestamp: formatTimestamp(frame.timestamp) })}
                  />
                  <div className="timestamp-label" style={{ fontSize: '12px', color: '#666', marginTop: '4px', paddingBottom: '4px' }}>
                    {formatTimestamp(frame.timestamp)}
                  </div>
                </div>
              ))}
              
              {/* 加载指示器：放在已加载内容的末尾 */}
              {group.isLoading && <LoadingSpinner />}
              
              {/* Spacer：用于撑开滚动条，模拟剩余未加载内容的宽度 */}
              {/* 注意：如果有 loading 状态，Spacer 依然存在，保持总长度恒定 */}
              {spacerWidth > 0 && (
                <div 
                  style={{ 
                    flex: '0 0 auto', 
                    width: `${spacerWidth}px`, 
                    height: '190px', // 与图片高度一致，确保滚动条正确计算
                    backgroundColor: 'transparent',
                    pointerEvents: 'none' // 不可交互，只是一个占位符
                  }}
                  data-spacer="true"
                  data-remaining-count={remainingCount}
                />
              )}
            </div>
          </div>
        )
      })}
      
      {loading && <LoadingSpinner />}
      
      {!hasMore && dateGroups.length > 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#888' }}>
          已加载全部记录
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

export default TimelineView