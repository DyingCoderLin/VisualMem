import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ImagePreview from '../components/ImagePreview'
import MarkdownRenderer from '../components/MarkdownRenderer'
import { apiClient } from '../services/api'
import type {
  RewindSegment,
  RewindTimelineFrame,
  TaskMemory,
  TaskMemoryAskResponse,
  TaskMemoryListItem
} from '../services/api'

const TIMELINE_PAGE_SIZE = 36

type TimelineSelection = {
  startFrameId: string | null
  endFrameId: string | null
}

const segmentKey = (segment: RewindSegment, index: number): string =>
  segment.segment_id || segment.frame_id || `${segment.timestamp || segment.start_time || 'segment'}-${index}`

const formatTimestamp = (timestamp?: string | null): string => {
  if (!timestamp) return ''
  const parsed = new Date(timestamp)
  if (Number.isNaN(parsed.getTime())) return timestamp
  const month = String(parsed.getMonth() + 1).padStart(2, '0')
  const day = String(parsed.getDate()).padStart(2, '0')
  const hours = String(parsed.getHours()).padStart(2, '0')
  const minutes = String(parsed.getMinutes()).padStart(2, '0')
  return `${month}-${day} ${hours}:${minutes}`
}

const formatLocalInput = (date: Date): string => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

const segmentTime = (segment: RewindSegment): string => {
  const start = segment.start_time || segment.timestamp
  const end = segment.end_time
  if (!end || end === start) return formatTimestamp(start)
  return `${formatTimestamp(start)} - ${formatTimestamp(end)}`
}

const segmentDuration = (segment: RewindSegment): string => {
  const start = new Date(segment.start_time || segment.timestamp || '').getTime()
  const end = new Date(segment.end_time || segment.timestamp || '').getTime()
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return ''
  const minutes = Math.round((end - start) / 60000)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest ? `${hours}h ${rest}m` : `${hours}h`
}

const evidenceLabel = (segment: RewindSegment): string => {
  return segment.activity_label || segment.title || segment.app_name || segment.window_name || segment.frame_id || 'Timeline evidence'
}

const evidenceAppWindow = (segment: RewindSegment): string => {
  const subFrameWithApp = segment.sub_frames?.find((sf) => sf.app_name || sf.window_name)
  const app = segment.app_name || subFrameWithApp?.app_name || ''
  const windowName = segment.window_name || subFrameWithApp?.window_name || ''
  if (app && windowName) return `应用：${app}-${windowName}`
  if (app) return `应用：${app}`
  if (windowName) return `窗口：${windowName}`
  return ''
}

const segmentSortValue = (segment: RewindSegment): number => {
  const value = segment.end_time || segment.timestamp || segment.start_time || ''
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : 0
}

const toApiTimestamp = (value: string): string | undefined => {
  if (!value) return undefined
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return undefined
  return date.toISOString()
}

const segmentImagePaths = (segment: RewindSegment): string[] => {
  const paths = [segment.image_path, ...(segment.sub_frames || []).map((sf) => sf.image_path)]
  return paths.filter((path): path is string => Boolean(path))
}

const getFrameImageUrl = (frame: RewindTimelineFrame): string => {
  if (frame.image_path) return apiClient.getImageUrl(frame.image_path)
  const subPath = frame.sub_frames?.find((sf) => sf.image_path)?.image_path
  return subPath ? apiClient.getImageUrl(subPath) : ''
}

const segmentApps = (segment: RewindSegment): string[] => {
  return Array.from(new Set([segment.app_name, ...(segment.sub_frames || []).map((sf) => sf.app_name)].filter(Boolean) as string[]))
}

const segmentActivities = (segment: RewindSegment): string[] => {
  return [segment.activity_label].filter(Boolean) as string[]
}

const ScrollTitle = ({ text, className = '' }: { text: string; className?: string }) => {
  const ref = useRef<HTMLDivElement>(null)
  const dragRef = useRef({ active: false, startX: 0, scrollLeft: 0 })

  return (
    <div
      ref={ref}
      className={`rewind-scroll-title ${className}`}
      title={text}
      onPointerDown={(event) => {
        if (!ref.current) return
        dragRef.current = {
          active: true,
          startX: event.clientX,
          scrollLeft: ref.current.scrollLeft
        }
        ref.current.setPointerCapture(event.pointerId)
      }}
      onPointerMove={(event) => {
        if (!dragRef.current.active || !ref.current) return
        const dx = event.clientX - dragRef.current.startX
        ref.current.scrollLeft = dragRef.current.scrollLeft - dx
      }}
      onPointerUp={(event) => {
        dragRef.current.active = false
        ref.current?.releasePointerCapture(event.pointerId)
      }}
      onPointerCancel={() => {
        dragRef.current.active = false
      }}
    >
      {text}
    </div>
  )
}

const LoadingLabel = ({ label }: { label: string }) => (
  <span className="rewind-loading-label" aria-live="polite">
    <span>{label}</span>
    <span className="rewind-loading-dots" aria-hidden="true">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  </span>
)

function RewindView() {
  const [sourceQuery, setSourceQuery] = useState('')
  const [startTimeLocal, setStartTimeLocal] = useState('')
  const [endTimeLocal, setEndTimeLocal] = useState('')
  const [segments, setSegments] = useState<RewindSegment[]>([])
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null)
  const [selectedApps, setSelectedApps] = useState<Set<string>>(new Set())
  const [selectedActivities, setSelectedActivities] = useState<Set<string>>(new Set())
  const [appFilterMode, setAppFilterMode] = useState<'include' | 'exclude'>('include')
  const [timelineFrames, setTimelineFrames] = useState<RewindTimelineFrame[]>([])
  const [timelineTotal, setTimelineTotal] = useState(0)
  const [timelineOffset, setTimelineOffset] = useState(0)
  const [timelineSelection, setTimelineSelection] = useState<TimelineSelection>({ startFrameId: null, endFrameId: null })
  const [isTimelineLoading, setIsTimelineLoading] = useState(false)
  const [previewImage, setPreviewImage] = useState<{ url: string; timestamp: string } | null>(null)
  const [memories, setMemories] = useState<TaskMemoryListItem[]>([])
  const [activeMemory, setActiveMemory] = useState<TaskMemory | null>(null)
  const [draftTitle, setDraftTitle] = useState('')
  const [draftMarkdown, setDraftMarkdown] = useState('')
  const [question, setQuestion] = useState('')
  const [askResult, setAskResult] = useState<TaskMemoryAskResponse | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [isBuilding, setIsBuilding] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isAsking, setIsAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const appOptions = useMemo(() => {
    return Array.from(new Set(segments.flatMap(segmentApps))).sort()
  }, [segments])

  const activityOptions = useMemo(() => {
    return Array.from(new Set(segments.flatMap(segmentActivities))).sort()
  }, [segments])

  const visibleSegments = useMemo(() => {
    return segments
      .filter((segment) => {
        const apps = segmentApps(segment)
        if (selectedApps.size > 0) {
          const hasSelectedApp = apps.some((app) => selectedApps.has(app))
          if (appFilterMode === 'include' && !hasSelectedApp) return false
          if (appFilterMode === 'exclude' && hasSelectedApp) return false
        }
        if (selectedActivities.size > 0) {
          const activities = segmentActivities(segment)
          if (!activities.some((activity) => selectedActivities.has(activity))) return false
        }
        return true
      })
      .sort((a, b) => segmentSortValue(b) - segmentSortValue(a))
  }, [appFilterMode, segments, selectedActivities, selectedApps])

  const selectedSegments = useMemo(() => {
    return segments
      .filter((segment, index) => selectedIds.has(segmentKey(segment, index)))
      .sort((a, b) => segmentSortValue(b) - segmentSortValue(a))
  }, [segments, selectedIds])

  const activeSegment = useMemo(() => {
    if (activeSegmentId) {
      const index = segments.findIndex((segment, idx) => segmentKey(segment, idx) === activeSegmentId)
      if (index >= 0) return segments[index]
    }
    return visibleSegments[0] || null
  }, [activeSegmentId, segments, visibleSegments])

  const timelineMaxOffset = Math.max(0, timelineTotal - TIMELINE_PAGE_SIZE)
  const timelineLoadedEnd = timelineOffset + timelineFrames.length

  const selectedTimelineFrames = useMemo(() => {
    if (!timelineSelection.startFrameId) return []
    const startIndex = timelineFrames.findIndex((frame) => frame.frame_id === timelineSelection.startFrameId)
    if (startIndex < 0) return []
    const rawEndIndex = timelineSelection.endFrameId
      ? timelineFrames.findIndex((frame) => frame.frame_id === timelineSelection.endFrameId)
      : startIndex
    const endIndex = rawEndIndex >= 0 ? rawEndIndex : startIndex
    const from = Math.min(startIndex, endIndex)
    const to = Math.max(startIndex, endIndex)
    return timelineFrames.slice(from, to + 1)
  }, [timelineFrames, timelineSelection])

  const selectedTimelineFrameIds = useMemo(() => {
    return new Set(selectedTimelineFrames.map((frame) => frame.frame_id))
  }, [selectedTimelineFrames])

  const loadMemories = useCallback(async () => {
    try {
      const response = await apiClient.listTaskMemories()
      setMemories(response.memories)
    } catch (err) {
      console.warn('Failed to load Task Memories:', err)
    }
  }, [])

  useEffect(() => {
    loadMemories()
  }, [loadMemories])

  useEffect(() => {
    const activeStillVisible = visibleSegments.some(
      (segment) => segmentKey(segment, segments.indexOf(segment)) === activeSegmentId
    )
    if ((!activeSegmentId || !activeStillVisible) && visibleSegments.length > 0) {
      setActiveSegmentId(segmentKey(visibleSegments[0], segments.indexOf(visibleSegments[0])))
    } else if (visibleSegments.length === 0 && activeSegmentId) {
      setActiveSegmentId(null)
    }
  }, [activeSegmentId, segments, visibleSegments])

  useEffect(() => {
    setTimelineOffset(0)
    setTimelineFrames([])
    setTimelineTotal(0)
    setTimelineSelection({ startFrameId: null, endFrameId: null })
  }, [activeSegmentId])

  useEffect(() => {
    setTimelineSelection({ startFrameId: null, endFrameId: null })
  }, [timelineOffset])

  useEffect(() => {
    if (!activeSegment?.start_time && !activeSegment?.timestamp) return
    const start = activeSegment.start_time || activeSegment.timestamp
    const end = activeSegment.end_time || activeSegment.timestamp || start
    if (!start || !end) return

    const timeout = window.setTimeout(async () => {
      setIsTimelineLoading(true)
      try {
        const response = await apiClient.getRewindTimelineFrames({
          start_time: start,
          end_time: end,
          offset: timelineOffset,
          limit: TIMELINE_PAGE_SIZE
        })
        setTimelineFrames(response.frames)
        setTimelineTotal(response.total_count)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load segment timeline.'
        setError(message)
      } finally {
        setIsTimelineLoading(false)
      }
    }, 180)

    return () => window.clearTimeout(timeout)
  }, [activeSegment, timelineOffset])

  const applyMemory = (memory: TaskMemory) => {
    setActiveMemory(memory)
    setDraftTitle(memory.title)
    setDraftMarkdown(memory.markdown)
    setAskResult(null)
    setQuestion('')
  }

  const getSearchTimeRange = (): { start_time?: string; end_time?: string } | null => {
    if (startTimeLocal && endTimeLocal) {
      const start = new Date(startTimeLocal)
      const end = new Date(endTimeLocal)
      if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
        setError('Invalid time range.')
        return null
      }
      if (start > end) {
        setError('Start time must be before end time.')
        return null
      }
    }

    return {
      start_time: toApiTimestamp(startTimeLocal),
      end_time: toApiTimestamp(endTimeLocal)
    }
  }

  const getTimeRangeLabel = (): string => {
    const start = startTimeLocal ? formatTimestamp(startTimeLocal) : ''
    const end = endTimeLocal ? formatTimestamp(endTimeLocal) : ''
    if (start && end) return `${start} - ${end}`
    if (start) return `From ${start}`
    if (end) return `Until ${end}`
    return ''
  }

  const getMemorySourceQuery = (): string => {
    const query = sourceQuery.trim()
    const rangeLabel = getTimeRangeLabel()
    if (query && rangeLabel) return `${query} (Time range: ${rangeLabel})`
    if (query) return query
    if (rangeLabel) return `Time range: ${rangeLabel}`
    return 'Selected Rewind evidence'
  }

  const applyQuickRange = (range: '1h' | '6h' | 'today' | 'yesterday') => {
    const now = new Date()
    const start = new Date(now)
    const end = new Date(now)
    if (range === '1h') start.setHours(now.getHours() - 1)
    if (range === '6h') start.setHours(now.getHours() - 6)
    if (range === 'today') {
      start.setHours(0, 0, 0, 0)
      end.setHours(23, 59, 0, 0)
    }
    if (range === 'yesterday') {
      start.setDate(now.getDate() - 1)
      start.setHours(0, 0, 0, 0)
      end.setDate(now.getDate() - 1)
      end.setHours(23, 59, 0, 0)
    }
    setStartTimeLocal(formatLocalInput(start))
    setEndTimeLocal(formatLocalInput(end))
  }

  const handleSearch = async () => {
    const query = sourceQuery.trim()
    const timeRange = getSearchTimeRange()
    if (!timeRange) return

    if (!query && !timeRange.start_time && !timeRange.end_time) {
      setError('Enter a query or time range first.')
      return
    }

    setIsSearching(true)
    setError(null)
    try {
      const response = await apiClient.searchRewindSegments({
        query,
        start_time: timeRange.start_time,
        end_time: timeRange.end_time,
        top_k: 12
      })
      setSegments(response.segments)
      setSelectedIds(new Set(response.segments.map((segment, index) => segmentKey(segment, index))))
      setActiveSegmentId(response.segments[0] ? segmentKey(response.segments[0], 0) : null)
      setSelectedApps(new Set())
      setSelectedActivities(new Set())
      if (response.segments.length === 0) {
        setError('No timeline evidence found.')
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed.'
      setError(message)
    } finally {
      setIsSearching(false)
    }
  }

  const handleToggleSegment = (key: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const handleTimelineFrameSelect = (frame: RewindTimelineFrame) => {
    setTimelineSelection((prev) => {
      if (!prev.startFrameId || prev.endFrameId) {
        return { startFrameId: frame.frame_id, endFrameId: null }
      }
      return { startFrameId: prev.startFrameId, endFrameId: frame.frame_id }
    })
  }

  const handleCreateSegmentFromTimeline = () => {
    if (!activeSegment || selectedTimelineFrames.length === 0) return
    const startFrame = selectedTimelineFrames[0]
    const endFrame = selectedTimelineFrames[selectedTimelineFrames.length - 1]
    const startTime = startFrame.timestamp <= endFrame.timestamp ? startFrame.timestamp : endFrame.timestamp
    const endTime = startFrame.timestamp <= endFrame.timestamp ? endFrame.timestamp : startFrame.timestamp
    const imagePath = startFrame.image_path || startFrame.sub_frames?.find((sf) => sf.image_path)?.image_path || null
    const appFromFrame = startFrame.sub_frames?.find((sf) => sf.app_name)?.app_name || null
    const windowFromFrame = startFrame.sub_frames?.find((sf) => sf.window_name)?.window_name || null
    const manualId = `manual_${Date.now()}_${startFrame.frame_id}`

    const manualSegment: RewindSegment = {
      segment_id: manualId,
      frame_id: startFrame.frame_id,
      timestamp: startFrame.timestamp,
      start_time: startTime,
      end_time: endTime,
      title: evidenceLabel(activeSegment),
      app_name: activeSegment.app_name || appFromFrame,
      window_name: activeSegment.window_name || windowFromFrame,
      activity_label: activeSegment.activity_label || null,
      image_path: imagePath,
      ocr_text: selectedTimelineFrames.map((frame) => frame.ocr_text).filter(Boolean).join('\n\n').slice(0, 4000),
      sub_frames: selectedTimelineFrames.flatMap((frame) => frame.sub_frames || []).slice(0, 12),
      metadata: {
        ...(activeSegment.metadata || {}),
        manual_segment: true,
        parent_segment_id: activeSegment.segment_id || activeSegment.frame_id || null,
        timeline_frame_refs: selectedTimelineFrames.map((frame) => ({
          frame_id: frame.frame_id,
          timestamp: frame.timestamp,
          image_path: frame.image_path || frame.sub_frames?.find((sf) => sf.image_path)?.image_path || null
        }))
      }
    }

    setSegments((prev) => [manualSegment, ...prev])
    setSelectedIds((prev) => {
      const next = new Set(prev)
      next.add(manualId)
      return next
    })
    setActiveSegmentId(manualId)
    setTimelineSelection({ startFrameId: null, endFrameId: null })
  }

  const toggleSetValue = (value: string, setter: (next: Set<string>) => void, current: Set<string>) => {
    const next = new Set(current)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    setter(next)
  }

  const handleBuildMemory = async () => {
    if (selectedSegments.length === 0) {
      setError('Select at least one evidence segment.')
      return
    }
    const timeRange = getSearchTimeRange()
    if (!timeRange) return

    setIsBuilding(true)
    setError(null)
    try {
      const memory = await apiClient.buildRewindContext({
        source_query: getMemorySourceQuery(),
        selected_segments: selectedSegments,
        start_time: timeRange.start_time,
        end_time: timeRange.end_time
      })
      applyMemory(memory)
      await loadMemories()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to build Task Memory.'
      setError(message)
    } finally {
      setIsBuilding(false)
    }
  }

  const handleOpenMemory = async (taskMemoryId: string) => {
    setError(null)
    try {
      const memory = await apiClient.getTaskMemory(taskMemoryId)
      applyMemory(memory)
      setSourceQuery(memory.source_query)
      setSegments(memory.selected_segments)
      setSelectedIds(new Set(memory.selected_segments.map((segment, index) => segmentKey(segment, index))))
      setActiveSegmentId(memory.selected_segments[0] ? segmentKey(memory.selected_segments[0], 0) : null)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to open Task Memory.'
      setError(message)
    }
  }

  const handleSaveMemory = async () => {
    if (!activeMemory) return
    const title = draftTitle.trim()
    if (!title) {
      setError('Title cannot be empty.')
      return
    }

    setIsSaving(true)
    setError(null)
    try {
      const saved = await apiClient.updateTaskMemory(activeMemory.task_memory_id, {
        title,
        markdown: draftMarkdown
      })
      applyMemory(saved)
      await loadMemories()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save Task Memory.'
      setError(message)
    } finally {
      setIsSaving(false)
    }
  }

  const handleAsk = async () => {
    if (!activeMemory) return
    const currentQuestion = question.trim()
    if (!currentQuestion) {
      setError('Enter a question first.')
      return
    }

    setIsAsking(true)
    setError(null)
    setAskResult(null)
    try {
      const response = await apiClient.askTaskMemory(activeMemory.task_memory_id, {
        question: currentQuestion,
        markdown: draftMarkdown
      })
      setAskResult(response)
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === 'AbortError'
          ? 'Ask timed out. The AI service did not return in time; try again or reduce the selected evidence.'
          : err instanceof Error
            ? err.message
            : 'Ask failed.'
      setError(message)
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="rewind-view">
      <aside className="rewind-filter-panel">
        <div className="rewind-panel-title">Search & Filters</div>
        <label className="rewind-field">
          <span>Selected query</span>
          <textarea
            className="rewind-query-input"
            value={sourceQuery}
            onChange={(event) => setSourceQuery(event.target.value)}
            placeholder="Ask VisualMem..."
            maxLength={500}
          />
        </label>

        <div className="rewind-time-filter-grid">
          <label className="rewind-time-field">
            <span>Start</span>
            <input
              type="datetime-local"
              value={startTimeLocal}
              onChange={(event) => setStartTimeLocal(event.target.value)}
            />
          </label>
          <label className="rewind-time-field">
            <span>End</span>
            <input
              type="datetime-local"
              value={endTimeLocal}
              onChange={(event) => setEndTimeLocal(event.target.value)}
            />
          </label>
        </div>

        <div className="rewind-filter-block">
          <div className="rewind-filter-title">Quick ranges</div>
          <div className="rewind-chip-grid">
            <button className="rewind-filter-chip" onClick={() => applyQuickRange('1h')}>Last 1h</button>
            <button className="rewind-filter-chip" onClick={() => applyQuickRange('6h')}>Last 6h</button>
            <button className="rewind-filter-chip" onClick={() => applyQuickRange('today')}>Today</button>
            <button className="rewind-filter-chip" onClick={() => applyQuickRange('yesterday')}>Yesterday</button>
          </div>
        </div>

        <div className="rewind-filter-block">
          <div className="rewind-filter-header">
            <span>App filter</span>
            <button className="rewind-link-button" onClick={() => setSelectedApps(new Set())}>Clear</button>
          </div>
          <div className="rewind-chip-list">
            {appOptions.map((app) => (
              <button
                key={app}
                className={`rewind-filter-chip ${selectedApps.has(app) ? 'active' : ''}`}
                onClick={() => toggleSetValue(app, setSelectedApps, selectedApps)}
              >
                {app}
              </button>
            ))}
            {appOptions.length === 0 && <div className="rewind-empty-state">Search first to populate apps.</div>}
          </div>
        </div>

        <div className="rewind-filter-block">
          <div className="rewind-filter-title">Activity types</div>
          <div className="rewind-chip-list">
            {activityOptions.map((activity) => (
              <button
                key={activity}
                className={`rewind-filter-chip ${selectedActivities.has(activity) ? 'active' : ''}`}
                onClick={() => toggleSetValue(activity, setSelectedActivities, selectedActivities)}
              >
                {activity}
              </button>
            ))}
            {activityOptions.length === 0 && <div className="rewind-empty-state">No activity labels yet.</div>}
          </div>
        </div>

        <div className="rewind-filter-block">
          <div className="rewind-filter-title">Include / Exclude</div>
          <label className="rewind-radio-row">
            <input
              type="radio"
              checked={appFilterMode === 'include'}
              onChange={() => setAppFilterMode('include')}
            />
            <span>Include selected apps</span>
          </label>
          <label className="rewind-radio-row">
            <input
              type="radio"
              checked={appFilterMode === 'exclude'}
              onChange={() => setAppFilterMode('exclude')}
            />
            <span>Exclude selected apps</span>
          </label>
        </div>

        <button
          className={`btn btn-primary rewind-search-button rewind-loading-button${isSearching ? ' is-loading' : ''}`}
          onClick={handleSearch}
          disabled={isSearching}
        >
          {isSearching ? <LoadingLabel label="Searching" /> : 'Search Timeline'}
        </button>
        <button
          className="rewind-link-button rewind-reset-button"
          onClick={() => {
            setSelectedApps(new Set())
            setSelectedActivities(new Set())
            setStartTimeLocal('')
            setEndTimeLocal('')
          }}
        >
          Reset all filters
        </button>
      </aside>

      <main className="rewind-timeline-pane">
        {error && <div className="rewind-error">{error}</div>}

        <section className="rewind-evidence-section">
          <div className="rewind-section-header">
            <div>
              <span>Evidence Timeline</span>
              <small>{visibleSegments.length} visible / {segments.length} matched</small>
            </div>
            <span className="rewind-count">{selectedSegments.length} selected</span>
          </div>

          <div className="rewind-evidence-list">
            {visibleSegments.map((segment) => {
              const originalIndex = segments.indexOf(segment)
              const key = segmentKey(segment, originalIndex)
              const imagePaths = segmentImagePaths(segment)
              const isActive = activeSegmentId === key
              const isSelected = selectedIds.has(key)
              return (
                <article
                  key={key}
                  className={`rewind-evidence-card ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveSegmentId(key)}
                >
                  <div className="rewind-evidence-card-main">
                    <label className="rewind-select-box" onClick={(event) => event.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleToggleSegment(key)}
                      />
                    </label>
                    <div className="rewind-evidence-body">
                      <ScrollTitle text={evidenceLabel(segment)} />
                      {evidenceAppWindow(segment) && (
                        <div className="rewind-evidence-app">{evidenceAppWindow(segment)}</div>
                      )}
                      <div className="rewind-evidence-time-row">
                        <span className="rewind-evidence-time">{segmentTime(segment)}</span>
                        {segmentDuration(segment) && <span className="rewind-duration">{segmentDuration(segment)}</span>}
                      </div>
                    </div>
                  </div>
                  <div className="rewind-thumb-strip">
                    {imagePaths.slice(0, 6).map((path, index) => (
                      <button
                        key={`${path}-${index}`}
                        className="rewind-thumb-button"
                        onClick={(event) => {
                          event.stopPropagation()
                          setPreviewImage({ url: apiClient.getImageUrl(path), timestamp: segmentTime(segment) })
                        }}
                        title="Open screenshot"
                      >
                        <img src={apiClient.getImageUrl(path)} alt="Timeline evidence" loading="lazy" />
                      </button>
                    ))}
                    {imagePaths.length > 6 && <span className="rewind-thumb-more">+{imagePaths.length - 6}</span>}
                  </div>
                </article>
              )
            })}
            {visibleSegments.length === 0 && <div className="rewind-empty-state">No evidence loaded.</div>}
          </div>
        </section>

        <section className="rewind-session-browser">
          <div className="rewind-section-header">
            <div>
              <span>Segment Timeline</span>
              <small>{activeSegment ? segmentTime(activeSegment) : 'Select an evidence segment'}</small>
            </div>
            <span className="rewind-count">
              {timelineTotal ? `${timelineOffset + 1}-${timelineLoadedEnd} / ${timelineTotal}` : '0 frames'}
            </span>
          </div>

          {activeSegment && (
            <>
              <div className="rewind-progress-row">
                <input
                  type="range"
                  min={0}
                  max={timelineMaxOffset}
                  step={Math.max(1, Math.floor(TIMELINE_PAGE_SIZE / 2))}
                  value={Math.min(timelineOffset, timelineMaxOffset)}
                  onChange={(event) => setTimelineOffset(Number(event.target.value))}
                  disabled={timelineTotal <= TIMELINE_PAGE_SIZE}
                />
                <button
                  className="rewind-icon-button"
                  disabled={timelineOffset <= 0}
                  onClick={() => setTimelineOffset(Math.max(0, timelineOffset - TIMELINE_PAGE_SIZE))}
                >
                  Earlier
                </button>
                <button
                  className="rewind-icon-button"
                  disabled={timelineOffset >= timelineMaxOffset}
                  onClick={() => setTimelineOffset(Math.min(timelineMaxOffset, timelineOffset + TIMELINE_PAGE_SIZE))}
                >
                  Later
                </button>
              </div>
              <div className="rewind-segment-editor-row">
                <span>
                  {selectedTimelineFrames.length > 0
                    ? `${selectedTimelineFrames.length} frame${selectedTimelineFrames.length > 1 ? 's' : ''} marked`
                    : 'No range marked'}
                </span>
                <button
                  className="rewind-icon-button"
                  disabled={selectedTimelineFrames.length === 0}
                  onClick={handleCreateSegmentFromTimeline}
                >
                  Create Segment
                </button>
                <button
                  className="rewind-icon-button"
                  disabled={selectedTimelineFrames.length === 0}
                  onClick={() => setTimelineSelection({ startFrameId: null, endFrameId: null })}
                >
                  Clear
                </button>
              </div>
            </>
          )}

          <div className="rewind-frame-strip">
            {timelineFrames.map((frame) => {
              const imageUrl = getFrameImageUrl(frame)
              const isMarked = selectedTimelineFrameIds.has(frame.frame_id)
              const isRangeStart = timelineSelection.startFrameId === frame.frame_id
              const isRangeEnd = timelineSelection.endFrameId === frame.frame_id
              return (
                <div
                  key={frame.frame_id}
                  className={`rewind-frame-tile ${isMarked ? 'marked' : ''} ${isRangeStart ? 'range-start' : ''} ${isRangeEnd ? 'range-end' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleTimelineFrameSelect(frame)}
                  onDoubleClick={() => imageUrl && setPreviewImage({ url: imageUrl, timestamp: formatTimestamp(frame.timestamp) })}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      handleTimelineFrameSelect(frame)
                    }
                  }}
                >
                  {imageUrl ? <img src={imageUrl} alt={frame.frame_id} loading="lazy" /> : <span>No image</span>}
                  <div className="rewind-frame-tile-footer">
                    <small>{formatTimestamp(frame.timestamp)}</small>
                    {imageUrl && (
                      <button
                        className="rewind-frame-open-button"
                        onClick={(event) => {
                          event.stopPropagation()
                          setPreviewImage({ url: imageUrl, timestamp: formatTimestamp(frame.timestamp) })
                        }}
                      >
                        Open
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
            {isTimelineLoading && <div className="rewind-frame-loading">Loading...</div>}
            {!isTimelineLoading && activeSegment && timelineFrames.length === 0 && (
              <div className="rewind-empty-state">No full-screen frames in this span.</div>
            )}
          </div>
        </section>
      </main>

      <aside className="rewind-workspace">
        <div className="rewind-workspace-header">
          <div className="rewind-title-editor">
            {activeMemory ? (
              <>
                <input
                  value={draftTitle}
                  onChange={(event) => setDraftTitle(event.target.value)}
                  className="rewind-title-input"
                  title={draftTitle}
                />
                <div className="rewind-memory-id">Saved - {activeMemory.task_memory_id}</div>
              </>
            ) : (
              <>
                <div className="rewind-panel-title">Task Memory</div>
                <div className="rewind-memory-id">No active memory</div>
              </>
            )}
          </div>
          {activeMemory && (
            <button className="btn" onClick={handleSaveMemory} disabled={isSaving}>
              {isSaving ? 'Saving' : 'Save'}
            </button>
          )}
        </div>

        <div className="rewind-memory-tools">
          <button
            className={`btn btn-primary rewind-build-button rewind-loading-button${isBuilding ? ' is-loading' : ''}`}
            onClick={handleBuildMemory}
            disabled={isBuilding || selectedSegments.length === 0}
          >
            {isBuilding ? <LoadingLabel label="Building" /> : 'Build Task Memory'}
          </button>
          <button className="rewind-icon-button" onClick={loadMemories}>Refresh saved</button>
        </div>

        <div className="rewind-saved-section">
          <div className="rewind-section-header">
            <span>Saved Memories</span>
            <span className="rewind-count">{memories.length}</span>
          </div>
          <div className="rewind-memory-list">
            {memories.map((memory) => (
              <button
                key={memory.task_memory_id}
                className={`rewind-memory-item ${
                  activeMemory?.task_memory_id === memory.task_memory_id ? 'active' : ''
                }`}
                onClick={() => handleOpenMemory(memory.task_memory_id)}
              >
                <ScrollTitle text={memory.title} className="rewind-memory-scroll-title" />
                <span className="rewind-memory-meta">
                  {formatTimestamp(memory.updated_at)} - {memory.selected_segment_count} segments
                </span>
              </button>
            ))}
            {memories.length === 0 && <div className="rewind-empty-state">No saved memories yet.</div>}
          </div>
        </div>

        {activeMemory ? (
          <>
            <div className="rewind-markdown-editor">
              <textarea
                value={draftMarkdown}
                onChange={(event) => setDraftMarkdown(event.target.value)}
                spellCheck={false}
              />
            </div>

            <div className="rewind-ask-panel">
              <div className="rewind-section-header">
                <span>Ask with this memory</span>
              </div>
              <div className="rewind-ask-row">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="What should I do next?"
                />
                <button
                  className={`btn btn-primary rewind-loading-button${isAsking ? ' is-loading' : ''}`}
                  onClick={handleAsk}
                  disabled={isAsking}
                >
                  {isAsking ? <LoadingLabel label="Asking" /> : 'Ask'}
                </button>
              </div>
              {askResult && (
                <div className="rewind-answer">
                  <MarkdownRenderer content={askResult.answer} />
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="rewind-no-memory">
            Select evidence and build a Task Memory.
          </div>
        )}
      </aside>

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

export default RewindView
