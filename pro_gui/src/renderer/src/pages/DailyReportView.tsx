import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { apiClient } from '../services/api'
import type { DailyReportPayload } from '../types/dailyReport'
import '../styles/DailyReport.css'

function localISODate(): string {
  const n = new Date()
  const y = n.getFullYear()
  const m = String(n.getMonth() + 1).padStart(2, '0')
  const d = String(n.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

function formatDateTitle(iso: string): string {
  const parts = iso.split('-').map(Number)
  const y = parts[0]
  const mo = parts[1]
  const day = parts[2]
  if (!y || !mo || !day) return iso
  return new Date(y, mo - 1, day).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

function CalendarGlyph() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M7 3v2M17 3v2M4 9h16M6 5h12a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V7a2 2 0 012-2z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function BulletList({ items }: { items: string[] }) {
  if (!items?.length) return <p className="daily-report-date-empty">暂无内容</p>
  return (
    <ul>
      {items.map((line, i) => (
        <li key={i}>{line}</li>
      ))}
    </ul>
  )
}

const DailyReportView: React.FC = () => {
  const [dates, setDates] = useState<string[]>([])
  /** 列表拉取完成且已根据列表选好默认日期后再展示详情 */
  const [listLoaded, setListLoaded] = useState(false)
  const [selectedDate, setSelectedDate] = useState<string | null>(null)
  const [report, setReport] = useState<DailyReportPayload | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [genLoading, setGenLoading] = useState(false)
  const [genError, setGenError] = useState<string | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  /** `onlyDates`: 只刷新侧边栏列表，不改动当前选中日期（生成日报后调用） */
  const refreshList = useCallback(async (options?: { onlyDates?: boolean }) => {
    const { dates: d } = await apiClient.listDailyReports()
    setDates(d)
    if (options?.onlyDates) {
      return
    }
    const today = localISODate()
    if (d.includes(today)) {
      setSelectedDate(today)
    } else if (d.length > 0) {
      setSelectedDate(d[0])
    } else {
      setSelectedDate(today)
    }
    setListLoaded(true)
  }, [])

  useEffect(() => {
    refreshList().catch((e) => console.error('listDailyReports', e))
  }, [refreshList])

  /** 仅当该日在列表中（磁盘上确有 JSON）时才请求详情，避免对「今天」无脑 GET 导致 404 */
  const selectedHasReport = useMemo(
    () => (selectedDate != null ? dates.includes(selectedDate) : false),
    [dates, selectedDate]
  )

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      if (!listLoaded || selectedDate === null) {
        return
      }
      if (!dates.includes(selectedDate)) {
        if (!cancelled) {
          setReport(null)
          setLoadError(null)
          setDetailLoading(false)
        }
        return
      }
      setDetailLoading(true)
      setLoadError(null)
      setReport(null)
      try {
        const data = await apiClient.getDailyReport(selectedDate)
        if (!cancelled) setReport(data)
      } catch (e) {
        if (cancelled) return
        setReport(null)
        setLoadError(e instanceof Error ? e.message : '加载失败')
      } finally {
        if (!cancelled) setDetailLoading(false)
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [listLoaded, selectedDate, dates])

  const titleDate = useMemo(
    () => (selectedDate ? formatDateTitle(selectedDate) : ''),
    [selectedDate]
  )

  const handleGenerate = async () => {
    if (!selectedDate) return
    setGenLoading(true)
    setGenError(null)
    try {
      const data = await apiClient.generateDailyReport(selectedDate)
      setReport(data)
      setLoadError(null)
      await refreshList({ onlyDates: true })
    } catch (e) {
      setGenError(e instanceof Error ? e.message : '生成失败')
    } finally {
      setGenLoading(false)
    }
  }

  const wm = report?.report?.work_module
  const lm = report?.report?.life_module
  const focusScore = lm?.focus_score
  const statusNote = report?.status
  const statusMessage = report?.message

  return (
    <div className="daily-report-page">
      <div className="daily-report-body">
        <aside className="daily-report-date-sidebar">
          <div className="daily-report-date-sidebar-title">日期</div>
          {dates.length === 0 ? (
            <p className="daily-report-date-empty">尚无日报。选择上方日期后点击「生成日报」即可创建。</p>
          ) : (
            dates.map((d) => (
              <div
                key={d}
                className={`daily-report-date-item ${d === selectedDate ? 'active' : ''}`}
                onClick={() => setSelectedDate(d)}
                role="button"
                tabIndex={0}
                onKeyDown={(ev) => {
                  if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault()
                    setSelectedDate(d)
                  }
                }}
              >
                {d}
              </div>
            ))
          )}
        </aside>

        <div className="daily-report-main">
          <div className="daily-report-toolbar">
            <div className="daily-report-date-field">
              <CalendarGlyph />
              <input
                className="daily-report-date-input"
                type="date"
                value={selectedDate ?? ''}
                onChange={(e) => setSelectedDate(e.target.value)}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
              <button
                type="button"
                className="daily-report-gen-btn"
                disabled={genLoading || !listLoaded || !selectedDate}
                onClick={handleGenerate}
              >
                {genLoading ? '生成中…' : '生成日报'}
              </button>
              {genError ? <div className="daily-report-error">{genError}</div> : null}
            </div>
          </div>

          <div className="daily-report-card-wrap">
            {!listLoaded ? (
              <div className="daily-report-empty-state">加载中…</div>
            ) : detailLoading ? (
              <div className="daily-report-empty-state">加载中…</div>
            ) : !selectedHasReport ? (
              <div className="daily-report-empty-state">
                该日尚无日报文件。请从左侧选已有日期，或选日期后点击「生成日报」。
              </div>
            ) : loadError && !report ? (
              <div className="daily-report-empty-state">{loadError}</div>
            ) : (
              <div className="daily-report-card">
                <div className="daily-report-card-header">
                  <div>
                    <div className="daily-report-kicker">Daily Report</div>
                    <div className="daily-report-title">{titleDate}</div>
                  </div>
                  <div className="daily-report-focus-box">
                    {focusScore != null ? (
                      <>
                        <div className="daily-report-focus-value">{focusScore}</div>
                        <div className="daily-report-focus-caption">Focus score</div>
                      </>
                    ) : (
                      <div className="daily-report-focus-caption">—</div>
                    )}
                  </div>
                </div>

                {statusNote === 'no_activity_sessions' && statusMessage ? (
                  <div className="daily-report-banner warn">{statusMessage}</div>
                ) : null}

                <section>
                  <h2 className="daily-report-section-title">App usage</h2>
                  {(report?.app_usage_summary?.length ?? 0) === 0 ? (
                    <p className="daily-report-date-empty">暂无应用前台时长数据。</p>
                  ) : (
                    report!.app_usage_summary.map((row) => {
                      const intent = row.purpose_keywords?.[0]?.trim() || '—'
                      const pct = Math.min(100, Math.max(0, Number(row.percentage) || 0))
                      return (
                        <div key={row.app} className="daily-report-app-row">
                          <div className="daily-report-app-head">
                            <span className="daily-report-app-name">{row.app}</span>
                            <div className="daily-report-app-meta">
                              <span className="daily-report-chip">
                                {row.focused_minutes} min · {pct}%
                              </span>
                              <span className="daily-report-pill" title={intent}>
                                {intent}
                              </span>
                            </div>
                          </div>
                          <div className="daily-report-bar-track">
                            <div className="daily-report-bar-fill" style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                      )
                    })
                  )}
                </section>

                <section style={{ marginTop: 'var(--spacing-lg)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                    <span className="daily-report-pill daily-report-pill-work">Work</span>
                  </div>
                  <div className="daily-report-panel">
                    <h4>核心产出</h4>
                    <BulletList items={wm?.core_accomplishments ?? []} />
                    <h4 style={{ marginTop: 'var(--spacing-md)' }}>研究与支撑</h4>
                    <BulletList items={wm?.supporting_research ?? []} />
                    <h4 style={{ marginTop: 'var(--spacing-md)' }}>阻碍与未完成</h4>
                    <BulletList items={wm?.blockers_and_unfinished ?? []} />
                    <h4 style={{ marginTop: 'var(--spacing-md)' }}>明日建议</h4>
                    <BulletList items={wm?.tomorrow_suggestions ?? []} />
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                    <span className="daily-report-pill daily-report-pill-life">Life</span>
                  </div>
                  <div className="daily-report-panel">
                    {lm?.focus_interpretation ? (
                      <>
                        <h4>专注解读</h4>
                        <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.5, color: 'var(--text-primary)' }}>
                          {lm.focus_interpretation}
                        </p>
                      </>
                    ) : null}
                    {lm?.fragmentation_diagnosis ? (
                      <>
                        <h4 style={{ marginTop: 'var(--spacing-md)' }}>碎片化诊断</h4>
                        <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.5, color: 'var(--text-primary)' }}>
                          {lm.fragmentation_diagnosis}
                        </p>
                      </>
                    ) : null}
                    {lm?.distraction_patterns ? (
                      <>
                        <h4 style={{ marginTop: 'var(--spacing-md)' }}>干扰模式</h4>
                        <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.5, color: 'var(--text-primary)' }}>
                          {lm.distraction_patterns}
                        </p>
                      </>
                    ) : null}
                    <h4 style={{ marginTop: 'var(--spacing-md)' }}>改进建议</h4>
                    <BulletList items={lm?.intervention_suggestions ?? []} />
                  </div>
                </section>

                <section style={{ marginTop: 'var(--spacing-lg)' }}>
                  <h2 className="daily-report-section-title">今日小结</h2>
                  <div className="daily-report-takeaway">
                    {(report?.report?.today_summary?.length ?? 0) === 0 ? (
                      <p className="daily-report-date-empty">暂无小结</p>
                    ) : (
                      report!.report.today_summary.map((para, i) => (
                        <p key={i}>{para}</p>
                      ))
                    )}
                  </div>
                </section>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DailyReportView
