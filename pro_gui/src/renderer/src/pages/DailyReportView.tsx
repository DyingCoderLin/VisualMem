import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiClient } from '../services/api'
import type {
  AssistantTone,
  DailyReportPayload,
  PlanningStyle,
  ReportPreferences
} from '../types/dailyReport'
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

function SettingsGearGlyph() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 15a3 3 0 100-6 3 3 0 000 6z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"
        stroke="currentColor"
        strokeWidth="1.5"
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

/* ------------------------------------------------------------------ */
/* Preferences Panel                                                  */
/* ------------------------------------------------------------------ */

const TONE_OPTIONS: { value: AssistantTone; label: string }[] = [
  { value: 'soft', label: '柔和' },
  { value: 'normal', label: '正常' },
  { value: 'push', label: 'Push' },
]

const PLANNING_OPTIONS: { value: PlanningStyle; label: string; desc: string }[] = [
  { value: 'detailed-present', label: '注重细致的当下', desc: '具体时段 · 可执行微行动' },
  { value: 'rough-overall', label: '粗略的总体规划', desc: '主题与方向 · 少细节' },
]

function PreferencesPanel({
  prefs,
  prefsSaving,
  soulLoading,
  selectedDate,
  onRefreshPrefs,
  onSetPrefsSaving,
  onUpdateDateGoal,
  onGenerateSoul,
}: {
  prefs: ReportPreferences | null
  prefsSaving: boolean
  soulLoading: boolean
  selectedDate: string | null
  onRefreshPrefs: () => Promise<void>
  onSetPrefsSaving: (v: boolean) => void
  onUpdateDateGoal: (date: string, goal: string) => void
  onGenerateSoul: () => Promise<void>
}) {
  const [goalsText, setGoalsText] = useState('')
  const [dateGoalDraft, setDateGoalDraft] = useState('')
  const [tone, setTone] = useState<AssistantTone>('normal')
  const [planning, setPlanning] = useState<PlanningStyle>('detailed-present')
  const [showSoul, setShowSoul] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const prevGoalsRef = useRef('')
  const prevToneRef = useRef<AssistantTone>('normal')
  const prevPlanningRef = useRef<PlanningStyle>('detailed-present')

  useEffect(() => {
    if (!prefs) return
    setGoalsText(prefs.long_term_goals.join('\n'))
    setDateGoalDraft(prefs.date_goals[selectedDate || ''] || '')
    setTone(prefs.assistant_tone)
    setPlanning(prefs.planning_style)
    prevGoalsRef.current = prefs.long_term_goals.join('\n')
    prevToneRef.current = prefs.assistant_tone
    prevPlanningRef.current = prefs.planning_style
  }, [prefs, selectedDate])

  const hasChanges =
    goalsText !== prevGoalsRef.current ||
    tone !== prevToneRef.current ||
    planning !== prevPlanningRef.current

  const handleSave = async () => {
    setLocalError(null)
    try {
      onSetPrefsSaving(true)
      const goalsParsed = goalsText
        .split('\n')
        .map((g) => g.trim())
        .filter(Boolean)
      await apiClient.patchReportPreferences({
        long_term_goals: goalsParsed,
        assistant_tone: tone,
        planning_style: planning,
      })
      await onRefreshPrefs()
      prevGoalsRef.current = goalsText
      prevToneRef.current = tone
      prevPlanningRef.current = planning
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '保存失败')
    } finally {
      onSetPrefsSaving(false)
    }
  }

  const handleSaveDateGoal = async () => {
    if (!selectedDate) return
    onUpdateDateGoal(selectedDate, dateGoalDraft.trim())
  }

  const handleGenerateSoul = async () => {
    setLocalError(null)
    try {
      await onGenerateSoul()
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '生成 soul.md 失败')
    }
  }

  return (
    <div className="daily-report-settings-panel">
      <div className="daily-report-settings-header">
        <span className="daily-report-settings-title">偏好设置</span>
      </div>

      {/* Long-term goals */}
      <div className="daily-report-setting-group">
        <label className="daily-report-setting-label">长期目标</label>
        <textarea
          className="daily-report-textarea"
          rows={4}
          placeholder="每行一个目标，如：&#10;- 完成 VisualMem 前端重构&#10;- 每周运动 3 次"
          value={goalsText}
          onChange={(e) => setGoalsText(e.target.value)}
        />
      </div>

      {/* Date-specific goal */}
      {selectedDate && (
        <div className="daily-report-setting-group">
          <label className="daily-report-setting-label">当日目标（{selectedDate}）</label>
          <div className="daily-report-date-goal-row">
            <input
              className="daily-report-date-goal-input"
              type="text"
              placeholder="今日想要重点完成的事…"
              value={dateGoalDraft}
              onChange={(e) => setDateGoalDraft(e.target.value)}
            />
            <button
              type="button"
              className="daily-report-small-btn"
              onClick={handleSaveDateGoal}
            >
              保存
            </button>
          </div>
        </div>
      )}

      {/* Assistant tone */}
      <div className="daily-report-setting-group">
        <label className="daily-report-setting-label">助手语气</label>
        <div className="daily-report-option-group">
          {TONE_OPTIONS.map((opt) => (
            <label key={opt.value} className="daily-report-option-btn">
              <input
                type="radio"
                name="tone"
                checked={tone === opt.value}
                onChange={() => setTone(opt.value)}
                className="daily-report-radio-hidden"
              />
              {opt.label}
            </label>
          ))}
        </div>
      </div>

      {/* Planning style */}
      <div className="daily-report-setting-group">
        <label className="daily-report-setting-label">规划风格</label>
        <div className="daily-report-option-group">
          {PLANNING_OPTIONS.map((opt) => (
            <label key={opt.value} className="daily-report-option-btn" title={opt.desc}>
              <input
                type="radio"
                name="planning"
                checked={planning === opt.value}
                onChange={() => setPlanning(opt.value)}
                className="daily-report-radio-hidden"
              />
              <span>{opt.label}</span>
              <span className="daily-report-option-desc">{opt.desc}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Save prefs button */}
      <div className="daily-report-setting-actions">
        <button
          type="button"
          className="daily-report-save-btn"
          disabled={!hasChanges || prefsSaving}
          onClick={handleSave}
        >
          {prefsSaving ? '保存中…' : '保存偏好'}
        </button>
      </div>

      {/* Soul.md */}
      <div className="daily-report-setting-group">
        <label className="daily-report-setting-label">soul.md（助手灵魂）</label>
        <p className="daily-report-setting-hint">
          根据长期目标和助手性格，自动生成助手的"灵魂"文档。生成后会在每次日报中引用。
        </p>
        <button
          type="button"
          className="daily-report-small-btn"
          disabled={soulLoading}
          onClick={handleGenerateSoul}
        >
          {soulLoading ? '生成中…' : '生成 / 更新 soul.md'}
        </button>
        {prefs?.soul_md && (
          <div className="daily-report-soul-actions">
            <span className="daily-report-soul-time">
              更新于 {prefs.soul_updated_at ? new Date(prefs.soul_updated_at).toLocaleString('zh-CN') : '未知'}
            </span>
            <button
              type="button"
              className="daily-report-tiny-btn"
              onClick={() => setShowSoul((v: boolean) => !v)}
            >
              {showSoul ? '收起' : '查看'} soul.md
            </button>
          </div>
        )}
        {showSoul && prefs?.soul_md && (
          <pre className="daily-report-soul-content">{prefs.soul_md}</pre>
        )}
      </div>

      {localError && <div className="daily-report-error">{localError}</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Main View                                                          */
/* ------------------------------------------------------------------ */

const DailyReportView: React.FC = () => {
  const [dates, setDates] = useState<string[]>([])
  const [listLoaded, setListLoaded] = useState(false)
  const [selectedDate, setSelectedDate] = useState<string | null>(null)
  const [report, setReport] = useState<DailyReportPayload | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [genLoading, setGenLoading] = useState(false)
  const [genError, setGenError] = useState<string | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  /* Preferences state */
  const [prefs, setPrefs] = useState<ReportPreferences | null>(null)
  const [prefsSaving, setPrefsSaving] = useState(false)
  const [soulLoading, setSoulLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  /* Load preferences */
  useEffect(() => {
    let cancelled = false
    apiClient.getReportPreferences()
      .then((p) => {
        if (!cancelled) setPrefs(p)
      })
      .catch((e) => {
        console.warn('[DailyReportView] load preferences failed', e)
      })
    return () => { cancelled = true }
  }, [])

  /* Refresh preferences after mutations */
  const refreshPrefs = useCallback(async () => {
    try {
      const p = await apiClient.getReportPreferences()
      setPrefs(p)
    } catch (e) {
      console.warn('[DailyReportView] refresh prefs failed', e)
    }
  }, [])

  /* Preferences mutation handlers */
  const handlePrefsSaving = useCallback((v: boolean) => {
    setPrefsSaving(v)
  }, [])

  const handleUpdateDateGoal = useCallback((date: string, goal: string) => {
    apiClient.setDateGoal(date, goal).then(() => {
      refreshPrefs()
    }).catch((e) => {
      console.warn('[DailyReportView] setDateGoal failed', e)
    })
  }, [refreshPrefs])

  const handleGenerateSoul = useCallback(async () => {
    setSoulLoading(true)
    try {
      await apiClient.generateSoul()
      await refreshPrefs()
    } finally {
      setSoulLoading(false)
    }
  }, [refreshPrefs])

  /* ---------- Existing report logic ---------- */

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
    refreshList().catch((e) => {
      if (e instanceof DOMException && e.name === 'AbortError') {
        console.warn('[DailyReportView] listDailyReports aborted (likely during recording stop flush)')
        return
      }
      console.error('listDailyReports', e)
    })
  }, [refreshList])

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
        if (e instanceof DOMException && e.name === 'AbortError') {
          return
        }
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
  const gc = report?.report?.goal_coaching
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
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  className="daily-report-small-btn"
                  onClick={() => setShowSettings((v: boolean) => !v)}
                  title="偏好设置"
                >
                  <SettingsGearGlyph />
                  偏好
                </button>
                <button
                  type="button"
                  className="daily-report-gen-btn"
                  disabled={genLoading || !listLoaded || !selectedDate}
                  onClick={handleGenerate}
                >
                  {genLoading ? '生成中…' : '生成日报'}
                </button>
              </div>
              {genError ? <div className="daily-report-error">{genError}</div> : null}
            </div>
          </div>

          {showSettings && (
            <PreferencesPanel
              prefs={prefs}
              prefsSaving={prefsSaving}
              soulLoading={soulLoading}
              selectedDate={selectedDate}
              onRefreshPrefs={refreshPrefs}
              onSetPrefsSaving={handlePrefsSaving}
              onUpdateDateGoal={handleUpdateDateGoal}
              onGenerateSoul={handleGenerateSoul}
            />
          )}

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

                {/* Goal Coaching section */}
                {gc && (
                  <section style={{ marginTop: 'var(--spacing-lg)' }}>
                    <h2 className="daily-report-section-title">目标教练</h2>
                    <div className="daily-report-goal-coaching">
                      {gc.progress_assessment && (
                        <div className="daily-report-gc-block">
                          <h4>目标进展评估</h4>
                          <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.5, color: 'var(--text-primary)' }}>
                            {gc.progress_assessment}
                          </p>
                        </div>
                      )}
                      {gc.suggestions?.length > 0 && (
                        <div className="daily-report-gc-block">
                          <h4>行动计划</h4>
                          <ul>
                            {gc.suggestions.map((s, i) => (
                              <li key={i}>{s}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {gc.push_message && (
                        <div className="daily-report-gc-push">
                          <strong>教练寄语：</strong>{gc.push_message}
                        </div>
                      )}
                    </div>
                  </section>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DailyReportView
