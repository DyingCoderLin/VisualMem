/** Mirrors `REPORT_LOG_DIR/daily_report_YYYY-MM-DD.json` from the reporting pipeline. */

export interface AppUsageRow {
  app: string
  focused_minutes: number
  percentage: number
  frame_count?: number
  purpose_keywords?: string[]
}

export interface WorkModule {
  core_accomplishments: string[]
  supporting_research: string[]
  blockers_and_unfinished: string[]
  tomorrow_suggestions: string[]
}

export interface LifeModule {
  focus_score: number
  focus_interpretation: string
  deep_work_blocks?: unknown[]
  fragmentation_diagnosis?: string
  distraction_patterns?: string
  intervention_suggestions?: string[]
}

export interface GoalCoaching {
  progress_assessment: string
  suggestions: string[]
  push_message: string
}

export interface ReportBody {
  date: string
  work_module: WorkModule
  life_module: LifeModule
  today_summary: string[]
  goal_coaching?: GoalCoaching
  raw_markdown?: string
}

export interface PipelineStats {
  data_fetch_ms?: number
  total_pipeline_ms?: number
  total_input_tokens?: number
  total_output_tokens?: number
  [key: string]: unknown
}

export interface DailyReportPayload {
  status?: string
  message?: string
  app_usage_summary: AppUsageRow[]
  metrics_semantics: string
  report: ReportBody
  pipeline_stats?: PipelineStats
}

export interface DailyReportListResponse {
  dates: string[]
}

export interface GenerateDailyReportResponse extends DailyReportPayload {
  ok: boolean
  date: string
}

/* ------------------------------------------------------------------ */
/* Report preferences                                                 */
/* ------------------------------------------------------------------ */

export type AssistantTone = 'soft' | 'normal' | 'push'
export type PlanningStyle = 'detailed-present' | 'rough-overall'

export interface ReportPreferences {
  long_term_goals: string[]
  date_goals: Record<string, string>
  assistant_tone: AssistantTone
  planning_style: PlanningStyle
  soul_updated_at: string | null
  soul_md?: string
}

export interface PreferencesPatch {
  long_term_goals?: string[]
  date_goals?: Record<string, string>
  assistant_tone?: AssistantTone
  planning_style?: PlanningStyle
}

export interface GenerateSoulResponse {
  ok: boolean
  content: string
  updated_at: string
}
