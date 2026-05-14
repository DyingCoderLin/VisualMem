import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import TimelineView from './pages/TimelineView'
import RealTimeTracing from './pages/RealTimeTracing.tsx'
import SmartTags from './pages/SmartTags'
import Settings from './pages/Settings'
import DailyReportView from './pages/DailyReportView'
import RewindView from './pages/RewindView'
import SearchResults from './components/SearchResults'
import MarkdownAnswerDropdown from './components/MarkdownAnswerDropdown'
import { SearchResult } from './components/SearchBar'
import { AppStoreProvider, useAppStore } from './store/AppStore'

function AppContent() {
  const {
    currentView,
    setCurrentView,
    rewindAskContext,
    rewindAskResult,
    rewindAskError,
    isRewindAsking
  } = useAppStore()
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null)
  const hasRewindAskOutput = Boolean(isRewindAsking || rewindAskResult || rewindAskError)

  const handleClearSearch = () => {
    setSearchResult(null)
  }

  return (
    <div className="app-container">
      <Sidebar currentView={currentView} onViewChange={setCurrentView} />
      <div className="main-content">
        <TopBar onSearchResult={setSearchResult} />
        <div className="content-area">
          {/* 全局搜索结果 - 仅在非实时追踪页面显示 */}
          {searchResult && (
            <div
              className="global-search-results-wrapper"
              style={{ display: currentView === 'timeline' ? undefined : 'none' }}
            >
              <SearchResults 
                result={searchResult} 
              />
            </div>
          )}
          {hasRewindAskOutput && (
            <div
              className="global-search-results-wrapper rewind-ask-results-wrapper"
              style={{ display: currentView === 'rewind' ? undefined : 'none' }}
            >
              <MarkdownAnswerDropdown
                title="Ask with this memory"
                subtitle={rewindAskContext?.title}
                content={rewindAskResult?.answer}
                error={rewindAskError}
                isLoading={isRewindAsking}
              />
            </div>
          )}
          <div className="view-container">
            {/* Keep heavy pages always mounted to preserve state and browser image cache */}
            <div className="timeline-view-container" style={{ display: currentView === 'timeline' ? undefined : 'none' }}>
              <TimelineView />
            </div>
            <div style={{ display: currentView === 'realtime' ? 'contents' : 'none' }}>
              <RealTimeTracing />
            </div>
            <div style={{ display: currentView === 'rewind' ? 'contents' : 'none' }}>
              <RewindView />
            </div>
            <div style={{ display: currentView === 'tags' ? 'contents' : 'none' }}>
              <SmartTags />
            </div>
            <div style={{ display: currentView === 'settings' ? 'contents' : 'none' }}>
              <Settings />
            </div>
            <div
              className="daily-report-view-container"
              style={{ display: currentView === 'daily' ? undefined : 'none' }}
            >
              <DailyReportView />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <AppStoreProvider>
      <AppContent />
    </AppStoreProvider>
  )
}

export default App
