import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import TimelineView from './pages/TimelineView'
import RealTimeTracing from './pages/RealTimeTracing.tsx'
import SmartTags from './pages/SmartTags'
import Settings from './pages/Settings'
import SearchResults from './components/SearchResults'
import { SearchResult } from './components/SearchBar'
import { AppStoreProvider, useAppStore } from './store/AppStore'

function AppContent() {
  const { currentView, setCurrentView } = useAppStore()
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null)

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
          {searchResult && currentView !== 'realtime' && (
            <div className="global-search-results-wrapper">
              <SearchResults 
                result={searchResult} 
              />
            </div>
          )}
          <div className="view-container">
            {currentView === 'timeline' && (
              <div className="timeline-view-container">
                <TimelineView />
              </div>
            )}
            {currentView === 'realtime' && <RealTimeTracing />}
            {currentView === 'tags' && <SmartTags />}
            {currentView === 'settings' && <Settings />}
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

