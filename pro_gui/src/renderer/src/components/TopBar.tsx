import React, { useState } from 'react'
import SearchBar, { SearchResult } from './SearchBar'

interface TopBarProps {
  onSearchResult: (result: SearchResult | null) => void
}

const TopBar: React.FC<TopBarProps> = ({ onSearchResult }) => {
  return (
    <div className="top-bar">
      <SearchBar onSearchResult={onSearchResult} />
    </div>
  )
}

export default TopBar

