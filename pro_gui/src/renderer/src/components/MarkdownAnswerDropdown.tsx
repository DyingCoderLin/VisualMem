import React, { useState } from 'react'
import MarkdownRenderer from './MarkdownRenderer'

interface MarkdownAnswerDropdownProps {
  title: string
  subtitle?: string
  content?: string | null
  error?: string | null
  isLoading?: boolean
}

const MarkdownAnswerDropdown: React.FC<MarkdownAnswerDropdownProps> = ({
  title,
  subtitle,
  content,
  error,
  isLoading = false
}) => {
  const [isMinimized, setIsMinimized] = useState(false)

  if (!isLoading && !content && !error) return null

  return (
    <div className={`markdown-answer-dropdown ${isMinimized ? 'minimized' : ''}`}>
      <div className="markdown-answer-header">
        <div>
          <div className="markdown-answer-title">{title}</div>
          {subtitle && <div className="markdown-answer-subtitle">{subtitle}</div>}
        </div>
        <button
          className="rag-toggle-button-small"
          onClick={() => setIsMinimized((prev) => !prev)}
          title={isMinimized ? 'Expand' : 'Minimize'}
        >
          {isMinimized ? '▲' : '▼'}
        </button>
      </div>
      {!isMinimized && (
        <div className="markdown-answer-body">
          {isLoading ? (
            <div className="markdown-answer-loading">Asking...</div>
          ) : error ? (
            <div className="markdown-answer-error">{error}</div>
          ) : (
            <MarkdownRenderer content={content || ''} />
          )}
        </div>
      )}
    </div>
  )
}

export default MarkdownAnswerDropdown
