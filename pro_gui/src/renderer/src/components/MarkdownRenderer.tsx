import React from 'react'

interface MarkdownRendererProps {
  content: string
}

/**
 * 简单的 Markdown 渲染器
 * 支持：加粗、斜体、代码块、行内代码、列表、标题
 */
const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  if (!content) return null

  // 分割文本为段落
  const paragraphs = content.split(/\n\n+/).filter(p => p.trim())

  const renderParagraph = (text: string, index: number) => {
    // 处理代码块
    if (text.trim().startsWith('```')) {
      const codeMatch = text.match(/```(\w+)?\n([\s\S]*?)```/)
      if (codeMatch) {
        const language = codeMatch[1] || ''
        const code = codeMatch[2]
        return (
          <pre key={index} className="markdown-code-block">
            {language && <span className="markdown-code-lang">{language}</span>}
            <code>{code}</code>
          </pre>
        )
      }
    }

    // 处理标题
    const headingMatch = text.match(/^(#{1,6})\s+(.+)$/)
    if (headingMatch) {
      const level = headingMatch[1].length
      const headingText = headingMatch[2]
      const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements
      return (
        <HeadingTag key={index} className={`markdown-heading markdown-h${level}`}>
          {renderInlineMarkdown(headingText)}
        </HeadingTag>
      )
    }

    // 处理列表
    if (text.trim().match(/^[\*\-\+]\s+/m) || text.trim().match(/^\d+\.\s+/m)) {
      const lines = text.split('\n')
      const listItems: string[] = []
      let currentItem = ''

      lines.forEach(line => {
        const listMatch = line.match(/^([\*\-\+]|\d+\.)\s+(.+)$/)
        if (listMatch) {
          if (currentItem) {
            listItems.push(currentItem)
          }
          currentItem = listMatch[2]
        } else if (line.trim()) {
          currentItem += ' ' + line.trim()
        }
      })
      if (currentItem) {
        listItems.push(currentItem)
      }

      if (listItems.length > 0) {
        return (
          <ul key={index} className="markdown-list">
            {listItems.map((item, i) => (
              <li key={i}>{renderInlineMarkdown(item)}</li>
            ))}
          </ul>
        )
      }
    }

    // 普通段落
    return (
      <p key={index} className="markdown-paragraph">
        {renderInlineMarkdown(text)}
      </p>
    )
  }

  // 渲染行内 Markdown（加粗、斜体、代码、链接）
  const renderInlineMarkdown = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = []
    let remaining = text
    let key = 0

    // 匹配模式：代码、加粗、斜体、链接
    const patterns = [
      { regex: /`([^`]+)`/g, render: (_match: string, code: string) => (
        <code key={key++} className="markdown-inline-code">{code}</code>
      )},
      { regex: /\*\*([^*]+)\*\*/g, render: (_match: string, bold: string) => (
        <strong key={key++} className="markdown-bold">{bold}</strong>
      )},
      { regex: /__([^_]+)__/g, render: (_match: string, bold: string) => (
        <strong key={key++} className="markdown-bold">{bold}</strong>
      )},
      { regex: /\*([^*]+)\*/g, render: (_match: string, italic: string) => (
        <em key={key++} className="markdown-italic">{italic}</em>
      )},
      { regex: /_([^_]+)_/g, render: (_match: string, italic: string) => (
        <em key={key++} className="markdown-italic">{italic}</em>
      )},
      { regex: /\[([^\]]+)\]\(([^)]+)\)/g, render: (_match: string, text: string, url: string) => (
        <a key={key++} href={url} target="_blank" rel="noopener noreferrer" className="markdown-link">
          {text}
        </a>
      )},
    ]

    let lastIndex = 0
    const matches: Array<{ index: number; length: number; render: () => React.ReactNode }> = []

    // 收集所有匹配
    patterns.forEach(({ regex, render }) => {
      let match
      regex.lastIndex = 0
      while ((match = regex.exec(remaining)) !== null) {
        // 立即保存匹配的值，避免闭包问题
        const matchValue = match[0]
        const matchGroup1 = match[1]
        const matchGroup2 = match[2]
        const matchIndex = match.index
        const matchLength = match[0].length
        
        matches.push({
          index: matchIndex,
          length: matchLength,
          render: () => render(matchValue, matchGroup1, matchGroup2)
        })
      }
    })

    // 按位置排序
    matches.sort((a, b) => a.index - b.index)

    // 处理重叠：优先处理更短的匹配（代码优先）
    const processed: Array<{ start: number; end: number; render: () => React.ReactNode }> = []
    matches.forEach(match => {
      // 检查是否与已处理的匹配重叠
      const overlaps = processed.some(p => 
        (match.index < p.end && match.index + match.length > p.start)
      )
      if (!overlaps) {
        processed.push({
          start: match.index,
          end: match.index + match.length,
          render: match.render
        })
      }
    })

    // 按位置排序
    processed.sort((a, b) => a.start - b.start)

    // 渲染
    processed.forEach(({ start, end, render }) => {
      // 添加之前的文本
      if (start > lastIndex) {
        parts.push(remaining.slice(lastIndex, start))
      }
      // 添加匹配的内容
      parts.push(render())
      lastIndex = end
    })

    // 添加剩余的文本
    if (lastIndex < remaining.length) {
      parts.push(remaining.slice(lastIndex))
    }

    return parts.length > 0 ? parts : [text]
  }

  return (
    <div className="markdown-content">
      {paragraphs.map((para, index) => renderParagraph(para, index))}
    </div>
  )
}

export default MarkdownRenderer

