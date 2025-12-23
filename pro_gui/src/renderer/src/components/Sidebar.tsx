import React from 'react'
import SystemStatus from './SystemStatus'

type ViewType = 'timeline' | 'realtime' | 'tags' | 'settings'

interface SidebarProps {
  currentView: ViewType
  onViewChange: (view: ViewType) => void
}

const Sidebar: React.FC<SidebarProps> = ({ currentView, onViewChange }) => {
  return (
    <div className="sidebar">
      <div className="sidebar-title">VisualMem</div>
      
      <ul className="nav-list">
        <li>
          <div
            className={`nav-item ${currentView === 'timeline' ? 'active' : ''}`}
            onClick={() => onViewChange('timeline')}
          >
            Timeline View
          </div>
        </li>
        <li>
          <div
            className={`nav-item ${currentView === 'realtime' ? 'active' : ''}`}
            onClick={() => onViewChange('realtime')}
          >
            Real-time Tracing
          </div>
        </li>
        {/* <li>
          <div
            className={`nav-item ${currentView === 'tags' ? 'active' : ''}`}
            onClick={() => onViewChange('tags')}
          >
            Smart Tags
          </div>
        </li>
        <li>
          <div
            className={`nav-item ${currentView === 'settings' ? 'active' : ''}`}
            onClick={() => onViewChange('settings')}
          >
            设置
          </div>
        </li> */}
      </ul>

      <SystemStatus />
    </div>
  )
}

export default Sidebar

