# VisualMem Professional - Electron Frontend

这是 VisualMem 的进阶版前端，使用 Electron + React + TypeScript 构建，提供更精美、更全面的用户界面。

## 技术栈

- **Electron**: 跨平台桌面应用框架
- **React**: UI 框架
- **TypeScript**: 类型安全
- **Vite**: 构建工具
- **@tanstack/react-virtual**: 虚拟滚动（用于高性能列表渲染）

## 项目结构

```
pro_gui/
├── src/
│   ├── main/           # Electron 主进程
│   │   └── index.ts    # 窗口管理和 Python 后端启动
│   ├── preload/        # 预加载脚本
│   │   └── index.ts    # 安全地暴露 Node API
│   └── renderer/       # 前端 UI (React)
│       ├── src/
│       │   ├── components/  # UI 组件
│       │   ├── pages/       # 页面组件
│       │   ├── services/    # API 客户端
│       │   └── styles/      # CSS 样式
│       └── index.html       # HTML 入口
├── package.json
└── tsconfig.json
```

## 安装依赖

```bash
cd pro_gui
npm install
```

## 开发模式

```bash
npm run dev
```

这将启动：
1. Electron 应用窗口
2. Vite 开发服务器（前端热重载）
3. 自动连接到 Python 后端（http://localhost:8080）

## 构建

```bash
npm run build
```

## 功能特性

### 1. AI 搜索结果展示
- 搜索结果左侧显示相关图片列表（可水平滚动）
- 右侧显示 AI 回答
- 图片按时间顺序排列，显示时间戳和相关度

### 2. 时间轴视图
- 按日期浏览历史截图
- 虚拟滚动，支持大量图片的高性能渲染
- 懒加载，随滚动动态加载图片

### 3. 系统状态面板
- 显示总帧数、磁盘使用量等信息
- 与现有系统状态风格保持一致

## 后端 API

前端通过 HTTP API 与 Python 后端通信：

- `GET /api/stats` - 获取系统统计信息
- `POST /api/query_rag_with_time` - RAG 查询（带时间范围）
- `GET /api/frames` - 获取时间范围内的帧列表
- `GET /api/image` - 获取图片文件

## 设计系统

UI 采用深色主题 + 黄色强调色的设计系统：

- 背景色：`#121212` (app), `#1A1A1A` (panel)
- 强调色：`#FFD700` (黄色)
- 文本色：`#FFFFFF` (主要), `#A0A0A0` (次要)

详细样式定义见 `src/renderer/src/styles/theme.css`

