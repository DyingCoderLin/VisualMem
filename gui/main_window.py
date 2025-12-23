#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisualMem GUI - 现代化 Spotlight 风格界面
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, QDateTime, QPropertyAnimation, QEasingCurve,
    QSize, Signal, QTimer, QPoint, QUrl
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextBrowser,
    QMessageBox, QMenu, QFrame, QGraphicsDropShadowEffect,
    QScrollArea, QSizePolicy, QApplication, QDateTimeEdit,
    QCalendarWidget
)
from PySide6.QtGui import (
    QFont, QTextCursor, QColor, QKeySequence, QShortcut,
    QPainter, QPainterPath, QBrush, QPen, QIcon, QPixmap
)

from config import config
from utils.logger import setup_logger
from core.storage.sqlite_storage import SQLiteStorage

from .workers import RecordWorker, QueryWorker

logger = setup_logger("main_window")


# ============================================
# 颜色主题
# ============================================
class Theme:
    # 背景
    BG_DARK = "#1a1a1e"
    BG_CARD = "#252529"
    BG_INPUT = "#2a2a2e"
    
    # 强调色
    ACCENT = "#f5c518"  # 黄色
    ACCENT_HOVER = "#ffd54f"
    
    # 文字
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#888888"
    TEXT_MUTED = "#666666"
    
    # 边框
    BORDER = "#3a3a3e"
    BORDER_FOCUS = "#f5c518"
    
    # 状态
    SUCCESS = "#4caf50"
    ERROR = "#f44336"
    WARNING = "#ff9800"


# ============================================
# 自定义控件
# ============================================
class RecordButton(QPushButton):
    """录制状态按钮 - 播放/停止风格，带点击特效"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self._is_pressed = False
        self._scale = 1.0
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()
        
        # 点击动画
        self._animation = QPropertyAnimation(self, b"scale")
        self._animation.setDuration(100)
        self._animation.setEasingCurve(QEasingCurve.OutQuad)
    
    def get_scale(self):
        return self._scale
    
    def set_scale(self, value):
        self._scale = value
        self.update()
    
    scale = property(get_scale, set_scale)
    
    def set_recording(self, recording: bool):
        self.is_recording = recording
        self._update_style()
        self.update()
    
    def _update_style(self):
        if self.is_recording:
            self.setToolTip("录制中 - 点击停止")
        else:
            self.setToolTip("点击开始录制")
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
            }}
        """)
    
    def mousePressEvent(self, event):
        self._is_pressed = True
        # 按下时缩小
        self._animation.stop()
        self._animation.setStartValue(self._scale)
        self._animation.setEndValue(0.85)
        self._animation.start()
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        self._is_pressed = False
        # 释放时恢复
        self._animation.stop()
        self._animation.setStartValue(self._scale)
        self._animation.setEndValue(1.0)
        self._animation.start()
        super().mouseReleaseEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 应用缩放
        center = self.rect().center()
        painter.translate(center)
        painter.scale(self._scale, self._scale)
        painter.translate(-center)
        
        # hover 背景
        if self.underMouse() and not self._is_pressed:
            painter.setBrush(QBrush(QColor(245, 197, 24, 40)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2, 2, 36, 36)
        
        # 绘制背景圆
        painter.setPen(QPen(QColor(Theme.ACCENT), 2))
        painter.setBrush(QBrush(Qt.transparent))
        painter.drawEllipse(4, 4, 32, 32)
        
        if self.is_recording:
            # 录制中 - 绘制方形（停止）
            painter.setBrush(QBrush(QColor(Theme.ACCENT)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(12, 12, 16, 16, 2, 2)
        else:
            # 未录制 - 绘制三角形（播放）
            painter.setBrush(QBrush(QColor(Theme.ACCENT)))
            painter.setPen(Qt.NoPen)
            path = QPainterPath()
            path.moveTo(14, 10)
            path.lineTo(14, 30)
            path.lineTo(30, 20)
            path.closeSubpath()
            painter.drawPath(path)


class ModeButton(QPushButton):
    """模式选择按钮 - RAG 和实时两种模式"""
    
    mode_changed = Signal(int)
    
    MODES = ["RAG", "实时"]
    MODES_FULL = ["RAG: 时间范围检索", "实时: 当前屏幕问答"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = 0
        self.setText(self.MODES[0])
        self.setFixedSize(60, 32)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.BG_INPUT};
                color: {Theme.ACCENT};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_CARD};
                border-color: {Theme.ACCENT};
            }}
            QPushButton::menu-indicator {{
                width: 0px;
            }}
        """)
        
        # 创建下拉菜单
        self.menu = QMenu(self)
        self.menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Theme.BG_CARD};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: 4px;
            }}
            QMenu::item {{
                color: {Theme.TEXT_PRIMARY};
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {Theme.ACCENT};
                color: {Theme.BG_DARK};
            }}
        """)
        
        for i, mode in enumerate(self.MODES_FULL):
            action = self.menu.addAction(mode)
            action.triggered.connect(lambda checked, idx=i: self._set_mode(idx))
        
        self.setMenu(self.menu)
    
    def _set_mode(self, index: int):
        self.current_mode = index
        self.setText(self.MODES[index])
        self.mode_changed.emit(index)


class SearchInput(QLineEdit):
    """搜索输入框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Ask VisualMem...")
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                border: none;
                color: {Theme.TEXT_PRIMARY};
                font-size: 16px;
                padding: 8px;
            }}
            QLineEdit::placeholder {{
                color: {Theme.TEXT_MUTED};
            }}
        """)


class OCRToggleButton(QPushButton):
    """OCR 模式切换按钮"""
    
    toggled_signal = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__("OCR", parent)
        self._active = False
        self.setFixedSize(50, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self._update_style()
        self.clicked.connect(self._on_clicked)
    
    def _on_clicked(self):
        self._active = self.isChecked()
        self._update_style()
        self.toggled_signal.emit(self._active)
    
    def is_active(self) -> bool:
        return self._active
    
    def _update_style(self):
        if self._active:
            # 激活状态 - 黄色高亮
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Theme.ACCENT};
                    color: {Theme.BG_DARK};
                    border: none;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {Theme.ACCENT_HOVER};
                }}
            """)
            self.setToolTip("OCR模式已开启 - 使用纯文本检索")
        else:
            # 未激活状态
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Theme.BG_INPUT};
                    color: {Theme.TEXT_MUTED};
                    border: 1px solid {Theme.BORDER};
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {Theme.BG_CARD};
                    color: {Theme.TEXT_PRIMARY};
                    border-color: {Theme.ACCENT};
                }}
            """)
            self.setToolTip("点击开启OCR模式 - 使用纯文本检索")


class StatusPanel(QFrame):
    """系统状态面板 - 简化版两行布局"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_CARD};
                border: none;
                border-radius: 12px;
            }}
        """)
        self.setMinimumHeight(120)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # 标题
        title = QLabel("SYSTEM STATUS")
        title.setStyleSheet(f"""
            color: {Theme.ACCENT};
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 2px;
        """)
        title.setFixedHeight(20)
        layout.addWidget(title)
        
        # 第一行：FRAMES STORED + DISK USAGE（左对齐）
        row1 = QHBoxLayout()
        row1.setSpacing(80)
        
        # 帧数
        frames_col = QVBoxLayout()
        frames_col.setSpacing(4)
        frames_label = QLabel("FRAMES STORED")
        frames_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        frames_label.setFixedHeight(16)
        self.frames_value = QLabel("0")
        self.frames_value.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 16px; font-weight: bold;")
        self.frames_value.setFixedHeight(22)
        frames_col.addWidget(frames_label)
        frames_col.addWidget(self.frames_value)
        row1.addLayout(frames_col)
        
        # 磁盘使用
        disk_col = QVBoxLayout()
        disk_col.setSpacing(4)
        disk_label = QLabel("DISK USAGE")
        disk_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        disk_label.setFixedHeight(16)
        self.disk_value = QLabel("—")
        self.disk_value.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-size: 13px;")
        self.disk_value.setFixedHeight(22)
        disk_col.addWidget(disk_label)
        disk_col.addWidget(self.disk_value)
        row1.addLayout(disk_col)
        
        row1.addStretch()  # stretch 放最后，让左边的元素左对齐
        layout.addLayout(row1)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {Theme.BORDER};")
        line.setFixedHeight(1)
        layout.addWidget(line)
        
        # 第二行：STORAGE + VLM MODEL（左对齐）
        row2 = QHBoxLayout()
        row2.setSpacing(80)
        
        # 存储
        storage_col = QVBoxLayout()
        storage_col.setSpacing(4)
        storage_label = QLabel("STORAGE")
        storage_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        storage_label.setFixedHeight(16)
        self.storage_value = QLabel("Local SQLite")
        self.storage_value.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-size: 13px;")
        self.storage_value.setFixedHeight(20)
        self.storage_value.setMinimumWidth(100)
        storage_col.addWidget(storage_label)
        storage_col.addWidget(self.storage_value)
        row2.addLayout(storage_col)
        
        # VLM 模型
        vlm_col = QVBoxLayout()
        vlm_col.setSpacing(4)
        vlm_label = QLabel("VLM MODEL")
        vlm_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; font-weight: bold;")
        vlm_label.setFixedHeight(16)
        self.vlm_value = QLabel("—")
        self.vlm_value.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-size: 13px;")
        self.vlm_value.setFixedHeight(20)
        vlm_col.addWidget(vlm_label)
        vlm_col.addWidget(self.vlm_value)
        row2.addLayout(vlm_col)
        
        row2.addStretch()  # stretch 放最后，让左边的元素左对齐
        
        layout.addLayout(row2)
    
    def update_status(self, frames: int, storage: str, vlm: str, disk: str = "—"):
        self.frames_value.setText(f"{frames:,}")
        self.storage_value.setText(storage)
        self.vlm_value.setText(vlm)
        self.disk_value.setText(disk)


class DiskUsageCalculator:
    """磁盘使用量计算器 - 统计整个 STORAGE_ROOT 目录的实际大小"""
    
    def __init__(self, storage_root: str = None):
        self.storage_root = Path(storage_root or config.STORAGE_ROOT)
        self._last_calculation = 0
        self._last_calculation_time = 0
    
    def _get_directory_size(self, directory: Path) -> int:
        """
        递归计算目录的总大小（字节）
        
        Args:
            directory: 目录路径
            
        Returns:
            目录总大小（字节）
        """
        total_size = 0
        try:
            if directory.exists() and directory.is_dir():
                for entry in directory.rglob('*'):
                    try:
                        if entry.is_file():
                            total_size += entry.stat().st_size
                    except (OSError, PermissionError):
                        # 忽略无法访问的文件
                        pass
        except (OSError, PermissionError):
            pass
        return total_size
    
    def calculate_initial(self) -> int:
        """启动时计算初始磁盘使用量（实际统计整个 STORAGE_ROOT 目录）"""
        try:
            total_bytes = self._get_directory_size(self.storage_root)
            self._last_calculation = total_bytes
            self._last_calculation_time = time.time()
            
            logger.info(f"磁盘使用量初始计算: {self.format_size(total_bytes)} ({self.storage_root})")
            
            return total_bytes
            
        except Exception as e:
            logger.error(f"计算磁盘使用量失败: {e}")
            return 0
    
    def add_frames(self, count: int = 1) -> int:
        """
        增量更新：标记需要重新计算（不实际计算，下次 get_usage 时再计算）
        为了提高性能，实际统计有缓存机制
        """
        # 不进行增量估算，而是标记需要重新计算
        # 实际计算会在 get_usage 中进行（带缓存）
        return self.get_usage()
    
    def get_usage(self) -> int:
        """
        获取当前磁盘使用量
        
        为了提高性能，使用缓存机制：
        - 如果距离上次计算不足 30 秒，返回缓存值
        - 否则重新计算
        """
        current_time = time.time()
        
        # 如果距离上次计算不足 30 秒，返回缓存值（避免频繁计算导致性能问题）
        if self._last_calculation > 0 and (current_time - self._last_calculation_time) < 30:
            return self._last_calculation
        
        # 重新计算
        try:
            total_bytes = self._get_directory_size(self.storage_root)
            self._last_calculation = total_bytes
            self._last_calculation_time = current_time
            return total_bytes
        except Exception as e:
            logger.error(f"计算磁盘使用量失败: {e}")
            return self._last_calculation if self._last_calculation > 0 else 0
    
    def get_formatted_usage(self) -> str:
        """获取格式化的磁盘使用量"""
        return self.format_size(self.get_usage())
    
    @staticmethod
    def format_size(bytes_size: int) -> str:
        """格式化字节大小"""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"


class ResultPanel(QFrame):
    """结果展示面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_CARD};
                border: 1px solid {Theme.BORDER};
                border-radius: 12px;
            }}
        """)
        self._init_ui()
        self._progress_value = 0
        self._progress_timer = None
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 进度条容器
        progress_container = QWidget()
        progress_container.setFixedHeight(6)
        progress_container.setStyleSheet(f"""
            background-color: {Theme.BG_INPUT};
            border-radius: 3px;
        """)
        progress_layout = QHBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QFrame()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(f"""
            background-color: {Theme.ACCENT};
            border-radius: 3px;
        """)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch()
        
        self.progress_container = progress_container
        self.progress_container.hide()
        layout.addWidget(self.progress_container)
        
        # 标题
        self.title_label = QLabel("VLM Analysis")
        self.title_label.setStyleSheet(f"""
            color: {Theme.TEXT_PRIMARY};
            font-size: 14px;
            font-weight: bold;
        """)
        layout.addWidget(self.title_label)
        
        # 结果文本（使用 QTextBrowser 支持滚动）
        self.result_text = QTextBrowser()
        self.result_text.setOpenExternalLinks(True)
        self.result_text.setStyleSheet(f"""
            QTextBrowser {{
                background-color: transparent;
                border: none;
                color: {Theme.TEXT_SECONDARY};
                font-size: 13px;
            }}
        """)
        self.result_text.setMinimumHeight(150)
        self.result_text.setMaximumHeight(400)
        layout.addWidget(self.result_text)
        
        # 缩略图区域
        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QHBoxLayout(self.thumbnails_widget)
        self.thumbnails_layout.setContentsMargins(0, 8, 0, 0)
        self.thumbnails_layout.setSpacing(8)
        self.thumbnails_widget.hide()
        layout.addWidget(self.thumbnails_widget)
    
    def show_progress(self, show: bool = True, animate: bool = True):
        self.progress_container.setVisible(show)
        if show and animate:
            self._start_progress_animation()
        else:
            self._stop_progress_animation()
    
    def _start_progress_animation(self):
        """启动进度动画"""
        self._progress_value = 0
        self.progress_bar.setFixedWidth(50)
        
        if self._progress_timer is None:
            self._progress_timer = QTimer(self)
            self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_timer.start(50)
    
    def _stop_progress_animation(self):
        """停止进度动画"""
        if self._progress_timer:
            self._progress_timer.stop()
    
    def _animate_progress(self):
        """进度条动画"""
        max_width = self.progress_container.width() - 4
        self._progress_value = (self._progress_value + 3) % max_width
        # 来回移动效果
        if self._progress_value < max_width // 2:
            width = 50 + self._progress_value
        else:
            width = 50 + max_width - self._progress_value
        self.progress_bar.setFixedWidth(min(width, max_width))
    
    def set_result(self, title: str, text: str):
        self.title_label.setText(title)
        self.result_text.setText(text)
        self.show_progress(False)
    
    def clear(self):
        self.title_label.setText("VLM Analysis")
        self.result_text.setText("")
        self.show_progress(False)
    
    def add_thumbnails(self, image_paths: list):
        """添加缩略图"""
        # 清除现有缩略图
        while self.thumbnails_layout.count():
            item = self.thumbnails_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for path in image_paths[:5]:  # 最多显示5张
            try:
                thumb = QLabel()
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    thumb.setPixmap(pixmap.scaledToHeight(80, Qt.SmoothTransformation))
                    thumb.setStyleSheet(f"""
                        border: 1px solid {Theme.BORDER};
                        border-radius: 4px;
                    """)
                    self.thumbnails_layout.addWidget(thumb)
            except Exception:
                pass
        
        self.thumbnails_layout.addStretch()
        self.thumbnails_widget.setVisible(len(image_paths) > 0)


class TimeInputWidget(QWidget):
    """时间输入控件（用于模式2）- 使用日期时间选择器"""
    
    DATETIME_STYLE = f"""
        QDateTimeEdit {{
            background-color: {Theme.BG_INPUT};
            border: 1px solid {Theme.BORDER};
            border-radius: 8px;
            color: {Theme.TEXT_PRIMARY};
            padding: 8px 12px;
            font-size: 13px;
            min-width: 160px;
        }}
        QDateTimeEdit:focus {{
            border-color: {Theme.ACCENT};
        }}
        QDateTimeEdit::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 20px;
            border: none;
        }}
        QDateTimeEdit::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {Theme.TEXT_MUTED};
        }}
        QDateTimeEdit QAbstractItemView {{
            background-color: {Theme.BG_CARD};
            color: {Theme.TEXT_PRIMARY};
            selection-background-color: {Theme.ACCENT};
            selection-color: {Theme.BG_DARK};
        }}
        QCalendarWidget {{
            background-color: {Theme.BG_CARD};
        }}
        QCalendarWidget QToolButton {{
            color: {Theme.TEXT_PRIMARY};
            background-color: {Theme.BG_INPUT};
            border: none;
            border-radius: 4px;
            padding: 4px;
        }}
        QCalendarWidget QToolButton:hover {{
            background-color: {Theme.ACCENT};
            color: {Theme.BG_DARK};
        }}
        QCalendarWidget QWidget#qt_calendar_navigationbar {{
            background-color: {Theme.BG_CARD};
        }}
        QCalendarWidget QAbstractItemView:enabled {{
            color: {Theme.TEXT_PRIMARY};
            background-color: {Theme.BG_CARD};
            selection-background-color: {Theme.ACCENT};
            selection-color: {Theme.BG_DARK};
        }}
        QCalendarWidget QAbstractItemView:disabled {{
            color: {Theme.TEXT_MUTED};
        }}
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)
        
        # 开始时间
        self.start_label = QLabel("从")
        self.start_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 14px;")
        layout.addWidget(self.start_label)
        
        self.start_input = QDateTimeEdit()
        self.start_input.setCalendarPopup(True)
        self.start_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_input.setStyleSheet(self.DATETIME_STYLE)
        layout.addWidget(self.start_input)
        
        # 结束时间
        self.end_label = QLabel("到")
        self.end_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 14px;")
        layout.addWidget(self.end_label)
        
        self.end_input = QDateTimeEdit()
        self.end_input.setCalendarPopup(True)
        self.end_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.end_input.setStyleSheet(self.DATETIME_STYLE)
        layout.addWidget(self.end_input)
        
        layout.addStretch()
        
        # 设置默认时间
        now = QDateTime.currentDateTime()
        self.end_input.setDateTime(now)
        self.start_input.setDateTime(now.addSecs(-24 * 3600))  # 24小时前
    
    def get_start_datetime(self) -> datetime:
        """获取开始时间"""
        return self.start_input.dateTime().toPython()
    
    def get_end_datetime(self) -> datetime:
        """获取结束时间"""
        return self.end_input.dateTime().toPython()


# ============================================
# 主窗口
# ============================================
class VisualMemMainWindow(QMainWindow):
    """VisualMem 主窗口 - Spotlight 风格"""
    
    def __init__(self):
        super().__init__()
        
        # 无边框窗口
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 状态
        self.is_recording = False
        self.current_mode = 0
        self.ocr_mode = False  # OCR 纯文本模式
        self.status_panel_visible = False
        self.result_panel_visible = False
        
        # 应用实例引用（用于调用退出功能）
        self.app_instance = None
        
        # 工作线程
        self.record_thread: Optional[QThread] = None
        self.record_worker: Optional[RecordWorker] = None
        self.query_thread: Optional[QThread] = None
        self.query_worker: Optional[QueryWorker] = None
        
        # 拖动
        self._drag_pos = None
        
        # 磁盘使用量计算器（统计整个 STORAGE_ROOT 目录）
        self.disk_calculator = DiskUsageCalculator(config.STORAGE_ROOT)
        
        self._init_ui()
        self._setup_shortcuts()
        self._load_initial_stats()
        
        # 居中显示
        self._center_on_screen()
    
    def _init_ui(self):
        """初始化 UI"""
        # 主容器
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)
        
        # 搜索栏
        self.search_bar = self._create_search_bar()
        main_layout.addWidget(self.search_bar)
        
        # 时间输入（RAG 模式下显示，默认显示）
        self.time_input = TimeInputWidget()
        main_layout.addWidget(self.time_input)
        
        # 状态面板（默认隐藏）
        self.status_panel = StatusPanel()
        self.status_panel.hide()
        main_layout.addWidget(self.status_panel)
        
        # 结果面板（默认隐藏）
        self.result_panel = ResultPanel()
        self.result_panel.hide()
        main_layout.addWidget(self.result_panel)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 10)
        self.central_widget.setGraphicsEffect(shadow)
        
        # 设置初始大小
        self._update_window_size()
    
    def _create_search_bar(self) -> QFrame:
        """创建搜索栏"""
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_CARD};
                border: 1px solid {Theme.BORDER};
                border-radius: 20px;
            }}
        """)
        bar.setFixedHeight(56)
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 16, 8)
        layout.setSpacing(14)
        
        # 录制按钮
        self.record_btn = RecordButton()
        self.record_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_btn)
        
        # 设置按钮
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(36, 36)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Theme.TEXT_MUTED};
                border: none;
                font-size: 22px;
            }}
            QPushButton:hover {{
                color: {Theme.TEXT_PRIMARY};
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 18px;
            }}
        """)
        self.settings_btn.clicked.connect(self._toggle_status_panel)
        layout.addWidget(self.settings_btn)
        
        # 模式选择
        self.mode_btn = ModeButton()
        self.mode_btn.mode_changed.connect(self._on_mode_changed)
        layout.addWidget(self.mode_btn)
        
        # 搜索输入
        self.search_input = SearchInput()
        self.search_input.returnPressed.connect(self._on_submit)
        layout.addWidget(self.search_input, 1)
        
        # OCR 模式切换按钮
        self.ocr_toggle = OCRToggleButton()
        self.ocr_toggle.toggled_signal.connect(self._on_ocr_mode_changed)
        layout.addWidget(self.ocr_toggle)
        
        # 关闭按钮（X）
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Theme.TEXT_MUTED};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: #ff4444;
                background-color: rgba(255, 68, 68, 0.15);
                border-radius: 14px;
            }}
        """)
        self.close_btn.clicked.connect(self._quit_app)
        layout.addWidget(self.close_btn)
        
        return bar
    
    def _setup_shortcuts(self):
        """设置快捷键"""
        # ESC 关闭/隐藏
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._on_escape)
        
        # Cmd+Enter 提交
        QShortcut(QKeySequence("Ctrl+Return"), self, self._on_submit)
        
        # Cmd+W 显示/隐藏窗口
        QShortcut(QKeySequence("Ctrl+W"), self, self._toggle_visibility)
    
    def _toggle_visibility(self):
        """切换窗口显示/隐藏"""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self.search_input.setFocus()
    
    def _center_on_screen(self):
        """居中显示"""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = screen.height() // 4  # 靠上显示
        self.move(x, y)
    
    def _update_window_size(self):
        """更新窗口大小"""
        base_height = 96  # 搜索栏 + 边距
        
        if self.time_input.isVisible():
            base_height += 60
        
        if self.status_panel.isVisible():
            base_height += 160  # 简化后的状态面板
        
        if self.result_panel.isVisible():
            base_height += 480
        
        self.setFixedSize(500, base_height)
    
    def _load_initial_stats(self):
        """加载初始统计信息和数据库时间范围"""
        try:
            # 在 remote GUI 模式下，不在本地加载 CLIP / LanceDB，只使用 SQLite 统计
            if config.GUI_MODE == "remote":
                sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
                stats = sqlite_storage.get_stats()
                total_frames = stats.get('total_frames', 0)
                disk_usage = self.disk_calculator.get_formatted_usage()
                storage_name = "Remote Backend"
                vlm_name = config.VLM_API_MODEL[:20] + "..." if len(config.VLM_API_MODEL) > 20 else config.VLM_API_MODEL
                self.status_panel.update_status(
                    frames=total_frames,
                    storage=storage_name,
                    vlm=vlm_name,
                    disk=disk_usage
                )
                self._load_db_time_range()
                return

            if config.STORAGE_MODE == "simple":
                from core.storage.simple_storage import SimpleStorage
                storage = SimpleStorage(storage_path=config.IMAGE_STORAGE_PATH)
            else:
                from core.storage.lancedb_storage import LanceDBStorage
                # 根据模型名称推断 embedding_dim，避免加载 CLIP 模型
                # 如果表已存在，LanceDBStorage 会正常加载（维度参数主要用于创建新表）
                model_name = config.CLIP_MODEL.lower()
                if "siglip-large" in model_name or "siglip-base-patch16-384" in model_name:
                    embedding_dim = 1024
                elif "siglip-base" in model_name:
                    embedding_dim = 768
                elif "large" in model_name or "vit-large" in model_name:
                    embedding_dim = 768
                else:
                    # 默认值（clip-base-patch32 等）
                    embedding_dim = 512
                storage = LanceDBStorage(db_path=config.LANCEDB_PATH, embedding_dim=embedding_dim)
            
            stats = storage.get_stats()
            total_frames = stats.get('total_frames', 0)
            
            # 计算磁盘使用量
            disk_usage = self.disk_calculator.get_formatted_usage()
            
            # 更新状态面板
            storage_name = "Local SQLite" if config.STORAGE_MODE == "simple" else "Vector DB"
            vlm_name = config.VLM_API_MODEL[:20] + "..." if len(config.VLM_API_MODEL) > 20 else config.VLM_API_MODEL
            
            self.status_panel.update_status(
                frames=total_frames,
                storage=storage_name,
                vlm=vlm_name,
                disk=disk_usage
            )
            
            # 从数据库获取时间范围
            self._load_db_time_range()
            
        except Exception as e:
            logger.error(f"加载统计失败: {e}", exc_info=True)
    
    def _load_db_time_range(self):
        """加载数据库的时间范围，设置为时间选择器的默认值"""
        try:
            sqlite_storage = SQLiteStorage(db_path=config.OCR_DB_PATH)
            
            # 获取最早和最晚的帧
            earliest = sqlite_storage.get_earliest_frame()
            latest = sqlite_storage.get_latest_frame()
            
            if earliest and latest:
                start_time = earliest.get('timestamp')
                end_time = latest.get('timestamp')
                
                if start_time and end_time:
                    self.time_input.start_input.setDateTime(QDateTime(start_time))
                    self.time_input.end_input.setDateTime(QDateTime(end_time))
                    logger.info(f"数据库时间范围: {start_time} - {end_time}")
            else:
                # 没有数据时使用默认值
                now = QDateTime.currentDateTime()
                self.time_input.end_input.setDateTime(now)
                self.time_input.start_input.setDateTime(now.addSecs(-24 * 3600))
                
        except Exception as e:
            logger.warning(f"加载数据库时间范围失败: {e}")
    
    # ============ 事件处理 ============
    
    def _on_escape(self):
        """ESC 键处理"""
        if self.result_panel.isVisible():
            self.result_panel.hide()
            self._update_window_size()
        elif self.status_panel.isVisible():
            self.status_panel.hide()
            self._update_window_size()
        else:
            self.hide()
    
    def _on_mode_changed(self, mode: int):
        """模式切换: 0=RAG, 1=实时"""
        self.current_mode = mode
        
        # RAG 模式显示时间输入
        self.time_input.setVisible(mode == 0)
        
        # 更新占位符
        self._update_placeholder()
        
        # 隐藏结果面板
        self.result_panel.hide()
        
        self._update_window_size()
    
    def _on_ocr_mode_changed(self, active: bool):
        """OCR 模式切换"""
        self.ocr_mode = active
        self._update_placeholder()
        logger.info(f"OCR模式: {'开启' if active else '关闭'}")
    
    def _update_placeholder(self):
        """根据当前模式更新输入框占位符"""
        if self.ocr_mode:
            placeholders = [
                "输入关键词搜索OCR文本...",
                "输入关于屏幕文字的问题..."
            ]
        else:
            placeholders = [
                "输入查询内容...",
                "问一个关于当前屏幕的问题..."
            ]
        self.search_input.setPlaceholderText(placeholders[self.current_mode])
    
    def _on_submit(self):
        """提交查询"""
        query = self.search_input.text().strip()
        
        if self.current_mode == 0:
            # RAG 模式: 时间范围 + 查询
            if query:
                self._do_rag_query(query, ocr_mode=self.ocr_mode)
            else:
                # 没有查询内容时，进行时间段总结
                self._do_time_summary()
        else:
            # 实时问答模式
            if query:
                self._do_realtime_query(query, ocr_mode=self.ocr_mode)
    
    def _toggle_status_panel(self):
        """切换状态面板显示"""
        self.status_panel_visible = not self.status_panel_visible
        self.status_panel.setVisible(self.status_panel_visible)
        
        # 隐藏结果面板
        if self.status_panel_visible:
            self.result_panel.hide()
        
        self._update_window_size()
    
    def _toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    # ============ 录制控制 ============
    
    def _start_recording(self):
        """开始录制"""
        try:
            self.record_btn.setEnabled(False)
            
            # 创建工作线程
            self.record_thread = QThread()
            self.record_worker = RecordWorker(config.STORAGE_MODE)
            self.record_worker.moveToThread(self.record_thread)
            
            # 连接信号
            self.record_thread.started.connect(self.record_worker.start_recording)
            self.record_worker.ready_signal.connect(self._on_recording_ready)
            self.record_worker.stats_signal.connect(self._on_stats_updated)
            self.record_worker.error_signal.connect(self._show_error)
            
            # 启动线程
            self.record_thread.start()
            
        except Exception as e:
            logger.error(f"启动录制失败: {e}", exc_info=True)
            self._show_error(f"启动录制失败: {str(e)}")
            self.record_btn.setEnabled(True)
    
    def _on_recording_ready(self):
        """录制准备就绪"""
        self.is_recording = True
        self.record_btn.setEnabled(True)
        self.record_btn.set_recording(True)
        
        # 重置录制帧计数
        self._last_recording_frames = 0
        
        # 更新状态面板
        self._load_initial_stats()
    
    def _on_stats_updated(self, stats: dict):
        """统计更新（录制时增量更新磁盘使用量）"""
        storage_name = "Local SQLite" if config.STORAGE_MODE == "simple" else "Vector DB"
        vlm_name = config.VLM_API_MODEL[:20] + "..." if len(config.VLM_API_MODEL) > 20 else config.VLM_API_MODEL
        
        # 增量更新磁盘使用量（每次录制新帧时调用）
        recording_frames = stats.get('recording_frames', 0)
        if recording_frames > 0:
            # 计算新增的帧数
            new_frames = recording_frames - getattr(self, '_last_recording_frames', 0)
            if new_frames > 0:
                self.disk_calculator.add_frames(new_frames)
                self._last_recording_frames = recording_frames
        
        disk_usage = self.disk_calculator.get_formatted_usage()
        
        self.status_panel.update_status(
            frames=stats.get('total_frames', 0),
            storage=storage_name,
            vlm=vlm_name,
            disk=disk_usage
        )
    
    def _stop_recording(self):
        """停止录制"""
        try:
            self.is_recording = False
            self.record_btn.setEnabled(False)
            
            if self.record_worker:
                self.record_worker.stop_recording()
            
            if self.record_thread and self.record_thread.isRunning():
                self.record_thread.quit()
                if not self.record_thread.wait(5000):
                    self.record_thread.terminate()
                    self.record_thread.wait()
            
            self.record_worker = None
            self.record_thread = None
            
            self.record_btn.setEnabled(True)
            self.record_btn.set_recording(False)
            
            # 更新状态面板
            self._load_initial_stats()
            
        except Exception as e:
            logger.error(f"停止录制失败: {e}", exc_info=True)
            self._show_error(f"停止录制失败: {str(e)}")
            self.record_btn.setEnabled(True)
    
    # ============ 查询操作 ============
    
    def _show_result_panel(self, title: str = "Processing..."):
        """显示结果面板"""
        self.result_panel.clear()
        self.result_panel.title_label.setText(title)
        self.result_panel.show_progress(True)
        self.result_panel.show()
        self.status_panel.hide()
        self._update_window_size()
    
    def _do_rag_query(self, query_text: str, ocr_mode: bool = False):
        """执行 RAG 查询（带时间范围）"""
        start_time = self.time_input.get_start_datetime()
        end_time = self.time_input.get_end_datetime()
        
        if ocr_mode:
            self._show_result_panel("📝 OCR Text Searching...")
            self._start_query_thread(
                lambda worker: worker.query_ocr_rag(query_text, start_time, end_time)
            )
        else:
            self._show_result_panel("🔍 Searching...")
            self._start_query_thread(
                lambda worker: worker.query_rag_with_time(query_text, start_time, end_time)
            )
    
    def _do_time_summary(self):
        """执行时间段总结"""
        try:
            start_time = self.time_input.get_start_datetime()
            end_time = self.time_input.get_end_datetime()
            
            if start_time >= end_time:
                self._show_error("开始时间必须早于结束时间")
                return
            
            self._show_result_panel("📅 Summarizing...")
            self._start_query_thread(lambda worker: worker.query_time_summary(start_time, end_time))
        
        except Exception as e:
            self._show_error(f"时间获取错误: {e}")
    
    def _do_realtime_query(self, question: str, ocr_mode: bool = False):
        """执行实时问答"""
        if ocr_mode:
            self._show_result_panel("📝 OCR Text Query...")
            self._start_query_thread(lambda worker: worker.query_realtime_ocr(question))
        else:
            self._show_result_panel("❓ Analyzing...")
            self._start_query_thread(lambda worker: worker.query_realtime(question))
    
    def _start_query_thread(self, query_func):
        """启动查询线程"""
        self.query_thread = QThread()
        self.query_worker = QueryWorker(config.STORAGE_MODE)
        self.query_worker.moveToThread(self.query_thread)
        
        self.query_thread.started.connect(lambda: query_func(self.query_worker))
        self.query_worker.result_signal.connect(self._display_result)
        self.query_worker.progress_signal.connect(self._update_progress)
        self.query_worker.error_signal.connect(self._show_query_error)
        self.query_worker.result_signal.connect(self.query_thread.quit)
        self.query_worker.error_signal.connect(self.query_thread.quit)
        
        self.query_thread.start()
    
    def _update_progress(self, text: str):
        """更新进度"""
        self.result_panel.result_text.setText(text)
    
    def _display_result(self, text: str):
        """显示结果"""
        self.result_panel.set_result("VLM Analysis", text)
    
    def _show_query_error(self, error: str):
        """显示查询错误"""
        self.result_panel.set_result("❌ Error", error)
    
    def _show_error(self, error: str):
        """显示错误"""
        QMessageBox.warning(self, "错误", error)
    
    # ============ 窗口拖动 ============
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
    
    # ============ 绘制圆角背景 ============
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制圆角矩形背景
        path = QPainterPath()
        path.addRoundedRect(
            self.rect().adjusted(10, 10, -10, -10),
            20, 20
        )
        
        painter.fillPath(path, QBrush(QColor(Theme.BG_DARK)))
    
    # ============ 应用实例引用 ============
    
    def set_app_instance(self, app_instance):
        """设置应用实例引用（用于调用退出功能）"""
        self.app_instance = app_instance
    
    # ============ 关闭功能 ============
    
    def _quit_app(self):
        """退出应用"""
        if self.app_instance:
            self.app_instance.quit_app()
        else:
            self.close()
    
    # ============ 关闭事件 ============
    
    def closeEvent(self, event):
        """关闭事件处理（清理资源）"""
        # 停止录制（如果正在录制）
        if self.is_recording:
            self._stop_recording()
        
        # 停止查询线程（如果正在运行）
        if self.query_thread and self.query_thread.isRunning():
            self.query_thread.quit()
            if not self.query_thread.wait(3000):
                self.query_thread.terminate()
                self.query_thread.wait()
        
        event.accept()
