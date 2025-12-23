#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模式UI组件
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QDateTimeEdit
)


class Mode1Widget(QWidget):
    """模式1: 快速检索"""
    
    search_clicked = Signal(str)  # 发送查询文本
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("输入查询内容:"))
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("例如: 我下午在看什么视频?")
        self.query_input.setMinimumHeight(40)
        self.query_input.returnPressed.connect(self._on_search)
        layout.addWidget(self.query_input)
        
        self.search_btn = QPushButton("搜索")
        self.search_btn.setMinimumHeight(40)
        self.search_btn.clicked.connect(self._on_search)
        layout.addWidget(self.search_btn)
        
        layout.addStretch()
    
    def _on_search(self):
        """搜索按钮点击"""
        query_text = self.query_input.text().strip()
        if query_text:
            self.search_clicked.emit(query_text)
    
    def get_search_button(self):
        """获取搜索按钮(用于外部控制enabled状态)"""
        return self.search_btn


class Mode2Widget(QWidget):
    """模式2: 时间总结"""
    
    summary_clicked = Signal(object, object)  # 发送 (start_time, end_time)
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("选择时间范围:"))
        
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel("从:"))
        
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        time_range_layout.addWidget(self.start_time_edit)
        
        time_range_layout.addWidget(QLabel("到:"))
        
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        time_range_layout.addWidget(self.end_time_edit)
        
        layout.addLayout(time_range_layout)
        
        self.summary_btn = QPushButton("总结")
        self.summary_btn.setMinimumHeight(40)
        self.summary_btn.clicked.connect(self._on_summary)
        layout.addWidget(self.summary_btn)
        
        layout.addStretch()
    
    def _on_summary(self):
        """总结按钮点击"""
        start_time = self.start_time_edit.dateTime().toPython()
        end_time = self.end_time_edit.dateTime().toPython()
        self.summary_clicked.emit(start_time, end_time)
    
    def get_summary_button(self):
        """获取总结按钮"""
        return self.summary_btn
    
    def get_time_edits(self):
        """获取时间编辑器"""
        return self.start_time_edit, self.end_time_edit


class Mode3Widget(QWidget):
    """模式3: 实时问答"""
    
    ask_clicked = Signal(str)  # 发送问题文本
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("关于当前活动的问题:"))
        
        self.realtime_input = QLineEdit()
        self.realtime_input.setPlaceholderText("例如: 我现在在做什么?")
        self.realtime_input.setMinimumHeight(40)
        self.realtime_input.returnPressed.connect(self._on_ask)
        layout.addWidget(self.realtime_input)
        
        self.ask_btn = QPushButton("提问")
        self.ask_btn.setMinimumHeight(40)
        self.ask_btn.clicked.connect(self._on_ask)
        layout.addWidget(self.ask_btn)
        
        layout.addStretch()
    
    def _on_ask(self):
        """提问按钮点击"""
        question = self.realtime_input.text().strip()
        if question:
            self.ask_clicked.emit(question)
    
    def get_ask_button(self):
        """获取提问按钮"""
        return self.ask_btn



