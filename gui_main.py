#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisualMem - PySide6 前端主入口
Spotlight 风格界面，支持全局快捷键
一个更简单、轻量的 GUI 界面，但是维护进度较慢，不推荐使用
"""

import sys
import signal
import os
import requests
from pathlib import Path
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QKeySequence, QAction, QPixmap, QShortcut
from PySide6.QtCore import Qt

from config import config
from gui import VisualMemMainWindow


def _set_process_name(name: str):
    """设置进程名称（macOS 上用于显示正确的应用名称）"""
    try:
        import platform
        if platform.system() == "Darwin":  # macOS
            # 在 macOS 上，可以通过设置 argv[0] 来影响进程名称
            if len(sys.argv) > 0:
                sys.argv[0] = name
            # 尝试使用 ctypes 设置进程标题（如果可用）
            try:
                import ctypes
                libc = ctypes.CDLL(None)
                libc.setproctitle(ctypes.c_char_p(name.encode('utf-8')))
            except (AttributeError, OSError):
                # setproctitle 不可用，仅使用 argv[0]
                pass
    except Exception:
        # 设置进程名称失败不影响程序运行
        pass


class VisualMemApp:
    """VisualMem 应用程序"""
    
    def __init__(self):
        # 在 macOS 上设置进程名称（必须在创建 QApplication 之前）
        _set_process_name("VisualMem")
        
        self.app = QApplication(sys.argv)
        
        # 设置应用信息
        self.app.setApplicationName("VisualMem")
        self.app.setOrganizationName("VLM-Research")
        self.app.setQuitOnLastWindowClosed(False)  # 关闭窗口不退出应用
        
        # 加载并设置应用图标（必须在创建窗口和托盘之前设置）
        app_icon = self._load_icon()
        if app_icon:
            self.app.setWindowIcon(app_icon)
            print("✅ 应用程序图标已设置")
        
        # 创建主窗口
        self.window = VisualMemMainWindow()
        # 设置窗口图标
        if app_icon:
            self.window.setWindowIcon(app_icon)
        # 将应用实例传递给主窗口，以便主窗口可以调用退出/最小化功能
        self.window.set_app_instance(self)
        
        # 创建系统托盘
        self._setup_tray(app_icon)
        
        # 设置全局快捷键
        self._setup_global_shortcuts()
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_icon(self):
        """加载应用图标"""
        logo_path = Path(__file__).parent / "logo.png"
        if logo_path.exists():
            try:
                pixmap = QPixmap(str(logo_path))
                if not pixmap.isNull():
                    # 创建一个支持多尺寸的图标
                    icon = QIcon()
                    # 首先添加原始尺寸（最高质量，不压缩）
                    icon.addPixmap(pixmap)
                    # 添加一些常用的大尺寸（使用更大尺寸以确保清晰度）
                    # macOS 系统托盘会从这些尺寸中选择合适的，不会过度压缩
                    sizes = [64, 128, 256]
                    for size in sizes:
                        scaled = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        icon.addPixmap(scaled)
                    return icon
                else:
                    print(f"⚠️  警告: 无法加载 logo.png 文件 ({logo_path})")
            except Exception as e:
                print(f"⚠️  警告: 加载图标时出错: {e}")
        else:
            print(f"⚠️  警告: 找不到 logo.png 文件 ({logo_path})")
        return None
    
    def _setup_tray(self, icon=None):
        """设置系统托盘"""
        # 检查系统托盘是否可用
        if not QSystemTrayIcon.isSystemTrayAvailable():
            print("⚠️  警告: 系统托盘不可用")
            return
        
        # 创建托盘图标
        self.tray = QSystemTrayIcon(self.app)
        
        # 创建托盘菜单
        menu = QMenu()
        
        show_action = QAction("显示 VisualMem", menu)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)
        
        menu.addSeparator()
        
        quit_action = QAction("退出", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)
        
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        
        # 设置托盘图标
        if icon:
            self.tray.setIcon(icon)
            print("✅ 系统托盘图标已设置")
        else:
            # 如果没有提供图标，尝试加载
            icon = self._load_icon()
            if icon:
                self.tray.setIcon(icon)
                print("✅ 系统托盘图标已加载并设置")
            else:
                print("⚠️  警告: 无法加载系统托盘图标")
        
        self.tray.setToolTip("VisualMem - 点击显示")
        self.tray.show()
    
    def _on_tray_activated(self, reason):
        """托盘图标激活"""
        # QSystemTrayIcon.ActivationReason.Trigger 表示单击
        # QSystemTrayIcon.ActivationReason.DoubleClick 表示双击
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # 单击托盘图标：切换窗口显示/隐藏
            self._toggle_window()
        elif reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            # 双击托盘图标：显示窗口
            self._show_window()
    
    def _toggle_window(self):
        """切换窗口显示（从托盘恢复或隐藏到托盘）"""
        # 检查窗口是否可见且不是最小化状态
        if self.window.isVisible() and not self.window.isMinimized():
            # 窗口可见且未最小化，隐藏到托盘
            self.window.hide()
        else:
            # 窗口不可见或最小化，显示窗口
            self._show_window()
    
    def _setup_global_shortcuts(self):
        """设置全局快捷键"""
        # Ctrl+Alt+W: 全局显示/隐藏窗口（即使窗口没有焦点也能工作）
        toggle_shortcut = QShortcut(QKeySequence("Ctrl+Alt+W"), self.window)
        toggle_shortcut.setContext(Qt.ApplicationShortcut)  # 设置为应用级全局快捷键
        toggle_shortcut.activated.connect(self._toggle_window)
        
        # Esc: 全局隐藏窗口（如果窗口可见）
        hide_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.window)
        hide_shortcut.setContext(Qt.ApplicationShortcut)  # 设置为应用级全局快捷键
        hide_shortcut.activated.connect(self._global_hide_window)
        
        print("✅ 全局快捷键已设置:")
        print("   - Ctrl+Alt+W: 显示/隐藏窗口")
        print("   - Esc: 隐藏窗口")
    
    def _global_hide_window(self):
        """全局隐藏窗口（Esc 键触发）"""
        if self.window.isVisible() and not self.window.isMinimized():
            self.window.hide()
    
    def _show_window(self):
        """显示窗口（从托盘恢复）"""
        # 使用 showNormal() 来恢复窗口（确保窗口状态正确）
        if self.window.isMinimized():
            self.window.showNormal()
        else:
            self.window.show()
        
        # 确保窗口显示在最前面
        self.window.raise_()
        self.window.activateWindow()
        # 确保窗口可见
        self.window.setVisible(True)
        # 设置窗口状态为正常（不是最小化）
        self.window.setWindowState(Qt.WindowNoState)
        # 聚焦到搜索框
        if hasattr(self.window, 'search_input'):
            self.window.search_input.setFocus()
    
    def quit_app(self):
        """退出应用（供主窗口调用）"""
        if self.window.is_recording:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self.window, '确认退出',
                "录制正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return False
        
        # 如果是 local 模式，需要同时关闭 backend_server
        if config.GUI_MODE == "local":
            self._stop_backend_server()
        
        self.window.close()
        self.tray.hide()
        self.app.quit()
        return True
    
    def _stop_backend_server(self):
        """
        停止 backend_server（仅在 local 模式下调用）
        尝试通过 PID 文件或进程名查找并停止 backend_server
        """
        try:
            # 方法1：尝试从 PID 文件读取（由 start_gui_with_backend.sh 创建）
            root_dir = Path(__file__).parent
            pid_file = root_dir / ".gui_backend_pid"
            
            if pid_file.exists():
                try:
                    with open(pid_file, "r") as f:
                        backend_pid = int(f.read().strip())
                    
                    # 检查进程是否还在运行
                    try:
                        os.kill(backend_pid, 0)  # 发送信号0检查进程是否存在
                        # 进程存在，尝试停止
                        print(f"正在停止 backend_server (PID: {backend_pid})...")
                        os.kill(backend_pid, signal.SIGTERM)
                        
                        # 等待进程退出（最多等待3秒）
                        import time
                        for _ in range(30):  # 30次，每次0.1秒
                            try:
                                os.kill(backend_pid, 0)
                                time.sleep(0.1)
                            except ProcessLookupError:
                                # 进程已退出
                                break
                        else:
                            # 如果进程还在，强制终止
                            try:
                                os.kill(backend_pid, signal.SIGKILL)
                                print(f"强制终止 backend_server (PID: {backend_pid})")
                            except ProcessLookupError:
                                pass  # 进程已退出
                        
                        # 删除 PID 文件
                        pid_file.unlink(missing_ok=True)
                        print("✅ backend_server 已停止")
                        return
                    except ProcessLookupError:
                        # 进程不存在，删除 PID 文件
                        pid_file.unlink(missing_ok=True)
                except (ValueError, FileNotFoundError):
                    pass  # PID 文件格式错误或不存在，尝试其他方法
            
            # 方法2：通过进程名查找（作为后备方案）
            import subprocess
            import platform
            import time
            
            system = platform.system()
            if system in ("Darwin", "Linux"):
                # 使用 pgrep 查找进程
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", "gui_backend_server.py"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid.strip()]
                        if pids:
                            print(f"正在停止 backend_server (找到 {len(pids)} 个进程)...")
                            for pid in pids:
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass  # 进程已不存在
                            
                            # 等待进程退出（最多等待3秒）
                            for _ in range(30):  # 30次，每次0.1秒
                                all_stopped = True
                                for pid in pids:
                                    try:
                                        os.kill(pid, 0)  # 检查进程是否存在
                                        all_stopped = False
                                        break
                                    except ProcessLookupError:
                                        pass  # 进程已退出
                                
                                if all_stopped:
                                    break
                                time.sleep(0.1)
                            
                            # 如果还有进程未退出，强制终止
                            for pid in pids:
                                try:
                                    os.kill(pid, 0)  # 检查进程是否存在
                                    os.kill(pid, signal.SIGKILL)
                                    print(f"强制终止 backend_server (PID: {pid})")
                                except ProcessLookupError:
                                    pass  # 进程已退出
                            
                            print("✅ backend_server 已停止")
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass  # pgrep 不可用或没有找到进程
            
        except Exception as e:
            # 停止 backend_server 失败不应该阻止 GUI 退出
            print(f"⚠️  停止 backend_server 时出错: {e}")
            print("   GUI 将继续退出，但 backend_server 可能仍在运行")
            print("   您可以手动停止它: pkill -f gui_backend_server.py")
    
    def _quit(self):
        """退出应用（托盘菜单调用）"""
        self.quit_app()
    
    def _signal_handler(self, sig, frame):
        """信号处理"""
        print("\n正在退出...")
        self.quit_app()
    
    def run(self):
        """运行应用"""
        # 显示窗口
        self._show_window()
        
        print("=" * 50)
        print("  VisualMem 已启动")
        print("  - 点击系统托盘图标切换显示")
        print("  - 按 Ctrl+Alt+W 全局显示/隐藏窗口")
        print("  - 按 ESC 隐藏窗口")
        print("  - 按 Ctrl+C 退出")
        print("=" * 50)
        
        return self.app.exec()


def check_backend_availability():
    """
    检查远程后端服务器是否可用
    
    Returns:
        bool: 如果后端可用返回 True，否则返回 False
    """
    if config.GUI_MODE != "remote":
        # 本地模式不需要检查
        return True
    
    if not config.GUI_REMOTE_BACKEND_URL:
        print("❌ 错误: GUI_MODE=remote 但 GUI_REMOTE_BACKEND_URL 未配置")
        print("   请在 .env 文件中设置 GUI_REMOTE_BACKEND_URL")
        return False
    
    backend_url = config.GUI_REMOTE_BACKEND_URL.rstrip("/")
    health_url = f"{backend_url}/health"
    
    print(f"🔍 正在检查后端服务器可用性: {backend_url}")
    
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        # 检查返回的数据格式
        data = response.json()
        if data.get("status") == "ok":
            print(f"✅ 后端服务器可用 (状态码: {response.status_code})")
            return True
        else:
            print(f"⚠️  后端服务器响应异常: {data}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 错误: 无法连接到后端服务器 {backend_url}")
        print("   请确认后端服务器是否已启动")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ 错误: 连接后端服务器超时 ({backend_url})")
        print("   请确认后端服务器是否正常运行")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ 错误: 后端服务器返回错误 (状态码: {e.response.status_code})")
        print(f"   请确认后端服务器是否正常运行")
        return False
    except Exception as e:
        print(f"❌ 错误: 检查后端服务器时发生未知错误: {e}")
        return False


def main():
    """主函数"""
    # 如果是远程模式，先检查后端服务器是否可用
    if not check_backend_availability():
        print("\n💡 提示: 如果是第一次启动，请先启动后端服务器")
        print("   然后重新运行 GUI")
        sys.exit(1)
    
    app = VisualMemApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
