#!/usr/bin/env bash

# Simple helper script to start backend_server + GUI on the same machine.
# 默认本地写盘 + 本地检索。如果希望 GUI 通过 HTTP 访问本地后端，可以设置 GUI_MODE=remote。
# 所有日志输出到 ./logs/gui_execution.log，GUI 在后台运行，不会阻塞终端。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

# 确保 logs 目录存在
mkdir -p "${ROOT_DIR}/logs"

LOG_FILE="${ROOT_DIR}/logs/gui_execution.log"

# 清空或创建日志文件
> "${LOG_FILE}"

# 将所有输出重定向到日志文件
exec >> "${LOG_FILE}" 2>&1

echo "Starting gui_backend_server on http://127.0.0.1:8080 ..."
echo "Logs will be written to: ${LOG_FILE}"

# 启动后端服务器，日志输出到文件
python gui_backend_server.py &
BACKEND_PID=$!

echo "Backend server started (PID: ${BACKEND_PID})"
sleep 2

echo "Starting GUI (local mode by default)..."
# 如需远程模式（GUI 只负责采集 + 上传），取消下面两行注释：
# export GUI_MODE=remote
# export GUI_REMOTE_BACKEND_URL=http://127.0.0.1:8080

# 保存后端 PID 到文件，方便后续停止
echo "${BACKEND_PID}" > "${ROOT_DIR}/.gui_backend_pid"

echo ""
echo "Backend server is running (PID: ${BACKEND_PID})"
echo "Logs are being written to: ${LOG_FILE}"
echo ""
echo "GUI is starting. Use the close button (X) in the GUI window to exit."
echo "To stop backend server manually: kill ${BACKEND_PID}"
echo "To view logs in real-time: tail -f ${LOG_FILE}"
echo ""

# 启动 GUI（所有输出已通过 exec 重定向到日志文件）
python gui_main.py

# GUI 退出后，停止后端服务器
echo ""
echo "GUI exited, stopping backend_server (pid=${BACKEND_PID})..."
kill "${BACKEND_PID}" 2>/dev/null || true

# 清理 PID 文件
rm -f "${ROOT_DIR}/.gui_backend_pid"


