#!/usr/bin/env bash
# COMPASS AI - macOS (Apple Silicon MPS / Intel) & Linux One-Click Installer & Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "======================================================================"
echo " COMPASS AI - Local Music Foundation Engine Setup & Launcher"
echo "======================================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 が見つかりません。Python 3.10 以上をインストールしてください。"
    exit 1
fi

if [ ! -f "${ROOT_DIR}/.venv/bin/python" ]; then
    echo "[COMPASS AI] 専用仮想環境 (.venv) を構築しています..."
    python3 -m venv "${ROOT_DIR}/.venv"
fi

exec "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/installer/bootstrap.py" "$@"
