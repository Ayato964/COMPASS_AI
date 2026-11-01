"""
COMPASS AI One-Click Hardware-Aware Bootstrapper (`installer/bootstrap.py`).
Designed for non-engineer users:
1. Detects OS (Windows / macOS / Linux) and GPU hardware (NVIDIA `nvidia-smi`, Apple Silicon `arm64`, or CPU).
2. Installs the optimal PyTorch build (CUDA 12.4 wheel for NVIDIA GPUs, native MPS/CPU wheel for macOS/CPU) if missing.
3. Installs lightweight runtime requirements (`requirements.txt`) without requiring `flash-attn`, `nvcc`, or `loralib`.
4. Initializes the user's autonomous model directory (`%APPDATA%\\COMPASS_AI\\models` on Windows,
   `~/Library/Application Support/COMPASS_AI/models` on macOS) on the physical host filesystem and seeds default model templates.
5. Verifies the local inference engine and optionally launches the FastAPI server on a verified conflict-free port or opens the models folder.
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _is_port_free(port: int) -> bool:
    """
    Verifies no existing listener occupies `port` on IPv4 (`127.0.0.1` / `0.0.0.0`) or
    IPv6 (`::1` / `[::]`), then confirms exclusive bindability on `127.0.0.1` (`[CRIT-02]`, `[HIGH-03]`).
    """
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)), (socket.AF_INET6, ("::1", port, 0, 0))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.05)
                if probe.connect_ex(addr) == 0:
                    return False
        except OSError:
            pass

    for bind_host in ("0.0.0.0", "127.0.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if excl is not None:
                    srv.setsockopt(socket.SOL_SOCKET, excl, 1)
                srv.bind((bind_host, port))
        except OSError:
            return False
    return True


def find_available_port(preferred_port: int = 8000, max_attempts: int = 30) -> int:
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        if 1024 <= candidate <= 65535 and _is_port_free(candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        return int(srv.getsockname()[1])


def has_nvidia_gpu() -> bool:
    """Detects whether an NVIDIA GPU is physically present via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi and sys.platform.startswith("win"):
        candidate = os.path.join(
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            "NVIDIA Corporation",
            "NVSMI",
            "nvidia-smi.exe",
        )
        if os.path.isfile(candidate):
            nvidia_smi = candidate
    if not nvidia_smi:
        return False
    try:
        res = subprocess.run(
            [nvidia_smi, "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        return res.returncode == 0 and b"GPU" in res.stdout
    except Exception:
        return False


def is_torch_ready(expect_cuda: bool) -> bool:
    try:
        import torch

        if expect_cuda and not torch.cuda.is_available():
            return False
        return True
    except ImportError:
        return False


def ensure_dependencies(skip_install: bool = False) -> None:
    nvidia_present = has_nvidia_gpu() and sys.platform != "darwin"
    print(f"[COMPASS AI] OS: {platform.system()} ({platform.machine()}) | NVIDIA GPU Detected: {nvidia_present}")

    already_has_cpu_torch = is_torch_ready(expect_cuda=False)
    if not skip_install and not is_torch_ready(expect_cuda=nvidia_present):
        print("[COMPASS AI] 最適な PyTorch ランタイムをインストールしています...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch"]
        if nvidia_present:
            cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu124"])
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            if already_has_cpu_torch:
                print(
                    "[COMPASS AI] [WARN] オフライン環境またはネットワーク制限のため CUDA 版 PyTorch の取得をスキップし、"
                    "既存の PyTorch ランタイムで継続します。"
                )
            else:
                raise

    req_file = os.path.join(ROOT_DIR, "requirements.txt")
    if not skip_install and os.path.isfile(req_file):
        try:
            import fastapi  # noqa: F401
            import mido  # noqa: F401
            import numpy  # noqa: F401
            import pretty_midi  # noqa: F401
            import pydantic  # noqa: F401
            import python_multipart  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            print("[COMPASS AI] 必須パッケージをインストールしています...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])


def initialize_models_directory() -> str:
    from model import (
        get_user_models_dir,
        resolve_physical_models_dir_for_shell,
        seed_user_models_dir,
    )

    user_models_dir = get_user_models_dir()
    bundled_models_dir = os.path.join(ROOT_DIR, "data", "models")
    seed_user_models_dir(user_models_dir, bundled_models_dir)
    physical_dir = resolve_physical_models_dir_for_shell(user_models_dir)
    print(f"[COMPASS AI] ユーザーモデルディレクトリを初期化しました: {physical_dir}")
    return physical_dir


def verify_engine() -> None:
    from model import ModelController

    controller = ModelController()
    status = controller.get_hardware_status()
    print("======================================================================")
    print(" COMPASS AI - ローカル推論エンジン 準備完了")
    print("======================================================================")
    print(f"  推論デバイス         : {status['device']} ({status['backend_class']})")
    print(f"  演算精度 (DType)     : {status['dtype']}")
    print(f"  モデル配置フォルダ   : {status['user_models_dir']}")
    print(f"  認識済みモデル定義数 : {status['available_models_count']}")
    print(f"  重み配置済みモデル   : {status['models_with_weights'] or '未配置 (model.pth を上記フォルダへ配置してください)'}")
    print("======================================================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="COMPASS AI One-Click Installer & Bootstrapper")
    parser.add_argument(
        "--setup-only",
        "--SetupOnly",
        dest="setup_only",
        action="store_true",
        help="環境構築とモデルフォルダ初期化のみ実行して終了します",
    )
    parser.add_argument(
        "--skip-pip",
        "--SkipPip",
        dest="skip_pip",
        action="store_true",
        help="pip インストールチェックをスキップします",
    )
    parser.add_argument(
        "--open-models-dir",
        "--OpenModelsDir",
        dest="open_models_dir",
        action="store_true",
        help="AppData の models フォルダをエクスプローラー/Finderで開きます",
    )
    default_host = os.environ.get("COMPASS_HOST", "").strip() or "127.0.0.1"
    raw_port = os.environ.get("COMPASS_PORT", "").strip()
    try:
        default_port = int(raw_port) if raw_port else 8000
    except ValueError:
        default_port = 8000

    parser.add_argument("--host", default=default_host, help="バインドホスト")
    parser.add_argument("--port", type=int, default=default_port, help="ポート番号")
    args = parser.parse_args()

    ensure_dependencies(skip_install=args.skip_pip)
    user_models_dir = initialize_models_directory()
    verify_engine()

    from model import resolve_physical_models_dir_for_shell

    shell_models_dir = resolve_physical_models_dir_for_shell(user_models_dir)

    if args.open_models_dir:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer.exe", shell_models_dir])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", shell_models_dir])
            else:
                subprocess.Popen(["xdg-open", shell_models_dir])
        except Exception as e:
            print(f"[COMPASS AI] フォルダを自動で開けませんでした ({shell_models_dir}): {e}")

    if args.setup_only:
        return 0

    import uvicorn

    selected_port = find_available_port(args.port)
    if selected_port != args.port:
        print(f"[COMPASS AI] ポート {args.port} が使用中のため、空きポート {selected_port} を選択しました。")
    os.environ["COMPASS_PORT"] = str(selected_port)

    print(f"[COMPASS AI] ローカル推論サーバーを起動しています: http://{args.host}:{selected_port}")
    uvicorn.run("app:app", host=args.host, port=selected_port, app_dir=ROOT_DIR, proxy_headers=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
