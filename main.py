"""
COMPASS AI Application Entrypoint (`main.py`).
Re-exports `app` and `model_controller` from `app.py` for compatibility with `uvicorn main:app`,
configures UTF-8 stdio streams, and runs a kernel-level parent-process watchdog when spawned by
Electron (`COMPASS_PARENT_PID`) so force-killing `electron.exe` never orphans `python.exe` (`[HIGH-02]`).
"""

import ctypes
import os
import sys
import threading
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_watchdog_lock = threading.Lock()
_watchdog_started = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def start_parent_watchdog(parent_pid: int) -> None:
    """
    Monitors `parent_pid` on a daemon thread and calls `os._exit(0)` immediately if the parent
    process terminates (even via `TerminateProcess` / Task Manager on non-MSIX Win32 Python).
    Started BEFORE importing PyTorch/`app` so parent crashes during cold import still self-terminate.
    """
    global _watchdog_started
    if parent_pid <= 0:
        return
    with _watchdog_lock:
        if _watchdog_started:
            return
        _watchdog_started = True

    def _watch_win32() -> None:
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, parent_pid)
        if not handle:
            err = ctypes.get_last_error()
            if err != 5:  # Not ERROR_ACCESS_DENIED -> parent process already exited
                os._exit(0)
            return

        try:
            while True:
                rc = kernel32.WaitForSingleObject(handle, 1000)
                if rc == WAIT_TIMEOUT:
                    continue
                if rc == WAIT_OBJECT_0:
                    os._exit(0)
                # WAIT_FAILED or abandoned -> exit conservatively
                os._exit(0)
        finally:
            kernel32.CloseHandle(handle)

    def _watch_posix() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() == 1:
                os._exit(0)
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass

    target = _watch_win32 if sys.platform.startswith("win") else _watch_posix
    t = threading.Thread(target=target, name="compass-parent-watchdog", daemon=True)
    t.start()


_boot_ppid = os.environ.get("COMPASS_PARENT_PID", "").strip()
if _boot_ppid.isdigit():
    start_parent_watchdog(int(_boot_ppid))

from app import app, model_controller  # noqa: E402

__all__ = ["app", "model_controller", "start_parent_watchdog"]


if __name__ == "__main__":
    import uvicorn

    raw_ppid = os.environ.get("COMPASS_PARENT_PID", "").strip()
    if raw_ppid.isdigit():
        start_parent_watchdog(int(raw_ppid))

    port = _env_int("COMPASS_PORT", 8000)
    host = os.environ.get("COMPASS_HOST", "").strip() or "127.0.0.1"
    uvicorn.run(app, host=host, port=port, proxy_headers=False)
