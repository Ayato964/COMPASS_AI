import threading
import time
from typing import Any, Callable, Dict, Optional
import torch
from mortm.models.modules.backend import HardwareBackendFactory, IHardwareBackend


class GenerationCancelledError(RuntimeError):
    """Raised when an in-flight autoregressive generation is cooperatively cancelled."""


class GenerationProgressTracker:
    """
    Thread-safe, run-scoped live inference telemetry store for `GET /generation_progress`.
    Guards updates by `run_id` so a cancelled worker thread can never overwrite telemetry
    belonging to a newer generation run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._state: Dict[str, Any] = {
            "run_id": None,
            "job_id": None,
            "active": False,
            "stage": "idle",
            "generated_tokens": 0,
            "max_tokens": 0,
            "message": "待機中",
            "updated_at": time.time(),
        }

    def start_run(self, run_id: str, message: str, job_id: Optional[str] = None) -> None:
        with self._lock:
            self._active_run_id = run_id
            self._state = {
                "run_id": run_id,
                "job_id": job_id,
                "active": True,
                "stage": "loading_model",
                "generated_tokens": 0,
                "max_tokens": 0,
                "message": message,
                "updated_at": time.time(),
            }

    def update(
        self,
        *,
        active: Optional[bool] = None,
        stage: Optional[str] = None,
        generated_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        message: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            if (
                run_id is not None
                and self._active_run_id is not None
                and run_id != self._active_run_id
            ):
                return
            if run_id is not None:
                self._state["run_id"] = run_id
            if active is not None:
                self._state["active"] = bool(active)
            if stage is not None:
                self._state["stage"] = stage
            if generated_tokens is not None:
                self._state["generated_tokens"] = int(generated_tokens)
            if max_tokens is not None:
                self._state["max_tokens"] = int(max_tokens)
            if message is not None:
                self._state["message"] = message
            self._state["updated_at"] = time.time()

    def snapshot(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            if job_id and snap.get("job_id") != job_id:
                return {
                    "run_id": None,
                    "job_id": job_id,
                    "active": False,
                    "stage": "idle",
                    "generated_tokens": 0,
                    "max_tokens": 0,
                    "message": "待機中",
                    "updated_at": snap.get("updated_at", time.time()),
                }
            return snap


GLOBAL_GENERATION_PROGRESS = GenerationProgressTracker()


class LearningProgress:
    def __init__(
        self,
        backend: Optional[IHardwareBackend] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self._backend = backend or HardwareBackendFactory.create("auto")
        self._step_callback = step_callback

    def get_device(self) -> torch.device:
        return self._backend.device

    def get_step(self) -> int:
        return 1

    def report_generation_step(
        self,
        generated_tokens: int,
        max_tokens: int,
        run_id: Optional[str] = None,
    ) -> None:
        GLOBAL_GENERATION_PROGRESS.update(
            active=True,
            stage="decoding",
            generated_tokens=generated_tokens,
            max_tokens=max_tokens,
            message=f"自己回帰トークン生成中 ({generated_tokens} / {max_tokens} tokens)...",
            run_id=run_id,
        )
        if self._step_callback is not None:
            self._step_callback(generated_tokens, max_tokens)


class _DefaultLearningProgress(LearningProgress):
    pass


class ProgressManager(LearningProgress):
    pass


class _LearnProgress:
    def get_progress_manager(self) -> ProgressManager:
        return ProgressManager()


class DefaultObserver(LearningProgress):
    pass
