"""GPU utilization metrics tracker for OpenRLHF training loops.

Tracks compute time, idle time, idle ratio, and threshold-based idle rate
by sampling GPU utilization in a background thread via pynvml.
"""

import threading
import time
from typing import Dict, List, Optional

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class GpuMetricsTracker:
    """Tracks GPU compute/idle timing and utilization-based idle rates.

    Usage::

        tracker = GpuMetricsTracker(gpu_index=0, idle_rate_thresholds=[90])

        # -- training loop --
        tracker.step_start()
        train_step(...)
        tracker.step_end()

        metrics = tracker.get_metrics()
        # metrics == {
        #     "gpu/compute_time": 12.3,
        #     "gpu/idle_time": 1.5,
        #     "gpu/idle_ratio": 0.108,
        #     "gpu/ir_90": 0.25,   # 25 % of samples had util < 90 %
        # }

    Args:
        gpu_index: CUDA device ordinal to monitor (default 0).
        idle_rate_thresholds: List of utilization thresholds for ``ir_T`` metrics.
            Each value *T* produces a metric ``gpu/ir_{T}`` equal to the
            fraction of utilization samples below *T* %.
        poll_interval_s: How often (seconds) to sample GPU utilization.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        idle_rate_thresholds: Optional[List[int]] = None,
        poll_interval_s: float = 0.1,
    ):
        self.gpu_index = gpu_index
        self.idle_rate_thresholds = idle_rate_thresholds or [90]
        self.poll_interval_s = poll_interval_s

        self._handle = None
        self._nvml_initialized = False
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._nvml_initialized = True
                name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                logger.info(f"GpuMetricsTracker: monitoring GPU {gpu_index} ({name})")
            except pynvml.NVMLError as e:
                logger.warning(f"GpuMetricsTracker: pynvml init failed ({e}); GPU metrics disabled")
        else:
            logger.warning("GpuMetricsTracker: pynvml not available; GPU metrics disabled")

        self._step_start_time: Optional[float] = None
        self._prev_step_end_time: Optional[float] = None
        self._compute_time: float = 0.0
        self._idle_time: float = 0.0

        self._util_samples: List[int] = []

        self._sampling = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def step_start(self) -> None:
        """Mark the beginning of a training step (compute phase)."""
        now = time.monotonic()
        self._step_start_time = now

        if self._prev_step_end_time is not None:
            self._idle_time = now - self._prev_step_end_time
        else:
            self._idle_time = 0.0

        self._util_samples = []
        self._start_sampling()

    def step_end(self) -> None:
        """Mark the end of a training step (compute phase)."""
        self._stop_sampling()

        now = time.monotonic()
        if self._step_start_time is not None:
            self._compute_time = now - self._step_start_time
        else:
            self._compute_time = 0.0

        self._prev_step_end_time = now

    def get_metrics(self) -> Dict[str, float]:
        """Return the latest step's metrics as a flat dict.

        Keys produced:
        - ``gpu/compute_time`` — seconds of active compute
        - ``gpu/idle_time`` — seconds of idle before this step
        - ``gpu/idle_ratio`` — idle / (idle + compute)
        - ``gpu/ir_{T}`` — fraction of utilization samples < T %
        """
        total = self._idle_time + self._compute_time
        idle_ratio = self._idle_time / total if total > 0 else 0.0

        metrics: Dict[str, float] = {
            "gpu/compute_time": round(self._compute_time, 4),
            "gpu/idle_time": round(self._idle_time, 4),
            "gpu/idle_ratio": round(idle_ratio, 4),
        }

        n_samples = len(self._util_samples)
        for t in self.idle_rate_thresholds:
            if n_samples > 0:
                count_below = sum(1 for u in self._util_samples if u < t)
                ir_value = count_below / n_samples
            else:
                ir_value = 0.0
            metrics[f"gpu/ir_{t}"] = round(ir_value, 4)

        return metrics

    def _start_sampling(self) -> None:
        if not self._nvml_initialized:
            return
        self._stop_event.clear()
        self._sampling = True
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()

    def _stop_sampling(self) -> None:
        if self._sampler_thread is not None and self._sampling:
            self._stop_event.set()
            self._sampler_thread.join(timeout=2.0)
            self._sampling = False
            self._sampler_thread = None

    def _sample_loop(self) -> None:
        """Periodically sample GPU utilization until stopped."""
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._util_samples.append(util.gpu)
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.poll_interval_s)

    def shutdown(self) -> None:
        """Stop sampler and release NVML."""
        self._stop_sampling()
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._nvml_initialized = False
