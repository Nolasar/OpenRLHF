"""GPU utilization metrics tracker for OpenRLHF training loops.

Tracks compute time, idle time, idle ratio, and threshold-based idle rate
by sampling GPU utilization in a background thread via pynvml.
Automatically detects GPUs from CUDA_VISIBLE_DEVICES and supports multi-GPU tracking.
"""

import os
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


def _get_gpu_indices() -> List[int]:
    """Parse CUDA_VISIBLE_DEVICES into a list of physical GPU indices.

    Returns:
        List of GPU indices to monitor.  Falls back to ``[0]`` when the
        environment variable is unset or empty.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible.strip():
        return [0]
    try:
        return [int(idx.strip()) for idx in cuda_visible.split(",") if idx.strip()]
    except ValueError:
        logger.warning(
            f"GpuMetricsTracker: failed to parse CUDA_VISIBLE_DEVICES='{cuda_visible}'; falling back to GPU 0"
        )
        return [0]


class GpuMetricsTracker:
    """Tracks GPU compute/idle timing and utilization-based idle rates.

    Automatically detects GPUs from ``CUDA_VISIBLE_DEVICES`` and monitors
    all of them.  Reports both per-GPU and averaged utilization metrics.

    Usage::

        tracker = GpuMetricsTracker(idle_rate_thresholds=[90])

        # -- training loop --
        tracker.step_start()
        train_step(...)
        tracker.step_end()

        metrics = tracker.get_metrics()
        # With CUDA_VISIBLE_DEVICES=0,1 and threshold 90:
        # metrics == {
        #     "gpu/compute_time": 12.3,
        #     "gpu/idle_time": 1.5,
        #     "gpu/idle_ratio": 0.108,
        #     "gpu/ir_90": 0.25,       # averaged across GPUs
        #     "gpu_0/ir_90": 0.20,     # per-GPU
        #     "gpu_1/ir_90": 0.30,     # per-GPU
        # }

    Args:
        idle_rate_thresholds: List of utilization thresholds for ``ir_T`` metrics.
            Each value *T* produces a metric ``gpu/ir_{T}`` equal to the
            fraction of utilization samples below *T* %.
        poll_interval_s: How often (seconds) to sample GPU utilization.
    """

    def __init__(
        self,
        idle_rate_thresholds: Optional[List[int]] = None,
        poll_interval_s: float = 0.1,
    ):
        self.idle_rate_thresholds = idle_rate_thresholds or [90]
        self.poll_interval_s = poll_interval_s

        self._gpu_indices: List[int] = []
        self._handles: Dict[int, object] = {}
        self._nvml_initialized = False

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_indices = _get_gpu_indices()
                for idx in self._gpu_indices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self._handles[idx] = handle
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    logger.info(f"GpuMetricsTracker: monitoring GPU {idx} ({name})")
                self._nvml_initialized = True
            except pynvml.NVMLError as e:
                logger.warning(f"GpuMetricsTracker: pynvml init failed ({e}); GPU metrics disabled")
        else:
            logger.warning("GpuMetricsTracker: pynvml not available; GPU metrics disabled")

        self._step_start_time: Optional[float] = None
        self._prev_step_end_time: Optional[float] = None
        self._compute_time: float = 0.0
        self._idle_time: float = 0.0

        # Per-GPU utilization samples: {gpu_index: [util_pct, ...]}
        self._util_samples: Dict[int, List[int]] = {idx: [] for idx in self._gpu_indices}

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

        self._util_samples = {idx: [] for idx in self._gpu_indices}
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
        - ``gpu/ir_{T}`` — fraction of utilization samples < T %, averaged across GPUs
        - ``gpu_{i}/ir_{T}`` — per-GPU fraction of utilization samples < T %
        """
        total = self._idle_time + self._compute_time
        idle_ratio = self._idle_time / total if total > 0 else 0.0

        metrics: Dict[str, float] = {
            "gpu/compute_time": round(self._compute_time, 4),
            "gpu/idle_time": round(self._idle_time, 4),
            "gpu/idle_ratio": round(idle_ratio, 4),
        }

        for t in self.idle_rate_thresholds:
            per_gpu_ir: List[float] = []
            for idx in self._gpu_indices:
                samples = self._util_samples.get(idx, [])
                n_samples = len(samples)
                if n_samples > 0:
                    count_below = sum(1 for u in samples if u < t)
                    ir_value = count_below / n_samples
                else:
                    ir_value = 0.0
                metrics[f"gpu_{idx}/ir_{t}"] = round(ir_value, 4)
                per_gpu_ir.append(ir_value)

            # Averaged ir across all GPUs
            if per_gpu_ir:
                metrics[f"gpu/ir_{t}"] = round(sum(per_gpu_ir) / len(per_gpu_ir), 4)
            else:
                metrics[f"gpu/ir_{t}"] = 0.0

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
        """Periodically sample GPU utilization for all monitored GPUs."""
        while not self._stop_event.is_set():
            for idx, handle in self._handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self._util_samples[idx].append(util.gpu)
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
