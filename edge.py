"""Edge-simulation helpers: latency, FPS, memory footprint and model size."""
from __future__ import annotations

import os
import time

import requests

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def rss_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class EdgeMonitor:
    """Context manager measuring wall-clock latency and process memory around inference."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.mem_before = rss_mb()
        return self

    def __exit__(self, *exc):
        self.latency_ms = (time.perf_counter() - self.t0) * 1000
        self.mem_after = rss_mb()
        return False

    def metrics(self) -> dict:
        return {
            "latency_ms": round(self.latency_ms, 1),
            "fps": round(1000 / self.latency_ms, 3) if self.latency_ms else 0.0,
            "app_memory_mb": round(self.mem_after, 1),
            "app_memory_delta_mb": round(self.mem_after - self.mem_before, 1),
        }


def ollama_model_info(host: str, model: str) -> dict:
    """On-disk (quantised) size and resident size of an Ollama model, if the server is reachable."""
    info = {"model_disk_mb": None, "model_loaded_mb": None, "quantization": None}
    try:
        tags = requests.get(f"{host}/api/tags", timeout=3).json().get("models", [])
        for m in tags:
            if m.get("name", "").startswith(model):
                info["model_disk_mb"] = round(m.get("size", 0) / 1e6, 0)
                info["quantization"] = m.get("details", {}).get("quantization_level")
        for m in requests.get(f"{host}/api/ps", timeout=3).json().get("models", []):
            if m.get("name", "").startswith(model):
                info["model_loaded_mb"] = round(m.get("size", 0) / 1e6, 0)
    except Exception:
        pass
    return info
