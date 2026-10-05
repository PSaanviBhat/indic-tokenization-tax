"""
GPU Memory Telemetry Tracker.
Measures baseline VRAM, model weight footprint, and peak allocation during TTS generation,
separating model weights from KV-cache and activation memory.
"""

import gc
import torch
from typing import Dict, Any, Optional


class GPUMemoryTracker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and "cuda" in device
        self.baseline_memory_gb = 0.0
        self.model_weights_memory_gb = 0.0

    def get_hardware_metadata(self) -> Dict[str, Any]:
        """Retrieve GPU metadata for reporting and reproducibility."""
        if not self.is_cuda:
            return {
                "device": "cpu",
                "gpu_name": "None (CPU Execution)",
                "total_vram_gb": 0.0,
            }
        
        props = torch.cuda.get_device_properties(self.device)
        total_vram_gb = props.total_memory / (1024 ** 3)
        return {
            "device": self.device,
            "gpu_name": props.name,
            "total_vram_gb": round(total_vram_gb, 2),
            "multi_processor_count": props.multi_processor_count,
        }

    def record_baseline(self) -> float:
        """Record VRAM allocated before loading model weights."""
        if not self.is_cuda:
            return 0.0
        gc.collect()
        torch.cuda.empty_cache()
        self.baseline_memory_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        return round(self.baseline_memory_gb, 4)

    def record_model_loaded(self) -> float:
        """Record VRAM after model weights are loaded on device."""
        if not self.is_cuda:
            return 0.0
        current_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        self.model_weights_memory_gb = current_alloc - self.baseline_memory_gb
        return round(self.model_weights_memory_gb, 4)

    def reset_peak_stats(self):
        """Reset PyTorch internal peak memory counters before a generation run."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_stats(self) -> Dict[str, float]:
        """
        Extract peak allocated and reserved memory during the last run.
        Separates model weights from transient generation memory (KV cache + activations).
        """
        if not self.is_cuda:
            return {
                "peak_allocated_gb": 0.0,
                "peak_reserved_gb": 0.0,
                "kv_and_activations_gb": 0.0,
                "model_weights_gb": 0.0,
            }

        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
        kv_activations = max(0.0, peak_allocated - (self.baseline_memory_gb + self.model_weights_memory_gb))

        return {
            "peak_allocated_gb": round(peak_allocated, 4),
            "peak_reserved_gb": round(peak_reserved, 4),
            "kv_and_activations_gb": round(kv_activations, 4),
            "model_weights_gb": round(self.model_weights_memory_gb, 4),
        }

    def clean_after_oom(self):
        """Emergency cleanup to recover CUDA context after an Out-Of-Memory error."""
        if self.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
