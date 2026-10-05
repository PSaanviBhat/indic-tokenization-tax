"""
Unit tests for GPU memory tracker and sweeper components.
"""

import torch
from src.tts.memory_tracker import GPUMemoryTracker
from src.tts.memory_wall_sweep import MemoryWallSweeper


def test_memory_tracker_basics():
    tracker = GPUMemoryTracker(device="cuda" if torch.cuda.is_available() else "cpu")
    meta = tracker.get_hardware_metadata()
    assert "device" in meta
    assert "gpu_name" in meta
    assert "total_vram_gb" in meta

    baseline = tracker.record_baseline()
    assert baseline >= 0.0

    stats = tracker.get_peak_stats()
    assert "peak_allocated_gb" in stats
    assert "kv_and_activations_gb" in stats
    print("test_memory_tracker_basics passed!")


def test_sweeper_instantiation():
    sweeper = MemoryWallSweeper(
        model_id="Qwen/Qwen2.5-0.5B",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    assert sweeper.model_id == "Qwen/Qwen2.5-0.5B"
    print("test_sweeper_instantiation passed!")


if __name__ == "__main__":
    test_memory_tracker_basics()
    test_sweeper_instantiation()
