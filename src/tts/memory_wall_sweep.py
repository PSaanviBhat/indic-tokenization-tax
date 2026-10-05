"""
Empirical Memory Wall Sweeper for Test-Time Scaling (TTS).
Runs batched generation of K candidate paths across English, Hindi, and Telugu,
logging peak GPU memory until hardware OOM boundaries are hit.
"""

import gc
import torch
import pandas as pd
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory_tracker import GPUMemoryTracker


class MemoryWallSweeper:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.tracker = GPUMemoryTracker(device=self.device)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load tokenizer and model weights while tracking VRAM footprint."""
        print(f"Initializing MemoryWallSweeper with model: {self.model_id}")
        self.tracker.record_baseline()

        print("  -> Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  -> Loading model weights onto {self.device} ({self.torch_dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        weights_gb = self.tracker.record_model_loaded()
        hw_info = self.tracker.get_hardware_metadata()
        print(f"Model successfully loaded. Weights footprint: {weights_gb:.2f} GB on {hw_info.get('gpu_name')}\n")

    def run_sweep(
        self,
        prompts: Dict[str, str],
        k_values: List[int] = [1, 2, 4, 8, 12, 16, 20, 24],
        max_new_tokens: int = 256,
    ) -> pd.DataFrame:
        """
        Execute memory sweep across languages and K paths.
        Catches OOM errors cleanly and records the exact failure boundary.
        """
        if self.model is None:
            self.load_model()

        hw = self.tracker.get_hardware_metadata()
        records = []

        print("=" * 70)
        print(f" STARTING MEMORY WALL SWEEP on {hw.get('gpu_name')} ({hw.get('total_vram_gb')} GB VRAM)")
        print(f" Target Max New Tokens per Path: {max_new_tokens}")
        print("=" * 70)

        for lang, prompt_text in prompts.items():
            print(f"\n>>> Profiling Language: {lang}")
            
            # Encode single prompt to measure input token length
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            prompt_tokens_len = encoded["input_ids"].shape[1]
            print(f"    Prompt Length: {prompt_tokens_len} tokens")

            oom_hit = False
            for k in k_values:
                if oom_hit:
                    # Skip higher K once OOM has been reached for this language
                    records.append({
                        "Language": lang,
                        "K Paths": k,
                        "Prompt Tokens": prompt_tokens_len,
                        "Max Gen Tokens": max_new_tokens,
                        "Total Trajectory Tokens": (prompt_tokens_len + max_new_tokens) * k,
                        "Peak VRAM (GB)": None,
                        "KV & Act (GB)": None,
                        "Status": "OOM (Skipped)",
                    })
                    continue

                print(f"    Testing K = {k:2d} ... ", end="", flush=True)

                # Batch prompt K times to simulate K parallel test-time reasoning trajectories
                batch_input_ids = encoded["input_ids"].repeat(k, 1).to(self.device)
                batch_attention_mask = encoded["attention_mask"].repeat(k, 1).to(self.device)

                self.tracker.reset_peak_stats()

                try:
                    with torch.no_grad():
                        _ = self.model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    stats = self.tracker.get_peak_stats()
                    peak_gb = stats["peak_allocated_gb"]
                    kv_act_gb = stats["kv_and_activations_gb"]

                    print(f"SUCCESS | Peak VRAM: {peak_gb:.2f} GB (KV+Act: {kv_act_gb:.2f} GB)")

                    records.append({
                        "Language": lang,
                        "K Paths": k,
                        "Prompt Tokens": prompt_tokens_len,
                        "Max Gen Tokens": max_new_tokens,
                        "Total Trajectory Tokens": (prompt_tokens_len + max_new_tokens) * k,
                        "Peak VRAM (GB)": peak_gb,
                        "KV & Act (GB)": kv_act_gb,
                        "Status": "COMPLETED",
                    })

                except torch.cuda.OutOfMemoryError:
                    print(f"FAILED (CUDA OOM!)")
                    self.tracker.clean_after_oom()
                    oom_hit = True

                    records.append({
                        "Language": lang,
                        "K Paths": k,
                        "Prompt Tokens": prompt_tokens_len,
                        "Max Gen Tokens": max_new_tokens,
                        "Total Trajectory Tokens": (prompt_tokens_len + max_new_tokens) * k,
                        "Peak VRAM (GB)": hw.get("total_vram_gb"),
                        "KV & Act (GB)": None,
                        "Status": "OOM (Memory Wall)",
                    })
                except Exception as e:
                    print(f"ERROR ({e})")
                    records.append({
                        "Language": lang,
                        "K Paths": k,
                        "Prompt Tokens": prompt_tokens_len,
                        "Max Gen Tokens": max_new_tokens,
                        "Total Trajectory Tokens": (prompt_tokens_len + max_new_tokens) * k,
                        "Peak VRAM (GB)": None,
                        "KV & Act (GB)": None,
                        "Status": f"ERROR: {str(e)[:30]}",
                    })

                # Clean garbage collection between K runs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        df = pd.DataFrame(records)
        return df
