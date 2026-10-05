"""
KV-Cache Memory Projector.
Calculates KV cache memory requirements across languages, sequence lengths, K trajectories,
and precisions (fp16, 8-bit, 4-bit, 2-bit), showing the impact of the tokenization tax.
"""

import pandas as pd
from typing import Dict, List, Optional


class KVMemoryProjector:
    def __init__(self):
        # Hardware memory budgets in GB (leaving headroom for model weights & activations)
        self.hardware_budgets = {
            "RTX 3050 (6 GB total)": 4.5,    # ~1.0 GB model weights + activations
            "RTX 4090 (24 GB total)": 20.0,  # ~2.5 GB model weights + activations
        }

        # Model architecture profiles
        self.model_profiles = {
            "Qwen2.5-0.5B (Active Local)": {
                "num_layers": 24,
                "num_kv_heads": 2,
                "head_dim": 64,
                # 2 * 24 * 2 * 64 * 2 bytes = 12,288 bytes = 12.0 KB / token in fp16
            },
            "Qwen3-0.6B (Proposal Target)": {
                "num_layers": 28,
                "num_kv_heads": 8,
                "head_dim": 128,
                # 2 * 28 * 8 * 128 * 2 bytes = 114,688 bytes = 112.0 KB / token in fp16
            },
        }

        self.precision_bytes = {
            "fp16": 2.0,
            "int8": 1.0,
            "int4": 0.5,
            "int2": 0.25,
        }

    def compute_kv_per_token_bytes(self, model_name: str, precision: str = "fp16") -> float:
        """Calculate bytes per token stored in KV cache."""
        prof = self.model_profiles[model_name]
        bytes_per_elem = self.precision_bytes[precision]
        # 2 represents both Key and Value matrices
        return 2 * prof["num_layers"] * prof["num_kv_heads"] * prof["head_dim"] * bytes_per_elem

    def generate_projection_table(
        self,
        inflation_rates: Dict[str, float],
        base_lengths: List[int] = [512, 1024, 2048, 4096],
        k_values: List[int] = [1, 4, 8, 16, 32, 64],
        precisions: List[str] = ["fp16", "int8", "int4", "int2"],
        model_name: str = "Qwen3-0.6B (Proposal Target)",
    ) -> pd.DataFrame:
        """
        Generate projected KV cache footprint in GB across languages, K, lengths, and precisions.
        inflation_rates: e.g., {'English': 1.0, 'Hindi': 2.1, 'Telugu': 2.6}
        """
        records = []
        for prec in precisions:
            bytes_per_token = self.compute_kv_per_token_bytes(model_name, prec)
            kb_per_token = bytes_per_token / 1024.0

            for base_len in base_lengths:
                for k in k_values:
                    row = {
                        "Model": model_name,
                        "Precision": prec,
                        "KB / Token": round(kb_per_token, 2),
                        "Base Trace Len": base_len,
                        "K Paths": k,
                    }
                    for lang, inflation in inflation_rates.items():
                        eff_len = base_len * inflation
                        total_tokens = eff_len * k
                        total_bytes = total_tokens * bytes_per_token
                        total_gb = total_bytes / (1024 ** 3)
                        row[f"{lang} KV (GB)"] = round(total_gb, 3)
                        
                        # Add fit status for 3050 and 4090
                        if lang.lower() == "telugu":
                            fits_3050 = "YES" if total_gb <= self.hardware_budgets["RTX 3050 (6 GB total)"] else "OOM"
                            fits_4090 = "YES" if total_gb <= self.hardware_budgets["RTX 4090 (24 GB total)"] else "OOM"
                            row["Telugu Fits 6GB?"] = fits_3050
                            row["Telugu Fits 24GB?"] = fits_4090

                    records.append(row)

        return pd.DataFrame(records)
