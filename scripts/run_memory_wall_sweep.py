"""
Runner script for Step 2: Empirical Memory Wall Curve.
Profiles Qwen on GPU across K sampling paths for English, Hindi, and Telugu,
logs peak VRAM until OOM occurs, and plots the empirical memory wall.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.tts.memory_wall_sweep import MemoryWallSweeper


def main():
    print("=" * 70)
    print(" STEP 2: EMPIRICAL MEMORY WALL SWEEP (TEST-TIME SCALING)")
    print("=" * 70)

    # 1. Load configuration
    config_path = "configs/memory_wall_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    # 2. Initialize Sweeper
    sweeper = MemoryWallSweeper(
        model_id=config["model"]["id"],
        device=config["model"]["device"],
        torch_dtype=torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.bfloat16,
    )

    # 3. Execute Sweep
    prompts = config["prompts"]
    k_values = config["sweep"]["k_values"]
    max_new_tokens = config["sweep"]["max_new_tokens"]

    results_df = sweeper.run_sweep(
        prompts=prompts,
        k_values=k_values,
        max_new_tokens=max_new_tokens,
    )

    # 4. Save structured outputs
    csv_path = "results/memory_wall_empirical.csv"
    md_path = "results/memory_wall_empirical.md"
    results_df.to_csv(csv_path, index=False)

    hw = sweeper.tracker.get_hardware_metadata()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Memory Wall Sweep Results\n\n")
        f.write(f"- **Hardware Profile:** {hw.get('gpu_name')} ({hw.get('total_vram_gb')} GB VRAM)\n")
        f.write(f"- **Model:** `{config['model']['id']}` (`{config['model']['torch_dtype']}`)\n")
        f.write(f"- **Max New Tokens:** {max_new_tokens}\n\n")
        f.write(results_df.to_markdown(index=False))

    print("\n--- Empirical Memory Wall Summary Table ---")
    print(results_df.to_string(index=False))
    print(f"\nSaved empirical results to {csv_path} and {md_path}")

    # 5. Plot Empirical Memory Wall Curve
    print("\nGenerating Empirical Memory Wall Plot...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    colors = {"English": "#2b5c8f", "Hindi": "#e7298a", "Telugu": "#d95f02"}
    markers = {"English": "o", "Hindi": "^", "Telugu": "s"}

    for lang in prompts.keys():
        lang_data = results_df[results_df["Language"] == lang]
        completed = lang_data[lang_data["Status"] == "COMPLETED"]
        oom = lang_data[lang_data["Status"] == "OOM (Memory Wall)"]

        # Plot completed trajectory line
        if not completed.empty:
            plt.plot(
                completed["K Paths"],
                completed["Peak VRAM (GB)"],
                linestyle="-",
                linewidth=2.5,
                marker=markers[lang],
                markersize=7,
                label=f"{lang} (Completed)",
                color=colors[lang],
            )

        # Mark OOM points with an 'X'
        if not oom.empty:
            plt.scatter(
                oom["K Paths"],
                oom["Peak VRAM (GB)"],
                color="red",
                s=120,
                marker="X",
                zorder=5,
                label=f"{lang} OOM Point" if lang == "Telugu" else None,
            )

    # Hardware VRAM threshold line
    total_vram = hw.get("total_vram_gb", 6.0)
    plt.axhline(
        y=total_vram,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Hardware Memory Ceiling ({hw.get('gpu_name')}: {total_vram} GB)",
    )

    plt.title(f"Empirical Memory Wall: Peak VRAM vs. K Paths ({config['model']['id']})", fontsize=13, weight="bold")
    plt.xlabel("Candidate Reasoning Paths (K)", fontsize=11)
    plt.ylabel("Peak GPU Memory Allocated (GB)", fontsize=11)
    plt.ylim(0, total_vram + 0.8)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    plot_path = "results/memory_wall_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved memory wall plot to {plot_path}")

    print("\n" + "=" * 70)
    print(" STEP 2 EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
