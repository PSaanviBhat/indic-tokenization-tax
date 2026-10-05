"""
Runner script for Step 3: Logit-Lens Pivot Check (Pillar A Go/No-Go Gate).
Projects intermediate layer representations through final RMSNorm + lm_head,
tracks emergence of the English pivot across layers, and generates heatmaps & trajectory plots.
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

from src.interpretability.logit_lens import LogitLensAnalyzer


def main():
    print("=" * 70)
    print(" STEP 3: LOGIT-LENS MULTILINGUAL PIVOT CHECK (PILLAR A)")
    print("=" * 70)

    # 1. Load config
    config_path = "configs/logit_lens_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    # 2. Initialize LogitLens Analyzer
    analyzer = LogitLensAnalyzer(
        model_id=config["model"]["id"],
        device=config["model"]["device"],
        torch_dtype=torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.bfloat16,
    )
    analyzer.load_model()

    # 3. Analyze Test Cases
    test_cases = config["test_cases"]
    print(f"Running Logit-Lens across {len(test_cases)} multilingual cloze test cases...")
    df = analyzer.analyze_dataset(test_cases)

    # Save detailed per-concept results
    detailed_csv = "results/logit_lens_detailed.csv"
    df.to_csv(detailed_csv, index=False)

    # 4. Aggregate layer-wise trajectory across languages
    summary = df.groupby(["Language", "Layer Index"]).agg({
        "Prob English Target": "mean",
        "Prob Native Target": "mean",
        "English vs Native Ratio": "mean",
    }).reset_index()

    summary_csv = "results/logit_lens_summary.csv"
    summary_md = "results/logit_lens_summary.md"
    summary.to_csv(summary_csv, index=False)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Logit-Lens Multilingual Pivot Analysis Summary\n\n")
        f.write(f"- **Model:** `{config['model']['id']}` (`{config['model']['torch_dtype']}`)\n")
        f.write(f"- **Method:** Intermediate Hidden State $\\to$ Final RMSNorm $\\to$ LM Head Projection\n\n")
        f.write(summary.to_markdown(index=False))

    print("\n--- Layer-wise Pivot Trajectory Summary ---")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_csv} and {summary_md}")

    # 5. Generate Figures
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Layer-wise Trajectory (English vs Native Probability across Layers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, lang in enumerate(["Hindi", "Telugu"]):
        ax = axes[idx]
        sub = summary[summary["Language"] == lang]

        ax.plot(
            sub["Layer Index"],
            sub["Prob English Target"],
            "o-",
            label="English Target Token Prob",
            color="#2b5c8f",
            linewidth=2.2,
        )
        ax.plot(
            sub["Layer Index"],
            sub["Prob Native Target"],
            "s--",
            label="Native Target Token Prob",
            color="#d95f02",
            linewidth=2.2,
        )

        ax.set_title(f"{lang} Prompts: Target Token Probability by Layer", fontsize=12, weight="bold")
        ax.set_xlabel("Transformer Layer Index (0 = Embedding, 24 = Final)", fontsize=10)
        ax.set_ylabel("Probability Mass" if idx == 0 else "")
        ax.axvspan(8, 16, color="yellow", alpha=0.15, label="Middle Layers (Pivot Zone)")
        ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    traj_plot_path = "results/logit_lens_trajectory.png"
    plt.savefig(traj_plot_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {traj_plot_path}")

    # Figure 2: Concept Heatmap (Probability of English Target across Concepts & Layers)
    pivot_table = df.pivot_table(
        index=["Language", "Concept"],
        columns="Layer Index",
        values="Prob English Target"
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_table,
        cmap="Blues",
        cbar_kws={"label": "P(English Target)"},
        linewidths=0.5,
    )
    plt.title("Logit-Lens Heatmap: English Token Emergence in Hidden States", fontsize=13, weight="bold")
    plt.xlabel("Transformer Layer Index", fontsize=11)
    plt.ylabel("Language & Concept", fontsize=11)
    plt.tight_layout()

    heatmap_path = "results/logit_lens_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    print("\n" + "=" * 70)
    print(" STEP 3 COMPLETE: Logit-Lens analysis and visual plots ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
