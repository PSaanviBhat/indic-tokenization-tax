"""
Main runner script for Step 1: Tokenization Tax Calculator and Analytical KV Scaling.
Executes on CPU, saves tables in CSV and Markdown, and generates visualization plots.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset

from src.tokenization_tax.calculator import TokenizationTaxCalculator
from src.tokenization_tax.memory_projection import KVMemoryProjector


def main():
    print("=" * 70)
    print(" STEP 1: TOKENIZATION TAX & KV CACHE SCALING ANALYSIS")
    print("=" * 70)

    # 1. Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # 2. Load Parallel Sentences from FLORES-200 (via Belebele test split)
    print("\n[1/4] Loading parallel FLORES-200 passages (English, Hindi, Telugu)...")
    num_samples = 300  # 300 parallel passages provides high statistical significance
    
    eng_ds = load_dataset("facebook/belebele", "eng_Latn", split=f"test[:{num_samples}]")
    hin_ds = load_dataset("facebook/belebele", "hin_Deva", split=f"test[:{num_samples}]")
    tel_ds = load_dataset("facebook/belebele", "tel_Telu", split=f"test[:{num_samples}]")

    parallel_data = {
        "English": [x["flores_passage"] for x in eng_ds],
        "Hindi": [x["flores_passage"] for x in hin_ds],
        "Telugu": [x["flores_passage"] for x in tel_ds],
    }
    print(f"Loaded {num_samples} parallel passages per language.")

    # 3. Evaluate Tokenization Tax
    print("\n[2/4] Running Tokenization Tax Calculator across tokenizers...")
    calc = TokenizationTaxCalculator()
    fertility_df = calc.evaluate_corpus(parallel_data)

    # Save fertility tables
    fertility_csv = "results/fertility_table.csv"
    fertility_md = "results/fertility_table.md"
    fertility_df.to_csv(fertility_csv, index=False)
    with open(fertility_md, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Fertility & Tokenization Tax Table\n\n")
        f.write(fertility_df.to_markdown(index=False))

    print("\n--- Measured Tokenization Tax Results ---")
    print(fertility_df.to_string(index=False))
    print(f"\nSaved fertility table to {fertility_csv} and {fertility_md}")

    # Extract measured inflation rates for Qwen
    qwen_rows = fertility_df[fertility_df["Tokenizer"].str.contains("Qwen")]
    inflation_rates = {}
    for _, row in qwen_rows.iterrows():
        inflation_rates[row["Language"]] = row["Word Inflation (x EN)"]
    print(f"\nEmpirically Measured Qwen Inflation Rates: {inflation_rates}")

    # 4. Generate KV-Cache Projections
    print("\n[3/4] Computing Analytical KV Cache Memory Projections...")
    projector = KVMemoryProjector()

    # Projection for Qwen3-0.6B (Proposal Target: 28 layers, 8 KV heads, dim 128 = 112 KB/tok fp16)
    proj_qwen3_df = projector.generate_projection_table(
        inflation_rates=inflation_rates,
        model_name="Qwen3-0.6B (Proposal Target)"
    )
    proj_qwen3_df.to_csv("results/kv_projection_qwen3.csv", index=False)
    with open("results/kv_projection_qwen3.md", "w", encoding="utf-8") as f:
        f.write("# Analytical KV Cache Projection: Qwen3-0.6B (112 KB/token in fp16)\n\n")
        # Write representative slice for markdown readability
        slice_df = proj_qwen3_df[proj_qwen3_df["Base Trace Len"].isin([1024, 2048])]
        f.write(slice_df.to_markdown(index=False))

    # Projection for Qwen2.5-0.5B (Local RTX 3050 Testbed: 24 layers, 2 KV heads, dim 64 = 12 KB/tok fp16)
    proj_qwen25_df = projector.generate_projection_table(
        inflation_rates=inflation_rates,
        model_name="Qwen2.5-0.5B (Active Local)"
    )
    proj_qwen25_df.to_csv("results/kv_projection_qwen25.csv", index=False)
    with open("results/kv_projection_qwen25.md", "w", encoding="utf-8") as f:
        f.write("# Analytical KV Cache Projection: Qwen2.5-0.5B (12 KB/token in fp16)\n\n")
        slice_df = proj_qwen25_df[proj_qwen25_df["Base Trace Len"].isin([1024, 2048])]
        f.write(slice_df.to_markdown(index=False))

    # 5. Generate Publication Plots
    print("\n[4/4] Generating Publication Figures for Slide Deck...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Figure 1: Fertility & Inflation Bar Chart
    plt.figure(figsize=(10, 5))
    bar_ax = sns.barplot(
        data=fertility_df,
        x="Tokenizer",
        y="Tokens / Word",
        hue="Language",
        palette="viridis"
    )
    plt.title("Tokenizer Fertility Comparison Across Languages (FLORES-200 Parallel Corpus)", fontsize=13, weight="bold")
    plt.ylabel("Tokens per Word (Lower is Better)")
    plt.xlabel("")
    for p in bar_ax.patches:
        height = p.get_height()
        if height > 0:
            bar_ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                            textcoords="offset points")
    plt.tight_layout()
    fert_plot_path = "results/fertility_comparison.png"
    plt.savefig(fert_plot_path, dpi=300)
    plt.close()
    print(f"Saved fertility bar plot to {fert_plot_path}")

    # Figure 2: KV Cache Memory Scaling vs K Paths (Qwen3-0.6B, 2048 trace length)
    plt.figure(figsize=(11, 6))
    subset = proj_qwen3_df[(proj_qwen3_df["Base Trace Len"] == 2048)]

    # Plot lines for English fp16, Telugu fp16, Telugu int8, Telugu int4
    k_vals = subset[subset["Precision"] == "fp16"]["K Paths"]
    en_fp16 = subset[subset["Precision"] == "fp16"]["English KV (GB)"]
    tel_fp16 = subset[subset["Precision"] == "fp16"]["Telugu KV (GB)"]
    tel_int8 = subset[subset["Precision"] == "int8"]["Telugu KV (GB)"]
    tel_int4 = subset[subset["Precision"] == "int4"]["Telugu KV (GB)"]

    plt.plot(k_vals, en_fp16, "o-", label="English (fp16)", color="#2b5c8f", linewidth=2.5)
    plt.plot(k_vals, tel_fp16, "s-", label=f"Telugu (fp16, {inflation_rates.get('Telugu', 2.6)}x Tax)", color="#d95f02", linewidth=2.5)
    plt.plot(k_vals, tel_int8, "^--", label="Telugu (8-bit KV Cache)", color="#7570b3", linewidth=2)
    plt.plot(k_vals, tel_int4, "d:", label="Telugu (4-bit KV Cache)", color="#1b9e77", linewidth=2)

    # Hardware memory limits
    plt.axhline(y=20.0, color="red", linestyle="--", linewidth=1.8, label="RTX 4090 Usable KV Limit (20 GB)")
    plt.axhline(y=4.5, color="purple", linestyle="--", linewidth=1.5, label="RTX 3050 Usable KV Limit (4.5 GB)")

    plt.title("The Tokenization Tax on KV Cache Memory (Qwen3-0.6B, 2k Base Length)", fontsize=13, weight="bold")
    plt.xlabel("Test-Time Sampling Paths (K)", fontsize=11)
    plt.ylabel("KV Cache Memory Footprint (GB)", fontsize=11)
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    kv_plot_path = "results/kv_memory_scaling_qwen3.png"
    plt.savefig(kv_plot_path, dpi=300)
    plt.close()
    print(f"Saved KV scaling plot to {kv_plot_path}")

    print("\n" + "=" * 70)
    print(" STEP 1 COMPLETE: All deliverables generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
