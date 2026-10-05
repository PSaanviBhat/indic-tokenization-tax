# The Tokenization Tax on Thinking: Latent vs. Explicit Test-Time Compute for Indic Reasoning Under Memory Constraints

**Current Phase:** Phase 0 / Phase 1 — Pillar B Foundation  

---

##  Executive Summary

Test-Time Scaling (TTS) enhances language model reasoning by generating $K$ independent candidate solution trajectories and selecting the best path via majority voting or verifiers. However, **every trajectory requires its own Key-Value (KV) cache**, causing memory to scale strictly with $K \times \text{trace length}$.

Under standard tokenizers, Indic scripts (like Hindi and Telugu) suffer from severe sub-optimal subword fragmentation, requiring **substantially more tokens** than English to express identical semantic meaning. This project quantifies this **"Tokenization Tax"** and evaluates memory-efficient alternatives (KV cache quantization, internal state branch pruning, and tokenizer-agnostic latent continuous reasoning).

---

##  Repository Structure (Step 1 Deliverable)

```text
phase_1/
├── src/
│   ├── __init__.py
│   └── tokenization_tax/
│       ├── __init__.py
│       ├── calculator.py         # Empirical fertility engine (tokens/word, char, byte)
│       └── memory_projection.py  # Analytical KV cache footprint & OOM boundary simulator
├── scripts/
│   └── run_fertility_check.py    # Main CPU runner: fetches FLORES-200, runs tests, plots figures
├── tests/
│   └── test_fertility.py         # Automated pytest/unit test verifying math & tokenizer contracts
├── results/
│   ├── fertility_table.csv       # Raw measured fertility & inflation data
│   ├── fertility_table.md        # Formatted markdown table for presentations
│   ├── kv_projection_qwen3.csv   # Target Qwen3-0.6B KV memory projections
│   ├── kv_projection_qwen3.md    # Target Qwen3-0.6B markdown table
│   ├── kv_projection_qwen25.csv  # Local Qwen2.5-0.5B KV memory projections
│   ├── kv_projection_qwen25.md   # Local Qwen2.5-0.5B markdown table
│   ├── fertility_comparison.png  # Publication-ready bar chart of tokenizer fertility
│   └── kv_memory_scaling_qwen3.png # Analytical memory curves vs K paths (log scale)
├── project_idea.MD               # Core research specification and protocol
├── Capstone_Project_Proposal.pdf # PES University official proposal document
├── capstone_execution_plan.pdf   # Phased timeline and milestones
├── One pager CAPSTONE.pdf        # One-page executive summary
└── README.md                     # This file
```

---

##  Detailed Breakdown: What Each File Does & Why It Was Built

### 1. `src/tokenization_tax/calculator.py`
* **What it does:** Implements the `TokenizationTaxCalculator` class. It loads tokenizers (`Qwen`, `Gemma`, `Sarvam-1`) and computes granular representation metrics over parallel text corpora.
* **Why it was built:** 
  - Simply counting "tokens per word" is insufficient and academically vulnerable in Indian languages because scripts like Telugu are highly agglutinative (words combine into long compounds) and Hindi uses complex vowel modifiers (*matras*).
  - This module calculates three complementary ratios:
    1. **Tokens / Word:** Standard readability metric.
    2. **Tokens / Character:** Measures fertility invariant to whitespace splitting conventions.
    3. **Tokens / Byte:** Measures efficiency relative to raw UTF-8 byte representation.
  - Computes the empirical **Inflation Factor** ($R_{\text{lang}} / R_{\text{en}}$) against the English baseline.

### 2. `src/tokenization_tax/memory_projection.py`
* **What it does:** Implements the `KVMemoryProjector` class to analytically compute KV cache memory requirements across sampling paths ($K \in \{1, 4, 8, 16, 32, 64\}$), base sequence lengths ($512, 1024, 2048, 4096$), and precisions (`fp16`, `int8`, `int4`, `int2`).
* **Why it was built:**
  - Connects tokenizer fertility directly to GPU memory (GB).
  - Uses the exact Grouped-Query Attention (GQA) KV memory formula:
    $$\text{KV Bytes per Token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_precision}$$
  - Evaluates memory against two hardware ceilings:
    - **RTX 3050 Laptop (6 GB VRAM):** Edge hardware ceiling (~4.5 GB usable for KV).
    - **RTX 4090 (24 GB VRAM):** Production target ceiling (~20.0 GB usable for KV).

### 3. `scripts/run_fertility_check.py`
* **What it does:** The primary execution pipeline.
  1. Pulls 300 semantically identical parallel sentences from **FLORES-200** (English, Hindi, Telugu via the `facebook/belebele` test split).
  2. Runs the `TokenizationTaxCalculator` across `Qwen-2.5`, `Gemma-2/3`, and `Sarvam-1`.
  3. Feeds the measured inflation rates into `KVMemoryProjector`.
  4. Exports structured tables to `results/` in both CSV and GitHub Markdown format.
  5. Uses `seaborn` and `matplotlib` to render high-resolution figures (`fertility_comparison.png` and `kv_memory_scaling_qwen3.png`).
* **Why it was built:** Serves as a single, reproducible command that produces the entire Phase 0 empirical deliverable without human intervention.

### 4. `tests/test_fertility.py`
* **What it does:** Contains unit tests verifying that:
  - Tokenizers encode and decode parallel strings accurately and non-trivially.
  - The GQA KV-cache byte formula matches theoretical specs (verifying that `Qwen3-0.6B` outputs exactly $114,688\text{ bytes} \approx 112\text{ KB/token}$ in `fp16` and half that in `int8`).
* **Why it was built:** Enforces software engineering rigor and test-driven reliability required for B.Tech capstone reviews.

---

##  How to Run the Code

### Step 1: Open Terminal and Activate Environment
Ensure you are using the dedicated Conda environment (`capstone`):
```powershell
conda activate capstone
```

### Step 2: Navigate to the Project Root
```powershell
cd d:\EBOOKS\Capstone\phase_1
```

### Step 3: Run the Complete Tokenization Tax Analysis
```powershell
# Set PYTHONPATH so Python can locate the 'src' package
$env:PYTHONPATH="d:\EBOOKS\Capstone\phase_1"

# Execute the runner script (Runs entirely on CPU in ~15-20 seconds)
python scripts/run_fertility_check.py
```

### Step 4: Run Unit Tests
```powershell
$env:PYTHONPATH="d:\EBOOKS\Capstone\phase_1"
python tests/test_fertility.py
```

---

##  Measured Findings for Review 1

### 1. Tokenizer Fertility Table (FLORES-200 Parallel Corpus)

| Tokenizer | Language | Total Tokens | Tokens / Word | Tokens / Char | Tokens / Byte | Word Inflation ($\times$ EN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen-2.5** *(151k vocab)* | **English** | 30,667 | **1.251** | 0.2079 | 0.2077 | **$1.00\times$** (Baseline) |
| **Qwen-2.5** *(151k vocab)* | **Hindi** | 136,058 | **4.794** | 0.9312 | 0.3634 | **$3.83\times$** |
| **Qwen-2.5** *(151k vocab)* | **Telugu** | 213,710 | **11.479** | 1.4496 | 0.5451 | **$9.18\times$** |
| **Gemma-2/3** *(256k vocab)*| **English** | 30,045 | **1.225** | 0.2036 | 0.2034 | **$1.00\times$** (Baseline) |
| **Gemma-2/3** *(256k vocab)*| **Hindi** | 54,758 | **1.935** | 0.3757 | 0.1467 | **$1.58\times$** |
| **Gemma-2/3** *(256k vocab)*| **Telugu** | 86,448 | **4.645** | 0.5868 | 0.2207 | **$3.79\times$** |
| **Sarvam-1** *(68k Indic vocab)* | **English** | 34,532 | **1.411** | 0.2342 | 0.2339 | **$1.00\times$** (Baseline) |
| **Sarvam-1** *(68k Indic vocab)* | **Hindi** | 39,838 | **1.408** | 0.2738 | 0.1070 | **$1.00\times$** |
| **Sarvam-1** *(68k Indic vocab)* | **Telugu** | 40,195 | **2.164** | 0.2737 | 0.1031 | **$1.53\times$** |

### 2. Analytical KV Scaling (Qwen3-0.6B, 2048 Base Trace Length)

| Precision | $K$ Paths | English KV | Hindi KV ($3.83\times$) | Telugu KV ($9.18\times$) | Telugu on RTX 3050 (6 GB)? | Telugu on RTX 4090 (24 GB)? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fp16** | 1 | 0.22 GB | 0.84 GB | 2.01 GB | **YES** | **YES** |
| **fp16** | 4 | 0.88 GB | 3.35 GB | 8.03 GB | **OOM** ($>4.5$ GB) | **YES** |
| **fp16** | 8 | 1.75 GB | 6.70 GB | 16.07 GB | **OOM** | **YES** |
| **fp16** | 16 | 3.50 GB | 13.41 GB | 32.13 GB | **OOM** | **OOM** ($>20$ GB) |
| **fp16** | 32 | 7.00 GB | 26.81 GB | 64.26 GB | **OOM** | **OOM** |
| **int8 (8-bit)** | 16 | 1.75 GB | 6.70 GB | 16.07 GB | **OOM** | **YES** *(Recovered)* |
| **int4 (4-bit)** | 32 | 1.75 GB | 6.70 GB | 16.07 GB | **OOM** | **YES** *(Recovered)* |

---

##  Strategic Key Takeaways for Review 1

1. **Empirical Proof of the Core Premise:**
   The literature estimated a $2\text{--}4\times$ token tax. On real parallel data, Qwen exhibits a **$3.83\times$** tax on Hindi and a massive **$9.18\times$** tax on Telugu.
2. **Tokenizer Allocation vs. Linguistic Difficulty:**
   Sarvam-1 achieves **$1.00\times$** (identical fertility to English) for Hindi and **$1.53\times$** for Telugu. This proves that Indic reasoning is hindered not by linguistic complexity, but by vocabulary starvation in standard multilingual tokenizers.
3. **The Test-Time Scaling Bottleneck:**
   At $K=16$ paths with a 2k trace, English requires only 3.5 GB of KV memory, while Telugu requires 32.13 GB, **completely breaching the 24 GB RTX 4090 hardware wall**.
4. **Quantization Recovery:**
   Quantizing the KV cache to 8-bit or 4-bit recovers significant headroom, allowing Telugu to scale up to $K=16$ and $K=32$ paths respectively.
