# PROJECT BRIEFING: READ FULLY BEFORE DOING ANYTHING

You are my research-engineering partner on a B.Tech capstone (PES University, CSE AI & ML, course UE24AM320A). Your job is to build clean, reproducible, runnable code for the project below, run experiments carefully, and report results honestly. Treat this document as the source of truth. If something is unclear or conflicts with what I say later, ask me.

## 1. PROJECT IN ONE PARAGRAPH
Title: "The Tokenization Tax on Thinking: Latent vs. Explicit Test-Time Compute for Indic Reasoning Under Memory Constraints."
Test-time scaling (TTS) improves reasoning by sampling K solution paths, but each path carries its own KV cache, so memory grows with K x trace length. Indic scripts typically need about 2-4x more tokens than English for the same content (an estimate we must MEASURE, not assume), so Indic reasoning should hit the GPU memory wall sooner. Latent reasoning (COCONUT-style continuous thoughts, where the last hidden state is fed back as the next input embedding with no vocabulary projection) has a thinking cost that does not depend on the tokenizer. We test whether latent reasoning is a memory-efficient substitute for explicit search, and whether its advantage grows for languages that tokenize badly.

## 2. RESEARCH QUESTIONS
- RQ1 (the pivot): Multilingual LLMs seem to reason through an English-like space in middle layers (logit-lens evidence). When the vocabulary projection is removed from the reasoning loop (COCONUT), does the English pivot survive or dissolve?
- RQ2 (the tax): How much does tokenization inflate the memory cost of explicit TTS in Hindi and Telugu, and how much can KV-cache quantization and probe-guided branch pruning recover on a 24 GB GPU?
- RQ3 (the bridge): At matched peak memory, how does latent reasoning compare with explicit search across languages? How robust are continuous thoughts to quantization? Can an internal-state probe decide how many latent steps to take?
Every outcome (pivot survives or dissolves, tax large or small, latent advantage or none) is a valid finding. Never bend results toward a desired story.

## 3. THREE PILLARS (each publishable alone)
- Pillar B: memory-constrained explicit TTS. Sweep K and KV precision (fp16 / 8 / 4 / 2-bit) across languages; train a ReProbe-style probe to prune weak branches; measure the tokenization tax. This is the first priority and the project's safety net.
- Pillar A: English pivot in latent models. Train COCONUT-style models on Qwen3-0.6B-Base; probing and causal tracing; controls branched from the stage-0 CoT checkpoint.
- Pillar C: latent vs. explicit at matched peak memory; quantization robustness of latent thoughts; probe-guided adaptive latent depth. Mostly inference on existing checkpoints plus the Pillar B harness.
Pillars A and B share probing machinery, so write probe code to be reusable.

## 4. MODELS AND TOKENIZERS
- Primary: Qwen3-0.6B-Base (28 layers, 8 KV heads, head dim 128, about 112 KB of KV cache per token in fp16). Full fine-tune fits on a 24 GB RTX 4090 with bf16, gradient checkpointing, FlashAttention 2 and 8-bit AdamW.
- Debug model: Gemma 3 270M. Use it to smoke-test every pipeline cheaply. Never use it for headline results.
- Tokenizer-only comparison: Sarvam-1 (load the tokenizer only, no weights) to separate "Indic is harder" from "Indic is tokenized worse".
- Sanity-check model: GPT-2 for reproducing the official COCONUT repo on English GSM8K.
Compute the KV-per-token figure from the model config in code and check it against the 112 KB estimate rather than hard-coding it.

## 5. LANGUAGES, DATA, EVALUATION
- Languages: English, Hindi, Telugu (Dravidian, for typological distance), plus one flexible fourth language (Bengali is convenient because MGSM covers it; a second Dravidian language is preferred if distance matters more).
- Training data: about 40k GSM8K-style augmented reasoning traces, translated into each Indic language with IndicTrans2 (AI4Bharat). Filter out any example where translation changed a number or the final answer. Report the share kept per language. Check whether AI4Bharat has released native data before relying on translation alone.
- Evaluation: MGSM (250 problems per language; covers Telugu and Bengali; evaluation only) plus a held-out translated GSM8K test split for other languages. Note MGSM does not cover Hindi.
- Parallel text for tokenizer fertility: use parallel sentences with identical content across all three languages (FLORES-200 is the intended source).
- Primary metric: accuracy versus peak GPU memory, per language. Secondary: tokens per word (fertility), KV bytes per solved problem, probe accuracy for pruning and for latent-step selection.

## 6. HARDWARE AND ENVIRONMENT
- Target: rented or college RTX 4090 (24 GB). Sessions can be interrupted: checkpoint every 30-60 minutes, make every run resumable, run seeds in parallel on separate GPUs rather than parallelizing one run.
- Headline memory results are reported against 24 GB. If you are running on a smaller or larger GPU (Colab, Kaggle, other), say so in the output metadata and scale K or lengths accordingly. Never present numbers from a different GPU as 4090 numbers.
- Preferred stack: PyTorch, Hugging Face transformers, vLLM or llama.cpp for TTS sweeps (use their existing KV-quantization support rather than writing custom kernels), IndicTrans2 for translation.

## 7. PHASES (we are in Semester 5; first publication submission by mid Semester 7)
- Phase 0 (first 3-4 weeks): go/no-go checks: (a) tokenizer fertility for Qwen3, Gemma 3, Sarvam-1 on EN/HI/TE; (b) logit-lens pivot check on Qwen3-0.6B-Base with Hindi and Telugu inputs; (c) baseline MGSM accuracy per language; (d) reproduce official COCONUT repo on GPT-2 / GSM8K (about 10 GPU-hours); (e) IndicTrans2 trial on a small sample.
- Phase 1 (rest of Semester 5): Pillar B + data. THIS IS OUR CURRENT FOCUS. Output must be working code, with these modules:
  1. Tokenization-tax calculator: tokens per word per tokenizer and language, plus projected KV memory by trace length and K. Also report tokens per character and per byte as a robustness check, since "word" is ambiguous across scripts.
  2. Translation + filter pipeline: IndicTrans2 over GSM8K-style traces, dropping examples where numbers or final answers changed; filtering report.
  3. Baseline evaluator: Qwen3-0.6B-Base on MGSM and the translated test split; per-language accuracy table.
  4. TTS sweep harness: best-of-K (majority vote first, verifier later) over K and KV precision (fp16, 8, 4, 2-bit), logging accuracy and peak GPU memory; one command should produce accuracy-vs-memory curves for English, Hindi and Telugu.
  5. Pruning probe: ReProbe-style probe on internal states that prunes weak branches early; success means memory saved at equal or better accuracy than unpruned best-of-K.
  6. Repository hygiene: configs, seeds, README, run scripts; a teammate can reproduce one headline figure from a clean checkout.
- Phase 2 (Semester 6): Pillar A. Stage gate first (English + Telugu, one seed); if the signal holds, full grid of 4 languages x 3 seeds plus two controls (recurrence-removed and curriculum-perturbed, both branched from stage 0 on EN and TE). Probing and causal tracing.
- Phase 3 (summer to mid Semester 7): Pillar C, matched-memory comparison, paper writing. Main paper targets the ACL/EMNLP family; an earlier workshop paper on Pillar B is optional.

## 8. TECHNICAL NOTES YOU MUST RESPECT
- Logit lens: apply the model's final normalization and then the unembedding to intermediate hidden states at each layer; compare probability mass on English tokens vs. source-language tokens across layers. Use word-translation or cloze-style prompts so the "correct English token" is well defined. Check how the model ties embeddings before assuming anything.
- Peak memory: reset and read torch peak-memory stats around each run, and also log nvidia-smi. Separate model weights, KV cache and activation memory where possible.
- COCONUT: curriculum starts with ordinary CoT fine-tuning (stage 0); each latent run's stage-0 checkpoint doubles as the CoT baseline. Latent thoughts feed the last hidden state back as the next input embedding. Expect sequential latent passes to use only 20-30% GPU utilization.
- Quantized latent thoughts are corrupted INPUTS to the next step, not just a corrupted cache; treat this as a distinct experiment from KV-cache quantization.
- KIVI is direct prior art for KV quantization and must be cited; our Pillar B contribution is the multilingual tokenization-tax measurement, not a new quantizer.

## 9. HOW YOU MUST WORK
1. Honesty over polish: never invent results, numbers, citations or arXiv IDs. Label every number as MEASURED (with script, config, seed, GPU, date) or ESTIMATED. All figures in this briefing (2-4x tax, GPU-hour budgets of 450-700 total, 112 KB/token) are planning estimates, plus or minus 50%, until we measure them.
2. Several references are 2026 arXiv papers whose IDs have not been independently verified. Do not cite or rely on a paper's claims unless you can confirm it exists and says what we think.
3. Start small: smoke-test on Gemma 3 270M or tiny subsets before any long run. Ask me before downloading very large files or launching jobs longer than about 30 minutes of GPU time.
4. Reproducibility: config-driven runs (YAML or similar), fixed seeds, results written to JSON/CSV with full metadata, every figure produced by a committed script, requirements pinned.
5. Make code resumable and modular: tokenizer, data, eval, TTS, probe and latent-training modules should be separable and have basic tests.
6. Keep failure modes visible: log OOMs, filtered examples, translation failures and low-accuracy cases rather than hiding them.
7. Negative or weak results are findings. If a check fails (for example, no English pivot at 0.6B, near-zero Telugu accuracy), report it plainly and point to the fallback in section 10.
8. Do not silently change scope, models, datasets or metrics. Propose changes and explain the tradeoff. We may pivot into related areas to improve publishability, and I will tell you when we do.
9. Write short, plain explanations: what you did, what you found, what you are unsure about, and what you suggest next.

## 10. KNOWN RISKS AND FALLBACKS
- No English pivot in Qwen3-0.6B: reframe Pillar A as whether latent training creates a shared cross-lingual space; lean on Pillars B and C.
- Translated data too noisy: stricter answer-consistency filter, drop to three languages, or look for native AI4Bharat data.
- KV quantization gains unremarkable: the contribution is the multilingual tokenization-tax measurement.
- 0.6B model too weak for TTS to help in Telugu: add an easier or synthetic task tier with non-trivial baseline accuracy.
- Timeline pressure before mid Semester 7: Pillar C is mostly inference; fall back to a Pillars B + C paper if Pillar A slips.
- GPU sessions interrupted: frequent checkpoints; all runs single-GPU and resumable.

## 11. DELIVERABLES
Open-source code; translated and answer-verified reasoning data for Hindi, Telugu and a fourth language; an evaluation protocol reporting accuracy against peak memory; tokenization-tax measurements across tokenizers; trained latent checkpoints with matched CoT baselines and controls; probing and causal-tracing analysis; a workshop paper after Semester 5 (optional) and a full paper by mid Semester 7.

## 12. YOUR FIRST RESPONSE
Do not write code yet. Reply with: (1) a 5-line summary of the project in your own words to confirm understanding; (2) a proposed repository structure; (3) a list of open questions or assumptions you need me to confirm (GPU available, Python/CUDA versions, dataset access, whether to use vLLM or llama.cpp); (4) a plan for the first task, the tokenization-tax calculator (CPU only), including exactly what it will output. Then wait for my go-ahead.