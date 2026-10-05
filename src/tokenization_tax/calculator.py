"""
Tokenizer fertility calculator comparing English, Hindi, and Telugu across tokenizers.
Calculates tokens per word, tokens per character, and tokens per byte.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from transformers import AutoTokenizer


class TokenizationTaxCalculator:
    def __init__(self, tokenizer_mapping: Dict[str, str] = None):
        """
        Initialize with a mapping of display names to HuggingFace model IDs.
        Default includes Qwen, Gemma, and Sarvam-1.
        """
        if tokenizer_mapping is None:
            self.tokenizer_mapping = {
                "Qwen-2.5 (151k vocab)": "Qwen/Qwen2.5-0.5B",
                "Gemma-2/3 (256k vocab)": "unsloth/gemma-2-2b",
                "Sarvam-1 (68k vocab)": "sarvamai/sarvam-1",
            }
        else:
            self.tokenizer_mapping = tokenizer_mapping

        self.tokenizers = {}

    def load_tokenizers(self):
        """Load all tokenizers into memory."""
        print("Loading tokenizers...")
        for name, hf_id in self.tokenizer_mapping.items():
            print(f"  -> Loading {name} ({hf_id})...")
            self.tokenizers[name] = AutoTokenizer.from_pretrained(
                hf_id, 
                trust_remote_code=True
            )
        print("All tokenizers loaded successfully.\n")

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words using whitespace and punctuation boundary splitting.
        Handles both Latin and Indic scripts.
        """
        # Split on whitespace; ensure non-empty tokens
        tokens = [w for w in re.split(r"\s+", text.strip()) if w]
        return max(len(tokens), 1)

    @staticmethod
    def count_chars(text: str) -> int:
        """Count characters excluding leading/trailing whitespaces."""
        return max(len(text.strip()), 1)

    @staticmethod
    def count_bytes(text: str) -> int:
        """Count raw UTF-8 encoded bytes."""
        return max(len(text.strip().encode("utf-8")), 1)

    def evaluate_corpus(self, parallel_data: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Evaluate tokenization statistics across parallel sentences.
        parallel_data: Dict where keys are language names (e.g., 'English', 'Hindi', 'Telugu')
                       and values are aligned lists of strings.
        """
        if not self.tokenizers:
            self.load_tokenizers()

        records = []
        languages = list(parallel_data.keys())
        num_samples = len(next(iter(parallel_data.values())))

        print(f"Evaluating {num_samples} parallel sentences across {len(languages)} languages...")

        for tok_name, tokenizer in self.tokenizers.items():
            en_tokens_per_word = None
            en_tokens_per_char = None
            en_tokens_per_byte = None

            for lang in languages:
                sentences = parallel_data[lang]
                
                total_tokens = 0
                total_words = 0
                total_chars = 0
                total_bytes = 0

                per_sentence_tpw = []
                per_sentence_tpc = []
                per_sentence_tpb = []

                for text in sentences:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    n_tokens = len(tokens)
                    n_words = self.count_words(text)
                    n_chars = self.count_chars(text)
                    n_bytes = self.count_bytes(text)

                    total_tokens += n_tokens
                    total_words += n_words
                    total_chars += n_chars
                    total_bytes += n_bytes

                    per_sentence_tpw.append(n_tokens / n_words)
                    per_sentence_tpc.append(n_tokens / n_chars)
                    per_sentence_tpb.append(n_tokens / n_bytes)

                mean_tpw = float(np.mean(per_sentence_tpw))
                mean_tpc = float(np.mean(per_sentence_tpc))
                mean_tpb = float(np.mean(per_sentence_tpb))

                if lang.lower() == "english":
                    en_tokens_per_word = mean_tpw
                    en_tokens_per_char = mean_tpc
                    en_tokens_per_byte = mean_tpb

                # Calculate inflation ratio against English baseline
                inflation_factor = (
                    mean_tpw / en_tokens_per_word if en_tokens_per_word else 1.0
                )

                records.append({
                    "Tokenizer": tok_name,
                    "Language": lang,
                    "Total Tokens": total_tokens,
                    "Total Words": total_words,
                    "Tokens / Word": round(mean_tpw, 3),
                    "Tokens / Char": round(mean_tpc, 4),
                    "Tokens / Byte": round(mean_tpb, 4),
                    "Word Inflation (x EN)": round(inflation_factor, 2),
                })

        df = pd.DataFrame(records)
        return df
