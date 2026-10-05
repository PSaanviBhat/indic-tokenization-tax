"""
Logit Lens Mechanistic Interpretability Engine.
Projects intermediate layer representations through final RMSNorm and unembedding head
to detect the emergence of the English pivot across transformer depth.
"""

import torch
import pandas as pd
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitLensAnalyzer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.norm_layer = None
        self.lm_head = None

    def load_model(self):
        """Load tokenizer and causal language model weights."""
        print(f"Loading LogitLens model: {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Locate final layer norm and unembedding matrix
        # For Qwen architecture: model.model.norm and model.lm_head
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self.norm_layer = self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            self.norm_layer = self.model.transformer.ln_f
        else:
            raise AttributeError("Could not identify final layer norm in model architecture.")

        self.lm_head = self.model.lm_head
        print("Model and unembedding layers successfully hooked.\n")

    def _get_first_token_id(self, text: str) -> int:
        """Encode target text and return the first subword token ID."""
        # Ensure leading space tokenization variation is captured
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            tokens = self.tokenizer.encode(" " + text.strip(), add_special_tokens=False)
        return tokens[0]

    def analyze_single_prompt(
        self,
        prompt: str,
        target_en: str,
        target_native: str,
        language: str,
        concept_id: str,
    ) -> pd.DataFrame:
        """
        Run forward pass on a cloze prompt and project all intermediate hidden states
        through final RMSNorm and lm_head to track layer-wise token probabilities.
        """
        if self.model is None:
            self.load_model()

        en_token_id = self._get_first_token_id(target_en)
        native_token_id = self._get_first_token_id(target_native)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states is tuple of (embedding_output, layer_1, ..., layer_L)
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1

        records = []
        for layer_idx, h in enumerate(hidden_states):
            # Focus on the last token position predicting the next word
            h_last = h[0, -1, :]

            # CRITICAL TECHNICAL NOTE: Apply final RMSNorm before unembedding
            h_norm = self.norm_layer(h_last)
            logits = self.lm_head(h_norm)
            probs = torch.softmax(logits, dim=-1)

            prob_en = probs[en_token_id].item()
            prob_native = probs[native_token_id].item()

            top1_id = torch.argmax(probs).item()
            top1_token = self.tokenizer.decode([top1_id]).strip()
            top1_prob = probs[top1_id].item()

            layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx}"

            records.append({
                "Concept": concept_id,
                "Language": language,
                "Layer Index": layer_idx,
                "Layer Name": layer_name,
                "Total Layers": num_layers,
                "Prob English Target": prob_en,
                "Prob Native Target": prob_native,
                "English vs Native Ratio": (prob_en + 1e-7) / (prob_native + 1e-7),
                "Top-1 Decoded Token": top1_token,
                "Top-1 Probability": top1_prob,
                "Target English Token": self.tokenizer.decode([en_token_id]).strip(),
                "Target Native Token": self.tokenizer.decode([native_token_id]).strip(),
            })

        return pd.DataFrame(records)

    def analyze_dataset(self, test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze a collection of multilingual cloze test cases."""
        all_dfs = []
        for case in test_cases:
            df = self.analyze_single_prompt(
                prompt=case["prompt"],
                target_en=case["target_en"],
                target_native=case["target_native"],
                language=case["language"],
                concept_id=case["concept"],
            )
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)
