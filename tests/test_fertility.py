"""
Unit test for Tokenization Tax calculator and KV memory projector.
"""

from src.tokenization_tax.calculator import TokenizationTaxCalculator
from src.tokenization_tax.memory_projection import KVMemoryProjector


def test_tokenization_tax_calculator_basic():
    data = {
        "English": ["Hello world! This is a test sentence."],
        "Hindi": ["नमस्ते दुनिया! यह एक परीक्षण वाक्य है।"],
        "Telugu": ["నమస్కారం ప్రపంచం! ఇది ఒక పరీక్ష వాక్యం."],
    }
    calc = TokenizationTaxCalculator()
    df = calc.evaluate_corpus(data)
    assert len(df) == 9  # 3 tokenizers * 3 languages
    assert "Tokens / Word" in df.columns
    assert "Tokens / Char" in df.columns
    assert "Tokens / Byte" in df.columns
    print("test_tokenization_tax_calculator_basic passed!")


def test_kv_memory_projector():
    proj = KVMemoryProjector()
    # Test fp16 bytes per token for Qwen3-0.6B (2 * 28 * 8 * 128 * 2 = 114688 bytes = 112 KB)
    bytes_tok = proj.compute_kv_per_token_bytes("Qwen3-0.6B (Proposal Target)", "fp16")
    assert bytes_tok == 114688
    # Test int8 is half
    assert proj.compute_kv_per_token_bytes("Qwen3-0.6B (Proposal Target)", "int8") == 57344
    print("test_kv_memory_projector passed!")


if __name__ == "__main__":
    test_tokenization_tax_calculator_basic()
    test_kv_memory_projector()
