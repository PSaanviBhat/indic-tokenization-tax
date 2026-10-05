"""
Unit tests for Logit-Lens interpretability engine.
"""

from src.interpretability.logit_lens import LogitLensAnalyzer


def test_analyzer_instantiation():
    analyzer = LogitLensAnalyzer(
        model_id="Qwen/Qwen2.5-0.5B",
        device="cpu",
    )
    assert analyzer.model_id == "Qwen/Qwen2.5-0.5B"
    print("test_analyzer_instantiation passed!")


if __name__ == "__main__":
    test_analyzer_instantiation()
