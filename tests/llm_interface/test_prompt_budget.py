from llm_interface import prompt_budget


def test_installed_vocabulary_counts_utf8_with_wrapper_reserve(monkeypatch):
    class Vocabulary:
        def tokenize(self, text, **kwargs):
            assert text == "α ≤ 5".encode("utf-8")
            assert kwargs == {"add_bos": True, "special": False}
            return [1, 2, 3]
    monkeypatch.setattr(prompt_budget, "_local_vocabulary", lambda *args: (Vocabulary(), 100))
    assert prompt_budget.prompt_input_tokens("α ≤ 5", "mistral:latest") == 103


def test_unavailable_vocabulary_uses_bytes_not_character_or_word_estimate(monkeypatch):
    monkeypatch.setattr(prompt_budget, "_local_vocabulary", lambda *args: (None, 1024))
    assert prompt_budget.prompt_input_tokens("α ≤ 5", "mistral:latest") == len("α ≤ 5".encode()) + 1024
