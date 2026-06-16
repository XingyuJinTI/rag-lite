"""
Shared test fixtures.

The chunking module derives its token budget from a HuggingFace tokenizer. To keep
these tests fast and offline (no model download), we substitute a deterministic
whitespace tokenizer: one token per whitespace-separated word, with a stable, growing
vocab so encode/decode round-trip across calls.
"""

import pytest


class FakeTokenizer:
    """Whitespace tokenizer: 1 token == 1 word. Deterministic, offline, reversible.

    `model_max_length` is configurable so `_model_max_tokens` clamping can be exercised.
    """

    def __init__(self, model_max_length: int = 512):
        self.model_max_length = model_max_length
        self._vocab = {}        # word -> id
        self._inv = {}          # id -> word

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            new_id = len(self._vocab)
            self._vocab[word] = new_id
            self._inv[new_id] = word
        return self._vocab[word]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids):
        return " ".join(self._inv[i] for i in ids)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def patch_tokenizer(monkeypatch):
    """Patch chunking._get_tokenizer to return a FakeTokenizer of a given max length."""
    from rag_lite import chunking

    def _apply(model_max_length: int = 512):
        tok = FakeTokenizer(model_max_length=model_max_length)
        monkeypatch.setattr(chunking, "_get_tokenizer", lambda _name: tok)
        return tok

    return _apply
