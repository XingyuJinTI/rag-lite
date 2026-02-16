"""
Shared utilities for RAG-Lite.
"""

import torch


def get_device() -> str:
    """
    Get the best available device for PyTorch inference.
    
    Returns:
        "cuda" if NVIDIA GPU available,
        "mps" if Apple Silicon GPU available,
        "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
