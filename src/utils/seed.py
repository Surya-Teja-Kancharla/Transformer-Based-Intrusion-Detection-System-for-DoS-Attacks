"""
seed.py
--------
Utility for setting random seeds across Python, NumPy, and PyTorch
to ensure reproducible experiments.

This is critical for research-grade implementations and paper reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility.

    Args:
        seed (int): Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hash seed (important for Python >= 3.11)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Global seed set to {seed}")
