# src/data/load_processed.py

import torch
import os

def load_processed(processed_path):
    train = torch.load(os.path.join(processed_path, "train", "data.pt"))
    val = torch.load(os.path.join(processed_path, "val", "data.pt"))
    test = torch.load(os.path.join(processed_path, "test", "data.pt"))
    return train, val, test
