# src/data/split_save.py

import os
import torch
from sklearn.model_selection import train_test_split

def split_and_save(X, y, processed_path, test_size=0.15, val_size=0.15, seed=42):
    os.makedirs(processed_path, exist_ok=True)

    train_path = os.path.join(processed_path, "train", "data.pt")
    val_path = os.path.join(processed_path, "val", "data.pt")
    test_path = os.path.join(processed_path, "test", "data.pt")

    # If already exists → reuse
    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        return

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    torch.save((X_train, y_train), train_path)
    torch.save((X_val, y_val), val_path)
    torch.save((X_test, y_test), test_path)
