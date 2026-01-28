import numpy as np
import torch

def create_windows(X, y, window_size, stride):
    sequences = []
    labels = []

    for i in range(0, len(X) - window_size, stride):
        sequences.append(X[i:i + window_size])
        labels.append(y[i + window_size - 1])

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return torch.from_numpy(sequences), torch.from_numpy(labels)
