import torch.nn.functional as F

def temperature_scale(logits, T):
    return logits / T
