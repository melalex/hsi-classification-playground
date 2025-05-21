import torch


def compute_confidence_interval(probs, z=1.96):
    n = 1  # Assuming a single observation
    l = probs - z * torch.sqrt((probs * (1 - probs)) / n)
    h = probs + z * torch.sqrt((probs * (1 - probs)) / n)
    l = torch.clamp(l, 0, 1)
    h = torch.clamp(h, 0, 1)
    return l, h
