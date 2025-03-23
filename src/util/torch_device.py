import torch


def resolve_torch_device() -> torch.device:
    return torch.device(__resolve_torch_device_str())


def __resolve_torch_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
