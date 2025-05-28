import torch

from torch import nn

from pathlib import Path

from torch.utils import data

from src.definitions import MODELS_FOLDER


def resolve_torch_device() -> torch.device:
    return torch.device(__resolve_torch_device_str())


def dataloader_from_prtototype(
    dataset: data.Dataset, dataloader: data.DataLoader
) -> data.DataLoader:
    return data.DataLoader(
        dataset=dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
    )


def save_model(
    model: nn.Module, model_name: str, base_path: Path = MODELS_FOLDER
) -> Path:
    base_path.mkdir(parents=True, exist_ok=True)

    save_path = base_path / f"{model_name}.bin"

    torch.save(model.state_dict(), base_path / model_name)

    return save_path


def __resolve_torch_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
