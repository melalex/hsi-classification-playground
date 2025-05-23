import torch

from torch.utils import data


def resolve_torch_device() -> torch.device:
    return torch.device(__resolve_torch_device_str())


def dataloader_from_prtototype(
    dataset: data.Dataset, dataloader: data.DataLoader
) -> data.DataLoader:
    return data.DataLoader(
        dataset=dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        sampler=dataloader.sampler,
        batch_sampler=dataloader.batch_sampler,
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


def __resolve_torch_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
