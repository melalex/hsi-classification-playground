from torch import Tensor
import torch
import torch.utils.data as data

from typing import Sequence

class UnlabeledDatasetDecorator(data.Dataset):

    def __init__(self, decorated: data.Dataset):
        super().__init__()

        self.decorated = decorated

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        return self.decorated[idx][0]


class LabeledDatasetDecorator(data.Dataset):

    def __init__(self, decorated: data.Dataset, labels: Sequence[int]):
        super().__init__()

        self.decorated = decorated
        self.labels = labels

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        return self.decorated[idx], self.labels[idx]


class BinaryDatasetDecorator(data.Dataset):

    def __init__(self, decorated: data.Dataset, label: int):
        super().__init__()

        self.decorated = decorated
        self.label = label

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        x, y = self.decorated[idx]
        return x, (
            torch.tensor(1, dtype=float)
            if y == self.label
            else torch.tensor(0, dtype=float)
        )
