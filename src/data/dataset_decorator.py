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
            torch.tensor(1, dtype=torch.float32)
            if y == self.label
            else torch.tensor(0, dtype=torch.float32)
        )


class PuDatasetDecorator(data.Dataset):

    def __init__(self, decorated: data.Dataset, label: int):
        super().__init__()

        self.decorated = decorated
        self.label = label

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        x, y = self.decorated[idx]

        if y == self.label:
            return x, torch.tensor(1, dtype=torch.int32)
        elif y == -1:
            return x, torch.tensor(-1, dtype=torch.int32)
        else:
            return x, torch.tensor(0, dtype=torch.int32)


class ConstLabelDataset(data.Dataset):

    def __init__(self, x: Tensor, label: int):
        super().__init__()

        self.decorated = UnlabeledDatasetDecorator(data.TensorDataset(x))
        self.label = label

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        x = self.decorated[idx]
        return x, torch.tensor(self.label, device=x.device, dtype=torch.float32)
