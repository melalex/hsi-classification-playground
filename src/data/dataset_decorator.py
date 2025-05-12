import torch.utils.data as data

class UnlabeledDatasetDecorator(data.Dataset):

    def __init__(self, decorated: data.Dataset):
        super().__init__()

        self.decorated = decorated

    def __len__(self):
        return len(self.decorated)

    def __getitem__(self, idx):
        x, _ = self.decorated[idx]

        return x
