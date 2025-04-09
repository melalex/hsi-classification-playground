from torch import nn
from torch.utils.data import DataLoader
from torch import optim


class ClassificationTrainer:
    num_epochs: int
    learning_rate: float
    loss_fun: nn.Module

    def __init__(
        self,
        num_epochs: int,
        learning_rate: float,
        loss_fun: nn.Module,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun

    def fit(self, model: nn.Module, train_loader: DataLoader):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        train_total_loss = 0
        epoch_loss = 0

        for _ in range(self.num_epochs):
            model.train()
            for x, y_true in train_loader:
                optimizer.zero_grad()

                y_pred = model(x)

                loss = self.loss_fun(y_pred, y_true)
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()

            epoch_loss = train_total_loss / len(train_loader)

        return epoch_loss
