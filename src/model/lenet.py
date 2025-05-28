from torch import nn


class FullyConvolutionalLeNet(nn.Module):

    def __init__(self, input_channels, num_classes, first_padding=0):
        super().__init__()

        self.params = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "first_padding": first_padding,
        }

        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 100, kernel_size=4, padding=first_padding),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(100, 200, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(200),
            nn.Conv2d(200, 500, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(500, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], self.num_classes)
        return x

    def get_params(self):
        return self.params


class PuLeNet(nn.Module):

    def __init__(self, input_channels, first_padding=0):
        super().__init__()

        self.params = {
            "input_channels": input_channels,
            "first_padding": first_padding,
        }

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 100, kernel_size=4, padding=first_padding),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(100, 200, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(200),
            nn.Conv2d(200, 500, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(500, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x.reshape(-1)

    def get_params(self):
        return self.params
