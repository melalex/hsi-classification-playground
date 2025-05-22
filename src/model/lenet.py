from torch import nn


class FullyConvolutionalLeNet(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(FullyConvolutionalLeNet, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 100, kernel_size=4),
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
