from torch import nn

from src.util.patches import extract_tensor_patches


ACTIVATIONS = {
    "relu": nn.ReLU(),
    "linear": nn.Identity(),
    "elu": nn.ELU(),
}


class BaseAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()


class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class SymmetricAutoEncoder(BaseAutoEncoder):

    def __init__(self, units: list[int], activation: str):
        super().__init__()

        encoder_units = []
        decoder_units = []

        for it, next in zip(units[:-2], units[1:-1]):
            encoder_units.append(self.provide_unit(it, next))
            encoder_units.append(ACTIVATIONS[activation])

        encoder_units.append(self.provide_unit(units[-2], units[-1]))

        self.encoder = nn.Sequential(*encoder_units)

        decoder_units = []

        for it, next in reversed(list(zip(units[2:], units[1:-1]))):
            decoder_units.append(self.provide_unit(it, next))
            decoder_units.append(ACTIVATIONS[activation])

        decoder_units.append(self.provide_unit(units[1], units[0]))

        self.decoder = nn.Sequential(*decoder_units)

    def provide_unit(self, in_f, out_f):
        pass


class AsymmetricAutoEncoder(BaseAutoEncoder):

    def __init__(
        self,
        encoder_units_def: list[int],
        decoder_units_def: list[int],
        activation: str,
    ):
        super().__init__()

        encoder_units = []
        decoder_units = []

        for it, next in zip(encoder_units_def[:-2], encoder_units_def[1:-1]):
            encoder_units.append(self.provide_unit(it, next))
            encoder_units.append(ACTIVATIONS[activation])

        encoder_units.append(
            self.provide_unit(encoder_units_def[-2], encoder_units_def[-1])
        )

        self.encoder = nn.Sequential(*encoder_units)

        decoder_units = []

        for it, next in zip(decoder_units_def[:-2], decoder_units_def[1:-1]):
            decoder_units.append(self.provide_unit(it, next))
            decoder_units.append(ACTIVATIONS[activation])

        decoder_units.append(
            self.provide_unit(decoder_units_def[-2], decoder_units_def[-1])
        )

        self.decoder = nn.Sequential(*decoder_units)

    def provide_unit(self, in_f, out_f):
        pass


class PointWiseAutoEncoder(BaseAutoEncoder):

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class LinearAutoEncoder(BaseAutoEncoder):

    def forward(self, x):
        b, c, h, w = x.shape
        x_r = x.permute(0, 2, 3, 1).reshape(b, -1, c)

        encoded = self.encoder(x_r)
        encoded_r = encoded.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        decoded = self.decoder(encoded)
        decoded_r = decoded.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return encoded_r, decoded_r


class SymmetricPointWiseAutoEncoder(SymmetricAutoEncoder, PointWiseAutoEncoder):

    def __init__(self, units: list[int], activation="relu"):
        super().__init__(units, activation)

    def provide_unit(self, in_f, out_f):
        return nn.Conv2d(in_f, out_f, kernel_size=1)


class SymmetricLinearAutoEncoder(SymmetricAutoEncoder, LinearAutoEncoder):

    def __init__(self, units: list[int], activation="relu"):
        super().__init__(units, activation)

    def provide_unit(self, in_f, out_f):
        return nn.Linear(in_f, out_f)


class AsymmetricPointWiseAutoEncoder(AsymmetricAutoEncoder, PointWiseAutoEncoder):

    def __init__(
        self,
        encoder_units_def: list[int],
        decoder_units_def: list[int],
        activation="relu",
    ):
        super().__init__(encoder_units_def, decoder_units_def, activation)

    def provide_unit(self, in_f, out_f):
        return nn.Conv2d(in_f, out_f, kernel_size=1)


class AsymmetricLinearAutoEncoder(AsymmetricAutoEncoder, LinearAutoEncoder):

    def __init__(
        self,
        encoder_units_def: list[int],
        decoder_units_def: list[int],
        activation="relu",
    ):
        super().__init__(encoder_units_def, decoder_units_def, activation)

    def provide_unit(self, in_f, out_f):
        return nn.Linear(in_f, out_f)


class SpatialAutoEncoder(nn.Module):

    def __init__(
        self,
        input_channels,
        embedding_size,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            ## Block 1 ##
            nn.Conv2d(input_channels, 100, kernel_size=4),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=2),
            ## Block 2 ##
            nn.Conv2d(100, 200, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(200),
            ## Block 3 ##
            nn.Conv2d(200, 500, kernel_size=1),
            nn.Sigmoid(),
            ## Block 4 ##
            nn.Conv2d(500, embedding_size, kernel_size=1),
            nn.BatchNorm2d(embedding_size),
            nn.Softmax(1),
        )
        self.decoder = nn.Sequential(
            ## Block 1 - Reverse of last encoder conv ##
            nn.ConvTranspose2d(embedding_size, 500, kernel_size=1),
            nn.BatchNorm2d(500),
            ## Block 2 ##
            nn.ConvTranspose2d(500, 200, kernel_size=1),
            nn.Sigmoid(),  # reverse of encoder Sigmoid input
            ## Block 3 ##
            nn.Upsample(scale_factor=2, mode="nearest"),  # reverse MaxPool2d
            nn.ConvTranspose2d(200, 100, kernel_size=2),
            nn.BatchNorm2d(100),
            ## Block 4 ##
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(100, input_channels, kernel_size=4),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
