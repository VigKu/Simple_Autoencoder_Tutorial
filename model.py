import torch.nn as nn


class Autoencoder(nn.Module):
    # last activatation fn in decoder
    # if pixel values range between 0 and 1 --> sigmoid
    # if pixel values range between -1 and 1 --> tanh
    def __init__(self):
        super(Autoencoder, self).__init__()
        # BS, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # BS, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # BS, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # BS, 64, 1, 1
        )
        # BS, 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # BS, 32, 7, 7
            nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # BS, 16, 13, 13
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # BS, 16, 14, 14
            nn.ReLU(),
            # nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1),  # BS, 1, 27, 27
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # BS, 1, 28, 28
            nn.Tanh()  # img pixel values range from -1 to 1 due to normalization
        )
        # nn.MaxPool2d() --> nn.MaxUnpool2d()

    def forward(self, x):
        x = self.decoder(self.encoder(x))
        return x
