from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: (32, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample: (32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downsample: (64, H/4, W/4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Output: (64, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample: (64, H/2, W/2)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample: (32, H, W)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Output: (1, H, W)
            nn.Sigmoid()  # Normalize the output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize the autoencoder model
model = Autoencoder()
print(model)
