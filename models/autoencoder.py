from torch import nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,1)), nn.ReLU(),
            nn.Upsample(size=(512, 219), mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

if __name__ == "__main__":
    model = DenoisingAutoencoder()
    print(model)