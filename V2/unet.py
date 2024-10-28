
#Src  https://medium.com/@sriskandaryan/autoencoders-demystified-audio-signal-denoising-32a491ab023a


import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, chnls_in=1, chnls_out=1, device="cuda"):
        super(UNet, self).__init__()
        self.device = device
        self.down_conv_layer_1 = DownConvBlock(chnls_in, 64, norm=False).to(device) # 128,1,512,216  -> 128,64 , 256, 108
        self.down_conv_layer_2 = DownConvBlock(64, 128).to(device) # 128,1,256,108 -> 128,64,128,54
        self.down_conv_layer_3 = DownConvBlock(128, 256).to(device) # 128,64,128,54 -> 128,128,64,27
        self.down_conv_layer_4 = DownConvBlock(256, 256, dropout=0.5).to(device) # 128,256,64,27 -> 128,256,32,13
        self.down_conv_layer_5 = DownConvBlock(256, 256, dropout=0.5).to(device)  # 128,256,32,13 -> 128,256,16,6
        self.down_conv_layer_6 = DownConvBlock(256, 256, dropout=0.5).to(device) # 128,256,16,6 -> 128,256,8,3

        self.up_conv_layer_1 = UpConvBlock(256, 256, kernel_size=(2, 2), stride=2, padding=0, dropout=0.5).to(device)
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=(2, 3), stride=2, padding=0, dropout=0.5).to(device)
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=(2, 3), stride=2, padding=0, dropout=0.5).to(device)
        self.up_conv_layer_4 = UpConvBlock(512, 128, dropout=0.5).to(device)
        self.up_conv_layer_5 = UpConvBlock(256, 64).to(device)

        self.upsample_layer = nn.Upsample(scale_factor=2).to(device)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)).to(device)
        self.conv_layer_1 = nn.Conv2d(128, chnls_out, 4, padding=1).to(device)
        self.activation = nn.Tanh().to(device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on GPU
        enc1 = self.down_conv_layer_1(x)
        enc2 = self.down_conv_layer_2(enc1)
        enc3 = self.down_conv_layer_3(enc2)
        enc4 = self.down_conv_layer_4(enc3)
        enc5 = self.down_conv_layer_5(enc4)
        enc6 = self.down_conv_layer_6(enc5)

        dec1 = self.up_conv_layer_1(enc6, enc5)
        dec2 = self.up_conv_layer_2(dec1, enc4)
        dec3 = self.up_conv_layer_3(dec2, enc3)
        dec4 = self.up_conv_layer_4(dec3, enc2)
        dec5 = self.up_conv_layer_5(dec4, enc1)

        final = self.upsample_layer(dec5)
        final = self.zero_pad(final)
        final = self.conv_layer_1(final)
        return final


class UpConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, stride=2, padding=1, dropout=0.0, device="cuda"):
        super(UpConvBlock, self).__init__()
        # Initialize layers directly on the specified device
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding).to(device),
            nn.InstanceNorm2d(op_sz).to(device),
            nn.ReLU().to(device),
        ]
        if dropout:
            self.layers.append(nn.Dropout(dropout).to(device))

        # Wrap all layers in a Sequential block, to be moved to device
        self.net = nn.Sequential(*self.layers).to(device)
        self.device = device

    def forward(self, x, enc_ip):
        # Ensure both inputs are on the correct device
        x = x.to(self.device)
        enc_ip = enc_ip.to(self.device)

        x = self.net(x)
        op = torch.cat((x, enc_ip), 1)
        return op


class DownConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, norm=True, dropout=0.0, device="cuda"):
        super(DownConvBlock, self).__init__()
        # Create layers and directly assign them to the specified device
        self.layers = [nn.Conv2d(ip_sz, op_sz, kernel_size, 2, 1).to(device)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz).to(device))
        self.layers.append(nn.LeakyReLU(0.2).to(device))
        if dropout:
            self.layers.append(nn.Dropout(dropout).to(device))

        # Wrap all layers in Sequential, to be moved to device
        self.net = nn.Sequential(*self.layers).to(device)
        self.device = device

    def forward(self, x):
        # Ensure input is also on the correct device
        x = x.to(self.device)
        return self.net(x)
