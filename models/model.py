import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class UNet(nn.Module):
    def __init__(self, encoder_channels=(1, 64, 128, 256, 512, 1024),
                 decoder_channels=(1024, 512, 256, 128, 64),
                 num_class=1, padding="same", dropout_rate=0.2, retain_dim=False, out_sz=(512, 512)):
        super().__init__()

        # A basic convolutional block with two Conv2d layers, BatchNorm, and ReLU activation
        class Block(nn.Module):
            def __init__(self, in_ch, out_ch, padding):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                return self.block(x)

        # Encoder: series of convolutional blocks with max pooling and dropout
        class Encoder(nn.Module):
            def __init__(self, chs, padding, dropout_rate=0.2):
                super().__init__()
                self.enc_blocks = nn.ModuleList(
                    [Block(chs[i], chs[i+1], padding) for i in range(len(chs)-1)]
                )
                self.pool = nn.MaxPool2d(2)  # Downsampling by a factor of 2
                self.dropout = nn.Dropout2d(p=dropout_rate)  # Dropout for regularization

            def forward(self, x):
                ftrs = []  # Store feature maps for skip connections
                for block in self.enc_blocks:
                    x = block(x)
                    ftrs.append(x)  # Store feature map before downsampling
                    x = self.dropout(x)  # Apply dropout
                    x = self.pool(x)  # Downsample
                return ftrs  # Return all feature maps for later use in skip connections

        # Decoder: Upsampling using transposed convolutions and concatenation with encoder features
        class Decoder(nn.Module):
            def __init__(self, chs, padding):
                super().__init__()
                self.upconvs = nn.ModuleList([
                    nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2) for i in range(len(chs)-1)
                ])
                self.dec_blocks = nn.ModuleList([
                    Block(chs[i], chs[i+1], padding) for i in range(len(chs)-1)
                ])

            def forward(self, x, encoder_features):
                for i in range(len(self.dec_blocks)):
                    x = self.upconvs[i](x)  # Upsample using transposed convolution
                    enc_ftrs = self.crop(encoder_features[i], x)  # Crop encoder feature to match upsampled size
                    x = torch.cat([x, enc_ftrs], dim=1)  # Concatenate with encoder features (skip connection)
                    x = self.dec_blocks[i](x)  # Apply convolutional block
                return x

            def crop(self, enc_ftrs, x):
                """ Crop the encoder features to match the size of the upsampled features. """
                _, _, H, W = x.shape
                enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
                return enc_ftrs

        # Instantiate Encoder and Decoder components
        self.encoder = Encoder(encoder_channels, padding, dropout_rate)
        self.decoder = Decoder(decoder_channels, padding)

        # Final 1x1 convolution to output segmentation mask with the desired number of classes
        self.head = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=num_class, kernel_size=1)

        # Option to retain original image dimensions
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)  # Encode input image
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])  # Decode using reversed feature maps
        out = self.head(out)  # Final convolution to get segmentation map

        # Resize output to match input dimensions if retain_dim is True
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out