import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, convs=2):
        super().__init__()
        if in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()

        stride = 1
        padding = kernel_size // 2

        def conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU()
            )
        self.block = nn.ModuleList([
            conv(in_channels, out_channels),
            *[
                conv(out_channels, out_channels)
                for _ in range(convs - 1)
            ]
        ])

    def forward(self, x):
        residual = self.projection(x)

        for module in self.block:
            x = module(x)

        return residual + x


def downsample(in_channels, out_channels, kernel_size=3, stride=2):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU()
    )


def upsample(in_channels, out_channels, kernel_size=3, stride=2):
    padding = (kernel_size - 1) // 2
    output_padding = stride - 1
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size, stride, padding, output_padding),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU()
    )


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            filters=[32, 48, 64, 96, 128, 192, 256, 384],
            bottleneck_filters=512,
            kernel_size=3,
            num_res_convs=2
    ):
        super().__init__()
        self.down_blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels,
                          kernel_size, num_res_convs)
            for in_channels, out_channels in zip([in_channels, *filters[1:]], filters)
        ])
        self.downsample = nn.ModuleList([
            downsample(in_channels, out_channels)
            for in_channels, out_channels in zip(filters, [*filters[1:], bottleneck_filters])
        ])

        self.bottleneck = ResidualBlock(
            bottleneck_filters, bottleneck_filters, 1, num_res_convs)
        self.upsample = nn.ModuleList([
            upsample(in_channels, out_channels)
            for out_channels, in_channels in zip(filters, [*filters[1:], bottleneck_filters])
        ])

        self.up_blocks = nn.ModuleList([
            # times 2 because of skip connection
            ResidualBlock(2 * out_channels, out_channels)
            for out_channels in filters
        ])

        self.output_layer = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x_skip = []

        for block, downsample in zip(self.down_blocks, self.downsample):
            x = block(x)
            x_skip.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        for up, block, x_down in reversed(list(zip(self.upsample, self.up_blocks, x_skip))):
            x = up(x)
            x = torch.cat([x, x_down], dim=1)
            x = block(x)

        x = self.output_layer(x)

        return x

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, a=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
