from torch import nn
from buildingblocks import create_encoders, DoubleConv, ExtResNetBlock


class BaseEncoder(nn.Module):
    def __init__(self, f_maps=(8, 16, 32), basic_module='resnet', layer_order='cge'):
        super().__init__()
        self.basic_module = basic_module
        if basic_module == 'resnet':
            basic_module = ExtResNetBlock
        elif basic_module == 'doubleconv':
            basic_module = DoubleConv
        else:
            basic_module = None

        self.f_maps = f_maps
        self.encoders = create_encoders(
            in_channels=1,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=3,
            conv_padding=1,
            layer_order=layer_order,
            num_groups=8,
            pool_kernel_size=2
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        return x
