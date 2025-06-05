import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50
from collections import OrderedDict
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch

# Path: Controller/Modules/segmentation_model.py

#### Segmentation Models
# import segmentation_models_pytorch as smp

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class Segmentation_Model(nn.Module):
    """
    A class representing a segmentation model.

    Args:
        encoder (nn.Module, optional): The encoder model to use. Defaults to resnet18.
        num_classes (int, optional): The number of output classes. Defaults to 1.
    """

    def __init__(self, num_classes=1, input_channel_change=False) -> None:
        super().__init__()

        model = resnet50()
        model = nn.Sequential(*list(model.children())[:-2])
        # model.load_state_dict(encoder_q_weights, strict=False)

        self.encoder = model
        if input_channel_change:
            self.encoder[0] = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder[0].load_state_dict({"weight": self.__convert_2_13_channels__()})
        
        self.decoder = nn.Sequential(
            DoubleConv(2048*2, 1024),
            DoubleConv(1024*2, 512),
            DoubleConv(512*2, 256),
            DoubleConv(256*2, 128),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def __convert_2_13_channels__(self):
        # Get the weights
        weights = self.encoder[0].state_dict()["weight"]

        # Repeat the values in the channel dimension
        new_weights = weights.repeat_interleave(13//weights.shape[1], dim=1)

        return new_weights

    def forward(self, t1, t2):
        
        skip_connections_t1 = []
        skip_connections_t2 = []

        skip_indexs = [4, 5, 6, 7]

        ## Encoder first image
        for idx, module in enumerate(self.encoder):
            t1 = module(t1)
            # check if the need to add skip connection
            if idx in skip_indexs:
                skip_connections_t1.append(t1)

        ## Encoder second image
        for idx, module in enumerate(self.encoder):
            t2 = module(t2)
            # check if the need to add skip connection
            if idx in skip_indexs:
                skip_connections_t2.append(t2)

        ## Difference between features of the two images
        x = torch.abs(t1 - t2)
        skip_connections = []
        for i in range(len(skip_connections_t1)):
            skip_connections.append(torch.abs(skip_connections_t1[i] - skip_connections_t2[i]))

        ## Decoder
        for idx, module in enumerate(self.decoder):
            if idx < 4:
                x = torch.cat([x, skip_connections.pop()], dim=1)

            if idx < 5: 
                x = F.interpolate(x, scale_factor=2)
                
            x = module(x)
            
        return x

if __name__ == "__main__":
    model = Segmentation_Model(input_channel_change=False)

    x_0 = torch.randn(2, 3, 256, 256)
    x_1 = torch.randn(2, 3, 256, 256)
    
    out = model(x_0, x_1)
    print(out.shape) # torch.Size([2, 1, 128, 128])
