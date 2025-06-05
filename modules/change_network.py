#author = "lstival"

# Import resnet18
import torch.nn as nn
import torch
from torchvision.models.resnet import resnet18, ResNet18_Weights
from modules.cnn import Up
from collections import OrderedDict
import os
import copy
from utils.utils import get_model_time

# Segmentation Decoder
class DecoderResNet18(nn.Module):
    def __init__(self, in_channels=512, net_dimension=64, out_channels=1):
        super().__init__()
        self.out_channels = out_channels

        self.up_1 = Up(in_channels, net_dimension*4)
        self.up_2 = Up((net_dimension*4)*2 , net_dimension*2)
        self.up_3 = Up((net_dimension*2)*2, net_dimension)
        self.up_4 = Up((net_dimension)*2, net_dimension)
        self.up_5 = Up(net_dimension, self.out_channels)

    def forward(self, x_t0, skips=None):

        x1,x2,x3, x4 = skips

        x_t0 = self.up_1(x_t0)
        x_t0 = self.up_2(x_t0, x3)
        x_t0 = self.up_3(x_t0, x2)
        x_t0 = self.up_4(x_t0, x1)
        x_t0 = self.up_5(x_t0)
        # x_t0 = torch.sigmoid(x_t0)

        return x_t0

class ChangeDetection(nn.Module):
    def __init__(self, encoder, decoder):
        super(ChangeDetection, self).__init__()

        # encoder model of original image
        self.encoder = encoder.to("cuda")
        # Encoder model of images in different time
        self.encoder_2 = copy.deepcopy(encoder).to("cuda")

        # Define the Decoder (segmentation model)
        decoder = DecoderResNet18(in_channels=512, net_dimension=64, out_channels=1)
        self.decoder = decoder.to("cuda")

        # Path to save the results
        self.paths = os.path.join("change_results", get_model_time())
        os.makedirs(self.paths, exist_ok=True)

        # Epoch counter
        self.epoch_counter = 0

    def forward(self, x0, x1):
        # Load the Encoder
        self.encoder.eval()
        self.encoder_2.eval()

        with torch.no_grad():
            _ = self.encoder(x0)
            output = self.encoder.model.cnn_out
            features = self.encoder.model.intermediate

            _ = self.encoder_2(x1)
            output_2 = self.encoder_2.model.cnn_out
            features_2 = self.encoder_2.model.intermediate

        decoder_input =  (output - output_2)

        decoder_features = [None] * len(features)
        for idx, i in enumerate(list(features.keys())):
            decoder_features[idx] = features[i] - features_2[i]
                
        seg_mask = self.decoder(decoder_input, decoder_features)
        return seg_mask
