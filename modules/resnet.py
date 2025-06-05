import torch
from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from lbp import lbp_histogram, lbp_values

class ResNet_encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet_encoder, self).__init__()
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool =nn.Sequential(list(resnet.children())[-2]) 
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(resnet.fc.in_features, 1024)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1024, 512)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(512, 256))
        ]))
        self.intermediate = {}

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x.type(torch.float32).to("cuda"))
            if idx in {4, 5, 6, 7}:
                self.intermediate[idx] = x
        self.cnn_out = x #save the output of the cnn
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet_enconder_LBP(ResNet_encoder):
    def __init__(self, pretrained=True):
        super(ResNet_enconder_LBP, self).__init__(pretrained=pretrained)
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(resnet.fc.in_features, 1024)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1024, 512)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(512, 256))
        ]))

    def forward(self, x, hist):
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in {4, 5, 6, 7}:
                self.intermediate[idx] = x
        self.cnn_out = x #save the output of the cnn
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x, hist), dim=1)
        x = self.fc(x)
        x = nn.Softmax(dim=1)(x)
        return x  
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = True
    # model = ResNet_encoder(pretrained=pretrained).to(device)
    model = ResNet_enconder_LBP(pretrained=pretrained).to(device)
    model.fc = nn.Linear(512+(3*16), 10).to(device)
    x = torch.randn(1, 3, 256, 256)
    hist = lbp_histogram(x).view(1, -1).type(torch.float32)

    x = x.to(device)
    hist = hist.to(device)

    out = model(x, hist)
    print(f"Output shape: {out.shape}")
    print(f"Shape of last convolutional layer: {model.cnn_out.shape}")
