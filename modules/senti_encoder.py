from modules import resnet
import torch.nn as nn
import torch
import modules.sentinel_bands as sentinel_bands
from collections import OrderedDict

class SentinelEncoder(nn.Module):
    """
    Use the self.model to process the sentinel 2 bands that are grouped
    by the self.bands_selector. The output of the model is the mean of the
    outputs of the model for each band.

    The group of bands have semantic meaning, so the model should be able to
    understand that the conjunt of the bands represents the same region single image.
    """
    def __init__(self, model=resnet.ResNet_encoder().to("cuda"), bands_selector=sentinel_bands.SentinelGroupBands()) -> None:
        super().__init__()

        # Get the last conv layer of the model
        # self.model = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
        self.model = model
        self.bands_selector = bands_selector

    def forward(self, x):
        bands_processed = self.bands_selector.sepair_bands(x)

        self.outs = []
        for band in bands_processed:
            self.outs.append(self.model(band))

        # Calculate the mean of all tensors in self.outs
        mean_tensor = torch.mean(torch.stack(self.outs), dim=0)

        # Return the mean tensor

        assert torch.isnan(mean_tensor).max() != True , "Output have Nan values"

        return mean_tensor

class SentinelEncoderOnlyBand(SentinelEncoder):
    def __init__(self, model=resnet.ResNet_encoder().to("cuda"), bands_selector=sentinel_bands.SentinelGroupBands()) -> None:
        super().__init__(model, bands_selector)

    def forward(self, x):
        bands_processed = self.bands_selector.sepair_bands(x)

        self.outs = []
        for band in bands_processed:
            self.outs.append(self.model(band))

        return self.outs

class SentinelEncoderLBP(SentinelEncoder):
    def __init__(self, model=resnet.ResNet_enconder_LBP().to("cuda"), bands_selector=sentinel_bands.EuroSATGroupBands()) -> None:
        super().__init__(model, bands_selector)

    def forward(self, x, hist):
        bands_processed = self.bands_selector.sepair_bands(x)

        self.outs = []
        for band in bands_processed:
            self.outs.append(self.model(band, hist))

        # Calculate the mean of all tensors in self.outs
        mean_tensor = torch.mean(torch.stack(self.outs), dim=0)

        # Return the mean tensor

        assert torch.isnan(mean_tensor).max() != True , "Output have Nan values"

        return mean_tensor


def load_model_and_weights(model_path, bands_selector=sentinel_bands.SentinelGroupBands(), model_resnet=resnet.ResNet_encoder(pretrained=True), lbp=False):
    # Load the model weights
    
    model_weights = torch.load(model_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if lbp:
        bands_selector = sentinel_bands.EuroSATGroupBands(ndvi_bands=False)
        model_resnet = resnet.ResNet_enconder_LBP(pretrained=True).to(device)
        model = SentinelEncoderLBP(model_resnet, bands_selector).to(device)
    else:
        model_resnet = resnet.ResNet_encoder(pretrained=True).to(device)
        model = SentinelEncoder(model_resnet, bands_selector).to(device)

    # model_resnet = resnet.ResNet_encoder(pretrained=pretrained).to(device)
    model.load_state_dict(model_weights)

    return model

if __name__ == "__main__":
    import copy

    # model_name = "20240312_191344"
    # model_path = f"./trained_models/{model_name}/q_weights.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_resnet = resnet.ResNet_enconder_LBP().to(device)
    bands_selector = sentinel_bands.EuroSATGroupBands(ndvi_bands=False)
    
    #Define if the Resnet model is pretrained
    pretrained = True

    # Create the models
    # model_q = SentinelEncoder(model_resnet, bands_selector).to(device)
    model_q = SentinelEncoderLBP(model_resnet, bands_selector).to(device)
    # model_q = SentinelEncoderOnlyBand(model_resnet, bands_selector).to(device)
    # model_q = load_model_and_weights(model_path, bands_selector, pretrained)
    model_k = copy.deepcopy(model_q).to(device)
    
    # Create the input
    x_0 = torch.randn(2, 13, 256, 256).to(device)
    x_1 = torch.randn(2, 13, 256, 256).to(device)

    hist_x0 = torch.randn(2, 16).to(device)
    hist_x1 = torch.randn(2, 16).to(device)
    
    # Define constrative loss between the two models
    criterion = nn.CosineEmbeddingLoss()

    # Forward pass
    out_q = model_q(x_0, hist_x0)
    out_k = model_k(x_1, hist_x1)

    # # Loop for compare all bands in self.outs between the models
    # loss = 0.0
    # for i in range(len(model_q.outs)):
    #     loss += criterion(model_q.outs[i], model_k.outs[i], torch.tensor([1.0]).to(device))
    #     loss /= len(model_q.outs)
    # print(loss)
    
    # # Loos between the outputs of the models (also cosine similarity)
    # loss_mean = criterion(out_q, out_k, torch.tensor([1.0]).to(device))
    # print(loss_mean)

    # print(model_q.model.cnn_out.shape)