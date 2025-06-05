from .cnn import Down
from .swin_transformer import StageModule
import torch
import torch.nn as nn
from collections import OrderedDict
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, layers, heads, head_dim, window_size, downscaling_factors, relative_pos_embedding= True, net_dimension=64, inter_channels=True):
        super().__init__()

        #Internal values (for skip connections), if true return the mid channels
        self.inter_channels = inter_channels
        self.net_dimension = net_dimension

        self.aglo_1 = StageModule(in_channels=1, hidden_dimension=3, layers=layers[0],
                        downscaling_factor=downscaling_factors[0], num_heads=2, head_dim=4,
                        window_size=window_size[0], relative_pos_embedding=relative_pos_embedding)

        self.down_1 = Down(net_dimension, net_dimension)
        self.att_1 = StageModule(in_channels=net_dimension, hidden_dimension=hidden_dim[0], layers=layers[0],
                                downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim[0],
                                window_size=window_size[0], relative_pos_embedding=relative_pos_embedding)
        
        self.down_2 = Down(net_dimension, net_dimension*2)
        self.att_2 = StageModule(in_channels=net_dimension*2, hidden_dimension=hidden_dim[1], layers=layers[1],
                                downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim[1],
                                window_size=window_size[1], relative_pos_embedding=relative_pos_embedding)

        self.down_3 = Down(net_dimension*2, net_dimension*4)
        self.att_3 = StageModule(in_channels=net_dimension*4, hidden_dimension=hidden_dim[2], layers=layers[2],
                                downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim[2],
                                window_size=window_size[2], relative_pos_embedding=relative_pos_embedding)

        self.down_4 = Down(net_dimension*4, 384)
        self.att_4 = StageModule(in_channels=384, hidden_dimension=hidden_dim[3], layers=layers[3],
                                downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim[3],
                                window_size=window_size[3], relative_pos_embedding=relative_pos_embedding)
        
        self.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(131072, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 50)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(50, 25))
    ]))
    
    def channel_agglutination(self, x):
        """
        Agglutinates the output chhannels over the batch dimension
        """
        num_channels = x.shape[1]
        temp_x = []
        for channel in range(num_channels):
            temp_x.append(self.aglo_1(x[:,channel,:,:].unsqueeze(1)))
        
        cosine_x = []
        for i in range(len(temp_x)-1):
            cosine_x.append(torch.cosine_similarity(temp_x[i], temp_x[i+1]).unsqueeze(1))
        
        t_temp_x = torch.cat(temp_x, dim=1)
        t_new_x = torch.cat(cosine_x, dim=1)
        new_x = torch.cat((t_temp_x, t_new_x, x), dim=1)
        new_x = self.channel_expansion(new_x, self.net_dimension)

        return new_x

    def channel_expansion(self, x, target_channels):
        x = x.repeat(1, target_channels//x.shape[1], 1, 1)
        while x.shape[1] < target_channels:
            x = torch.cat((x, x[:,0,].unsqueeze(1)), dim=1)
        return x

    def forward(self, x):
        x = self.channel_agglutination(x)
        x = self.down_1(x)
        x1 = self.att_1(x)
        x = self.down_2(x1)
        x2 = self.att_2(x)
        x = self.down_3(x2)
        x3 = self.att_3(x)
        x = self.down_4(x3)
        x = self.att_4(x)

        out = self.fc(torch.flatten(x, 1))

        if self.inter_channels:
            return out, (x1, x2, x3)
        else:
            return out

if __name__ == "__main__":
    from cnn import Down
    from swin_transformer import StageModule
    # Models parameters
    in_channels = 3
    net_dimension = 64
    hidden_dim = [net_dimension, net_dimension*2, net_dimension*4, 384]
    layers = [2, 2, 2, 2]
    heads = [4, 4, 4, 4]
    head_dim = [32, 32, 32, 32]
    window_size = [2, 2, 2, 2]
    downscaling_factors = [1, 1, 1, 1]

    # Instantiate model
    model = Encoder(in_channels=in_channels, hidden_dim=hidden_dim, layers=layers, heads=heads, 
                    head_dim=head_dim, window_size=window_size, downscaling_factors=downscaling_factors).to("cuda")

    # Test model
    img = torch.zeros((5,13,256,256)).to("cuda")
    out = model(img)
    print(out[0].shape)

    img_SAR = torch.zeros((5,2,256,256)).to("cuda")
    out_SAR = model(img_SAR)
    print(out_SAR[0].shape)
