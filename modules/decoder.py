from .cnn import Up
from .swin_transformer import StageModule
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, layers, heads, head_dim, window_size, downscaling_factors, relative_pos_embedding= True, net_dimension=64, with_skips=True):
        super().__init__()
        self.with_skips = with_skips

        self.down_1 = Up(in_channels, net_dimension*4)
        self.att_1 = StageModule(in_channels=net_dimension*4, hidden_dimension=hidden_dim[0], layers=layers[0],
                                downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim[0],
                                window_size=window_size[0], relative_pos_embedding=relative_pos_embedding)
        
        if self.with_skips:
            self.down_2 = Up((net_dimension*4)*2 , net_dimension*2)
            self.att_2 = StageModule(in_channels=net_dimension*2, hidden_dimension=hidden_dim[1], layers=layers[1],
                                    downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim[1],
                                    window_size=window_size[1], relative_pos_embedding=relative_pos_embedding)

            self.down_3 = Up((net_dimension*2)*2, net_dimension)
            self.att_3 = StageModule(in_channels=net_dimension, hidden_dimension=hidden_dim[2], layers=layers[2],
                                    downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim[2],
                                    window_size=window_size[2], relative_pos_embedding=relative_pos_embedding)

            self.down_4 = Up((net_dimension)*2, net_dimension)
            self.att_4 = StageModule(in_channels=net_dimension, hidden_dimension=hidden_dim[3], layers=layers[3],
                                    downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim[3],
                                    window_size=window_size[3], relative_pos_embedding=relative_pos_embedding)
        else:
            self.down_2 = Up(net_dimension*4 , net_dimension*2)
            self.att_2 = StageModule(in_channels=net_dimension*2, hidden_dimension=hidden_dim[1], layers=layers[1],
                                    downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim[1],
                                    window_size=window_size[1], relative_pos_embedding=relative_pos_embedding)

            self.down_3 = Up((net_dimension*2)*2, net_dimension)
            self.att_3 = StageModule(in_channels=net_dimension, hidden_dimension=hidden_dim[2], layers=layers[2],
                                    downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim[2],
                                    window_size=window_size[2], relative_pos_embedding=relative_pos_embedding)

            self.down_4 = Up(net_dimension, net_dimension)
            self.att_4 = StageModule(in_channels=net_dimension, hidden_dimension=hidden_dim[3], layers=layers[3],
                                    downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim[3],
                                    window_size=window_size[3], relative_pos_embedding=relative_pos_embedding)

    def forward(self, x_t0, x_t1=None, skips=None):

        if skips != None:
            x1,x2,x3 = skips

            x_t0 = self.down_1(x_t0)
            x_t0 = self.att_1(x_t0)
            x_t0 = self.down_2(torch.cat([x_t0,x3], dim=1))
            x_t0 = self.att_2(x_t0)
            x_t0 = self.down_3(torch.cat([x_t0,x2], dim=1))
            x_t0 = self.att_3(x_t0)
            x_t0 = self.down_4(torch.cat([x_t0,x1], dim=1))
            x_t0 = self.att_4(x_t0)
            x = x_t0

        else:
            # Process t_0 img
            x = self.down_1(x_t0)
            x = self.att_1(x)
            x = self.down_2(x)
            x = self.att_2(x)

            # Process t_1 img
            x_t1 = self.down_1(x_t1)
            x_t1 = self.att_1(x_t1)
            x_t1 = self.down_2(x_t1)
            x_t1 = self.att_2(x_t1)
            
            # Concatenate t_0 and t_1
            x = torch.cat([x, x_t1], dim=1)

            x = self.down_3(x)
            x = self.att_3(x)
            x = self.down_4(x)
            x = self.att_4(x)
        
        # x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # Models parameters
    from cnn import Up
    from swin_transformer import StageModule

    in_channels = 512
    net_dimension = 64
    hidden_dim = [net_dimension*4, net_dimension*2, net_dimension, 2]
    layers = [2, 2, 2, 2]
    heads = [4, 4, 4, 4]
    head_dim = [32, 32, 32, 32]
    window_size = [2, 2, 2, 2]
    downscaling_factors = [1, 1, 1, 1]

    # Instantiate model
    model = Decoder(in_channels=in_channels, hidden_dim=hidden_dim, layers=layers, heads=heads, 
                    head_dim=head_dim, window_size=window_size, downscaling_factors=downscaling_factors, with_skips=False).to("cuda")

    # Test model
    img = torch.zeros((5,512,16,16)).to("cuda")
    skips = [torch.zeros((5,64,128,128)).to("cuda"), torch.zeros((5,128,64,64)).to("cuda"), torch.zeros((5,256,32,32)).to("cuda")]
    out = model(img, img)
    print(out.shape)