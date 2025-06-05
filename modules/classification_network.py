# Author: L. Stival
# This files creates the network for the classification of the EuroSAT dataset

import torch.nn as nn
import torch.nn.functional as F

class ClassificationNetwork(nn.Module):
    """
    A neural network model for classification of the EuroSAT dataset.

    Args:
        img_size (int): The size of the input image.
        num_classes (int): The number of classes in the dataset.

    Attributes:
        fc1 (nn.Linear): The fully connected layer.
        softmax (nn.Softmax): The softmax activation function.

    """

    def __init__(self, num_classes):
        super(ClassificationNetwork, self).__init__()
        self.fc1 = nn.Linear(512 * 8 * 8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    import torch

    num_classes = 10

    network = ClassificationNetwork(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(5, 512, 8, 8).to(device) # Shape of output of Resnet

    network.to(device)

    output = network(x)
    
    # Print the index of the maximum value of the output
    print(output.argmax(dim=1))
    # for i in output:
    #     print(i.argmax().item())    
