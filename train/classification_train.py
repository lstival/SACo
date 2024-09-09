#Author: L. Stival
# This file trainig the network for the classification of the EuroSAT dataset
# using different backbones

from modules.senti_encoder import load_model_and_weights
from modules.classification_network import ClassificationNetwork
from data_reader.EuroSAT_ms_reader import MSReadData
from modules import resnet
import modules.sentinel_bands as sentinel_bands
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils.utils import get_model_time
from torch.utils.tensorboard import SummaryWriter

# dataset_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

class ClassificationTrain():
    def __init__(self) -> None:
        #Create folder to save the trained models
        self.model_name = get_model_time()
        self.model_path = f"../trained_models/classifcation/{self.model_name}"
        os.makedirs(self.model_path, exist_ok=True)

    def validate(self, encoder, decoder, test_loader, criterion, device):
        """
        This method validate the network

        Args:
        - encoder: SentinelEncoder - The encoder model
        - decoder: ClassificationNetwork - The decoder model
        - test_loader: DataLoader - The test data loader
        - criterion: nn.CrossEntropyLoss - The loss function
        - device: torch.device - The device to use for the training
        """
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                _ = encoder(images)
                features = encoder.model.cnn_out
                outputs = decoder(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the network on the test images: {accuracy} %')

    def train(self, encoder, decoder, train_loader, test_loader, num_epochs, criterion, optimizer, device):
        """
        This method train the network

        Args:
        - encoder: SentinelEncoder - The encoder model
        - decoder: ClassificationNetwork - The decoder model
        - train_loader: DataLoader - The train data loader
        - test_loader: DataLoader - The test data loader
        - num_epochs: int - The number of epochs
        - criterion: nn.CrossEntropyLoss - The loss function
        - optimizer: optim.Adam - The optimizer
        - device: torch.device - The device to use for the training
        """
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (data, labels) in pbar:
                data = data.to(device)
                labels = labels.to(device)
                
                # Forward pass
                _ = encoder(data)
                features = encoder.model.cnn_out
                # features = encoder(data)
                outputs = decoder(features)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

            # Validate the network
            self.validate(encoder, decoder, test_loader, criterion, device)
            # Save the model
            torch.save(decoder.state_dict(), f"{self.model_path}/classification.pt")

        print('Finished Training')


if __name__ == "__main__":
    # Create the writer
    from modules.senti_encoder import load_model_and_weights, SentinelEncoder
    writer = SummaryWriter()
    
    # parameters
    image_size = 256
    num_classes = 10
    batch_size = 10
    num_epochs = 10
    train=True

    model_name = "20240315_204706"
    model_path = f"../trained_models/encoder/{model_name}/q_weights.pt"
    bands_selector = sentinel_bands.SentinelGroupBands()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # encoder = load_model_and_weights(model_path, bands_selector)
    # encoder.eval()

    encoder = SentinelEncoder(resnet.ResNet_encoder(), bands_selector).to(device)
    encoder.eval()

    decoder = ClassificationNetwork(num_classes).to(device)

    # Load the data
    root_path = "./data/EuroSAT"
    dataset = MSReadData()
    dataloader, test_dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=2e-4, weight_decay=1e-3)

    # Train the network
    train = ClassificationTrain()
    train.train(encoder, decoder, dataloader, test_dataloader, num_epochs, criterion, optimizer, device)
