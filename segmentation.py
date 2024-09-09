#author: L. Stival
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from modules.cnn import Up
from eval_metrics import Eval
from data_reader.OSCD_ms_reader import MSReadData
from utils.utils import get_model_time, plot_images, python_files, to_img, is_notebook
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Loss
import monai

# Eval de model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# from train.change_train import DiceLoss
from torchmetrics.classification import Dice
from train.change_train import DiceLoss

class SegementationNetwork(nn.Module):
    def __init__(self):
        super(SegementationNetwork, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        #Change the number of input channels to 13
        self.encoder.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        #Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # List of residuals
        self.intermediate = {6: None, 5: None, 4: None}

        # Define the Decoder (segmentation model)
        self.up1 = Up(512, 256)
        self.up2 = Up(2*256, 128)
        self.up3 = Up(2*128, 64)
        self.up4 = Up(2*64, 64)
        self.up5 = Up(64, 1)


    def forward(self, x1, x2):
        # x = torch.sub(x1, x2)
        
        x1_intermediate = {6: None, 5: None, 4: None}
        x2_intermediate = {6: None, 5: None, 4: None}

        for idx, layer in enumerate(self.encoder):
            x1 = layer(x1)
            x2 = layer(x2)
            if idx in {4, 5, 6}:
                x1_intermediate.update({idx: x1})
                x2_intermediate.update({idx: x2})

        self.intermediate = {6: torch.abs(x1_intermediate[6] - x2_intermediate[6]),
                             5: torch.abs(x1_intermediate[5] - x2_intermediate[5]),
                             4: torch.abs(x1_intermediate[4] - x2_intermediate[4])}

        x = torch.abs(torch.sub(x1, x2))

        x = self.up1(x)
        x = self.up2(x, self.intermediate[6])
        x = self.up3(x, self.intermediate[5])
        x = self.up4(x, self.intermediate[4])
        x = self.up5(x)
        
        return x
    
class TrainSegementationNetwork():
    def __init__(self, model_name=None) -> None:
        self.model = SegementationNetwork()
        if model_name:
            self.model.load_state_dict(torch.load(f"./trained_models/segmentation/{model_name}/seg_model.pth"))
        self.model.train()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = DiceLoss().to(device)
        self.criterion = monai.losses.GeneralizedDiceLoss(sigmoid=True).to(device)
        # self.criterion = Dice(average='micro').to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.model = self.model.to(self.device)
        self.name_to_save = get_model_time()
    
    def save(self, path):
        file_name = "seg_model.pt"
        torch.save(self.model.state_dict(), os.path.join(path, file_name))

    def train(self, dataloader, val_dataloader, epochs=200):
        pbar = tqdm(range(epochs))
        loss = 0.
        val_acc = 0.
        acc = 0.
        pbar.set_description(f"Loss: {loss:.4f}, F1: {acc:.4f}, Val F1: {val_acc:.4f}")
        for epoch in pbar:
            for i, (x1, x2, labels) in enumerate(dataloader):
                acc = 0.
                self.model.train()
                self.optimizer.zero_grad()
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device)
                out = self.model(x1, x2)
                loss = self.criterion(out, labels.int())
                loss.backward()
                self.optimizer.step()
                acc += self.__accuracy__(out, labels)
                pbar.set_description(f"Loss: {loss:.4f}, F1: {acc:.4f}, Val F1: {val_acc:.4f}")
                pbar.update(1)
                
            if epoch % 50 == 0:
                out = torch.where(out > 0.5, 1, 0)
                self.__plot__(out, labels)

                os.makedirs(f"./trained_models/segmentation/{self.name_to_save}", exist_ok=True)
                self.save(f"./trained_models/segmentation/{self.name_to_save}")

            val_acc = self.evaluate(val_dataloader)
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('F1/train', acc, epoch)
            writer.add_scalar('F1/val', val_acc, epoch)

            pbar.set_description(f"Loss: {loss:.4f}, F1: {acc:.4f}, Val F1: {val_acc:.4f}")
            pbar.update(1)
            # self.scheduler.step()

        os.makedirs(f"./trained_models/segmentation/{self.name_to_save}", exist_ok=True)
        self.save(f"./trained_models/segmentation/{self.name_to_save}")

        print("Finished Training")

    def __plot__(self, out, y):
        pic = to_img(out[:5].cpu().data)
        y_pic = to_img(y[:5].cpu().data)

        if is_notebook():
            plot_images(pic[:5], cmap="gray")
            plot_images(y_pic[:5], cmap="gray")
    
    def __accuracy__(self, outputs, labels):
        predicted = outputs.argmax(dim=1)
        predicted = predicted.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        acc = eval.f1_score(labels.flatten().astype(int), predicted.flatten())
        return acc 
    
    def evaluate(self, val_dataloader):
        val_acc = 0.
        with torch.no_grad():
            self.model.eval()
            for i, (x1, x2, labels) in enumerate(val_dataloader):
                val_acc = 0.
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device)
                out = self.model(x1, x2)
                val_acc += self.__accuracy__(out, labels)
                val_acc /= len(labels)

            return val_acc 
    
class EvalSegementationNetwork():
    def __init__(self, model_name) -> None:
        self.eval = Eval()
        self.df_metrics = pd.DataFrame(columns=["F1", "Precision", "Recall"])
        self.model = SegementationNetwork()
        self.model.load_state_dict(torch.load(f"./trained_models/segmentation/{model_name}/seg_model.pt"))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self, dataloader):
        metrics = {"f1": [], "precision": [], "recall": []}
        all_outputs = []
        all_labels = []

        for i, (x1, x2, labels) in enumerate(dataloader):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            labels = labels.to(self.device)
            out = self.model(x1, x2)
            predicted = torch.where(out > 0.5, 1, 0)
            predicted = predicted.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            y_pred, f1, precision, recall = self.eval.evaluate(labels.flatten().astype(int), predicted.flatten())
            all_outputs.append(y_pred)
            all_labels.append(labels.flatten().astype(int))

            metrics["f1"].append(f1)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
        
        df = pd.DataFrame(metrics)
        df.columns = ["F1", "Precision", "Recall"]

        conf_mtx = eval_metrics.confusion_matrix(np.array(all_labels).flatten(), np.array(all_outputs).flatten())
        return df, conf_mtx, all_outputs, all_labels

if __name__ == "__main__":
    # Load the eval
    eval = Eval()
    train = True
    batch_size = 20
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define tensorboard log directory
    os.makedirs("logs/segmentation", exist_ok=True)
    log_dir = f"logs/segmentation/{get_model_time()}"
    writer = SummaryWriter(log_dir)

    # Load the data
    dataset = MSReadData()
    if train:
        dataloader, val_dataloader = dataset.create_dataLoader("./data/OSCD", 256, batch_size, train=True, num_workers=6)

        # Train the model   
        trainer = TrainSegementationNetwork()
        trainer.train(dataloader, val_dataloader, 1000)
    else:
        batch_size = 1
        dataloader = dataset.create_dataLoader("./data/OSCD", 256, 1, train=False, num_workers=1)

    # Create the model
    model = SegementationNetwork().to(device)

    model_name = "20240331_152901"

    # #  Evaluate the model
    from eval_metrics import Eval
    eval_metrics = Eval()
    eval = EvalSegementationNetwork(model_name)
    df, cf_mtx, predictions, all_labels = eval.evaluate(dataloader)
    print(df.mean())
    print(f"True Negative: {cf_mtx[0][0]}")
    print(f"False Positive: {cf_mtx[0][1]}")
    print(f"False Negative: {cf_mtx[1][0]}")
    print(f"True Positive: {cf_mtx[1][1]}")

    plot_images(torch.Tensor(predictions[5:10]).view(-1,1,256,256), cmap="gray")
    plot_images(torch.Tensor(all_labels[5:10]).view(-1,1,256,256), cmap="gray")