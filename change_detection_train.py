#### Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.models import resnet18, ResNet18_Weights

#### My packages
from modules.u_net import Segmentation_Model
from data_reader.OSCD_ms_reader import  RemoteImagesDataset
from utils import plot_images, get_model_time
from CaCo_seg import get_segmentation_model

#### Others packages
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from collections import OrderedDict
# from torchmetrics.classification import Dice
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Aplicar a função de ativação sigmoid para converter as previsões em probabilidades
        inputs = torch.sigmoid(inputs)
        
        # Achatar as dimensões [batch, channels, height, width] para [batch, -1]
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calcular a interseção
        intersection = (inputs_flat * targets_flat).sum()
        
        # Calcular o coeficiente Dice e subtrair de 1 para obter a perda Dice
        dice_coefficient = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coefficient
        
        return dice_loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=2.0, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        
        dice_coefficient = (2. * self.weight * intersection + self.smooth) / (self.weight * targets_flat.sum() + inputs_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coefficient
        
        return dice_loss

def split_image(img):
    patches = []
    b, c, h, w = img.shape
    patch_size = 96

    # Calculate the number of patches needed
    num_patches_height = (h + patch_size - 1) // patch_size
    num_patches_width = (w + patch_size - 1) // patch_size

    # Calculate the padding required
    pad_height = num_patches_height * patch_size - h
    pad_width = num_patches_width * patch_size - w

    # Pad the image if necessary
    img = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height))

    for i in range(0, h + pad_height, patch_size):
        for j in range(0, w + pad_width, patch_size):
            patch = img[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return torch.cat(patches, dim=0)

class EvaluationMetrics:
    def __init__(self):
        self.predictions = []
        self.labels = []

    def update(self, outputs, labels):
        self.predictions.append(outputs)
        self.labels.append(labels)

    def sklearn_precision(self):
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return precision_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5, zero_division=1.0)

    def sklearn_recall(self):
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return recall_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5)

    def sklearn_f1_score(self):
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return f1_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5)
    
def train_model():
    #### Train process
    pbar = tqdm(range(epochs), desc="Epochs")
    best_loss = float('inf')
    best_f1 = 0
    for epoch in pbar:
        model.train()
        total_loss = 0
        # eval_metrics = EvaluationMetrics()
        for i, (t1, t2, labels) in enumerate(dataloader):
            
            t1 = torch.cat((t1[:,3:4], t1[:,2:3], t1[:,1:2]), dim=1) # RGB
            t2 = torch.cat((t2[:,3:4], t2[:,2:3], t2[:,1:2]), dim=1)
            
            t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
            labels = torch.where(labels > 0.5, torch.tensor(1.0), torch.tensor(0.0))

            t1_slices = split_image(t1)
            t2_slices = split_image(t2)
            labels_slices = split_image(labels)

            outputs = model(t1_slices, t2_slices)
            loss = criterion(outputs, labels_slices)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        ## Eval model
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            eval_metrics_val = EvaluationMetrics()
            for i, (t1, t2, labels) in enumerate(val_datalaoder):
                
                t1 = torch.cat((t1[:,3:4], t1[:,2:3], t1[:,1:2]), dim=1)
                t2 = torch.cat((t2[:,3:4], t2[:,2:3], t2[:,1:2]), dim=1)
            
                t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
                val_labels = torch.where(labels > 0.5, torch.tensor(1.0), torch.tensor(0.0))
                # labels = labels.to(torch.int)
                # val_labels = labels

                ## Slices
                t1_slices = split_image(t1)
                t2_slices = split_image(t2)
                val_labels_slices = split_image(labels)
                outputs_val = model(t1_slices, t2_slices)
                loss = criterion(outputs_val, val_labels_slices)
                eval_metrics_val.update(outputs_val, val_labels_slices)

                val_total_loss += loss.item()
                
        precision = eval_metrics_val.sklearn_precision()
        recall = eval_metrics_val.sklearn_recall()
        f1 = eval_metrics_val.sklearn_f1_score()

        pbar.set_description(f"Epoch: {epoch} - Val Loss: {val_total_loss/len(val_datalaoder):.5f} - Loss: {total_loss/len(dataloader):.5f} - Precision: {precision:.5f} - Recall: {recall:.5f} - F1: {f1:.5f}")
    
        # Save the model
        model_path_save =  f"{path_save}/{model_name}"
        os.makedirs(model_path_save, exist_ok=True)

        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(model.state_dict(), f"{model_path_save}/best_model.pth")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{model_path_save}/best_f1.pth")

        if epoch % 10 == 0:
            # Binarization of the output
            outputs = torch.where(outputs > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            plots = torch.cat((outputs[:2].cpu().detach(), labels_slices[:2].cpu().detach()), dim=0)
            plot_images(plots, cmap="gray")

            # Plot the val 
            outputs_val = torch.where(outputs_val > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            try:
                plots_val = torch.cat((outputs_val[:2].cpu().detach(), val_labels[:2].cpu().detach()), dim=0)
            except:
                plots_val = torch.cat((outputs_val[:2].cpu().detach(), val_labels_slices[:2].cpu().detach()), dim=0)
            plot_images(plots_val, cmap="gray")

            torch.save(model.state_dict(), f"{model_path_save}/model.pth")

    print(model_name)
    return best_loss, best_f1

def get_model(my_model=True, CaCo=False):
    model = Segmentation_Model(input_channel_change=False)

    if my_model:
        ## Load weights
        if fine_tunning_name is not None:
            print("SaCo")
            new_weights = OrderedDict()
            weights_path = f"./remote_features/trained_models/encoder/{fine_tunning_name}/best_model.pth"
            weights = torch.load(weights_path)

            for key in list(weights.keys()):
                if "model.model." in key:
                    new_weights[key.replace("model.model.", "encoder.")] = weights[key]

            for key in model.state_dict().keys():
                if key not in new_weights.keys():
                    new_weights[key] = model.state_dict()[key]

            model.load_state_dict(new_weights)

    if CaCo:
        print("CaCo model")
        weights = torch.load(f"C:/Users/stiva/Downloads/seco_resnet18_1m.ckpt")

        ### SeCo
        print("Seco")
        encoder_q_weights = OrderedDict()
        for key in weights["state_dict"].keys():
            if "encoder_q" in key:
                new_key = key.replace("encoder_q.", "")
                encoder_q_weights[new_key] = weights["state_dict"][key]

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-2])
        model.load_state_dict(encoder_q_weights, strict=False)
        backbone = model
        model = get_segmentation_model(backbone, [2, 4], [64, 64, 128, 256, 512])
    model.to(device)
    return model

def get_datasets(path):
    dataset = RemoteImagesDataset(path, batch_size, img_size, train=train)
    val_dataset = RemoteImagesDataset(path, batch_size, img_size, train=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_datalaoder = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader, val_datalaoder

if __name__ == "__main__":
    print("main")
    torch.manual_seed(42)

    #### Instantiate the dataset
    path = "./data/OSCD/"
    batch_size = 1
    img_size=512
    train=True
    dataloader, val_datalaoder = get_datasets(path)

    fine_tunning_name = "20240513_130522"
    # fine_tunning_name = None

    #### Path to save
    path_save = f"./remote_features/trained_models/change_detection"
    os.makedirs(path_save, exist_ok=True)

    #### Intance the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # #### Train paramets
    epochs = 100
    
    ####### Define hyperparameter grid
    learning_rates = [1e-3, 1e-4]
    optimizers = ['Adam', 'AdamW']
    my_model = False
    CaCo = False

    hyperparameter_combinations = list(itertools.product(learning_rates, optimizers))

    # Placeholder for tracking best performance and hyperparameters
    best_performance = float('inf')
    best_hyperparameters = {}
    best_encoder_method = {}

    for lr, optimizer_name in hyperparameter_combinations:
        # Initialize model, criterion, and optimizer for each combination
        model_name = get_model_time()
        model = get_model(my_model=my_model, CaCo=CaCo)# Load the model and weights
        criterion = DiceLoss().to(device)

        ### Frezze encoder weights
        for param in model.encoder.parameters():
            param.requires_grad = False
    
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Reset scheduler for each combination
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        current_performance, current_f1 = train_model()

        # Update best performance and hyperparameters
        if current_performance < best_performance:
            best_performance = current_performance
            best_hyperparameters = {'lr': lr, 'optimizer': optimizer_name}
            best_model_name = model_name
            best_model = model

    # Print best hyperparameters
    print(f"Best Hyperparameters: {best_hyperparameters}")
    print(f"Best Performance: {best_performance}")
    print(f"Best Encoder Method: {best_encoder_method}")
    print(f"Best Model Name: {best_model_name}")