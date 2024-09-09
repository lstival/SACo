#Author: Leandro Stival    
#### Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#### Other Packages
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

#### My packages
from data_reader.PASTIS_datalaoder import PASTIS_Dataset
from modules.segmentation_network import Segmentation_Model
from criterions import DiceLoss, FocalLoss, FocalCELoss, WeightedFocalLoss
from utils import plot_images, get_model_time, PASTIS_color_map, get_mIou, get_accuracy, collate_fn_PASTIS, get_PASTIS_class_weights, split_image

def get_model(fine_tunning_name=None, input_channel_change=False, unet=False):

    if fine_tunning_name is None and not unet:
        model = Segmentation_Model(input_channel_change=input_channel_change, num_classes=20)

        if input_channel_change:
            model.encoder[0] = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.encoder[0].load_state_dict({"weight": model.__convert_2_10_channels__()})
        print("SeCo")
        return model

    elif unet:
        model = smp.Unet(
            encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=10,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=20,                      # model output channels (number of classes in your dataset)
        )
        print("Unet")
        return model

    else:
        model = Segmentation_Model(input_channel_change=True, num_classes=20)
        ## Fine Tuinning segmentation SACo
        # weights = f"./remote_features/train_models/segmentation/grid_search/{fine_tunning_name}/best_pixel_segmentation.pth"
        weights = f"./remote_features/train_models/segmentation/{fine_tunning_name}/best_pixel_segmentation.pth"
        model.load_state_dict(torch.load(weights))
        print("fine tunning SACo")

        ## New SACO
        # new_weights = OrderedDict()
        # weights_path = f"./remote_features/trained_models/encoder/{fine_tunning_name}/best_model.pth"
        # weights = torch.load(weights_path)

        # for key in list(weights.keys()):
        #     if "model.model." in key:
        #         new_weights[key.replace("model.model.", "encoder.")] = weights[key]

        # for key in model.state_dict().keys():
        #     if key not in new_weights.keys():
        #         new_weights[key] = model.state_dict()[key]

        # model.load_state_dict(new_weights)

        # if input_channel_change:
        #     model.encoder[0] = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     model.encoder[0].load_state_dict({"weight": model.__convert_2_10_channels__()})

    
        # print("SACo")
    
        return model

def get_data_loaders(data_path, train_folds, val_folds, batch_size):
    use_mono_date = params["mono_date"]
    if use_mono_date:
        train_dataset = PASTIS_Dataset(data_path, norm=True, folds=train_folds, mono_date="2018-09-01")
        val_dataset = PASTIS_Dataset(data_path, norm=True, folds=val_folds, mono_date="2018-09-01")
    else:
        train_dataset = PASTIS_Dataset(data_path, norm=True, folds=train_folds)
        val_dataset = PASTIS_Dataset(data_path, norm=True, folds=val_folds)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_PASTIS
    )

    val_dataset = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_PASTIS
    )

    return dataloader, val_dataset

if __name__ == "__main__":
    import json
    #### Check if cuda is available
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cmap = PASTIS_color_map()

    ## Load Json file with training parameters
    json_train_file = "./segmentation_PASTIS_params.json"
    with open(json_train_file, "r") as file:
        params = json.load(file)
    print(params)

    # "20240513_130522",

    ## Define the train params
    batch_size = params["batch_size"]
    epochs = params["epochs"]

    ## Define the model params
    encoder_name = params["encoder_name"]
    model_name = get_model_time()
    model = get_model(encoder_name, input_channel_change=True, unet=False)
    for param in model.encoder.parameters():
        param.requires_grad = params["fine_tune_encoder"]
    model.train()
    model.to(device)

    ## Define the data params
    data_path = params["data_path"]
    train_folds = params["train_folds"]
    val_folds = params["val_folds"]

    ## Define the log directory
    log_path = params["log_path"]   
    log_dir = f"{log_path}/{model_name}"
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    ## Define the path to save the model
    path_to_save = f"{params['path_to_save']}/{model_name}"
    os.makedirs(path_to_save, exist_ok=True)

    ## Define the dataloaders
    split_images = params["split_data"] # Split the images in patches of 32x32
    dataloader, val_dataset = get_data_loaders(data_path, train_folds, val_folds, batch_size)

    ## Define the loss function
    use_class_weights = params["use_class_weights"]
    use_dice_loss = params["use_dice_loss"]

    if use_class_weights:
        class_weights = get_PASTIS_class_weights()
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=torch.tensor(class_weights["weights"])).to(device)
        # criterion = FocalCELoss(ignore_index=0,gamma=2).to(device)
        # criterion = WeightedFocalLoss(gamma=2, weight=torch.tensor(class_weights["weights"]).to(device)).to(device)
        # criterion = DiceLoss(use_class_weights=True).to(device)
        if use_dice_loss:
            # criterion_dice = DiceLoss(use_class_weights=True).to(device)
            criterion_dice = DiceLoss(use_class_weights=False).to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)
        # criterion = FocalCELoss(ignore_label=0,gamma=2).to(device)
        if use_dice_loss:
            criterion_dice = DiceLoss().to(device)

    ## Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ## Define the alpha for the dice loss
    alpha = 0.01

    ## Define the best validation loss
    best_val_loss = np.inf
    val_total_mIoU = 0
    val_total_OA = 0

    ## Train the model
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        train_total_loss = 0
        train_total_mIoU = 0
        train_total_OA = 0
        model.train()
        for idx, (images, targets) in enumerate(dataloader):
            if split_images:
                images = split_image(images)
                targets = split_image(targets)
            img = images.to(device)
            # rgb_img = torch.stack((img[:,2], img[:,1], img[:,0]), dim=1)
            rgb_img = img
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_img)
            loss = criterion(outputs, targets.long())
            if use_dice_loss:
                loss += alpha * criterion_dice(outputs, targets.long())
            
            train_total_loss += loss.item()   
            loss.backward()
            optimizer.step()

            ## Metrics
            predictions = torch.argmax(outputs, dim=1)

            miou = get_mIou(predictions, targets)
            oa = get_accuracy(predictions, targets)
            
            train_total_mIoU += miou.item()
            train_total_OA += oa

            pbar.set_postfix({"Loss": train_total_loss / (idx + 1),
                                "train mIoU": train_total_mIoU / (idx + 1),
                                "train OA": train_total_OA / (idx + 1),
                                "val mIoU": val_total_mIoU / val_dataset.__len__(),
                                "val OA": val_total_OA / val_dataset.__len__()})

        #Log the metrics
        writer.add_scalar('Loss/train', train_total_loss / (idx + 1), epoch)
        writer.add_scalar('mIoU/train', train_total_mIoU / (idx + 1), epoch)
        writer.add_scalar('OA/train', train_total_OA / (idx + 1), epoch)

        torch.save(model.state_dict(), f"{path_to_save}/pixel_segmentation.pth")

        #### Eval method
        val_total_mIoU = 0
        val_total_OA = 0
        model.eval()

        with torch.no_grad():
            for idx, (images, val_targets) in enumerate(val_dataset):
                if split_images:
                    images = split_image(images)
                    val_targets = split_image(val_targets)

                img = images.to(device)
                # rgb_img = torch.stack((img[:,2], img[:,1], img[:,0]), dim=1)
                rgb_img = img
                val_targets = val_targets.to(device)

                outputs = model(rgb_img)
                val_predictions = torch.argmax(outputs, dim=1)
                val_loss = criterion(outputs, val_targets.long())
                if use_dice_loss:
                    val_loss += alpha * criterion_dice(outputs, val_targets.long())

                mIoU = get_mIou(val_predictions, val_targets)
                oa = get_accuracy(val_predictions, val_targets)

                val_total_mIoU += mIoU.item()
                val_total_OA += oa

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{path_to_save}/best_pixel_segmentation.pth")
            
            pbar.set_postfix({"Loss": train_total_loss / dataloader.__len__(),
                                "train mIoU": train_total_mIoU / dataloader.__len__(),
                                "train OA": train_total_OA / dataloader.__len__(),
                                "val mIoU": val_total_mIoU / (idx +1),
                                "val OA": val_total_OA / (idx + 1)})
            
            # Log validation metrics
            writer.add_scalar('Loss/val', val_loss.item(), epoch)
            writer.add_scalar('mIoU/val', val_total_mIoU, epoch)
            writer.add_scalar('OA/val', val_total_OA, epoch)

        alpha += 0.01
        
        # Plot the masks
        if epoch % 10 == 0:
            print("Train predictions")
            plot_images(predictions.unsqueeze(1)[:5,], cmap=cmap)
            plot_images(targets.unsqueeze(1)[:5], cmap=cmap)
            print("Val predictions")
            plot_images(val_predictions.unsqueeze(1)[:5,], cmap=cmap)
            plot_images(val_targets.unsqueeze(1)[:5], cmap=cmap)

    print(model_name)
    print("Finished Training")