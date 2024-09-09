#Author: Leandro Stival    
#### Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#### Other Packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp

#### My packages
from data_reader.PASTIS_datalaoder import PASTIS_Dataset
from modules.segmentation_network import Segmentation_Model
from utils import plot_images, PASTIS_color_map, get_mIou, get_accuracy, collate_fn_PASTIS

def get_segmentation_model(model_name, unet=False, grid=False):

    if unet:
        model_path = f"./remote_features/train_models/segmentation/{model_name}/best_pixel_segmentation.pth"

        weights = torch.load(model_path)

        model = smp.Unet(
        encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=10,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=20,                      # model output channels (number of classes in your dataset)
        )

        model.load_state_dict(weights)

        print("Unet")
        return model

    else:
        model = Segmentation_Model(input_channel_change=False, num_classes=20)
        if grid:
            model_path = f"./remote_features/train_models/segmentation/grid_search/{model_name}/best_pixel_segmentation.pth"
        else:
            model_path = f"./remote_features/train_models/segmentation/{model_name}/best_pixel_segmentation.pth"
        # model_path = f"./remote_features/train_models/segmentation/grid_search/{model_name}/best_pixel_segmentation.pth"
        weights = torch.load(model_path)

        model.encoder[0] = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.encoder[0].load_state_dict({"weight": model.__convert_2_10_channels__()})
        model.load_state_dict(weights)

        print("SACo")

        return model

def get_test_dataloader(data_path, batch_size, folds=[1], mono_date=None):
    dataset = PASTIS_Dataset(data_path, norm=True, folds=folds, mono_date=mono_date)

    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
        collate_fn=collate_fn_PASTIS
    )

    return test_dataloader

if __name__ == "__main__":
    #### Check if cuda is available
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PASTIS_cmap = PASTIS_color_map()

    #### Eval Features
    batch_size = 1
    image_size = 128
    data_path = "./data/PASTIS"
    model_name = "20240818_125009"

    print(f"Loading model: {model_name}")
    # model = get_segmentation_model(model_name)
    model = get_segmentation_model(model_name, unet=False, grid=False)
    model.to(device)

    # dataset = PASTIS_Dataset(data_path, norm=True, folds=[1], mono_date="1")
    #### DataLoader
    dataloader = get_test_dataloader(data_path, batch_size, folds=[5])

    total_mIoU = 0
    total_OA = 0
    model.eval()
    all_predictions = []
    all_targets = []
    all_inputs = []

    tqdm_dataloader = tqdm(dataloader)
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm_dataloader):
            img = images.to(device)
            # rgb_img = torch.stack((img[:,2], img[:,1], img[:,0]), dim=1)
            rgb_img = img
            # all_inputs.append(rgb_img)
            targets = targets.to(device)

            outputs = model(rgb_img)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    miou = get_mIou(all_predictions, all_targets)
    oa = get_accuracy(all_predictions, all_targets)

    predict_img = all_predictions[:2].unsqueeze(1)
    target_img = all_targets[:2].unsqueeze(1)
    im2plot = torch.cat((predict_img, target_img), dim=0)

    acc = all_predictions == all_targets
    acc = acc.float().mean()

    plot_images(im2plot, cmap=PASTIS_cmap)

    print(f"mIoU: {miou}")
    print(f"OA: {oa}")
    print("Done")

    # #### Plot the best samples

    # for idx, (images, targets) in enumerate(dataloader):
    #     all_inputs.append(images.detach().cpu())

    # individual_acc = {}
    # threshold = 1 # Nao achou nada com mais de 7 classes
    # for idx in range(all_predictions.shape[0]):
    #     unique_values = torch.unique(all_targets[idx])
    #     predict_unique_values = torch.unique(all_predictions[idx])
    #     if len(unique_values) > threshold and len(predict_unique_values) > threshold:
    #         acc = get_accuracy(all_predictions[idx], all_targets[idx])
    #         individual_acc[idx] = acc

    # # Sort the dictionary by the value
    # individual_acc = OrderedDict(sorted(individual_acc.items(),
    #                 key=lambda x: x[1],
    #                 reverse=True))

    # ## Top 3 samples
    # top3_indices = []
    # top3_accs = []
    # top3 = list(individual_acc.items())[:3]
    # for img_idx, acc in top3:
    #     top3_indices.append(img_idx)
    #     top3_accs.append(acc)

    # ## Img top 3 to plot
    # predict_img = all_predictions[top3_indices].unsqueeze(1)
    # target_img = all_targets[top3_indices].unsqueeze(1)
    # tensor_all_inputs = torch.cat(all_inputs, dim=0)
    # rgb_img = tensor_all_inputs[top3_indices]

    # rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    # print("Top 3 samples")
    # plot_images(rgb_img)

    # print(f"Predictions acc: {top3_accs}")
    # plot_images(predict_img, cmap=PASTIS_cmap)

    # print("Targets")
    # plot_images(target_img, cmap=PASTIS_cmap)

    ## Confusion matrix for the pixels
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(all_targets.flatten(), all_predictions.flatten())
    labels = list(dataloader.dataset.labels_dict.values())
    
    # Calculate accuracy per class
    row_sums = cm.sum(axis=1)
    cm_normalized = cm / row_sums[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(25, 25))
    im = ax.imshow(cm_normalized, cmap="Blues")
    
    # Add text annotations inside each point
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{cm_normalized[i, j]*100:.1f}%', ha='center', va='center', color='black')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(im)
    plt.show()
