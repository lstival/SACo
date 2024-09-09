# Author: Leandro Stival

#### Torch
import torch
from torch.utils.data import DataLoader

#### My packages
from modules.u_net import Segmentation_Model
from data_reader.OSCD_ms_reader import  RemoteImagesDataset
from utils import plot_images
from change_detection_train import EvaluationMetrics, DiceLoss, split_image
from utils import get_model_time, plot_images

#### Others packages
from tqdm import tqdm
import os

if __name__ == "__main__":
    torch.manual_seed(42)   
    #### Instantiate the dataset
    path = "./data/OSCD/"
    batch_size = 1
    img_size=512
    train=False
    # mode = "best_loss"
    mode = "best_f1"

    #### Define the model to eval
    trained_model_name = "20240722_094110"
    model_name = get_model_time()

    path_save = f"./remote_features/trained_models/change_detection"

    # Create the dataloader
    dataset = RemoteImagesDataset(path, batch_size, img_size, train=train)
    val_dataset = RemoteImagesDataset(path, batch_size, img_size, train=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_datalaoder = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    ## Define the model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Segmentation_Model()

    ## Eval Best Loss
    print("Best Loss")
    if mode == "best_loss":
        weights_path = f"./remote_features/trained_models/change_detection/{trained_model_name}/best_model.pth"
    elif mode == "best_f1":
        weights_path = f"./remote_features/trained_models/change_detection/{trained_model_name}/best_f1.pth"
    else:
        print("Invalid mode")
        exit()
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
 
    for param in model.encoder.parameters():
        param.requires_grad = False

    #### Train paramets
    epochs = 100 
    learning_rate = 1e-4
    criterion = DiceLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    #### Train process
    pbar = tqdm(range(epochs), desc="Epochs")
    best_loss = float('inf')
    best_f1 = 0
    for epoch in pbar:
        model.train()
        total_loss = 0
        # eval_metrics = EvaluationMetrics()
        for i, (t1, t2, labels) in enumerate(dataloader):
            t1 = torch.cat((t1[:,3:4], t1[:,2:3], t1[:,1:2]), dim=1)
            t2 = torch.cat((t2[:,3:4], t2[:,2:3], t2[:,1:2]), dim=1)
            t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
            labels = torch.where(labels > 0.5, torch.tensor(1.0), torch.tensor(0.0))

            optimizer.zero_grad()
            outputs = model(t1, t2)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
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

                ## Slices
                t1_slices = split_image(t1)
                t2_slices = split_image(t2)
                val_labels_slices = split_image(labels)

                ## No slice
                outputs_val = model(t1, t2)
                loss = criterion(outputs_val, val_labels)
                eval_metrics_val.update(outputs_val, val_labels)

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
            # plots = torch.cat((outputs[:2].cpu().detach(), labels_slices[:2].cpu().detach()), dim=0)
            plots = torch.cat((outputs[:2].cpu().detach(), labels[:2].cpu().detach()), dim=0)
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