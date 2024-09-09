# Author: Leandro Stival

#### Torch
import torch
from torch.utils.data import DataLoader

#### My packages
from modules.u_net import Segmentation_Model
from data_reader.OSCD_ms_reader import  RemoteImagesDataset
from utils import plot_images

#### Others packages
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

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
        return precision_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5)

    def sklearn_recall(self):
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return recall_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5)

    def sklearn_f1_score(self):
        predictions = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        return f1_score(labels.cpu().detach().numpy().flatten().astype(int)> 0.5, predictions.cpu().detach().numpy().flatten().astype(int)> 0.5)

class EvalMethod() :
    def __init__(self, model, dataloader, slices):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.slices = slices

    def eval(self):
        eval_metrics = EvaluationMetrics()

        pbar = tqdm(self.dataloader, desc="Eval")

        with torch.no_grad():
            self.model.eval()
            for i, (t1, t2, labels) in enumerate(pbar):
                
                t1 = torch.cat((t1[:,3:4], t1[:,2:3], t1[:,1:2]), dim=1)
                t2 = torch.cat((t2[:,3:4], t2[:,2:3], t2[:,1:2]), dim=1)
                t1, t2, labels = t1.to(self.device), t2.to(self.device), labels.to(self.device)
                labels = torch.where(labels > 0.5, torch.tensor(1.0), torch.tensor(0.0))

                #### Slices
                if self.slices:
                    sliced_t1 = split_image(t1)
                    sliced_t2 = split_image(t2)
                    sliced_labels = split_image(labels)

                    outputs = self.model(sliced_t1, sliced_t2)
                    outputs = torch.sigmoid(outputs)
                    outputs = torch.where(outputs > 0.5, torch.tensor(1.0), torch.tensor(0.0))

                    # Eval the outputs
                    eval_metrics.update(outputs, sliced_labels)
                    plots = torch.cat((outputs[:3].cpu().detach(), sliced_labels[:3].cpu().detach()), dim=0)

                #### No slices
                else:
                    ## Predict the change
                    outputs = self.model(t1, t2)
                    outputs = torch.sigmoid(outputs)
                    outputs = torch.where(outputs > 0.5, torch.tensor(1.0), torch.tensor(0.0))
                    eval_metrics.update(outputs, labels)
                    plots = torch.cat((outputs.cpu().detach(), labels.cpu().detach()), dim=0)
                    
                plot_images(plots, cmap="gray")

        # Estimate the metrics
        avg_precision = eval_metrics.sklearn_precision()
        avg_recall = eval_metrics.sklearn_recall()
        avg_f1 = eval_metrics.sklearn_f1_score()

        # Print the metrics
        print(f"Precision: {avg_precision:.5f}")
        print(f"Recall: {avg_recall:.5f}")
        print(f"F1: {avg_f1:.5f}")

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

if __name__ == "__main__":
    torch.manual_seed(42)   
    #### Instantiate the dataset
    path = "./data/OSCD/"
    batch_size = 1
    img_size=512
    train=False
    slices = True

    #### Define the model to eval
    trained_model_name = "20240722_094110"
    print(trained_model_name)

    # Create the dataloader
    dataset = RemoteImagesDataset(path, batch_size, img_size, train=False)
    dataloader = DataLoader(dataset,
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
    weights_path = f"./remote_features/trained_models/change_detection/{trained_model_name}/best_model.pth"
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    eval_model = EvalMethod(model, dataloader, slices)
    eval_model.eval()
