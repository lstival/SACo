# Description: Evaluate the naive classification model with LBP features
# Author: Leandro Stival

# My libraries
from data_reader.EuroSAT_ms_reader import MSReadData
from classification import EvalClassification
from classification_lbp import ClassificationNetworkLBP
from modules.senti_encoder import SentinelEncoderLBP
from modules.sentinel_bands import EuroSATGroupBands
from modules.resnet import ResNet_enconder_LBP
from lbp import lbp_histogram
from tqdm import tqdm

# Base libraries
import os
import torch
import torch.nn as nn

class EvalClassificationLBP(EvalClassification):
    """
    A class for evaluating the naive classification model with LBP features.

    Args:
        model_name (str): The name of the model to load.
        model (nn.Module): The pre-trained model to use for evaluation.

    Attributes:
        model (nn.Module): The naive classification model with LBP features.
        device (str): The device to use for evaluation (cuda or cpu).

    Methods:
        evaluate(dataloader): Evaluates the model on the given dataloader.
    """

    def __init__(self, model_name=None, model=None):
        """
        Initializes the EvalClassificationLBP class.

        Args:
            model_name (str): The name of the model to load.
            model (nn.Module): The pre-trained model to use for evaluation.
        """
        if model:
            self.model = model

        if model_name:
            encoder = SentinelEncoderLBP(bands_selector=EuroSATGroupBands(), model=ResNet_enconder_LBP())
            self.model = ClassificationNetworkLBP(encoder)
            self.model.resnet.model.fc =  nn.Linear(512+(13*16), 10)
            try:
                self.model.load_state_dict(torch.load(f"./trained_models/classification_lbp/{model_name}/best_model.pth"))
                print("Loaded best model")
            except:
                self.model.load_state_dict(torch.load(f"./trained_models/classification_lbp/{model_name}/model.pth"))
                print("Loaded model")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self, dataloader):
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate the model on.

        Returns:
            acc (float): The accuracy of the model.
            all_outs (torch.Tensor): The predicted outputs of the model.
            all_labels (torch.Tensor): The true labels of the data.
        """
        self.model.eval()
        all_outs = []
        all_labels = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                x, labels = data

                hist = lbp_histogram((x*255).type(torch.int)).view(x.shape[0], -1).type(torch.float32)
                img = x.to(self.device)
                hist = hist.to(self.device)

                outputs = self.model(img, hist)
                labels = labels.to(self.device)

                all_outs.append(outputs)
                all_labels.append(labels)

        all_outs = torch.cat(all_outs)
        all_labels = torch.cat(all_labels)

        correct = (all_outs.argmax(dim=1) == all_labels).sum().item()
        acc = correct / len(all_labels)
        return acc, all_outs, all_labels

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix

    print("Main")
    root_path = "./data/EuroSAT_test"
    batch_size = 1
    image_size = 64
    train = False
    num_classes = 10

    # Data Reader
    dataset = MSReadData()
        
    dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train, num_workers=1)	

    # Eval to load model
    name = "20240501_094026"
    eval = EvalClassificationLBP(model_name=name)
    acc, all_outs, all_labels = eval.evaluate(dataloader)
    print(f"Accuracy: {acc}")

    df = pd.DataFrame(columns=["label", "predicted"])
    for i, (out, label) in enumerate(zip(all_outs, all_labels)):
        predicted = out.argmax(dim=0).cpu().numpy()
        label = label.cpu().numpy().item()
        df.loc[i] = [label, predicted]

    path = f"./results/classification/{name}/"
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}results.csv")

    # Count the number of correct predictions per class and the % of correct predictions

    classes = dataset.dataset.labels_list
    df = df.astype(int)
    df["correct"] = df["label"] == df["predicted"]
    df_correct = df.groupby("label").sum()
    df_total = df.groupby("label").count()
    df_correct["total"] = df_total["predicted"]
    df_correct["predicted"] = df_correct["predicted"]
    df_correct["percentage"] = df_correct["correct"] / df_correct["total"]
    df_correct["correct"].sum() / df_correct["total"].sum()

    # Put the class name considering the order of the classes
    df_correct["class"] = classes

    df_correct.to_csv(f"{path}results_per_class.csv")
    print(df_correct.sort_values(by="percentage"))

    y_true = df["label"]
    y_pred = df["predicted"]

    cm = confusion_matrix(y_true, y_pred)

    df_confusion_matrix = pd.DataFrame(cm, columns=classes, index=classes)

    # Convert absolute values to percentages per class
    df_confusion_matrix = df_confusion_matrix.div(df_confusion_matrix.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=df_confusion_matrix.values, display_labels=classes)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90)  # Rotate x labels by 90 degrees
    plt.show()

    forest_sum = df_confusion_matrix.loc["Forest"].sum()
    print(f"Sum of values for 'Forest': {forest_sum}")
