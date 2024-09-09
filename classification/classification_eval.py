# Author: L. Stival
"""
This file reads the classification model and calculates the accuracy, precision, and F1 score of the model.
"""

from modules.classification_network import ClassificationNetwork
from modules.senti_encoder import load_model_and_weights, sentinel_bands
from data_reader.EuroSAT_ms_reader import MSReadData
import torch
import pandas as pd
from tqdm import tqdm

list_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def read_encoder(model_name = "20240315_204706"):
    """
    Reads and returns an encoder model.

    Args:
        model_name (str): The name of the model to read. Defaults to "20240315_204706".

    Returns:
        encoder: The loaded encoder model.

    """
    model_path = f"../trained_models/encoder/{model_name}/q_weights.pt"
    bands_selector = sentinel_bands.SentinelGroupBands()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = load_model_and_weights(model_path, bands_selector)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder

def read_classification_model(model_name):
    """
    Reads the classification model from the specified model_name.

    Args:
        model_name (str): The name of the model.

    Returns:
        model (ClassificationNetwork): The loaded classification model.
    """
    model_path = f"../trained_models/classifcation/{model_name}/classification.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationNetwork(10)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_metrics(outputs, labels):
    """
    Calculates the accuracy, precision, recall, and F1 score based on the model outputs and labels.

    Args:
        outputs (torch.Tensor): The predicted outputs from the model.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        accuracy (float): The accuracy of the model.
        precision (float): The precision of the model.
        recall (float): The recall of the model.
        f1 (float): The F1 score of the model.
    """
    predicted = outputs.argmax(dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total

    # Calculate precision and F1 score
    tp = (predicted * labels).sum().item()
    fp = ((1 - labels) * predicted).sum().item()
    fn = (labels * (1 - predicted)).sum().item()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

def classification_eval(model, encoder, data_path, image_size=256, batch_size=16):
    """
    Evaluate the performance of a classification model on a given dataset.

    Args:
        model (torch.nn.Module): The classification model to evaluate.
        encoder (torch.nn.Module): The feature encoder model.
        data_path (str): The path to the dataset.
        image_size (int, optional): The size of the input images. Defaults to 256.
        batch_size (int, optional): The batch size for evaluation. Defaults to 16.

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation metrics (Accuracy, Precision, Recall, F1).
    """

    dataset = MSReadData()
    dataloader = dataset.create_dataLoader(data_path, image_size, batch_size, train=False, num_workers=1)

    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1 = []

    df_metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pbar = tqdm(dataloader, total=len(dataloader))
    pbar.set_description("Evaluating model")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            _ = encoder(images)
            outputs = model(encoder.model.cnn_out)

        accuracy, precision, recall, f1 = calculate_metrics(outputs, labels)
        total_accuracy.append(accuracy)
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)

    df_metrics['Accuracy'] = total_accuracy
    df_metrics['Precision'] = total_precision
    df_metrics['Recall'] = total_recall
    df_metrics['F1'] = total_f1

    return df_metrics

if __name__ == '__main__':
    print("Main")
    model_name = "20240325_173841"
    encoder_name = "20240315_204706"
    model = read_classification_model(model_name)
    encoder = read_encoder()

    data_path = "./data/EuroSAT"
    image_size = 256
    batch_size = 16

    df_metrics = classification_eval(model, encoder, data_path, image_size, batch_size)

    print(df_metrics.head())
    print(f"Average Accuracy: {df_metrics['Accuracy'].mean()}")
    print(f"Average Precision: {df_metrics['Precision'].mean()}")
    print(f"Average Recall: {df_metrics['Recall'].mean()}")
    print(f"Average F1: {df_metrics['F1'].mean()}")