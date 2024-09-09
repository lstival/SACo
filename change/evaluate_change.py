import os
import torch
import pytorch_lightning as pl
from modules.change_network import DecoderResNet18, ChangeDetection
from modules.senti_encoder import SentinelEncoder, load_model_and_weights

from utils.utils import get_model_time
    
class EvaluateChangeDetection:
    """Class for evaluating change detection models."""

    def __init__(self, encoder_path: str, decoder_path: str, device="cuda"):


        # self.encoder = load_model_and_weights(encoder_path)
        self.__load_model__(decoder_path)
    
    def __load_model__(self, model_path: str):
        """
        Loads the trained model.

        Args:
            model_path (str): Path to the trained model checkpoint.
        """
        self.encoder = SentinelEncoder().to("cuda")
        self.decoder = DecoderResNet18().to("cuda")

        weights = torch.load(model_path)

        encoder_dict = OrderedDict()
        decoder_dict = OrderedDict()

        for key, value in weights.items():
            if key.startswith('encoder') and not key.startswith('encoder_2'):
                encoder_dict[key.replace('encoder.', '')] = value
            elif key.startswith('decoder'):
                decoder_dict[key.replace('decoder.', '')] = value
        
        self.decoder.load_state_dict(decoder_dict)
        self.decoder.to("cuda")
        self.decoder.eval()

        self.encoder.load_state_dict(encoder_dict)
        self.encoder.to("cuda")
        self.encoder.eval()

        self.model = ChangeDetection(self.encoder, self.decoder).to("cuda")

    def precision(self, y_true, y_pred):
        """
        Calculates the precision score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Precision score.
        """
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-7)
        return precision

    def recall(self, y_true, y_pred):
        """
        Calculates the recall score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Recall score.
        """
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-7)
        return recall
    
    def f1(self, y_true, y_pred):
        """
        Calculates the F1 score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: F1 score.
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1

    def evaluate(self, x0, x1, y_true):
        """
        Evaluates the change detection model.

        Args:
            x0 (torch.Tensor): Input data for the first time step.
            x1 (torch.Tensor): Input data for the second time step.
            y_true (torch.Tensor): True labels.

        Returns:
            tuple: A tuple containing the segmentation mask, precision, recall, and F1 score.
        """
        with torch.no_grad():
            seg_mask = self.model(x0, x1)
        
        precision = self.precision(y_true, seg_mask)
        recall = self.recall(y_true, seg_mask)
        f1 = self.f1(y_true, seg_mask)
        return seg_mask, precision, recall, f1

if __name__ == "__main__":
    # Read the dataset
    from data_reader.OSCD_ms_reader import MSReadData
    from torch.utils.data import DataLoader
    from utils.utils import to_img, plot_images
    import os
    from tqdm import tqdm
    from collections import OrderedDict

    # Define the data reader
    data_path = r".\data\OSCD"
    dataset = MSReadData()

    # Define the data loaderr
    data_loader = dataset.create_dataLoader(data_path, batch_size=1, shuffle=True, train=False, image_size=256)

    # Define the model
    decoder_name = "20240327_203022"
    decoder_path = f"C:/BEPE/remote_features/trained_models/change/{decoder_name}/change_model.pt"

    # read enconder
    encoder_name = "20240315_204706"
    encoder_path = f"./trained_models/encoder/{encoder_name}/q_weights.pt"

    method_eval = EvaluateChangeDetection(encoder_path, decoder_path)

    # Evaluate the model
    # Get the first batch

    dict_results = {"seg_mask": [], "y_true": [], "precision": 0, "recall": 0, "f1": 0}
    for sample in tqdm(data_loader):
        x_1, x_2, y = sample
        x_1 = x_1.to("cuda")
        x_2 = x_2.to("cuda")
        y = y.to("cuda")
        seg_mask, precision, recall, f1 = method_eval.evaluate(x_1, x_2, y)
        dict_results["seg_mask"].append(seg_mask.cpu().data)
        dict_results["y_true"].append(y.cpu().data)
        dict_results["precision"] += precision
        dict_results["recall"] += recall
        dict_results["f1"] += f1

    # Calculate the mean
    precision = dict_results["precision"] / len(data_loader)
    recall = dict_results["recall"] / len(data_loader)
    f1 = dict_results["f1"] / len(data_loader)

    # Plot the results
    pic = [to_img(x) for x in dict_results["seg_mask"][:5]]
    y_pic = [to_img(x) for x in dict_results["y_true"][:5]]

    plot_images(torch.stack(pic).squeeze(1), cmap="gray")
    plot_images(torch.stack(y_pic).squeeze(1), cmap="gray")

    # Print the results
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall: {recall*100:.2f}")
    print(f"F1: {f1*100:.2f}")
