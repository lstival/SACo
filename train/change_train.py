# Logger
from torchvision.utils import save_image
import torch.nn as nn

# Libraries
import torch
import os
# from modules import Decoder
from utils.utils import get_model_time, plot_images, python_files, to_img, is_notebook
from data_reader.OSCD_ms_reader import MSReadData
from modules.change_network import DecoderResNet18, ChangeDetection

from torchmetrics.classification import Dice
from tqdm import tqdm
# Trainer

class DiceLoss(nn.Module):
    """
    Dice loss function for binary segmentation tasks.
    """
    __name__ = 'dice_loss'

    def f_score(self, pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
        """
        Calculate the F-score (Dice coefficient) between predicted and ground truth tensors.

        Args:
            pr (torch.Tensor): A tensor of predicted elements.
            gt (torch.Tensor): A tensor of ground truth elements.
            beta (float): Beta value for F-score calculation (default: 1).
            eps (float): Epsilon value to avoid zero division (default: 1e-7).
            threshold: Threshold for binarization of predicted outputs.
            activation (str): Activation function to be applied to predicted outputs (default: 'sigmoid').

        Returns:
            float: F-score (Dice coefficient) between predicted and ground truth tensors.
        """

        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = torch.nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = torch.nn.Softmax2d()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
            )

        pr = activation_fn(pr)

        if threshold is not None:
            pr = (pr > threshold).float()

        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp

        score = ((1 + beta ** 2) * tp + eps) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

        return score

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        """
        Calculate the Dice loss between predicted and ground truth tensors.

        Args:
            y_pr (torch.Tensor): Predicted tensor.
            y_gt (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Dice loss between predicted and ground truth tensors.
        """
        return 1 - self.f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)

def validation(model, val_loader, criterion):
    """
    Perform validation on the given model using the validation data loader.

    Args:
        model: The model to be validated.
        val_loader: The validation data loader.
        criterion: The loss criterion.

    Returns:
        tuple: A tuple containing the validation loss and the Dice score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    val_loss = 0.0
    dice = Dice().to(device)
    

    with torch.no_grad():
        for batch_idx, (x_0, x_1, y) in enumerate(val_loader):
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            y = y.to(device)

            out = model.forward(x_0, x_1)
            loss = criterion(out, y.int())

            out = torch.where(out > 0.5, 1, 0)
            dice.update(out, y.int())

            val_loss += loss.item()

    val_loss /= len(val_loader)
    dice_score = dice.compute()

    return val_loss, dice_score

def training(model, train_loader, val_loader, criterion, max_epochs=100):
    """
    Train the given model using the training data loader.

    Args:
        model: The model to be trained.
        train_loader: The training data loader.
        criterion: The loss criterion.
        max_epochs (int): The maximum number of epochs to train (default: 100).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dt_str = get_model_time()
    
    pbar = tqdm(total=max_epochs * len(train_loader))
    val_loss = 0.0
    dice_score = 0.0
        
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (x_0, x_1, y) in enumerate(train_loader):
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            out = model.forward(x_0, x_1)
            loss = criterion(out, y.int())

            out = torch.where(out > 0.5, 1, 0)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Dice Score": dice_score})
            pbar.update(1)
            
        train_loss /= len(train_loader)

        val_loss, dice_score = validation(model, val_loader, criterion)
        dice_score = dice_score.item()

        pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Dice Score": dice_score})

        # Plot the image if batch_idx == xxx
        if epoch % 100 == 0:
            pic = to_img(out[:5].cpu().data)
            y_pic = to_img(y[:5].cpu().data)

            if is_notebook():
                plot_images(pic[:5], cmap="gray")
                plot_images(y_pic[:5], cmap="gray")

            path_save_image =  f'../change_results/{str(dt_str)}/image_{batch_idx}.png'
            os.makedirs(os.path.dirname(path_save_image), exist_ok=True)

            save_image(pic, path_save_image)
            # img_grid = torchvision.utils.make_grid(pic)

            # Save the model
            path_to_save = f"../trained_models/change/{dt_str}/"
            os.makedirs(path_to_save, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path_to_save, f"change_model.pt"))

            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(path_to_save, f"checkpoint_{epoch}.pt"))

    pbar.close()

if __name__ == "__main__":
    import modules.sentinel_bands as sentinel_bands
    from modules.senti_encoder import load_model_and_weights
    from modules.change_network import DecoderResNet18, ChangeDetection

    image_size = 256
    train = True
    dt_str = get_model_time()

    # Instantiate the dataset
    dataset = MSReadData()

    path = "./data/OSCD/"
    batch_size=20

    # Create the dataloader
    if train:
        dataloader, val_dataloader = dataset.create_dataLoader(path, image_size, batch_size, num_workers=8, train=train)
    else:
        dataloader = dataset.create_dataLoader(path, image_size, batch_size, num_workers=8, train=train)

    # Encoder name
    model_name = "20240315_204706"
    model_path = f"../trained_models/encoder/{model_name}/q_weights.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_resnet = resnet.ResNet_encoder().to(device)
    bands_selector = sentinel_bands.SentinelGroupBands()
    
    #Define if the Resnet model is pretrained
    pretrained = False

    # Create the models
    encoder = load_model_and_weights(model_path, bands_selector, pretrained)
    encoder = encoder.to(device)
    decoder = DecoderResNet18().to("cuda")

    # Train
    model = ChangeDetection(encoder, decoder).to(device)
    criterion = DiceLoss()
    training(model, dataloader, val_dataloader, criterion, max_epochs=1000)
