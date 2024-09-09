import torch
import torch.nn as nn
from torch.nn import functional as F

## My packages
from utils import get_PASTIS_class_weights

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_indices=[], use_class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_indices = ignore_indices
        self.use_class_weights = use_class_weights
        weights = get_PASTIS_class_weights()
        self.class_weights = torch.tensor(weights["weights"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        
        # Create a mask to ignore the specified classes
        mask = torch.ones_like(targets, dtype=torch.bool)
        for ignore_index in self.ignore_indices:
            mask &= (targets != ignore_index)
        
        # Convert targets to one-hot encoding
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply mask
        inputs = inputs * mask.unsqueeze(1)
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over all classes except the ignored ones
        valid_classes = [i for i in range(inputs.shape[1]) if i not in self.ignore_indices]
        dice = dice[:, valid_classes]  # Ignore specified classes
        dice_loss = 1 - dice.mean()
        
        if self.use_class_weights is not None:
            class_weights = self.class_weights[valid_classes].to(self.device)
            dice_loss = (class_weights * dice_loss).mean()

            weighted_dice_loss = (class_weights * dice_loss).mean()
        
            return weighted_dice_loss
        
        return dice_loss

class FocalLoss(nn.Module):
    """
    Focal loss from: github.com/TotalVariation/Exchanger4SITS
    """
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, preds, target):
        if preds.dim() > 2:
            preds = preds.flatten(2).transpose(1, 2).flatten(0, 1)
        target = target.view(-1, 1)

        if self.ignore_label is not None:
            keep = target[:, 0] != self.ignore_label
            preds = preds[keep, :]
            target = target[keep, :]

        if preds.squeeze(1).dim() == 1:
            logpt = torch.sigmoid(preds)
            logpt = logpt.view(-1)
        else:
            logpt = F.log_softmax(preds, dim=-1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
        pt = logpt.exp()

        """
        if self.alpha is not None:
            if self.alpha.type() != preds.data.type():
                self.alpha = self.alpha.type_as(preds.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        """

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.class_weights = weight

    def forward(self, inputs, targets, ):
        if self.class_weights is not None:
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=self.class_weights)
        else:
            CE_loss = nn.CrossEntropyLoss(reduction='none')
        CE_loss = CE_loss(inputs,targets)
        targets = targets.long()
        sm = inputs.softmax(dim=1)
        selector = nn.functional.one_hot(targets, num_classes=21).bool()
        pt = sm[selector]
        F_loss = (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

class FocalCELoss(nn.Module):

    """
    FocalLoss copied from github.com/VSainteuf/utae-paps
    """
    def __init__(self, gamma=1.0, size_average=True, ignore_index: int = -100, weight = None):
         super(FocalCELoss, self).__init__()
         self.gamma = gamma
         self.size_average = size_average
         self.ignore_index = ignore_index
         self.weight = weight

    def forward(self, preds, target):
        # preds shape (B, C), target shape (B,)
        target = target.view(-1,1)

        if preds.ndim > 2: # e.g., (B, C, H, W)
            preds = preds.permute(0, 2, 3, 1).flatten(0, 2)

        keep = target[:, 0] != self.ignore_index
        preds = preds[keep, :]
        target = target[keep, :]

        logpt = F.log_softmax(preds, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            w = self.weight.expand_as(preds)
            w = w.gather(1, target)
            loss = -1 * (1 - pt) ** self.gamma * w * logpt
        else:
            loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def get_optimizer(optimizer_name):
    if optimizer_name == "CrossEntropy":
        optimizer = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    elif optimizer_name == "FocalLoss":
        optimizer = FocalLoss(gamma=2, alpha=None, size_average=True, ignore_label=0)

    elif optimizer_name == "DiceLoss":
        optimizer = DiceLoss(ignore_indices=[0])

    elif optimizer_name == "FocalCELoss":
        optimizer = FocalCELoss(gamma=1.0, size_average=True, ignore_index=0)

    return optimizer

def get_optimizer_with_class_weights(optimizer_name):
    if optimizer_name == "CrossEntropy":
        class_weights = get_PASTIS_class_weights()
        optimizer = nn.CrossEntropyLoss(ignore_index=0, reduction='mean', weight=class_weights)

    elif optimizer_name == "FocalLoss":
        print("Weights not supoorted for FocalLoss")
        class_weights = get_PASTIS_class_weights()
        optimizer = FocalLoss(gamma=2, alpha=None, size_average=True, ignore_label=0)

    elif optimizer_name == "DiceLoss":
        optimizer = DiceLoss(ignore_indices=[0], use_class_weights=True)

    return optimizer

def get_scheduler(optimizer, schedule_name):
    if schedule_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif schedule_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif schedule_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif schedule_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    return scheduler

def get_criterion(optimizer_name, schedule_name,use_class_weights=False):
    
    if use_class_weights:
        optimizer_list = []
        if type(optimizer_name) == list:
            for optimizer in optimizer_name:
                optimizer = get_optimizer_with_class_weights(optimizer)
                optimizer_list.append(optimizer)
        else:
            optimizer = get_optimizer_with_class_weights(optimizer_name)
    else:
        optimizer_list = []
        if type(optimizer_name) == list:
            for optimizer in optimizer_name:
                optimizer = get_optimizer(optimizer)
                optimizer_list.append(optimizer)
        else:
            optimizer = get_optimizer(optimizer_name)

    if optimizer_list == []:
        scheduler = get_scheduler(optimizer, schedule_name)
        return optimizer, scheduler
    else:
        scheduler_list = []
        for optimizer in optimizer_list:
            scheduler_list.append(get_scheduler(optimizer, schedule_name))
        return optimizer_list, scheduler_list

if __name__ == "__main__":
    print("main")

    use_class_weights = True
    optimizer_name = ["FocalCELoss", "DiceLoss"]
    schedule_name = "ExponentialLR"
    
    criterion, schedule = get_criterion(optimizer_name, schedule_name, use_class_weights=use_class_weights)
    