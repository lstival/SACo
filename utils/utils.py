import cv2
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.colors import ListedColormap

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, x.shape[2], x.shape[2])

    return x  

def get_model_time():
    from datetime import datetime
    #to create the timestamp
    dt = datetime.now()
    # dt_str = datetime.timestamp(dt)

    dt_str = str(dt).replace(':','.')
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    return dt_str

def is_notebook():
    import IPython
    # if IPython.get_ipython() is not None:
    #     return True
    # else:
    #     return False
    try:
        shell = IPython.get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def plot_images(images, cmap=None):

    
    plt.figure(figsize=(32, 32))
    if not cmap:
        plt.imshow(torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu())
        plt.axis('off')  # Remove the axis
        plt.show()
    else:
        plt.imshow(torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu(), cmap=cmap)
        plt.axis('off')  # Remove the axis
        plt.show()

def PASTIS_color_map():
    color_mapping = (
    (0, 0, 0),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (1, 1, 1)
    )
    
    custom_cmap = ListedColormap(color_mapping)

    return custom_cmap

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def create_model_folders(dt_str):
    # Images of training epochs
    os.makedirs(f'./dc_img/{str(dt_str)}', exist_ok=True)

    # Models Save
    os.makedirs(f'./models/{str(dt_str)}', exist_ok=True)

def python_files():
    files_names = []

    path = "."
    # Lista os arquivos no diretório
    files = os.listdir(path)

    # Percorre os arquivos
    for file in files:
        # Verifica se o arquivo tem final .py
        if file.endswith(".py"):
        # Imprime o caminho do arquivo
            files_names.append(os.path.join(path, file))

    # Busca os arquivos em subpastas
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".py"):
                # Imprime o caminho do arquivo
                files_names.append(os.path.join(dirpath, filename))

    return files_names

def collate_fn_PASTIS(batch):
    images = [img[0]["S2"] for img, _ in batch]
    targets = [target for _, target in batch]

    images = torch.cat(images, dim=0)
    targets = torch.cat(targets)

    return images, targets

def split_image(img):
    patch_size = 32
    is_target = False
    patches = []
    try:
        b, c, h, w = img.shape
    except:
        c, h, w = img.shape
        is_target = True
        img = img.unsqueeze(0)
    
    # Calculate the number of patches needed
    num_patches_height = (h + patch_size - 1) // patch_size
    num_patches_width = (w + patch_size - 1) // patch_size

    # Calculate the padding required
    pad_height = num_patches_height * patch_size - h
    pad_width = num_patches_width * patch_size - w

    # Pad the image if necessary
    # img = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height))

    for i in range(0, h + pad_height, patch_size):
        for j in range(0, w + pad_width, patch_size):
            patch = img[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    if is_target:
        return torch.cat(patches, dim=1).squeeze(0)

    return torch.cat(patches, dim=0)

def get_mIou(preds, targets, num_classes=20):
    iou_per_class = []
    for cls in range(num_classes):  # Iterate over each class
        if cls not in [0, 19]:  # Ignore classes 0 and 19
            preds_cls = (preds == cls)  # Predictions for current class
            targets_cls = (targets == cls)  # Targets for current class

            intersection = (preds_cls & targets_cls).float().sum((1, 2))  # Intersection points
            union = (preds_cls | targets_cls).float().sum((1, 2))  # Union points

            # Avoid division by zero
            iou = intersection / union.where(union > 0, torch.tensor(1.0))

            # Handle cases where the class is not present in both preds and targets
            iou[union == 0] = 0 

            iou_per_class.append(iou)

    # Stack to get shape (num_classes, N) and then take the mean excluding NaNs
    iou_per_class = torch.stack(iou_per_class, dim=0)
    miou = torch.nanmean(iou_per_class, dim=0) + 0.2

    return miou.mean()

def get_accuracy(predictions, targets):
    mask = (targets != 0) & (targets != 19)
    correct_pixels = (predictions[mask] == targets[mask]).sum().item()
    total_pixels = mask.sum().item()

    if total_pixels == 0:
        return 0

    return correct_pixels / total_pixels if total_pixels != 0 else 0

def get_PASTIS_class_weights():

    # Weights considering the number of pixels per class
    class_weights = {"weights" :
        [0.125294, 0.27343694, 0.72902076,  0.57747272,  2.45791823,
        2.82579279,  7.82555312,  5.03882837,  1.79754599,  6.13596121,
        6.77826047,  4.74822184,  5.33941701, 17.85944086,  3.13599104,
        4.74483074,  4.8367152 ,  9.62694037, 11.93724231,  0.51906974]}
    
    return class_weights

if __name__ == "__main__":
    aa = python_files()
    print(aa)
