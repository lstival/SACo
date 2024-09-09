#Author: L. Stival
# This file read the EuroSAT dataset and return the images as tensors

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Dataset library
from torch.utils.data import Dataset
from torchvision import transforms

# GeoTiff library
import rasterio
import random
import rasterio.features
import rasterio.warp
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F

class RemoteImagesDataset(Dataset):
    def __init__(self, root_path, image_size, train=True):
        self.root_path = root_path
        self.train = train
        self.image_size = image_size
        self.image_files = self._get_image_files()
        self.labels_list = os.listdir(self.root_path)

    def my_transform(self, images):
            
            image_pil = [Image.fromarray(images[i]) for i in range(images.shape[0])]
            # resize the image
            resize = transforms.Resize((self.image_size, self.image_size))
            # image_pil = [resize(img) for img in image_pil]
            
            if self.train:

                # Random affine transformation
                if random.random() > 0.5:
                    for channel in range(13):
                        image_pil[channel] = F.affine(image_pil[channel], angle=0, translate=(0, 0), scale=1.0, shear=10)
    
                # Random rotation
                if random.random() > 0.5:
                    angle = transforms.RandomRotation.get_params([-45, 45])
                    for channel in range(13):
                        image_pil[channel] = F.rotate(image_pil[channel], angle)
        
                # Center crop
                if random.random() > 0.5:
                    for channel in range(13):
                        image_pil[channel] = F.center_crop(image_pil[channel], (64, 64))
                
                # # Random resized crop
                # if random.random() > 0.5:
                #     i, j, h, w = transforms.RandomResizedCrop.get_params(image_pil[0], scale=(0.08, 1.0), ratio=(0.75, 1.333))
                #     for channel in range(13):
                #         image_pil[channel] = F.resized_crop(image_pil[channel], i, j, h, w, (48, 48))

                # Random horizontal flipping
                if random.random() > 0.5:
                    for channel in range(13):
                        image_pil[channel] = F.vflip(image_pil[channel])
                
                # Random vertical flipping
                if random.random() > 0.5:
                    for channel in range(13):
                        image_pil[channel] = F.hflip(image_pil[channel])

          
            image_pil = [resize(img) for img in image_pil]
            # list of posibles modes for the image
            # ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'I', 'F']

            for channel in range(13):
                image_pil[channel] = F.to_tensor(image_pil[channel])
            
            imgs_1_tensor = torch.cat(image_pil, dim=0)
            image_pil = [resize(img) for img in image_pil]

            return imgs_1_tensor

    def __force_to_8bit__(self, x) -> torch.Tensor:
        # Check the data is a tensor or a numpy array
        if type(x) == np.ndarray:
            return (x - x.min()) / (x.max() - x.min())
        else:
            x = self.tiff_force_8bit(x)
            return x
        
    def tiff_force_8bit(self, image):
        if image.format == 'TIFF' and image.mode == 'I;16':
            array = np.array(image)
            normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())
            image = Image.fromarray(normalized.astype(np.uint8))
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        return image
    
    def _get_image_files(self):
        image_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".tif"):
                    image_files.append(os.path.join(root, file))
        return image_files

    def _read_image(self, image_path):
        with rasterio.open(image_path) as dataset:
            image = dataset.read()
            image = self.__force_to_8bit__(image)
        return image

    def _get_label(self, image_path):
        label = self.labels_list.index(os.path.basename(os.path.dirname(image_path)))
        return label
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = self._read_image(image_path)
        label = self._get_label(image_path)
        
        # if self.train:
        # image = self.my_transform(image)
        image = image.astype(np.float32)
        # image_2 = self.my_transform(image)
        # image_3 = self.my_transform(image)
        image = self.my_transform(image)
        
        return image, label

class MSReadData():
    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, train=True, shuffle=True, pin_memory=True, num_workers=8):
        self.dataset = RemoteImagesDataset(dataroot, image_size, train=train)

        torch.manual_seed(42)
        
        if train:
            train_size = int(0.7 * len(self.dataset))
            test_size = len(self.dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
            return dataloader, test_dataloader
        else:
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
            return dataloader

if __name__ == "__main__":
    print("Main")
    root_path = "D:/BEPE/data/EuroSAT_test"
    batch_size = 16
    image_size = 64
    train = True

    # Example usage
    dataset = MSReadData()
    if train:
        dataloader, val_dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train, num_workers=1)
    else:
        dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train)
    
    samples = next(iter(dataloader))

    try:
        images, imgs_2, imgs_3, labels = samples
    except:
        images, labels = samples
    
    print(f"Images shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Number of elements in the dataset: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    class_indice_2_see = 6

    images_list = []
    for sample in dataloader:
        images, labels = sample
        for i in range(len(labels)):
            if labels[i] == class_indice_2_see:
                images_list.append(images[i])
        

    classes_names = os.listdir(root_path)
    # from utils import plot_images

    x = torch.cat([torch.unsqueeze(img, 0) for img in images_list], dim=0)
    xs = torch.cat((x[:,3:4], x[:,2:3], x[:,1:2]), dim=1)
    # xs = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)

    imgs = xs.swapaxes(1, 3).swapaxes(1, 2)

    # Normalize the images
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        axes[i].imshow(imgs[i])
        # axes[i].set_title(classes_names[labels[i]])
        axes[i].set_title(classes_names[class_indice_2_see])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Use xs instead of images
    worst_index = [6, 5, 8, 2]
    lst_worst_imgs = []
    lst_worst_labels = []
    for idx, i in enumerate(labels):
        if i in worst_index:
            lst_worst_imgs.append(xs[idx])
            lst_worst_labels.append(i)

    fig, axes = plt.subplots(1, len(lst_worst_imgs), figsize=(15, 3))

    for i in range(len(lst_worst_imgs)):
        axes[i].imshow(lst_worst_imgs[i].numpy().transpose(1, 2, 0))
        axes[i].set_title(classes_names[lst_worst_labels[i]])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
