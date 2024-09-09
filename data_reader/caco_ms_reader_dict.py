# author: Leandro Stival
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Dataset library
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

# GeoTiff library
import rasterio
import random
import rasterio.features
import rasterio.warp
from PIL import Image
from torchvision import transforms
import random

# Create the dataset
class RemoteImagesDatasetDict(Dataset):
    """
    This class create a dataset from a path, where the data multi scpetral images from caco dataset
    that will read and return as a tuple with 3 tensors. The image, an positive sample and a negative sample.
    """
    def __init__(self, path: str, batch_size: int, image_size: int, train= True, mode: str = "seco") -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.train = train
        self.samples = []
        self.mode = mode
        self.__list_the_times__()
        self.all_samples = self.get_all_samples()


    def get_all_samples(self):
        all_samples= {}
        for region in self.samples:
            samples = []
            for sample in self.samples:
                if region in sample:
                    original_image = self.__read_original_image__(sample)
                    image = self.__read_image__(sample)
                    positive = self.__read_positive__(sample)
                    samples.append((original_image, image, positive))
            all_samples[region] = samples
        return all_samples
    
    def tiff_force_8bit(self, image):
        if image.format == 'TIFF' and image.mode == 'I;16':
            array = np.array(image)
            normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())
            image = Image.fromarray(normalized.astype(np.uint8))
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image
    
    def __get_label_class__(self, sample: str) -> str:
        """
        Return the label of the class
        """
        region = sample.split(os.path.sep)[-2]
        return region
        
    def __list_the_regions__(self):
        """
        List all the regions in the path
        """
        regions = [folder for folder in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, folder))]
        return regions

    def __list_the_times__(self):
        """
        List all the samples by time in the regions
        and add to the samples list
        """
        regions = self.__list_the_regions__()
        for region in regions:
            current_path = os.path.join(self.path, region)
            for time in os.listdir(current_path):
                sample = os.path.join(current_path, time)
                if os.path.isdir(sample):
                    self.samples.append(sample)
                    
    def __transform__(self, image, flip=True, crop_params=None, do_jilter=False):
        
        if self.train:
            # Rotate the image
            if self.do_rotation:
                image = transforms.functional.rotate(image, self.rotation_angle)
                image = transforms.CenterCrop((self.image_size//2, self.image_size//2))(image)
            if flip:
                flip = transforms.RandomHorizontalFlip(p=1.)
                image = flip(image)
            if crop_params is not None:
                image = transforms.CenterCrop((self.image_size//2, self.image_size//2))(image)
            if do_jilter:
                jilter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                image = jilter(image)

        # Convert the image to tensor
        image = transforms.ToTensor()(image)

        return image
        
    def __read_image__(self, sample: str) -> torch.Tensor:
        """
        Read the image from the sample
        """
        options = [True, False]
        do_flip = random.choice(options)
        do_crop = random.choice(options)
        do_jilter = random.choice(options)
        self.do_rotation = random.choice(options)
        if self.do_rotation:
            self.rotation_angle = random.randint(0, 360)

        img = []
        imgs = [f for f in os.listdir(sample) if f.endswith(".tif")]

        for img_idx in imgs:
            temp_img = Image.open(os.path.join(sample, img_idx))

            # resize the image
            resize = transforms.Resize((self.image_size, self.image_size))
            # temp_img = resize(temp_img)

            if do_crop:
                i,j,h,w = 10, 10, self.image_size//2, self.image_size//2
                temp_img = self.__transform__(temp_img, flip=do_flip, crop_params=(i, j, h, w), do_jilter=do_jilter)
                
            else:
                temp_img = self.__transform__(temp_img, flip=do_flip, crop_params=None, do_jilter=do_jilter)
            
            temp_img = resize(temp_img)
            img.append(temp_img)

        return torch.cat(img, dim=0)
    
    def __read_original_image__(self, sample: str) -> torch.Tensor:
        """
        Read the main image from the sample
        """
        img = []
        imgs = [f for f in os.listdir(sample) if f.endswith(".tif")]
        for i in imgs:
            temp_img = Image.open(os.path.join(sample, i))
            temp_img = self.tiff_force_8bit(temp_img)
            temp_img = transforms.Resize((self.image_size, self.image_size))(temp_img)
            temp_img = transforms.ToTensor()(temp_img)
            # temp_img = transforms.Normalize(mean=[0.5], std=[0.5])(temp_img)
            img.append(temp_img)

        return torch.cat(img, dim=0)
    
    def __read_positive__(self, sample: str) -> torch.Tensor:
        """
        Read a positive sample from the sample
        """
        region = sample.split(os.path.sep)[-2]
        positive = random.choice(os.listdir(os.path.join(self.path, region)))
        while positive == sample or positive.endswith(".tif"):
            positive = random.choice(os.listdir(os.path.join(self.path, region)))
        return self.__read_image__(os.path.join(self.path, region, positive))

    def __read_negative__(self, sample: str) -> torch.Tensor:
        """
        Read a negative sample from the sample
        """
        negative = random.choice(self.samples)
        while negative == sample:
            negative = random.choice(self.samples)
        return self.__read_image__(negative)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Return the image, a positive sample and a negative sample
        """
        # Get the sample (Region)
        sample = self.samples[index]

        # Get the class label
        class_label = self.__get_label_class__(sample)
        
        if self.mode is not None:
            mode = self.mode.upper()

        # Read the Main img, the argumented version of original image and the main image in another time
        if self.mode == None:
            positive = None
        elif mode == "seco".upper():
            image, argumented_positive, temporal_positive = self.all_samples[sample][0]
            return image, argumented_positive, temporal_positive, class_label

class MSReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, train=True, mode="temporal", shuffle=True, pin_memory=True, num_workers=8):
        self.datas = RemoteImagesDatasetDict(dataroot, batch_size, image_size, train=train, mode=mode)
        
        if train:
            train_size = int(0.7 * len(self.datas))
            val_size = len(self.datas) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.datas, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            self.dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            self.dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            return self.dataloader_train, self.dataloader_val
        else:
            self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle)
            return self.dataloader
    
if __name__ == "__main__":
    path = r".\data\mini_caco10k\clean_10k_geography"
    dataset = MSReadData()
    batch_size = 10
    img_size=64
    # mode = ["SeCo", "MoCo", "Temporal", "Eval"]
    mode = "SECO"
    train = True

    if train:
        dataloader, val_dataloader = dataset.create_dataLoader(path, img_size, batch_size, train=train, mode=mode.upper(), num_workers=12)
    else:
        dataloader = dataset.create_dataLoader(path, img_size, batch_size, train=train, mode=mode, num_workers=1)
    
    print(f"len of the dataset: {len(dataloader.dataset)}")
    # datas = RemoteImagesDataset(path, batch_size, img_size, train=True)
    # samples = datas.__getitem__(0)
    # sample = next(iter(dataloader))
    import time
    start_time = time.time()
    samples = []

    for i, sample in enumerate(dataloader):
        samples.append(sample[0])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # import matplotlib.pyplot as plt
    # def plot_images(images, cmap=None):

    #     plt.figure(figsize=(32, 32))
    #     if not cmap:
    #         plt.imshow(torch.cat([
    #             torch.cat([i for i in images.cpu()], dim=-1),
    #         ], dim=-2).permute(1, 2, 0).cpu())
    #         plt.axis('off')  # Remove the axis
    #         plt.show()
    #     else:
    #         plt.imshow(torch.cat([
    #             torch.cat([i for i in images.cpu()], dim=-1),
    #         ], dim=-2).permute(1, 2, 0).cpu(), cmap=cmap)
    #         plt.axis('off')  # Remove the axis
    #         plt.show()

    # # x = sample[0][0:1]
    # x = sample[1]
    # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
    # plot_images(img)