# This file contains the class to read the multi spectral images
# Utils Libaries
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

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

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B10': 2256.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B10': 1202.0605464,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}
# Create the dataset
class RemoteImagesDataset(Dataset):
    """
    This class create a dataset from a path, where the data multi scpetral images
    that will read and return as a tensor.
    """
    def __init__(self, path: str, batch_size: int, image_size: int, train= True) -> None:
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.train = train
        self.train_cities = ["aguasclaras","bercy","bordeaux","nantes","paris","rennes","saclay_e","abudhabi","cupertino","pisa","beihai","hongkong","beirut","mumbai"]
    
        # Each scene is a city with images
        if train:
            self.scenes = self.train_cities
        else:
            # Get only cities that no the label folder and not in training set of cities
            self.scenes = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder != "labels" and folder not in self.train_cities]

    def normalize(self, img, mean, std):
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img
    
    # def tiff_force_8bit(self, image):
    #     if image.format == 'TIFF' and image.mode == 'I;16':
    #         array = np.array(image)
    #         normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())
    #         image = Image.fromarray(normalized.astype(np.uint8))
    #         image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

    #     return image
    
    def __read_labels__(self, img_path: str, scene: str):
        """
        Open the label image
        from the folder labels/cm/cm.png
        and return it as a binary PIL image with mode 1
        """

        label_path = os.path.join(img_path, "labels", scene, "cm", "cm.png")
        label = Image.open(label_path)
        
        # label = label.convert("1")  # Convert label to mode 1
        label = label.convert("L")  # Convert label to mode L
        return label

    def __open_images__(self, img_path: str, scene: str):
        """
        Open the time_0 and time_1 images
        from the folder imgs_1_rect and imgs_2_rect
        and return them as a tensor of all bands concatenated
        """

        #Loop for 13 channels of sentinel 2 data
        imgs_1 = []
        imgs_2 = []

        for i in range(12):
            # Open the image
            img_band = f"B{str(i+1).zfill(2)}"
            img_1_path = os.path.join(img_path, scene, "imgs_1_rect", f"{img_band}.tif")
            img = Image.open(img_1_path)
            img = self.normalize(np.array(img), 
                                            BAND_STATS['mean'][img_band], 
                                            BAND_STATS['std'][img_band])
            # img = self.__force_to_8bit__(img)
            imgs_1.append(img)

            # Open the image
            img_2_path = os.path.join(img_path, scene, "imgs_2_rect", f"B{str(i+1).zfill(2)}.tif")
            img = Image.open(img_2_path)
            img = self.normalize(np.array(img), 
                                    BAND_STATS['mean'][img_band], 
                                    BAND_STATS['std'][img_band])
            # img = self.__force_to_8bit__(img)
            imgs_2.append(img)

        # The 8A channel add
        img = os.path.join(img_path, scene, "imgs_1_rect", f"B{str('8A').zfill(2)}.tif")
        img = Image.open(img_1_path)
        # img = self.__force_to_8bit__(img)
        img = self.normalize(np.array(img), 
                        BAND_STATS['mean'][img_band], 
                        BAND_STATS['std'][img_band])
        imgs_1.append(img)

        img = os.path.join(img_path, scene, "imgs_2_rect", f"B{str('8A').zfill(2)}.tif")
        img = Image.open(img_2_path)
        # img = self.__force_to_8bit__(img)
        img = self.normalize(np.array(img), 
                                BAND_STATS['mean'][img_band], 
                                BAND_STATS['std'][img_band])
        imgs_2.append(img)
        
        # imgs_1_tensor = torch.cat(imgs_1, dim=0)
        # imgs_2_tensor = torch.cat(imgs_2, dim=0)
        
        # return imgs_1_tensor, imgs_2_tensor
            
        return imgs_1, imgs_2

    def my_segmentation_transform(self, input, input2, target):

        target = F.resize(target, (self.image_size, self.image_size))

        if self.train:
    
            i, j, h, w = transforms.RandomCrop.get_params(target, (round(self.image_size*0.8), round(self.image_size*0.8)))

            # Random rotation
            if random.random() > 0.5:
                for channel in range(13):
                    input[channel] = F.rotate(input[channel], self.angle_rotation)
                    input2[channel] = F.rotate(input2[channel], self.angle_rotation)
                target = F.rotate(target, self.angle_rotation)

            # Apply the same crop to the input images
            for channel in range(13):
                # input[channel] = F.resize(input[channel], (self.image_size, self.image_size))
                input[channel] = F.crop(input[channel], i, j, h, w)

                # input2[channel] = F.resize(input2[channel], (self.image_size, self.image_size))
                input2[channel] = F.crop(input2[channel], i, j, h, w)

            target = F.crop(target, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                for channel in range(13):
                    input[channel] = F.vflip(input[channel])
                    input2[channel] = F.vflip(input2[channel])
                target = F.vflip(target)

            # Random vertical flipping
            if random.random() > 0.5:
                for channel in range(13):
                    input[channel] = F.hflip(input[channel])
                    input2[channel] = F.hflip(input2[channel])
                target = F.hflip(target)

            # # Random jitter
            # if random.random() > 0.5:
            #     for channel in range(13):
            #         input[channel] = F.adjust_brightness(input[channel], random.uniform(0.1, 0.3))
            #         input[channel] = F.adjust_contrast(input[channel], random.uniform(0.1, 0.3))

            #         input2[channel] = F.adjust_brightness(input2[channel], random.uniform(0.1, 0.3))
            #         input2[channel] = F.adjust_contrast(input2[channel], random.uniform(0.1, 0.3))

            # # Random zoom
            # if random.random() > 0.5:
            #     for channel in range(13):
            #         input[channel] = F.affine(input[channel], angle=0, translate=(0, 0), scale=1, shear=0)
            #         input2[channel] = F.affine(input2[channel], angle=0, translate=(0, 0), scale=1, shear=0)
            #     target = F.affine(target, angle=0, translate=(0, 0), scale=1, shear=0)


        for channel in range(13):
            input[channel] = F.to_tensor(input[channel])
            input2[channel] = F.to_tensor(input2[channel])

        # Resize
        for channel in range(13):
            input[channel] = F.resize(input[channel], (self.image_size, self.image_size))
            input2[channel] = F.resize(input2[channel], (self.image_size, self.image_size))
        
        target = F.resize(target, (self.image_size, self.image_size))

        target = F.to_tensor(target)
        imgs_1_tensor = torch.cat(input, dim=0)
        imgs_2_tensor = torch.cat(input2, dim=0)

        return imgs_1_tensor, imgs_2_tensor, target

    def __force_to_8bit__(self, x) -> torch.Tensor:
        # Check the data is a tensor or a numpy array
        if type(x) == np.ndarray:
            return (x - x.min()) / (x.max() - x.min())
        else:
            x = self.tiff_force_8bit(x)
            return x

    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # Read the geotiff image

        if self.train:
            self.angle_rotation = random.randint(0, 360)
        else:
            self.angle_rotation = 0

        scene = self.scenes[index]
        img_path = self.path+"/"

        sample_t1, sample_t2 = self.__open_images__(img_path, scene)
        labels = self.__read_labels__(img_path, scene)

        sample_t1, sample_t2, labels = self.my_segmentation_transform(sample_t1, sample_t2, labels)

        # return sample, sample_other_time, img_paths
        return sample_t1, sample_t2, labels

if __name__ == "__main__":
    print("main")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # Instantiate the dataset
    path = "./data/OSCD/"
    batch_size = 2
    img_size=512
    train=True
    dataset = RemoteImagesDataset(path, batch_size, img_size, train=train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    # Iterate over the dataset
    sample = next(iter(dataloader))
    print("Number of elements in the batch: ", len(sample))

    print("Shape of the Img in T1: ", sample[0].shape)
    print("Shape of the Img in T2: ", sample[1].shape)
    print("Shape of the Label: ", sample[2].shape)

    # Len of dataset
    print(f"len of the dataset: {len(dataloader.dataset)}")

    # print("len val_datalaoder: ", len(val_datalaoder.dataset))

    from utils.utils import plot_images

    # Plot the images
    # plt.imshow(sample[2][0][0],cmap="gray")
    plot_images(sample[2][:1], cmap="gray")

    color_img = x = sample[0][:1]
    channels_3 = torch.cat((x[:,3:4], x[:,2:3], x[:,1:2]), dim=1)
    # channels_3 = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
    plot_images(channels_3)

    color_img2 = x = sample[1][:1]
    channels_3_2 = torch.cat((x[:,3:4], x[:,2:3], x[:,1:2]), dim=1)
    # channels_3_2 = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
    plot_images(channels_3_2)
