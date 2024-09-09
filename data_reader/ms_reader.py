# This file contains the class to read the multi spectral images
# Utils Libaries
import glob
import torch
import numpy as np
import re
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

# Create the dataset
class RemoteImagesDataset(Dataset):
    """
    This class create a dataset from a path, where the data multi scpetral images
    that will read and return as a tensor.
    """
    def __init__(self, path: str, batch_size: int, image_size: int, multi_spectral: bool = True) -> None:
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.multi_spectral = multi_spectral

        self.scenes = os.listdir(path)
        self.image_names = self.__find_images__()
        self.image_names = self.__data_filer__(self.image_names)

    def __find_images__(self) -> list:
        """
        Find the images in the path
        """
        # Loop to find the images in subfolders
        self.image_names= []
        deep_counter = 1

        while self.image_names == [] and deep_counter < 10:
            deeper = "/*"
            self.image_names = glob.glob(f"{self.path}{deeper*deep_counter}.tif")
            deep_counter += 1

        return self.image_names

    def __data_filer__(self, image_names: list) -> list:
        """
        Filter the data to only get the images in S1 (13 channels)
        """
        # Check if the data is multi spectral or optical
        # where the multi spectral data is S2 and the optical is S1

        if self.multi_spectral:
            pattern = re.compile(r'S2')
            filtered_image_names = [name for name in image_names if pattern.search(name)]
        else:
            pattern = re.compile(r'S1')
            filtered_image_names = [name for name in image_names if pattern.search(name)]

        # If the filtered data is empty, return the original data
        if filtered_image_names == []:
            filtered_image_names = image_names

        return filtered_image_names
    
    def __open_image__(self, img_path: str):
        with rasterio.open(img_path) as dataset:
        # Read the dataset's valid data mask as a ndarray.
            mask = dataset.dataset_mask()

            # Extract feature shapes and values from the array.
            for geom, val in rasterio.features.shapes(
                    mask, transform=dataset.transform):

                # Transform shapes from the dataset's own coordinate
                # reference system to CRS84 (EPSG:4326).
                geom = rasterio.warp.transform_geom(
                    dataset.crs, 'EPSG:4326', geom, precision=6)

                # Print GeoJSON shapes to stdout.
                # print(geom)

            # Read a single band, storing the data in a 2D array.
            array = dataset.read()

        return array
    
    def __read_geotiff__(self, index: int):
        """
        Read a single geotiff image from a path.
        """
        img_path = self.image_names[index]
        substring = self.__pathc_number__(img_path)+"."
        
        matching_itens = [item for item in self.image_names if substring in item and item != img_path]
        match_item = self.__get_random_sample__(matching_itens)

        # Read the images
        img = self.__open_image__(img_path)
        match_img = self.__open_image__(match_item)

        img_paths = [img_path, match_item]
        
        return img, match_img, img_paths

    def __get_random_sample__(self, samples):
        index = random.randint(0, len(samples) - 1)
        return samples[index]

    def __pathc_number__(self, string):

        substring = string.split("_")[-1].split(".")[0]
        actual_patch = f"{string.split('_')[-2]}_{substring}"

        return actual_patch

    def __transform__(self, x) -> torch.Tensor:
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Check the data is a tensor or a numpy array
        if type(x) == np.ndarray:
            # Check if all values of x are 0
            if np.all(x == 0):
                return x
            else:
                x_mean = (x - x.min()) / (x.max() - x.min())
                if np.isnan(x_mean).any():
                    x_mean = np.nan_to_num(x_mean)
            return x_mean
        else:
            return transform(x)

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # Read the geotiff image
        sample, sample_other_time, img_paths = self.__read_geotiff__(index)

        # Resize and Normalize the image between 0 and 1
        sample = self.__transform__(sample)
        sample = torch.tensor(sample.astype("int16")).float()

        sample_other_time = self.__transform__(sample_other_time)
        sample_other_time = torch.tensor(sample_other_time.astype("int16")).float()

        # return sample, sample_other_time, img_paths
        return sample, sample_other_time, img_paths

# Instantiate the dataset
class MSReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, multi_spectral=True, shuffle=True, pin_memory=True, num_workers=0):

        self.datas = RemoteImagesDataset(dataroot, batch_size, image_size, multi_spectral)

        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
        return self.dataloader

if __name__ == "__main__":
    print("main")
    # https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/

    # path = "./data/ROIs1158_spring_s1/ROIs1158_spring_s1"
    # path = "./data/s1_america/america/ROIs1158/106/"
    # path = "E:/BEPE dataset/s2_africa_test/ROIs2017"
    path = "F:/SEN12MSCRTS"
    # path = "./data/s1_europa/europa/ROIs1868/17/"
    # path = "./data/OSCD/beirut/imgs_1"
    # path = "C:/Users/stiva/OneDrive/Área de Trabalho/Remote_IMG_test/"
    batch_size = 4
    # dataSet = RemoteImagesDataset(path, batch_size, 256)

    # dataloader = torch.utils.data.DataLoader(dataSet, batch_size=10, shuffle=True, pin_memory=True)
    # data = next(iter(dataloader))
    dataset = MSReadData()
    # SAR = dataset.create_dataLoader(path, 256, batch_size, multi_spectral=False, shuffle=False)
    multi_dataloader = dataset.create_dataLoader(path, 256, batch_size, multi_spectral=True, shuffle=False)

    # dataloader = zip(multi_dataloader, SAR)
    dataloader = multi_dataloader

    data = next(iter(dataloader))
    print(len(data))
    """
    Return 0: multi_spectral
    Return 1: multi_spectral in another time
    """
    print(f"multi_dataloader shape: {data[0][0].shape}")
    # print(f"optical_dataloader shape: {data[1][0].shape}")

    def plot_images(images):
        plt.figure(figsize=(32, 32))
        plt.imshow(torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu(), cmap="gray")  # Set cmap="gray"
        plt.show()

    # plot_images(data[0][-2:,1:4])
    # plot_images(data[0][:,1:4][5:6])
    plot_images(data[0][:,1:4])
    plot_images(data[1][:,1:4])

    # img_path = r"F:\SEN12MSCRTS\ROIs1868\139\S2\13\s2_ROIs1868_139_ImgNo_13_2018-06-14_patch_9.tif"
    # with rasterio.open(img_path) as dataset:
    #     # Read the dataset's valid data mask as a ndarray.
    #     mask = dataset.dataset_mask()

    #     # Extract feature shapes and values from the array.
    #     for geom, val in rasterio.features.shapes(
    #             mask, transform=dataset.transform):

    #         # Transform shapes from the dataset's own coordinate
    #         # reference system to CRS84 (EPSG:4326).
    #         geom = rasterio.warp.transform_geom(
    #             dataset.crs, 'EPSG:4326', geom, precision=6)

    #         # Print GeoJSON shapes to stdout.
    #         # print(geom)

    #     # Read a single band, storing the data in a 2D array.
    #     array = dataset.read()
    

# s1_ROIs1158_106_ImgNo_0_2018-01-04_patch_0
# s1_ROIs1158_106_ImgNo_1_2018-01-16_patch_0
# s1_ROIs1158_106_ImgNo_2_2018-02-09_patch_0 
# s1_ROIs1158_106_ImgNo_2_2018-02-09_patch_237

# Sentinel-2 Bands info
# https://gisgeography.com/sentinel-2-bands-combinations/
# Band	Resolution	Central Wavelength	Description
# B1	60 m	443 nm	Ultra Blue  (Coastal and Aerosol)
# B2	10 m	490 nm	Blue
# B3	10 m	560 nm	Green
# B4	10 m	665 nm	Red
# B5	20 m	705 nm	Visible and Near Infrared (VNIR)
# B6	20 m	740 nm	Visible and Near Infrared (VNIR)
# B7	20 m	783 nm	Visible and Near Infrared (VNIR)
# B8	10 m	842 nm	Visible and Near Infrared (VNIR)
# B8a	20 m	865 nm	Visible and Near Infrared (VNIR)
# B9	60 m	940 nm	Short Wave Infrared (SWIR)
# B10	60 m	1375 nm	Short Wave Infrared (SWIR)
# B11	20 m	1610 nm	Short Wave Infrared (SWIR)
# B12	20 m	2190 nm	Short Wave Infrared (SWIR)


# img_path = os.path.join(path, "0", "s1_ROIs1158_106_ImgNo_0_2018-01-04_patch_0.tif")
# with rasterio.open(img_path) as dataset:

# # Read the dataset's valid data mask as a ndarray.
#     mask = dataset.dataset_mask()

#     # Extract feature shapes and values from the array.
#     for geom, val in rasterio.features.shapes(
#             mask, transform=dataset.transform):

#         # Transform shapes from the dataset's own coordinate
#         # reference system to CRS84 (EPSG:4326).
#         geom = rasterio.warp.transform_geom(
#             dataset.crs, 'EPSG:4326', geom, precision=6)

#         # Print GeoJSON shapes to stdout.
#         # print(geom)

#     # Read a single band, storing the data in a 2D array.
#     array = dataset.read()