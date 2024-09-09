from caco_ms_reader import RemoteImagesDataset
from lbp import lbp_values
import matplotlib.pyplot as plt
import torch
import os

class CaCOLBP(RemoteImagesDataset):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def lbp_image(self, images):
        """
        Calculate the LBP image of the given image
        """
        return lbp_values((images.unsqueeze(0)*255).type(torch.int)).squeeze(0).type(torch.float32)
    
    def save_lbp_image(self, lbp_image, path_to_save):
        """
        Save the LBP image to the given path
        """
        bands = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
        for i, band in enumerate(bands):
            try:
                plt.imsave(f"{path_to_save}/{band}.png", lbp_image[i].numpy(), cmap='gray')
            except:
                print(f"Error: Could not save the image {band} in the path {path_to_save}")

    def __getitem__(self, index) -> dict:
        """
        Return the image, a positive sample and a negative sample
        """
        # Get the sample (Region)
        sample = self.samples[index]
        path_to_save = sample.replace(sample.split("/")[3], sample.split('/')[3]+"_LBP")
        # path_to_save = path_to_save.replace("C:", "F:")
        os.makedirs(path_to_save, exist_ok=True)
        
        image = self.__read_original_image__(sample)
        lbp_image = self.lbp_image(image)
        self.save_lbp_image(lbp_image, path_to_save)
        
        return lbp_image

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    # path = r".\data\caco10k\clean_10k_geography"
    # path = "./data/BigEarthNet/test"
    path = "./data/BigEarthNet/19/test"
    # path = r"E:\seasonal_contrast_1m\clean_1m_geography"
    batch_size = 128
    image_size = 64

    # Create an instance of the CaCOLBP dataset
    dataset = CaCOLBP(path, image_size=image_size, batch_size=batch_size)

    # Create a data loader for the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

    # Iterate over all samples in the dataset
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Processing LBP images")
    for index, data in pbar:
        pass

# if __name__ == '__main__':
#     import os
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm
# import os
# from torch.utils.data import DataLoader
# from tqdm import tqdm

#     path = r".\data\caco10k\clean_10k_geography"
#     batch_size = 1
#     image_size = 256

#     # Create an instance of the CaCOLBP dataset
#     dataset = CaCOLBP(path, image_size=image_size, batch_size=batch_size)

#     # Create a data loader for the dataset
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)

#     # Iterate over all samples in the dataset
#     all_lbp_images = []
#     pbar = tqdm(enumerate(dataloader), total=len(dataloader))
#     pbar.set_description("Processing LBP images")
#     for index, data in pbar:
#         # Get the image and path to save
#         all_lbp_images.append(data)
