import torch
from data_reader.caco_ms_reader import MSReadData
from lbp import lbp_values

# ['B1.tif', 'B11.tif', 'B12.tif', 'B2.tif', 'B3.tif', 'B4.tif', 'B5.tif', 'B6.tif', 'B7.tif', 'B8.tif', 'B8A.tif', 'B9.tif']

class SentinelGroupBands():
    def __init__(self, ndvi_bands=False) -> None:
        self.ndvi_bands = ndvi_bands

    def sepair_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sepair the bands from the image
        """
        self.natural_colors = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
        self.infrared = torch.cat((x[:,10:11], x[:,5:6], x[:,4:5]), dim=1)
        self.urban = torch.cat((x[:,2:3], x[:,1:2], x[:,5:6]), dim=1)
        self.agriculture = torch.cat((x[:,2:3], x[:,9:10], x[:,3:4]), dim=1)
        self.atmospheric_penetration = torch.cat((x[:,2:3], x[:,1:2], x[:,11:12]), dim=1)
        
        # Not used bands agrouped in groups of 3
        self.no_mean_0 = torch.cat((x[:,0:1], x[:,6:7], x[:,7:8]), dim=1)
        self.no_mean_1 = torch.cat((x[:,8:9], x[:,9:10], x[:,11:12]), dim=1)

        if self.ndvi_bands:
            ndvi = (x[:,9:10] - x[:,5:6]) / (x[:,9:10] + x[:,5:6])

            # Evi
            evi_top = (x[:,9:10] - x[:,5:6])
            
            evi_down = (x[:,9:10] + 6 * x[:,5:6] - 7.5 * x[:,3:4] + 1)
            
            # Adjust EVI calculation to avoid division by zero
            # evi_down = torch.where(evi_down == 0, torch.ones_like(evi_down), evi_down)
            evi_down = torch.where(evi_down == 0, torch.full_like(evi_down, 0.001), evi_down)
            
            evi = 2.5 * (evi_top / evi_down + 1)
            
            savi = ((x[:,9:10] - x[:,5:6]) / (x[:,9:10] + x[:,5:6] + 0.5)) * 1.5
            self.output = torch.cat((ndvi, evi, savi), dim=1)

            assert torch.isnan(self.output).max() != True , "Output have Nan values"

            return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1, self.output
        
        else:

            # Check if all bands have the same shape
            assert self.natural_colors.shape == self.agriculture.shape == self.infrared.shape == self.urban.shape == self.atmospheric_penetration.shape == self.no_mean_0.shape == self.no_mean_1.shape, "All bands must have the same shape"

        return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1

class EuroSATGroupBands(SentinelGroupBands):
    def sepair_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sepair the bands from the image
        """
        self.natural_colors = torch.cat((x[:,3:4], x[:,2:3], x[:,1:2]), dim=1)  # B4 (Red), B3 (Green), B2 (Blue)
        self.infrared = torch.cat((x[:,7:8], x[:,3:4], x[:,2:3]), dim=1)  # B8 (NIR), B4 (Red), B3 (Green)
        self.urban = torch.cat((x[:,11:12], x[:,10:11], x[:,3:4]), dim=1)  # B12 (SWIR2), B11 (SWIR1), B4 (Red)
        self.agriculture = torch.cat((x[:,11:12], x[:,8:9], x[:,1:2]), dim=1)  # B12 (SWIR2), B8A (Narrow NIR), B2 (Blue)
        self.atmospheric_penetration = torch.cat((x[:,11:12], x[:,10:11], x[:,0:1]), dim=1)  # B12 (SWIR2), B11 (SWIR1), B1 (Coastal aerosol)

        # Not used bands agrouped in groups of 3
        self.no_mean_0 = torch.cat((x[:,0:1], x[:,4:5], x[:,5:6]), dim=1)  # B1 (Coastal aerosol), B5 (Red Edge1), B6 (Red Edge2)
        self.no_mean_1 = torch.cat((x[:,6:7], x[:,8:9], x[:,9:10]), dim=1)  # B7 (Red Edge3), B8A (Narrow NIR), B9 (Water vapor)

        if self.ndvi_bands:
            ndvi = (x[:,9:10] - x[:,5:6]) / (x[:,9:10] + x[:,5:6])

            # Evi
            evi_top = (x[:,9:10] - x[:,5:6])
            
            evi_down = (x[:,9:10] + 6 * x[:,5:6] - 7.5 * x[:,3:4] + 1)
            
            # Adjust EVI calculation to avoid division by zero
            # evi_down = torch.where(evi_down == 0, torch.ones_like(evi_down), evi_down)
            evi_down = torch.where(evi_down == 0, torch.full_like(evi_down, 0.001), evi_down)
            
            evi = 2.5 * (evi_top / evi_down + 1)
            
            savi = ((x[:,9:10] - x[:,5:6]) / (x[:,9:10] + x[:,5:6] + 0.5)) * 1.5
            self.output = torch.cat((ndvi, evi, savi), dim=1)

            assert torch.isnan(self.output).max() != True , "Output have Nan values"

            return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1, self.output
        
        else:

            # Check if all bands have the same shape
            assert self.natural_colors.shape == self.agriculture.shape == self.infrared.shape == self.urban.shape == self.atmospheric_penetration.shape == self.no_mean_0.shape == self.no_mean_1.shape, "All bands must have the same shape"

        return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1

class EuroSATGroupBandsLBP(EuroSATGroupBands):
    def sepair_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sepair the bands from the image
        """
        device = x.device
        self.natural_colors = torch.cat((x[:,3:4], x[:,2:3], x[:,1:2]), dim=1)  # B4 (Red), B3 (Green), B2 (Blue)
        self.infrared = torch.cat((x[:,7:8], x[:,3:4], x[:,2:3]), dim=1)  # B8 (NIR), B4 (Red), B3 (Green)
        self.urban = torch.cat((x[:,11:12], x[:,10:11], x[:,3:4]), dim=1)  # B12 (SWIR2), B11 (SWIR1), B4 (Red)
        self.agriculture = torch.cat((x[:,11:12], x[:,8:9], x[:,1:2]), dim=1)  # B12 (SWIR2), B8A (Narrow NIR), B2 (Blue)
        self.atmospheric_penetration = torch.cat((x[:,11:12], x[:,10:11], x[:,0:1]), dim=1)  # B12 (SWIR2), B11 (SWIR1), B1 (Coastal aerosol)

        # Not used bands agrouped in groups of 3
        self.no_mean_0 = torch.cat((x[:,0:1], x[:,4:5], x[:,5:6]), dim=1)  # B1 (Coastal aerosol), B5 (Red Edge1), B6 (Red Edge2)
        self.no_mean_1 = torch.cat((x[:,6:7], x[:,8:9], x[:,9:10]), dim=1)  # B7 (Red Edge3), B8A (Narrow NIR), B9 (Water vapor)

        self.lbp_0 = lbp_values((x[:,:3]*255).type(torch.int).cpu())
        self.lbp_0 = (self.lbp_0 - self.lbp_0.min()) / (self.lbp_0.max() - self.lbp_0.min())
        self.lbp_1 = lbp_values((x[:,3:6]*255).type(torch.int).cpu())
        self.lbp_1 = (self.lbp_1 - self.lbp_1.min()) / (self.lbp_1.max() - self.lbp_1.min())
        self.lbp_2 = lbp_values((x[:,6:9]*255).type(torch.int).cpu())
        self.lbp_2 = (self.lbp_2 - self.lbp_2.min()) / (self.lbp_2.max() - self.lbp_2.min())
        self.lbp_3 = lbp_values((x[:,9:12]*255).type(torch.int).cpu())
        self.lbp_3 = (self.lbp_3 - self.lbp_3.min()) / (self.lbp_3.max() - self.lbp_3.min())
   
        return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1, self.lbp_0, self.lbp_1, self.lbp_2, self.lbp_3

class SentinelGroupBandsLBP(SentinelGroupBands):
    def sepair_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sepair the bands from the image
        """
        self.natural_colors = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
        self.infrared = torch.cat((x[:,10:11], x[:,5:6], x[:,4:5]), dim=1)
        self.urban = torch.cat((x[:,2:3], x[:,1:2], x[:,5:6]), dim=1)
        self.agriculture = torch.cat((x[:,2:3], x[:,9:10], x[:,3:4]), dim=1)
        self.atmospheric_penetration = torch.cat((x[:,2:3], x[:,1:2], x[:,11:12]), dim=1)
        
        # Not used bands agrouped in groups of 3
        self.no_mean_0 = torch.cat((x[:,0:1], x[:,6:7], x[:,7:8]), dim=1)
        self.no_mean_1 = torch.cat((x[:,8:9], x[:,9:10], x[:,11:12]), dim=1)

        self.lbp_0 = lbp_values((x[:,:3]*255).type(torch.int))
        self.lbp_0 = (self.lbp_0 - self.lbp_0.min()) / (self.lbp_0.max() - self.lbp_0.min()).type(torch.float32)
        self.lbp_0 = self.lbp_0.to(x.device)
        self.lbp_1 = lbp_values((x[:,3:6]*255).type(torch.int))
        self.lbp_1 = (self.lbp_1 - self.lbp_1.min()) / (self.lbp_1.max() - self.lbp_1.min()).type(torch.float32)
        self.lbp_1 = self.lbp_1.to(x.device)
        self.lbp_2 = lbp_values((x[:,6:9]*255).type(torch.int))
        self.lbp_2 = (self.lbp_2 - self.lbp_2.min()) / (self.lbp_2.max() - self.lbp_2.min()).type(torch.float32)
        self.lbp_2 = self.lbp_2.to(x.device)
        self.lbp_3 = lbp_values((x[:,9:12]*255).type(torch.int))
        self.lbp_3 = (self.lbp_3 - self.lbp_3.min()) / (self.lbp_3.max() - self.lbp_3.min()).type(torch.float32)
        self.lbp_3 = self.lbp_3.to(x.device)

        return self.natural_colors, self.agriculture, self.infrared, self.urban, self.atmospheric_penetration, self.no_mean_0, self.no_mean_1, self.lbp_0, self.lbp_1, self.lbp_2, self.lbp_3

if __name__ == "__main__":
    path = r"C:\BEPE\data\mini_caco10k\clean_10k_geography"
    dataset = MSReadData()
    batch_size = 2
    img_size=256
    train=False
    dataloader = dataset.create_dataLoader(path, img_size, batch_size, num_workers=1, train=train, mode=None)
    print(f"len of the dataset: {len(dataloader.dataset)}")
    img, positive = next(iter(dataloader))

    sentinel = SentinelGroupBandsLBP()
    # natural_colors, agriculture, infrared, urban, atmospheric_penetration, no_mean_0, no_mean_1, lbp_0, lbp_1, lbp_2, lbp_3 = sentinel.sepair_bands(img)
    output = sentinel.sepair_bands(img)
    
    # sentinel = EuroSATGroupBands(ndvi_bands=True)
    # natural_colors, agriculture, infrared, urban, atmospheric_penetration, no_mean_0, no_mean_1, ndvi = sentinel.sepair_bands(img)
    # print(f"Natural colors: {natural_colors.shape}")
    # print(f"Agriculture: {agriculture.shape}")
    # print(f"Infrared: {infrared.shape}")
    # print(f"Urban: {urban.shape}")
    # print(f"Atmospheric penetration: {atmospheric_penetration.shape}")
    # print(f"No mean 0: {no_mean_0.shape}")
    # print(f"No mean 1: {no_mean_1.shape}")
    # print(f"NDVI : {ndvi.shape}")

    # print("End of the script")

    # from utils.utils import *