from skimage.feature import local_binary_pattern
from data_reader.EuroSAT_ms_reader import MSReadData
import torch

def lbp_values(image, features_size = 16):
    all_features = []
    for img in image:
        img_features = []
        for channel in img:
            lbp = local_binary_pattern(channel.cpu().numpy(), features_size, 2, method='uniform')
            lbp = torch.tensor(lbp).unsqueeze(0)
            img_features.append(lbp)
        lbp_concat = torch.cat(img_features, dim=0)
        all_features.append(lbp_concat)
    all_features = torch.stack(all_features)
    return all_features

def lbp_histogram(image, features_size=16):
    all_features = lbp_values(image)
    all_hist = []
    for i in all_features:
        hist = []
        for feature_channel in i:
            hist.append(torch.histc(feature_channel, bins=features_size, min=0, max=features_size-1))

        hist_tensor = torch.stack(hist)
        all_hist.append(hist_tensor)
    hits = torch.stack(all_hist)
    hits_normalized = (hits - hits.min()) / (hits.max() - hits.min())
    return hits_normalized

if __name__ == "__main__":
    root_path = "./data/EuroSAT_permanent"
    batch_size = 5
    image_size = 256
    train = False

    # Example usage
    dataset = MSReadData()
    dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train, num_workers=1)
            
    # sample = dataloader.dataset[2]
    sample = next(iter(dataloader))

    img = sample[0]
    features_size = 16
    hits = lbp_histogram(img, features_size)
    hist = hits.view(batch_size, -1)

    print(f"Histogram shape {hist.shape}")

    lbp_values = lbp_values(img, features_size)

    print(f"Features shape {lbp_values.shape}")