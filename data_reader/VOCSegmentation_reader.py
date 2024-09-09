import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the VOCSegmentation dataset
dataset = VOCSegmentation(root='./data', year='2012', image_set='train', transform=transform, target_transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Example usage
for images, masks in dataloader:
    print(images.shape, masks.shape)
    break