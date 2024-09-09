import os
import random
import shutil

# Path to the dataset folder
dataset_path = r".\data\EuroSAT"

# Path to the test folder
test_path = r".\data\test_EuroSAT"
os.makedirs(test_path, exist_ok=True)

# List of class names
class_names = os.listdir(dataset_path)

# Percentage of folders to move to the test folder
percentage = 20

# Random seed
random.seed(42)

# Iterate over each class
for class_name in class_names:
    # Path to the class folder
    class_path = os.path.join(dataset_path, class_name)
    
    # List of image files in the class folder
    image_files = os.listdir(class_path)
    
    # Number of images to move to the test folder
    num_images = int(len(image_files) * percentage / 100)
    
    # Randomly select images to move
    images_to_move = random.sample(image_files, num_images)
    
    # Move the selected images to the test folder
    for image in images_to_move:
        src = os.path.join(class_path, image)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
        dst = os.path.join(test_path, class_name, image)
        shutil.move(src, dst)
