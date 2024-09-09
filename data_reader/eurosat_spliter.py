import os
from datasets import load_dataset
EuroSAT_MSI = load_dataset("blanchon/EuroSAT_MSI")

# Open the text file in write mode
with open("eurosat_test_files.txt", "w") as file:
    # Loop through each sample in the dataset
    for sample in EuroSAT_MSI["test"]:
        # Write the filename of the sample to the text file
        file.write(sample["filename"] + "\n")


root_path = r".\data\EuroSAT_full"
destination_folder = r".\data\EuroSAT"

import shutil

# Open the text file in read mode
with open("eurosat_test_files.txt", "r") as file:
    # Read all lines from the file and store them in a list
    test_files = file.read().splitlines()

# Loop through all files in the root_path
for root, dirs, files in os.walk(root_path):
    for file in files:
        # Construct the full path of the file
        file_path = os.path.join(root, file)
        
        # Check if the file is not in the test_files list
        if file not in test_files:
            # Split the file path to get the folder name
            folder_name = os.path.basename(os.path.dirname(file_path))
            
            # Construct the destination path using the folder_name
            destination_path = os.path.join(destination_folder, folder_name, file)
            
            # Create the destination folder if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy the file to the destination folder
            shutil.copy(file_path, destination_path)

root_path = r".\data\EuroSAT_full"
destination_folder = r".\data\EuroSAT_test"

# Open the text file in read mode
with open("eurosat_test_files.txt", "r") as file:
    # Loop through each line in the file
    for line in file:
        # Remove the newline character from the line
        filename = line.strip()
        
        # Construct the full path of the file in the root_path
        # Split the filename using "_" as the separator
        filename_parts = filename.split("_")

        # Use the first part of the filename as the folder name
        folder_name = filename_parts[0]

        # Construct the full path of the file in the root_path
        file_path = os.path.join(root_path, folder_name, filename)
        # file_path = os.path.join(root_path, filename)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Construct the destination path using the folder_name
            destination_path = os.path.join(destination_folder, folder_name, filename)
            
            # Create the destination folder if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy the file to the destination folder
            shutil.copy(file_path, destination_path)
