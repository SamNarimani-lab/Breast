# import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import splitfolders
import random
import shutil





# spliting folders into train/test 

# Define the root directory containing the 100 folders
root_directory = 'final_nifti_files'
train_directory = 'Stavanger_data/train'
val_directory = 'Stavanger_data//val'
test_directory = 'Stavanger_data/test'

# Ensure the destination directories exist
os.makedirs(train_directory, exist_ok=True)
os.makedirs(val_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# List all subdirectories (patient folders) in the root directory
subdirectories = os.listdir(root_directory)

# Shuffle the list of subdirectories to randomize the order
random.shuffle(subdirectories)

# Calculate the number of folders for each set
total_folders = len(subdirectories)
num_train_folders = int(0.8 * total_folders)
num_val_folders = int(0.1 * total_folders)
num_test_folders = int(0.1 * total_folders)

# Assign folders to each set
train_folders = subdirectories[:num_train_folders]
val_folders = subdirectories[num_train_folders:num_train_folders + num_val_folders]
test_folders = subdirectories[num_train_folders + num_val_folders:]

# Function to copy folders from source to destination
def copy_folders(source_directory, destination_directory, folders_to_copy):
    for folder in folders_to_copy:
        source_path = os.path.join(source_directory, folder)
        destination_path = os.path.join(destination_directory, folder)
        shutil.copytree(source_path, destination_path)

# Copy folders to their respective sets
copy_folders(root_directory, train_directory, train_folders)
copy_folders(root_directory, val_directory, val_folders)
copy_folders(root_directory, test_directory, test_folders)

print("Dataset splitting completed.")







# auxiliary functions


# function to convert nifti file to array to use in the model



# class to convert data 



