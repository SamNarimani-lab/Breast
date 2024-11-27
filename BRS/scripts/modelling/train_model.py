import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import costumdataset
import model
from torchmetrics.functional.classification import dice
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train and val path
train_path_folders = 'Stavanger_data/train'
val_path_folders   = 'Stavanger_data/val'

# Define your CustomNiftiDataset3D class here (as previously defined)

# Create DataLoader for the 3D NIfTI training dataset
train_dataset = costumdataset.CustomNiftiDataset3D(train_path_folders)
train_dataloader = DataLoader(train_dataset  ,  shuffle=False)
# len(train_dataloader)
# for images, masks in train_dataloader:
#     images, masks = images.to(device).float(), masks.to(device).float()
#     print(images.shape , masks.shape)

# Create DataLoader for the 3D NIfTI validation dataset
val_dataset = costumdataset.CustomNiftiDataset3D(val_path_folders )
val_dataloader = DataLoader(val_dataset , shuffle=False)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# shape =[]
# mask= []
# for images, masks in train_dataloader:
#     images, masks = images.to(device ), masks.to(device ) 
#     shape.append(images.shape)
#     mask.append(masks.shape)

# Instantiate the 3D U-Net model and move it to the GPU if available
model_U = model.UNet2D(in_channels=2, out_channels=1).to(device)

# Define loss function (e.g., Dice coefficient for segmentation)
criterion = nn.L1Loss()
#weight = torch.tensor([0.9]) # higher weight for positive class
#criterion = nn.BCEWithLogitsLoss()
# Define optimizer with a learning rate
optimizer = optim.Adam(model_U.parameters(), lr=0.001)

# Define a learning rate scheduler ( ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

# Training loop

num_epochs = 5

for epoch in range(num_epochs):
    model_U.train()  # Set the model to training mode
    total_loss = 0.0
    for images, masks in train_dataloader:
        images, masks = images[0].to(device).float(), masks[0].to(device).float()  # Convert to float32 and move data to GPU if available
        
        for idx in range(0 , images.shape[0] ) :           
            optimizer.zero_grad()
            outputs = model_U(images[idx , ...] )
            loss = criterion(outputs, masks[idx  , ...])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    
        # Validation loop
    model_U.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_dataloader:
            val_images, val_masks = val_images[0].to(device).float(), val_masks[0].to(device).float()  # Convert to float32 and move data to GPU if available
            for idx in range(0 , images.shape[0] ) :
                val_outputs = model_U(val_images[idx , ...])
                val_loss += criterion(val_outputs, val_masks[idx, ...]).item()

    # Calculate and print the average training and validation loss
    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    #avg_dice_train = dice_train / len(train_dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.7f} Validation Loss: {avg_val_loss:.7f} ')
    

    # Update the learning rate scheduler based on validation loss
    scheduler.step(avg_val_loss)

# Save the trained 2D U-Net model
torch.save(model_U.state_dict(), 'unet2d_model.pth')