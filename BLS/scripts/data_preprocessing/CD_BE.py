import os
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import *
from torch.utils.data import Dataset 
import json
from torch.utils.data import DataLoader
import pandas as pd


class CustomDataForBreast (Dataset):
    def __init__(self , json_file_path , oversample=False, downsample=False, forced_downsample=False, transform = False
                   , use_subtraction = False ,
                  min_slices = None , max_slices= None , target_slices =None , seg_type = None , data_type = None):
        
        self.oversample = oversample
        self.downsample = downsample
        self.forced_downsample = forced_downsample
        self.transform = transform 
        self.use_subtraction = use_subtraction
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.target_slices = target_slices
        self.seg_type = seg_type
        self.data_type = data_type

        # Load JSON file containing file paths
        with open(json_file_path, 'r') as f:
            self.file_paths = json.load(f)

    # def reorientation (self , nifti_file , ):

    def oversample_to_majority (self , image , indices_to_repeat , current_slices ):

        if current_slices >= self.max_slices:
            return image[:, :, : self.max_slices]
        # Combine original slices with the oversampled slices
        oversampled_image = np.concatenate((image, image[:, :, indices_to_repeat]), axis=-1)
        return oversampled_image

 
    def downsample_to_minority(self , image , current_slices , indices_to_keep  ):

        if current_slices <= self.min_slices:
            return image[: , : ,  : self.min_slices]
        downsampled_image = image[:, :, indices_to_keep]
        return downsampled_image

    def forced_downsample_to_target(self , image , current_slices ):
        


        if current_slices <= self.target_slices:
            return image[:, :, : self.target_slices]

        slices_to_remove = current_slices - self.target_slices
        slices_to_remove_each_end = slices_to_remove // 2
        # Calculate indices to keep
        start_index = slices_to_remove_each_end
        end_index = current_slices - slices_to_remove_each_end
        # Adjust if the number of slices to remove is odd
        if slices_to_remove % 2 != 0:
            end_index -= 1
        downsampled_image = image[:, :, start_index:end_index]
        return downsampled_image

    def __len__(self):
        return len(self.file_paths)
    
     
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        pre_contrast_data  , post_contrast_data , mask_data = [] , []  , []
        mask_nifti = None
        mask_B_nifti = None
        if self.seg_type == 'L':
            for file_name in os.listdir(img_path):
                if file_name.startswith('S-mask-L'):                    
                    mask_path = os.path.join(img_path, file_name)
                    mask_nifti = nib.load(mask_path).get_fdata()

                elif file_name.startswith('S-PRE'):
                    pre_contrast_path = os.path.join(img_path, file_name)
                    pre_contrast_nifti = ((nib.load(pre_contrast_path).get_fdata()))

                elif file_name.startswith('S-POST1'):
                    post_contrast_path  = os.path.join(img_path, file_name)
                    post_contrast_nifti = (nib.load(post_contrast_path).get_fdata())

        elif self.seg_type == 'LB':
            if len(os.listdir(img_path)) > 3 :
                for file_name in os.listdir(img_path):
                    if file_name.startswith('S-mask-B'):
                        mask_B = os.path.join(img_path, file_name)
                        mask_B_nifti = nib.load(mask_B).get_fdata()

                    if file_name.startswith('S-mask-L'):
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()

                if self.data_type == 'PP1':
                    for file_name in os.listdir(img_path):
                        if file_name.startswith('S-PRE'):
                            pre_contrast_path = os.path.join(img_path, file_name)
                            pre_contrast_nifti = ((nib.load(pre_contrast_path).get_fdata())) * mask_B_nifti
                        if file_name.startswith('S-POST'):
                            post_contrast_path  = os.path.join(img_path, file_name)
                            post_contrast_nifti = (nib.load(post_contrast_path).get_fdata()) * mask_B_nifti
                                        
                    
        elif self.seg_type == 'LBM':
            if len(os.listdir(img_path)) > 3 :
                for file_name in os.listdir(img_path):
                    if file_name.startswith('S-mask-B'):
                        mask_B = os.path.join(img_path, file_name)
                        mask_B_nifti = nib.load(mask_B).get_fdata()
                    if file_name.startswith('S-mask-L' ):
                        
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()
                        min_slice , max_slice = mask_selected_volume (mask_nifti)
                        # print(min_slice , max_slice)
                        mask_nifti = mask_nifti[: , : , min_slice -2 : max_slice + 2]
                        # print(mask_nifti.shape[-1])

                if self.data_type == 'PP1':
                    for file_name in os.listdir(img_path):
                        if file_name.startswith('S-PRE'):
                            pre_contrast_path = os.path.join(img_path, file_name)
                            pre_contrast_nifti = ((nib.load(pre_contrast_path).get_fdata()))[: , : , min_slice -2 : max_slice + 2] * mask_B_nifti[: , : , min_slice -2 : max_slice + 2]
                        if file_name.startswith('S-POST'):
                            post_contrast_path  = os.path.join(img_path, file_name)
                            post_contrast_nifti = (nib.load(post_contrast_path).get_fdata())[: , : , min_slice -2 : max_slice + 2] * mask_B_nifti[: , : , min_slice -2 : max_slice + 2]
        elif self.seg_type == 'LBM_pred':
            if len(os.listdir(img_path)) > 3 :
                for file_name in os.listdir(img_path):
                    if file_name.startswith('S-mask-B'):
                        mask_B = os.path.join(img_path, file_name)
                        mask_B_nifti = nib.load(mask_B).get_fdata()
                    if file_name.startswith('S-mask-L' ):
                        
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()

                if self.data_type == 'PP1':
                    for file_name in os.listdir(img_path):
                        if file_name.startswith('S-PRE'):
                            pre_contrast_path = os.path.join(img_path, file_name)
                            pre_contrast_nifti = ((nib.load(pre_contrast_path).get_fdata())) * mask_B_nifti
                        if file_name.startswith('S-POST'):
                            post_contrast_path  = os.path.join(img_path, file_name)
                            post_contrast_nifti = (nib.load(post_contrast_path).get_fdata()) * mask_B_nifti
                                        
        elif self.seg_type == 'LBMO':

            df = pd.read_excel(r'Directory of excel file for LMBO')

            if len(os.listdir(img_path)) > 3 :
                for file_name in os.listdir(img_path):
                    if file_name.startswith('S-mask-B'):
                        mask_B = os.path.join(img_path, file_name)
                        mask_B_nifti = nib.load(mask_B).get_fdata()
                    if file_name.startswith('S-mask-L' ):
                        
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()
                        
                

                if self.data_type == 'PP1':
                    for file_name in os.listdir(img_path):
                        if file_name.startswith('S-PRE'):
                            pre_contrast_path = os.path.join(img_path, file_name)
                            pre_contrast_nifti = nib.load(pre_contrast_path).get_fdata() * mask_B_nifti
                        if file_name.startswith('S-POST'):
                            post_contrast_path  = os.path.join(img_path, file_name)
                            post_contrast_nifti = nib.load(post_contrast_path).get_fdata() * mask_B_nifti          
                
                min_slice , max_slice = mask_selected_volume (mask_nifti)
                index =  df[df['File Path'] == img_path].index
                H_min = int(df.loc[index, 'H_min_final'].values[0])  # Convert to an integer
                H_max = int(df.loc[index, 'H_max_final'].values[0])  # Convert to an integer
                pre_contrast_nifti = pre_contrast_nifti  [ : , H_min : H_max + 1 , min_slice -1 : max_slice + 1 ]
                post_contrast_nifti = post_contrast_nifti[ : , H_min : H_max + 1 , min_slice -1 : max_slice + 1 ]
                mask_nifti = mask_nifti[: , H_min : H_max + 1 , min_slice -1 : max_slice + 1]    
        elif self.seg_type == 'LBMO_pred':

            df = pd.read_excel(r'CDirectory of excel file for LMBO_pred')

            if len(os.listdir(img_path)) > 3 :
                for file_name in os.listdir(img_path):
                    if file_name.startswith('S-mask-B'):
                        mask_B = os.path.join(img_path, file_name)
                        mask_B_nifti = nib.load(mask_B).get_fdata()
                    if file_name.startswith('S-mask-L' ):
                        
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()
                        
                

                if self.data_type == 'PP1':
                    for file_name in os.listdir(img_path):
                        if file_name.startswith('S-PRE'):
                            pre_contrast_path = os.path.join(img_path, file_name)
                            pre_contrast_nifti = nib.load(pre_contrast_path).get_fdata() * mask_B_nifti
                        if file_name.startswith('S-POST'):
                            post_contrast_path  = os.path.join(img_path, file_name)
                            post_contrast_nifti = nib.load(post_contrast_path).get_fdata() * mask_B_nifti          
                
                
                index =  df[df['File Path'] == img_path].index
                H_min = int(df.loc[index, 'H_min_final'].values[0])  # Convert to an integer
                H_max = int(df.loc[index, 'H_max_final'].values[0])  # Convert to an integer
                pre_contrast_nifti = pre_contrast_nifti  [ : , H_min : H_max + 1 , : ]
                post_contrast_nifti = post_contrast_nifti[ : , H_min : H_max + 1 , : ]
                mask_nifti = mask_nifti[: , H_min : H_max + 1 , :]    

        else:
            if self.data_type == 'P':
                for file_name in os.listdir(img_path):

                    if file_name.startswith('S-mask-' + self.seg_type):
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()
                    if file_name.startswith('S-PRE'):
                        pre_contrast_path = os.path.join(img_path, file_name)
                        pre_contrast_nifti = (nib.load(pre_contrast_path).get_fdata())

            elif self.data_type == 'PP1':
                for file_name in os.listdir(img_path):

                    if file_name.startswith('S-mask-' + self.seg_type):
                        mask_path = os.path.join(img_path, file_name)
                        mask_nifti = nib.load(mask_path).get_fdata()
                    if file_name.startswith('S-PRE'):
                        pre_contrast_path = os.path.join(img_path, file_name)
                        pre_contrast_nifti = (nib.load(pre_contrast_path).get_fdata())
                    if file_name.startswith('S-POST'):
                        post_contrast_path  = os.path.join(img_path, file_name)
                        post_contrast_nifti = nib.load(post_contrast_path).get_fdata()

  
        current_slices = pre_contrast_nifti.shape[-1]
        
        if self.oversample :
            additional_slices_needed = self.max_slices - current_slices
            indices_to_repeat = np.random.choice(current_slices, additional_slices_needed, replace=True)
            if self.data_type == 'P':
                pre_image  = self.oversample_to_majority (pre_contrast_nifti , indices_to_repeat , current_slices )
            elif self.data_type == 'PP1':
                pre_image  = self.oversample_to_majority (pre_contrast_nifti , indices_to_repeat , current_slices )
                post_image = self.oversample_to_majority (post_contrast_nifti , indices_to_repeat , current_slices )
            mask_image = self.oversample_to_majority (mask_nifti , indices_to_repeat , current_slices )
            for i in range(pre_image.shape[-1]):
                
                if self.data_type == 'P':

                    pre_img = pre_image [: , : , i]
                    pre_contrast_data. append(pre_img )
                elif self.data_type == 'PP1':
                    pre_img = pre_image [: , : , i]
                    pre_contrast_data. append(pre_img )
                    post_img = post_image[: , : , i]
                    post_contrast_data.append(post_img)

                mask_data.append(mask_image[... , i])

        if self.downsample:
            indices_to_keep = np.random.choice(current_slices, self.min_slices , replace=False)
            indices_to_keep.sort()
            pre_image  = self.downsample_to_minority (pre_contrast_nifti , current_slices , indices_to_keep)
            post_image = self.downsample_to_minority (post_contrast_nifti , current_slices , indices_to_keep)
            mask_image = self.downsample_to_minority (mask_nifti , current_slices , indices_to_keep)
            for i in range(pre_image.shape[-1]):
                pre_img = pre_image [: , : , i]
                # mean_pre = pre_img.mean()
                # std_pre =pre_img.std()
                # pre_img = (pre_img - mean_pre)/ std_pre
                pre_contrast_data. append(pre_img )

                post_img = post_image[: , : , i]
                # mean_post = post_img.mean()
                # std_post = post_img.std()
                # post_img = (post_img - mean_post)/ std_post
                post_contrast_data.append(post_img)

                mask_data.append(mask_image[... , i])

        if self.forced_downsample:
            pre_image  = self.forced_downsample_to_target (pre_contrast_nifti , current_slices )
            post_image = self.forced_downsample_to_target (post_contrast_nifti , current_slices )
            mask_image = self.forced_downsample_to_target (mask_nifti , current_slices )
            for i in range(pre_image.shape[-1]):
                pre_img = pre_image [: , : , i]
                # mean_pre = pre_img.mean()
                # std_pre =pre_img.std()
                # pre_img = (pre_img - mean_pre)/ std_pre
                pre_contrast_data. append(pre_img )

                post_img = post_image[: , : , i]
                # mean_post = post_img.mean()
                # std_post = post_img.std()
                # post_img = (post_img - mean_post)/ std_post
                post_contrast_data.append(post_img)

                mask_data.append(mask_image[... , i])

        
        if self.data_type == 'P':
            pre_contrast_data = np.array(pre_contrast_data)
        elif self.data_type == 'PP1':
            pre_contrast_data = np.array(pre_contrast_data)
            post_contrast_data = np.array(post_contrast_data)

        mask_data = np.array(mask_data)

        mask = torch.tensor(mask_data).unsqueeze(1)
    



        if self.use_subtraction:
            
            subtraction_data = pre_contrast_data- post_contrast_data
            pre_contrast_data = torch.tensor(pre_contrast_data).unsqueeze(1)
            post_contrast_data = torch.tensor(post_contrast_data).unsqueeze(1)
            subtraction_data = torch.tensor(subtraction_data).unsqueeze(1)
            input = torch.cat((pre_contrast_data, post_contrast_data  , subtraction_data), dim=1) 
        else:
            if self.data_type == 'P':
                pre_contrast_data = torch.tensor(pre_contrast_data).unsqueeze(1)
                input = pre_contrast_data
            elif self.data_type == 'PP1':
                pre_contrast_data = torch.tensor(pre_contrast_data).unsqueeze(1)
                post_contrast_data = torch.tensor(post_contrast_data).unsqueeze(1)
                input = torch.cat((pre_contrast_data, post_contrast_data ), dim=1)
    
        
        mask = mask.permute(1,2,3,0)
        input = input.permute(1,2,3,0)

        if self.transform:
            # Apply transformations if specified
            transform1 = RandomRotation(degrees=180)
            aug1_input = transform1(input)
            aug1_mask= transform1(mask)
            input = torch.cat((input , aug1_input) , dim =-1)
            mask = torch.cat((mask,aug1_mask) , dim =-1)

        return input , mask 



def find_min_max_slices(json_file_path):
    # Define the path to the JSON file
    
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    min_slices = float('inf')  # Initialize to a very large number
    max_slices = float('-inf')  # Initialize to a very small number

    # Iterate through each file path in the JSON data
    for file_path in data:  # Adjust the key based on your JSON structure
        for file_name in os.listdir(file_path):
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):  # Check for NIfTI files
        
                # Load the NIfTI file
                img_path= os.path.join(file_path, file_name)
                img = nib.load(img_path)
                img_data = img.get_fdata()  # Get the image data

                num_slices = img_data.shape[2]  # Assuming slices are along the 3rd dimension

                # Update min and max slices
                min_slices = min(min_slices, num_slices)
                max_slices = max(max_slices, num_slices)

    return min_slices, max_slices

def find_min_max_L(json_file_path):
    # Define the path to the JSON file
    
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    min_slices = float('inf')  # Initialize to a very large number
    max_slices = float('-inf')  # Initialize to a very small number

    # Iterate through each file path in the JSON data
    for file_path in data:  # Adjust the key based on your JSON structure
        for file_name in os.listdir(file_path):
            if file_name.startswith('S-mask-L') :  # Check for NIfTI files
        
                # Load the NIfTI file
                img_path= os.path.join(file_path, file_name)
                img = nib.load(img_path)
                img_data = img.get_fdata()  # Get the image data

                min_L , max_L = mask_selected_volume (img_data)
                num_slices = max_L - min_L

                # Update min and max slices
                min_slices = min(min_slices, num_slices)
                max_slices = max(max_slices, num_slices)


    return min_slices , max_slices

# this function should be edited later becasu if we have two lesions and they are far from each other, it chooses almost the breast volume which is undesirable
def mask_selected_volume(mask_nifti):
    selected_slices = []
    for i in range(mask_nifti.shape[-1]):
        if mask_nifti[..., i].any() == 1 :
            selected_slices.append(i)
    min_slice = min(selected_slices)
    max_slice = max(selected_slices)

    return min_slice , max_slice

# Loading k_fold dataset
def load_kfold_data(seg_type, data_type, dataset_dir):
    json_file_path = f"{dataset_dir}/data_kfold_{seg_type}.json"
    if seg_type == 'B' or seg_type == 'L' or seg_type == 'LB' or seg_type ==' LBMO_pred' or  seg_type ==' LBM_pred':
        min_slices, max_slices = find_min_max_slices(json_file_path=json_file_path)
        print(f'min_slices is {min_slices} and max_slices is {max_slices}')
        return CustomDataForBreast(
            json_file_path, 
            oversample=True,
            min_slices=min_slices, 
            max_slices=max_slices,
            target_slices=None,
            seg_type=seg_type, 
            data_type=data_type
        )

    elif seg_type == 'LBM' or seg_type=='LBMO':
        min_slices, max_slices = find_min_max_L(json_file_path=json_file_path)
        print(f'min_slices is {min_slices} and max_slices is {max_slices}')

        return CustomDataForBreast(
            json_file_path, 
            oversample=True,
            min_slices=min_slices, 
            max_slices=max_slices + 4,
            target_slices=None,
            seg_type=seg_type, 
            data_type=data_type
        )

# Loading 3-subset data (Random splitting)
def load_3_subset_data(seg_type, batch_size, data_type, dataset_dir):
    root_dir_all   = f"{dataset_dir}/data_kfold_{seg_type}.json"
    root_dir_train = f"{dataset_dir}/train_files_{seg_type}.json"
    root_dir_val   = f"{dataset_dir}/val_files_{seg_type}.json"
    root_dir_test  = f"{dataset_dir}/test_files_{seg_type}.json"
    
    # Retrieve min and max slices from the full dataset
    min_slices, max_slices = find_min_max_slices(root_dir_all)
    
    # Define datasets
    train_dataset = CustomDataForBreast(
        root_dir_train, 
        oversample=True, 
        min_slices=min_slices, 
        max_slices=max_slices,
        target_slices=None, 
        seg_type=seg_type,
        data_type=data_type
    )
    val_dataset = CustomDataForBreast(
        root_dir_val, 
        oversample=True, 
        min_slices=min_slices, 
        max_slices=max_slices,
        target_slices=None, 
        seg_type=seg_type,
        data_type=data_type
    )
    test_dataset = CustomDataForBreast(
        root_dir_test, 
        oversample=True, 
        min_slices=min_slices, 
        max_slices=max_slices,
        target_slices=None, 
        seg_type=seg_type,
        data_type=data_type
    )
    
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader