import os
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import *
from torch.utils.data import Dataset 
import json


class CustomDataForBreast (Dataset):
    def __init__(self , json_file_path , oversample=False, downsample=False, 
                forced_downsample=False, transform = False , use_subtraction = False ,
                min_slices = None , max_slices= None , target_slices =None , seg_type = ''):
        
        self.oversample = oversample
        self.downsample = downsample
        self.forced_downsample = forced_downsample
        self.transform = transform 
        self.use_subtraction = use_subtraction
        self.min_slices = min_slices
        self.max_slices = max_slices
        self.target_slices = target_slices
        self.seg_type = seg_type

        # Load JSON file containing file paths
        with open(json_file_path, 'r') as f:
            self.file_paths = json.load(f)

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
        start_index = slices_to_remove_each_end
        end_index = current_slices - slices_to_remove_each_end
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
            pre_image  = self.oversample_to_majority (pre_contrast_nifti , indices_to_repeat , current_slices )
            post_image = self.oversample_to_majority (post_contrast_nifti , indices_to_repeat , current_slices )
            mask_image = self.oversample_to_majority (mask_nifti , indices_to_repeat , current_slices )

            for i in range(pre_image.shape[-1]):

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
                pre_contrast_data. append(pre_img )
                post_img = post_image[: , : , i]
                post_contrast_data.append(post_img)
                mask_data.append(mask_image[... , i])

        if self.forced_downsample:

            pre_image  = self.forced_downsample_to_target (pre_contrast_nifti , current_slices )
            post_image = self.forced_downsample_to_target (post_contrast_nifti , current_slices )
            mask_image = self.forced_downsample_to_target (mask_nifti , current_slices )

            for i in range(pre_image.shape[-1]):

                pre_img = pre_image [: , : , i]
                pre_contrast_data. append(pre_img )
                post_img = post_image[: , : , i]
                post_contrast_data.append(post_img)
                mask_data.append(mask_image[... , i])

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
            pre_contrast_data = torch.tensor(pre_contrast_data).unsqueeze(1)
            post_contrast_data = torch.tensor(post_contrast_data).unsqueeze(1)
            input = torch.cat((pre_contrast_data, post_contrast_data ), dim=1)
    
        
        mask = mask.permute(1,2,3,0)
        input = input.permute(1,2,3,0)

        if self.transform:
            transform1 = RandomRotation(degrees=180)
            aug1_input = transform1(input)
            aug1_mask= transform1(mask)
            input = torch.cat((input , aug1_input) , dim =-1)
            mask = torch.cat((mask,aug1_mask) , dim =-1)

        return input , mask 
