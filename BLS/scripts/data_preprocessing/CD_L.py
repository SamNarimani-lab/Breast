import os
import nibabel as nib
import numpy as np
import torch
from torchvision import transforms

class CustomDataForLesion:
    def __init__(self , root_dir):
        self.root_dir = root_dir
        #self.patient_folders = os.listdir(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_data(self):
        # patient_folder = self.patient_folders[idx]
        pre_contrast_data , post_contrast_data , mask_data = [] , [] , []
        for folder_name in os.listdir(self.root_dir):
            patient_path = os.path.join(self.root_dir, folder_name)

            #patient_path = os.path.join(self.root_dir, self.patient_folders)
        
            file_name = os.listdir(patient_path)

            for file_name in os.listdir(patient_path):
                
                if file_name.startswith('S-PRE'):
                    pre_contrast_path = os.path.join(patient_path, file_name)
                    pre_contrast_nifti = nib.load(pre_contrast_path).get_fdata()
                    for i in range (pre_contrast_nifti.shape[2]):
                        pre_contrast_data. append(pre_contrast_nifti [: , : , i])
                    #pre_contrast_data.append( pre_contrast_nifti.get_fdata())
                        
                elif file_name.startswith('S-POST'):
                    post_contrast_path = os.path.join(patient_path, file_name)
                    post_contrast_nifti = nib.load(post_contrast_path).get_fdata()
                    for i in range (post_contrast_nifti.shape[2]):
                        post_contrast_data .append( post_contrast_nifti [: , : , i])
                    #post_contrast_data.append (post_contrast_nifti.get_fdata())
                        
                elif file_name.startswith('S-mask-B'):
                    mask_path = os.path.join(patient_path, file_name)
                    mask_nifti = nib.load(mask_path).get_fdata()
                    #mask_data = mask_data.append( mask_nifti.get_fdata())              
                    for i in range (mask_nifti.shape[2]):
                        mask_data .append ( mask_nifti [ : , : , i])
                    
        pre_contrast_data = np.array(pre_contrast_data)
        post_contrast_data = np.array(post_contrast_data)
        mask = np.array(mask_data)

        pre_contrast_data = torch.tensor(pre_contrast_data).unsqueeze(1)
        post_contrast_data = torch.tensor(post_contrast_data).unsqueeze(1)
        mask = torch.tensor(mask_data).unsqueeze(1)
        # #return pre_contrast_data , post_contrast_data , mask_data

        input = torch.cat((pre_contrast_data, post_contrast_data), dim=1)     
        
        return input , mask


