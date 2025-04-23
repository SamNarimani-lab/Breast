import numpy as np
import os
import nibabel as nib
import csv
import pandas

data_path = 'final_nifti_files'


n = len(next(os.walk (data_path))[1])
print(n)

def mask_path_function(root_dir):

    patient_folders = os.listdir(root_dir)
    mask_files = []
    mask_path = []
    for patient_folder in patient_folders:
        patient_folder_path = os.path.join(root_dir, patient_folder)
        m_files = [f for f in os.listdir(patient_folder_path) if f.startswith('MASK')]
        # using the first element of the m_files since it is a list to extract the tring name of the mask
        # mask_files.append(m_files[0])
        patient_number = patient_folder  
        mask_path.append((patient_number , os.path.join(patient_folder_path , m_files[0])))  
        
                                                        

    return mask_path 

mask_path_list = mask_path_function (data_path)
#print(mask_path_list)


# Checking that we have all mask data and there is no missing data
if n == len(mask_path_list) :
    print('Nomber of mask files are ok')
else:
    print('Check for the missing mask files')


def mask_loader(mask_path):
    mask_number = mask_path[:][0]
    mask_array = (nib.load(mask_path[:][1])) . get_fdata()

    

    # Find indices where the image is equal to 1
    indices = np.argwhere(mask_array == 1)
    # Find the minimum and maximum indices in rows and columns  
    # be carefule!!!! If you wanna use the result in an array, they should be one less than these amounts
    min_row_indices = np.min(indices , axis = 0)[0] + 1
    max_row_indices = np.max(indices , axis =0)[0] + 1
    min_column_indices = np.min(indices , axis = 0)[1] + 1
    max_column_indices = np.max(indices , axis =0)[1] + 1
    min_slice_indices = np.min(indices , axis = 0)[2] + 1
    max_slice_indices = np.max(indices , axis =0)[2] + 1

    return mask_number , min_row_indices  , max_row_indices   , min_column_indices  , max_column_indices , min_slice_indices , max_slice_indices



#creating csv file of annotation box for patients
with open('Annotation box.csv', 'w', newline='') as patient_file:
    patient_writer = csv.writer(patient_file)
    patient_writer.writerow(['patient_number' ,'min_row_indices'  , 'max_row_indices '  , 'min_column_indices'  , 'max_column_indices' , 'min_slice_indices' , 'max_slice_indices'])
    for i in range(n):
        patient_number , min_row_indices  , max_row_indices   , min_column_indices  , max_column_indices , min_slice_indices , max_slice_indices = mask_loader(mask_path_function (data_path)[i])
        patient_writer.writerow([patient_number ,min_row_indices  , max_row_indices   , min_column_indices  , max_column_indices , min_slice_indices , max_slice_indices])
        

# converting to excel file    

csv_file = pandas.read_csv ('Annotation box.csv')

csv_file.to_excel ('Annotation box.xlsx' , index = True)
    




