import os
import nibabel as nib

# This class help to merge masks for patients having more than one lesion in their breast image
# S-mask shows the mask file for Stavanger data. S indicates Stavanger dataset and it should be changed for various dataset

class MaskMerger:
    # initialize the root directory
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def count_mask_files(self):
        patient_folders = os.listdir(self.root_dir)
        mask_files = {}
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.root_dir, patient_folder)
            m_files = [f for f in os.listdir(patient_folder_path) if f.startswith('S-mask')]
            mask_files[patient_folder] = m_files  # Store the mask filenames as a list
        return mask_files

    def sum_masks_and_save(self):
        mask_files = self.count_mask_files()

        for patient_folder, mask_filenames in mask_files.items():
            if len(mask_filenames) > 1:
                combined_mask = None
                for mask_filename in mask_filenames:
                    mask_path = os.path.join(self.root_dir, patient_folder, mask_filename)
                    mask_nifti = nib.load(mask_path)
                    mask_data = mask_nifti.get_fdata()

                    if combined_mask is None:
                        combined_mask = mask_data
                    else:
                        combined_mask += mask_data

                # Save the combined mask as a new NIfTI file
                combined_mask_nifti = nib.Nifti1Image (combined_mask, mask_nifti.affine)
                combined_mask_path = os.path.join (self.root_dir, patient_folder,'S-mask-{}.nii'.format(patient_folder) )
                nib.save(combined_mask_nifti, combined_mask_path)



