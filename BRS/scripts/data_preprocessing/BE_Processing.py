import os
import nibabel as nb
import shutil as sh

class BreastPreprocessing:

    def __init__(self, root_dir , dest_dir):
        
        self.root_dir = root_dir
        self.dest_dir = dest_dir
    

    def load_and_save_nifti(self):          
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            dest_path = os.path.join(self.dest_dir , folder_name)
            os.mkdir( dest_path )
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    if any(file_name.startswith(prefix) for prefix in ['S-PRE', 'S-POST1', 'S-mask-B']):
                        sh.copy2(file_path , dest_path)
