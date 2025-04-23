import os
import nibabel as nib


class GetDimension:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.nifti_dimensions = []

    def load_nifti_dimensions(self):
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    
                    if file_name.startswith("S-PRE"):
                        img_pre = nib.load(file_path)
                        dimensions_pre = img_pre.shape 
                        self.nifti_dimensions.append((file_name, dimensions_pre))

    def get_third_part_statistics(self):
        unique_third_parts = set()
        count_120, count_140, count_150 = 0, 0, 0
        
        for item in self.nifti_dimensions:
            third_part = item[1][2]
            unique_third_parts.add(third_part)

            if third_part == 120:
                count_120 += 1
            elif third_part == 140:
                count_140 += 1
            else:
                count_150 += 1

        return unique_third_parts, [count_120, count_140, count_150]
