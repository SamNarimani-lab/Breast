import torch
from scripts.data_preprocessing.CD_BE import CustomDataForBreast
from trainer import BreastTrainer
from monai.losses import *
from torch.optim import RAdam 
from torch.optim.lr_scheduler import *
from monai.networks.nets import *
from torch.nn import *
from torchvision.models.segmentation import   *
from torchvision.transforms import *
from seg_models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#kfold 
root_dir_kfold = r'path/to/data/jason_file'
kfold_dataset = CustomDataForBreast ( root_dir_kfold ,  oversample = True  , min_slices = 120 , max_slices = 150 , target_slices = 100 , seg_type = 'B' )


# Loading Model
model = CustomFCN(weight = None ,  version = 'resnet50' , num_input_channels = 1 , num_classes = 1 ).to(device=device)

# optimizer and scheduler
optimizer = RAdam(model.parameters(), lr=0.00001 , weight_decay= 0.00003)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='True')

# Criteria
criterion1 = MaskedDiceLoss (include_background= False , sigmoid = True)
criterion2 = DiceCELoss (include_background= False , sigmoid = True , jaccard=False , lambda_dice = 0.1 , lambda_ce = 0.9)


trainer = BreastTrainer(model, criterion1, criterion2, optimizer , scheduler , alpha = 1 , beta = 0 , k_fold = 10  )
trainer.train_cv( kfold_dataset , epochs=100 , patience_model = 10 , batch_size=8 , model_name = 'FCN-Pre-BRS' )

torch.save(model, 'FCN-BRS.pt')



