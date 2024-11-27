import os
import sys
import torch
from scripts.data_preprocessing.CD_BE import CustomDataForBreast
from trainer import BreastTrainer
from monai.losses import *
from torch.optim import RAdam , Adam , SGD
from torch.optim.lr_scheduler import *
from monai.networks.nets import *
from torch.nn import *
from torchvision.models.segmentation import   *
from torchvision.transforms import *
from torch.utils.data import  DataLoader
from MONAI import *
# from DenseNetme import DenseNet



# sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))  # Add 'data' folder to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))  # Add 'models' folder to sys.path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory of json files
# root_dir_train = r'dataset\processed_data\train_files_B.json'
# root_dir_val   = r'dataset\processed_data\val_files_B.json'
# root_dir_test  = r'dataset\processed_data\test_files_B.json'

# # Loading data
# ###### min and max slices should be found before any further process and target slice can be voluntary chosen
# ##### train/val/test method
# train_dataset = CustomDataForBreast( root_dir_train,  oversample = True  , min_slices = 120 , max_slices = 150 , target_slices = 100 , seg_type = 'B' )

# val_dataset   = CustomDataForBreast( root_dir_val  ,  oversample = True  , min_slices = 120 , max_slices = 150 , target_slices = 100 , seg_type = 'B' ) 

# test_dataset  = CustomDataForBreast( root_dir_test ,  oversample = True  , min_slices = 120 , max_slices = 150 , target_slices = 100 , seg_type = 'B' )


# # Loading data
# train_loader = DataLoader(train_dataset, batch_size = 8 , shuffle=True )
# val_loader   = DataLoader(val_dataset  , batch_size = 8 , shuffle=False )
# test_loader  = DataLoader(test_dataset , batch_size = 8 , shuffle=False )


#kfold method
root_dir_kfold = r'dataset\\processed_data\data_kfold_B.json'
kfold_dataset = CustomDataForBreast ( root_dir_kfold ,  oversample = True  , min_slices = 120 , max_slices = 150 , target_slices = 100 , seg_type = 'B' )
# kfold_dataset = CustomDataForBreast (root_dir_kfold  ,  oversample = True  , min_slices = 12  , max_slices = 41  , target_slices = 10 , seg_type = 'L')


# Loading Model
# MONAI Models
# model = ModefiedBasicUNet (spatial_dims = 2 , in_channels =2  , out_channels =1 ,
#                              features = (4, 8 ,16 ,31, 64 , 4) , dropout = 0.05).to(device=device)

# model = ModefiedBasicUNetPlusPlus (spatial_dims = 2 , in_channels =2  , out_channels =1 ,
#                             features = (32, 64 ,128 ,256, 512 , 32) , dropout = 0.0).to(device=device)
# model = ModefiedDenseNet(spatial_dims=2 , in_channels=2  , out_channels=1 , block_config  = (6, 12, 64, 48) ).to(device=device)

# model = DoubledDenseNet (spatial_dims=2 , in_channels1=2 , in_channels2=6 , out_channels1=4 , out_channels2=1 ,
#                           block_config1 = (6, 12, 32, 32) , block_config2  = (6, 12, 32, 32) ).to(device=device)

#model = AttentionUnet(spatial_dims = 2 , in_channels =2 , out_channels= 1, channels = (64, 128 ,256 ,512, 1024 ), strides =(2,2,2,2,2) ,
 #                       kernel_size=3, up_kernel_size=3, dropout=0.05).to(device=device)
# model = DoubleUNet (spatial_dims1 = 2 , spatial_dims2=2 , in_channels1 = 2 , out_channels1 =64 , features1 = (64 ,128 ,256, 512 ,1024, 64), dropout1 =0.05
#                     , in_channels2 = 64 , out_channels2 = 1 , features2 =(64 ,128 ,256, 512 ,1024, 64) , dropout2 = 0.05).to(device=device)
### pytorch semantic segmentation models
# model =  CustomDeepLabv3( weight = None  , version ='resnet50', num_input_channels=2 , num_classes=1 ).to(device=device)
# model = lraspp_mobilenet_v3_large(progress= True, num_classes = 1  ).to(device=device)
model = CustomFCN(weight = None ,  version = 'resnet50' , num_input_channels =  1 , num_classes = 1  ).to(device=device)


optimizer = RAdam(model.parameters(), lr=0.00001 , weight_decay= 0.00003)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='True')
# scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.00004, mode='triangular', cycle_momentum= False)

criterion1 = MaskedDiceLoss (include_background= False , sigmoid = True)
# criterion1 =  DiceFocalLoss(include_background= False , sigmoid = True , gamma =2 , lambda_dice = 0.1 , lambda_focal = 0.9 )
criterion2 = DiceCELoss (include_background= False , sigmoid = True , jaccard=False , lambda_dice = 0.1 , lambda_ce = 0.9)
#criterion1 =  DiceFocalLoss(include_background= False , sigmoid = True , gamma =2 , lambda_dice = 0.1 , lambda_focal = 0 )
#criterion2 = DiceCELoss (include_background= False , sigmoid = True , jaccard=False , lambda_dice = 0.1 , lambda_ce = 0.9)
# criterion2 = BCELoss()


trainer = BreastTrainer(model, criterion1, criterion2, optimizer , scheduler , alpha = 1 , beta = 0 , k_fold = 10  )
# trainer.train(train_loader, val_loader, test_loader, epochs=100, patience=10 , model_name = 't')
# trainer = BreastTrainer(model, criterion1, criterion2, optimizer , scheduler , alpha=1.0, beta=0.0 , k_fold=5 )
trainer.train_cv( kfold_dataset , epochs=100 , patience_model = 10 , batch_size=8 , model_name = 'FCN-Pre-BRS' )

torch.save(model, 'FCN-Pre-BRS.pt')



