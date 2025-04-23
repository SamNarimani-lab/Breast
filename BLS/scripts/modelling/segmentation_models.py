import torch.nn as nn
from monai.networks.nets import  BasicUNetPlusPlus 
from torchvision.models.segmentation import *



class ModefiedBasicUNetPlusPlus(nn.Module):
    def __init__(self , spatial_dims , in_channels , out_channels , features , dropout ):
        super(ModefiedBasicUNetPlusPlus , self). __init__()
        self.modefiedUNetPlusPlus = BasicUNetPlusPlus (spatial_dims= spatial_dims, in_channels=in_channels , out_channels=out_channels , features=features
                                       , act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
                                       norm=('instance', {'affine': True}), bias=True, dropout= dropout , upsample='deconv')
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , x):

        x = self.modefiedUNetPlusPlus(x)[0]
        return x
    

