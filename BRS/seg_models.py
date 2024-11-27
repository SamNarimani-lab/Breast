import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet , BasicUNetPlusPlus , DenseNet
from torchvision.models.segmentation import *
import torch.nn.functional as F


# Using models in monai website and changing the parameters based on our approach considerations

# UNet Model
class ModefiedBasicUNet(nn.Module):
    def __init__(self , spatial_dims , in_channels , out_channels , features , dropout ):
        super(ModefiedBasicUNet , self). __init__()
        self.modefiedUNet = BasicUNet (spatial_dims= spatial_dims, in_channels=in_channels , out_channels=out_channels , features=features
                                       , act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
                                       norm=('instance', {'affine': True}), bias=True, dropout= dropout , upsample='deconv')
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , x):
        x = self.modefiedUNet(x)
        return x

# UNet++ Model
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

# DenseNet
class ModefiedDenseNet(nn.Module):
    def __init__(self , spatial_dims , in_channels , out_channels , block_config ):
        super(ModefiedDenseNet , self). __init__()
        self.modefiedDenseNet = DenseNet (spatial_dims = spatial_dims, in_channels = in_channels , out_channels = out_channels,
                                               init_features= 64, growth_rate=32, block_config=block_config
                                              , bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.05)                  
    
    def forward(self , x):
        x = self.modefiedDenseNet(x)
        return x
  

# Using models in Pytorch and modifying them based on our approach considerations

# DeepLabV3 with three feature extractors
class CustomDeepLabv3(nn.Module):
    def __init__(self, weight  , version, num_input_channels , num_classes):
        super(CustomDeepLabv3, self).__init__()

        if version == 'mobilnet':
            self.model = deeplabv3_mobilenet_v3_large(weight)
        elif version == 'resnet50':
            self.model = deeplabv3_resnet50(weight)
        elif version == 'resnet101':
            self.model = deeplabv3_resnet101(weight)
        
        # Modify the first convolutional layer to accept num_input_channels
        self.model.backbone.conv1 = nn.Conv2d(
            num_input_channels, 
            self.model.backbone.conv1.out_channels, 
            kernel_size=self.model.backbone.conv1.kernel_size, 
            stride=self.model.backbone.conv1.stride, 
            padding=self.model.backbone.conv1.padding, 
            bias=False
        )
        
        # Modify the classifier to match the desired number of output classes
        self.model.classifier[-1] = nn.Conv2d(
            self.model.classifier[-1].in_channels, 
            num_classes, 
            kernel_size=self.model.classifier[-1].kernel_size
        )

    def forward(self, x):
        return self.model(x)

# FCN model with two feature extractors 
class CustomFCN(nn.Module):
    def __init__(self, weight , version,  num_input_channels, num_classes):
        super(CustomFCN, self).__init__()
        
        if version == 'resnet50':
            self.model = fcn_resnet50(weight)
        elif version == 'resnet101':
            self.model = fcn_resnet101(weight)

        # Modify the first convolutional layer to accept the specified number of input channels
        first_conv_layer = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=first_conv_layer.out_channels, 
            kernel_size=first_conv_layer.kernel_size, 
            stride=first_conv_layer.stride, 
            padding=first_conv_layer.padding, 
            bias=False
        )
        
        # Ensure the weights of the new first convolutional layer are initialized appropriately
        nn.init.kaiming_normal_(self.model.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Update the classifier to produce the desired number of output classes
        self.model.classifier[-1] = nn.Conv2d(
            in_channels=self.model.classifier[-1].in_channels, 
            out_channels=num_classes, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )

    def forward(self, x):
        return self.model(x)
    