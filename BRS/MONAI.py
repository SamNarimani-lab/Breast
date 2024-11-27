import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet , BasicUNetPlusPlus 
from torchvision.models.segmentation import *
from DenseNetme import DenseNet
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=64):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, num_filters, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(num_filters)
        
        self.conv_3x3_1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(num_filters)
        
        self.conv_3x3_2 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(num_filters)
        
        self.conv_3x3_3 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(num_filters)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1_2 = nn.Conv2d(in_channels, num_filters, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(num_filters)
        
        self.conv_1x1_3 = nn.Conv2d(num_filters * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        
        x2 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        
        x3 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))
        
        x4 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = self.conv_1x1_2(x5)
        x5 = self.bn_conv_1x1_2(x5)
        x5 = x5.expand(-1, -1, x.shape[2], x.shape[3])
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))
        
        return x


class DoubleUNet (nn.Module):
    def __init__(self  , spatial_dims1 , spatial_dims2 , in_channels1 , out_channels1 , features1 , dropout1 , in_channels2 , out_channels2 , features2 , dropout2):
        super(DoubleUNet , self).__init__()
        self.UNet1 = BasicUNet (spatial_dims= spatial_dims1, in_channels=in_channels1 , out_channels=out_channels1 , features=features1
                                       , act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
                                       norm=('instance', {'affine': True}), bias=True, dropout= dropout1 , upsample='deconv')
        self.UNet2 = BasicUNet (spatial_dims= spatial_dims2, in_channels=in_channels2 , out_channels=out_channels2 , features=features2
                                       , act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
                                       norm=('instance', {'affine': True}), bias=True, dropout= dropout2 , upsample='deconv')
    def forward(self , x):
        x = self.UNet1 (x)
        x = self.UNet2 (x)

        return x

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


class ModefiedDenseNet(nn.Module):
    def __init__(self , spatial_dims , in_channels , out_channels , block_config ):
        super(ModefiedDenseNet , self). __init__()
        self.modefiedDenseNet = DenseNet (spatial_dims = spatial_dims, in_channels = in_channels , out_channels = out_channels,
                                               init_features= 64, growth_rate=32, block_config=block_config
                                              , bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.05)   
        # self.ASPP = ASPP (in_channels=4, out_channels = out_channels, num_filters=4)               
    
    def forward(self , x):

        x = self.modefiedDenseNet(x)
        # x = self.ASPP(x)

        return x

class DoubledDenseNet(nn.Module):
    def __init__(self , spatial_dims , in_channels1 , in_channels2 , out_channels1 , out_channels2 , block_config1 , block_config2 ):
        super(DoubledDenseNet , self). __init__()
        self.modefiedDenseNet1 = DenseNet (spatial_dims = spatial_dims, in_channels = in_channels1, out_channels = out_channels1,
                                               init_features= 64, growth_rate=32, block_config=block_config1
                                              , bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.05)                   
        self.modefiedDenseNet2 = DenseNet (spatial_dims = spatial_dims, in_channels = in_channels2, out_channels = out_channels2,
                                        init_features= 64, growth_rate=32, block_config=block_config2
                                        , bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.05)  
    def forward(self , x):

        x1 = self.modefiedDenseNet1(x)
        x2 = torch.cat((x1 , x) , dim = 1)
        x = self.modefiedDenseNet2(x2)

        return x      

# Pretrained models in pytorch
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

class CustomLRASPPMobileNetV3Large(nn.Module):
    def __init__(self, weight , num_input_channels, num_classes):
        super(CustomLRASPPMobileNetV3Large, self).__init__()
        self.model = lraspp_mobilenet_v3_large(weight)
        
        # Modify the first convolutional layer to accept num_input_channels
        self.model.backbone.features[0][0] = nn.Conv2d(
            num_input_channels, 
            self.model.backbone.features[0][0].out_channels, 
            kernel_size=self.model.backbone.features[0][0].kernel_size, 
            stride=self.model.backbone.features[0][0].stride, 
            padding=self.model.backbone.features[0][0].padding, 
            bias=False
        )
        
        # Modify the classifier to match the desired number of output classes
        self.model.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        return self.model(x)

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
    