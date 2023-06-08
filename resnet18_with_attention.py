# The following is my implementation of the resnet18 architecture
# combined with the attention mechanism.
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mmdet.registry import MODELS
from torchvision.transforms import InterpolationMode

class ECAModule(nn.Module):
    """
    From the 2020 paper "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks."
    Also this is more computationally efficient than the SE block, so that's noice.
    This block helps incorporate channel-wise attention to refine feature maps.
    """
    def __init__(self):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(0.5)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.view(b, 1, c))
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionGate(nn.Module):
    """
    This block helps incorporate spatial attention to refine feature maps.
    """
    def __init__(self,in_channels,gate_channels,inter_channels=None):
        super(AttentionGate,self).__init__()
        self.in_channels = in_channels
        self.gate_channels = gate_channels

        if inter_channels == None:
            inter_channels = in_channels // 2

        self.conv_g = nn.Sequential(
            nn.Conv2d(gate_channels,inter_channels,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Adding dropout to prevent overfitting.
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Adding dropout to prevent overfitting.
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels,1,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Gives probabilities to scale by.
        )

    def forward(self,x, g):
        # Logic for what kind of attention you want to implement.
        x1 = self.conv_x(x) # After this, x will have inter_channel number of channels.
        g = self.conv_g(g) # After this, g will have inter_channel number of channels.
        g = nn.functional.interpolate(g, size=x1.size()[2:], mode='bilinear', align_corners=True) # To make the spatial dimensions of g and x the same.
        psi = self.psi(F.relu(x1 + g, inplace=True))
        return x * psi.expand_as(x)  # Returns a feature map with the same spatial dimensions and num. channels as x.


class NonLocalBlock(nn.Module):
    """
    Implemented according to the paper : "Non-local Neural Networks" (Wang et al., 2018). There is a ready made module for this in the
    torchvision library, but implementing it from scratch makes it more customizable and doesn't require depending on the torchvision library.
    This block helps capture long-range dependencies.
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = 4 # If not specified, take inter_channels to be a fourth of the number of input channels.
                                                   # This 4 and cutting the spatial dimensions by 4 of k,v are UNRELATED.
        self.g = nn.Conv2d(self.in_channels,self.inter_channels, kernel_size=1, stride=1, padding=0) # kerne_size = 1,stride=1,padding=0 keeps the spatial dims the same.
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels,self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(self.inter_channels,self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu') # Doing the kaiming initialization, which is guess is the default but I'm experimenting.

        if sub_sample: # Make the input dimensions smaller to reduce the computational complexity. By default I'm setting this to True.
            self.g_pool = nn.MaxPool2d(2)
            self.phi_pool = nn.MaxPool2d(2)

    def forward(self, x):
        b, c, _, _ = x.size()

        v = self.g(x)
        if hasattr(self, 'g_pool'):
            v = self.g_pool(v)
        v = v.view(b, self.inter_channels, -1)
        v = v.permute(0, 2, 1)

        q = self.theta(x)
        q = q.view(b, self.inter_channels, -1)
        q = q.permute(0, 2, 1)

        k = self.phi(x)
        if hasattr(self, 'phi_pool'):
            k = self.phi_pool(k)
        k = k.view(b, self.inter_channels, -1)
        # Not permuting (in this case, transposing) k is necessary for matrix multiplication.

        """
        Using matmul to compute the pairwise correlations between 
        different spatial locations in the input feature map x.
        """
        attention_map = torch.matmul(q, k)
        attention_map_normalized = F.softmax(attention_map, dim=-1)

        attended_v = torch.matmul(attention_map_normalized, v)
        attended_v = attended_v.permute(0, 2, 1).contiguous()
        attended_v = attended_v.view(b, self.inter_channels, *x.size()[2:])
        if hasattr(self, 'g_pool'):
            attended_v = F.interpolate(attended_v, size=x.size()[2:], mode='bilinear', align_corners=False) # Interpolating to get to the same spatial dims as x.
        W_attended_v = self.W(attended_v)
        z = W_attended_v + x # Gives an output of the same spatial dims and channels as x.

        return z


    

# Registering my custom backbone
@MODELS.register_module()
class ResNet18Attention(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet18Attention, self).__init__(**kwargs)
        print("Setting up the custom backbone with attention gates, ECA and non-local blocks (with a Resnet18 base.)")
        # Getting a pretrained resnet18 model
        # self.res18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        self.res18 = models.resnet18(weights = None)

        # Initializing the AttentionGate, ECA and NonLocalBlocks. We don't need to specify the number of input channels to ECA since it computes em 
        # using x.shape() anyways. Even the nonlocalblock can do this but lite, im defined it to take the param and am passing the param anyways.
        self.attention1 = AttentionGate(64, 512)
        self.eca1 = ECAModule() 
        # self.non_local1 = NonLocalBlock(64)

        self.attention2 = AttentionGate(128, 512)
        self.eca2 = ECAModule()
        # self.non_local2 = NonLocalBlock(128)

        self.attention3 = AttentionGate(256, 512)
        self.eca3 = ECAModule()
        # self.non_local3 = NonLocalBlock(256)

        self.attention4 = AttentionGate(512, 512)
        self.eca4 = ECAModule()
        # self.non_local4 = NonLocalBlock(512)

        # self.freeze_all_layers('model') # Freezing all params of the backbone so that the weights won't be updated during training.

        print("Successfully initialized the attention, ECA and non-local blocks.")

    def freeze_all_layers(self,param_type = 'model'):
        """In case you want to freeze the layers of the backbone. param_type specifies whether you want
        to freeze the entire backbone or just the resnet parts leaving the other ones learnable (like the eca, attention etc.)"""
        # Freeze the backbone layers
        if param_type == "all":
            for param in self.res18.parameters():
                param.requires_grad = False

            # Freeze the custom layers
            for module in [self.eca1, self.eca2, self.eca3, self.eca4,
                        self.attention1, self.attention2, self.attention3, self.attention4]:
                for param in module.parameters():
                    param.requires_grad = False
        elif param_type == 'model':
            for param in self.res18.parameters():
                param.requires_grad = False
    


    def forward(self, x):
        x = self.res18.conv1(x)
        x = self.res18.bn1(x)
        x = self.res18.relu(x)
        x = self.res18.maxpool(x)

        x1 = self.res18.layer1(x) # Feature map after layer 1.
        x2 = self.res18.layer2(x1) # Feature map after layer 2.
        x3 = self.res18.layer3(x2) # Feature map after layer 3.
        x4 = self.res18.layer4(x3) # Feature map after layer 4.

        
        # x4 = self.non_local4(x4_attn)

        # x5 = x4_attn.clone() # Use this x5 as a gating signal. Might help improve performance.

        # Makes sense to apply ECA first, attention gates next
        # and NonLocal Block after that.
        x1_attn = self.attention1(x1,x4)
        x1 = self.eca1(x1_attn)
        # x1 = self.non_local1(x1)

        x2_attn = self.attention2(x2,x4)
        x2 = self.eca2(x2_attn)
        # x2 = self.non_local2(x2)

        x3_attn = self.attention3(x3,x4)
        x3 = self.eca3(x3_attn)
        # x3 = self.non_local3(x3_attn)

        x4_attn = self.attention4(x4,x4)
        x4 = self.eca4(x4_attn)

        # x1_att = self.attention1(x1, x4)
        # x1 = self.eca1(x1_att)
        # x1 = self.non_local1(x1)

        # x2_att = self.attention2(x2, x4)
        # x2 = self.eca2(x2_att)
        # x2 = self.non_local2(x2)

        # x3_att = self.attention3(x3, x4)
        # x3 = self.eca3(x3_att)
        # x3 = self.non_local3(x3)

        # x4_att = self.attention4(x4, x4)
        # x4 = self.eca4(x4_att)
        # x4 = self.non_local4(x4)

        outs = [x1, x2, x3, x4]

        return tuple(outs)





        
        

        
        
        


