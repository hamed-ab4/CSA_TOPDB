import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50, resnet101
from torchvision.transforms import functional
from .resnet import ResNet
#from .eca_module import eca_layer
from .se_module import SELayer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class CSA(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """
    def __init__(self, in_channels, channels,
                 radix=2, reduction_factor=4,
                norm_layer=nn.BatchNorm2d):
        super(CSA, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=1)


    def forward(self, x):

        batch, channel = x.shape[:2]
        splited = torch.split(x, channel//self.radix, dim=1)

        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel//self.radix, dim=1)

        out= torch.cat([att*split for (att, split) in zip(atten, splited)],1)
        return out.contiguous()
        

class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)  
        
        
class BatchDropTop(nn.Module):
    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio
    
    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h*w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda: mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x    
        
class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x                      



class CSA_TOPDB(nn.Module):
    def __init__(self, num_classes, height_ratio=0.5 , width_ratio=0.5):
        super(CSA_TOPDB, self).__init__()
        resnet = resnet50(pretrained=True)
        self.radix = 2

        self.base_1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,resnet.maxpool)
        self.base_2 = resnet.layer1
        self.base_3 = resnet.layer2
        self.base_4 = resnet.layer3
        self.base_5 = resnet.layer4  

        self.att1 = SELayer(64,8)
        self.att2 = SELayer(256,32)
        self.att3 = SELayer(512,64)
        self.att4 = SELayer(1024,128)
        self.att5 = SELayer(2048,256)
        

        self.att_s1=CSA(64,int(64/self.radix),radix=self.radix)
        self.att_s2=CSA(256,int(256/self.radix),radix=self.radix)
        self.att_s3=CSA(512,int(512/self.radix),radix=self.radix)
        self.att_s4=CSA(1024,int(1024/self.radix),radix=self.radix)

        self.BN1 = BN2d(64)
        self.BN2 = BN2d(256)
        self.BN3 = BN2d(512)
        self.BN4 = BN2d(1024)
        
      

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes          

        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )

        self.res_part.load_state_dict(self.base_5.state_dict())

        reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU()
        )


         # global branch
         
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes)  
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)


        # part branch
        
        self.res_part2 = Bottleneck(2048, 512)    
        self.part_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.batch_crop = BatchDropTop(height_ratio )
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)

        

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.base_1(x) 
        x = self.att_s1(x)
        x = self.BN1(x)   
        y = self.att1(x)
        x=x*y.expand_as(x)   

        x = self.base_2(x)   
        x = self.att_s2(x)
        x = self.BN2(x)  
        y = self.att2(x)
        x=x*y.expand_as(x)

        x = self.base_3(x)
        x = self.att_s3(x)
        x = self.BN3(x)
        y = self.att3(x)
        x=x*y.expand_as(x)

        x = self.base_4(x)
        x = self.att_s4(x)
        x = self.BN4(x)
        y = self.att4(x)
        x=x*y.expand_as(x)

        x = self.res_part(x)


        predict = []
        triplet_features = []
        softmax_features = []

        #global branch
        
        glob = self.global_avgpool(x)       
        global_triplet_feature = self.global_reduction(glob).squeeze()  
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)
       
        #part branch
        
        x_nodrop = self.res_part2(x)

        x_droped = self.batch_crop(x_nodrop)

        
        triplet_feature = self.part_maxpool(x_droped).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)

        #third branch
        
        triplet_feature3 = self.part_maxpool(x_nodrop).squeeze()
        feature3 = self.reduction(triplet_feature3)
        softmax_feature3 = self.softmax(feature3)
        triplet_features.append(feature3)
        softmax_features.append(softmax_feature3)
        predict.append(feature3)


        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [           
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.base_1.parameters()},
            {'params': self.base_2.parameters()},
            {'params': self.base_3.parameters()},
            {'params': self.base_4.parameters()},
            {'params': self.base_5.parameters()},
            {'params': self.att1.parameters()},
            {'params': self.att2.parameters()},
            {'params': self.att3.parameters()},
            {'params': self.att4.parameters()},           
            {'params': self.att_s1.parameters()},
            {'params': self.att_s2.parameters()},
            {'params': self.att_s3.parameters()},
            {'params': self.att_s4.parameters()},                                  
            {'params': self.BN1.parameters()},
            {'params': self.BN2.parameters()},
            {'params': self.BN3.parameters()},
            {'params': self.BN4.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
        ]
        return params
