#!/usr/bin/python

from collections import OrderedDict
import torch
import copy
import torch.nn as nn
from torch.functional import einsum
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from thop import profile, clever_format
from einops import rearrange, repeat
from ffc import FFC_BN_ACT, ConcatTupleLayer
from models.utils.utils import img2df, feature_fusion
from models.utils.utils import Basconv, UnetConv, UnetUp, UnetUp4, GloRe_Unit
from models.utils.init_weights import init_weights
from bidirectional_cross_attention import BidirectionalCrossAttention
from typing import Tuple, Union
from common import *
from position_encoding import PositionEmbeddingSine
from msdeformattn import MSDeformAttnPixelDecoder
from shape_spec import ShapeSpec
from transformer import Decoder
from time import time
import torchvision.transforms as T
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness
from scipy.optimize import linear_sum_assignment
def get_model(model_name, in_channels=1, num_classes=9, ratio=0.5):
    if model_name == "unet":
        model = UNet(in_channels, num_classes)
    elif model_name == "y_net_gen":
        model = YNet_general(in_channels, num_classes, ffc=False)
    elif model_name == "y_net_gen_ffc":
        model = YNet_general(in_channels, num_classes, ffc=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance":
        model = YNet_advance1(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance_gcn":
        model = YNet_advance_gcn(in_channels, num_classes, ffc=True, attention=True, gcn=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance2":
        model = YNet_advance2(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance_double":
        model = YNet_advance_double(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance2_gcn":
        model = YNet_advance2_gcn(in_channels, num_classes, ffc=True, attention=True, gcn=True, ratio_in=ratio)
    elif model_name == "y_net_gen_advance2_cat":
        model = YNet_advance2_cat(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)

    #newer
    elif model_name == "y_net_gen_advance2_cat_double":
        model = YNet_advance2_cat_double(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
    #branch
    elif model_name == "y_net_gen_advance2_branch":
        model = YNet_advance2_branch(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
   #2ynet: aadd graph before attention 
    elif model_name == "y_net_gen_advance2_double_graph":
        model = YNet_advance2_branch_graph(in_channels, num_classes, ffc=True, attention=True, gcn=True, ratio_in=ratio)
    #replace cross attention with ATT layer
    elif model_name == "y_net_gen_att_cross":
        model = YNet_advance_double_add_att_cross(in_channels, num_classes, ffc=True, attention=True,  ratio_in=ratio) 
 
    elif model_name == "y_net_gen_att_layer":
        model = YNet_advance2_branch_graph_cs(in_channels, num_classes, ffc=True, attention=True,  ratio_in=ratio) 
 
  
 
  
     ####cross cross cross
    #connect encoder first 
    elif model_name == "y_net_gen_advance2_combine":
        model = YNet_advance2_combine(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)
      
    elif model_name == "y_net_gen_advance2_cross":
        model = YNet_advance2_branch_cross(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)  
      
    elif model_name == "y_net_gen_advance2_double_graph_true":
        model = YNet_advance2_branch_graph_true(in_channels, num_classes, ffc=False, attention=True, ratio_in=ratio)
       
    elif model_name == "y_net_gen_ffc_cs":
        model = YNet_general_cs(in_channels, num_classes, ffc=True, ratio_in=ratio)
       
    elif model_name == "y_net_gen_ffc_cs_cat":
        model = YNet_general_cs_cat(in_channels, num_classes, ffc=True, ratio_in=ratio)         
         
    elif model_name == "y_net_gen_cat_cs":
        model = YNet_advance2_cat_double_cross(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)      
     
    elif model_name == "y_net_gen_cat_cs_channel":
        model = YNet_advance2_cat_double_cscat(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)      
  ###///layer/////// 
    elif model_name == "y_net_two_cs":
        model = YNet_advance2_branch_layer(in_channels, num_classes, ffc=True, attention=True,  ratio_in=ratio)    
        
    elif model_name == "y_net_gen_cat_cs_final":
        model = YNet_advance2_cat_double_cross_cat(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)          
    elif model_name == "y_net_gen_cat_cs_final":
        model = YNet_advance2_cat_double_cross_cat(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)    
    elif model_name == "y_net_layer_add":
        model = YNet_advance2_cat_add_layer(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)           
            
  ########################UNETR UNETR UNETR
    elif model_name == "unetatt":
        model = UNETR2D(in_channels=1, out_channels=9, img_size=(224,224))    
           
           
    elif model_name == "y_net_add_att_in":
        model = YNet_advance_double_add_att_four(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)           
     #############mp2former
     
    elif model_name == "y_net_layer_mp2":
        model = YNet_advance2_cat_layer_mp2(in_channels, num_classes, ffc=True, attention=True, ratio_in=ratio)        
             
                  
    else:
        print("Model name not found")
        assert False

    return model
  

def parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_class):
    check_list = [[],[]]
    for i in range(len(expand_sizes)):
        check_list[0].extend(expand_sizes[i])
        check_list[1].extend(out_channels[i])
    for i in range(len(check_list[0]) - 1):
        assert check_list[0][i + 1] % check_list[1][i] == 0 , 'The out_channel should be divisible by expand_size of the next block, due to the expanded DW conv'
    assert num_token > 0, 'num_token should be larger than 0'
    assert d_model > 0, 'd_model should be larger than 0'
    assert in_channel > 0, 'in_channel should be larger than 0'
    assert project_demension > 0, 'project_demension should be larger than 0'
    assert fc_demension > 0, 'fc_demension should be larger than 0'
    assert num_class > 0, 'num_class should be larger than 0'


class BottleneckLite(nn.Module):
    '''Proposed in Yunsheng Li, Yinpeng Chen et al., MicroNet, arXiv preprint arXiv: 2108.05894v1'''
    def __init__(self, in_channel, expand_size, out_channel, kernel_size=3, stride=1, padding=1):
        super(BottleneckLite, self).__init__()
        self.in_channel = in_channel
        self.expand_size = expand_size
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bnecklite = nn.Sequential(
            nn.Conv2d(self.in_channel, self.expand_size, kernel_size=self.kernel_size, 
                            stride=self.stride, padding=self.padding, groups=self.in_channel).cuda(),
            nn.ReLU6(inplace=True).cuda(),
            nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1).cuda(),
            nn.BatchNorm2d(self.out_channel).cuda()
        )
    
    def forward(self, x):
        return self.bnecklite(x)



class MLP(nn.Module):
    '''widths [in_channel, ..., out_channel], with ReLU within'''
    def __init__(self, widths, bn=True, p=0.5):
        super(MLP, self).__init__()
        self.widths = widths
        self.bn = bn
        self.p = p
        self.layers = []
        for n in range(len(self.widths) - 2):
            layer_ = nn.Sequential(
                nn.Linear(self.widths[n], self.widths[n + 1]).cuda(),
                nn.Dropout(p=self.p).cuda(),
                nn.ReLU6(inplace=True).cuda(),
            )
            self.layers.append(layer_)
        self.layers.append(
            nn.Sequential(
                nn.Linear(self.widths[-2], self.widths[-1]),
                nn.Dropout(p=self.p)
            )
        )
        self.mlp = nn.Sequential(*self.layers).cuda()
        if self.bn:
            self.mlp = nn.Sequential(
                *self.layers,
                nn.BatchNorm1d(self.widths[-1]).cuda()
            )

    def forward(self, x):
        return self.mlp(x)



class DynamicReLU(nn.Module):
    '''channel-width weighted DynamticReLU '''
    '''Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU, arXiv preprint axXiv: 2003.10027v2'''
    def __init__(self, in_channel, control_demension, k=2):
        super(DynamicReLU, self).__init__()
        self.in_channel = in_channel
        self.k = k
        self.control_demension = control_demension
        self.Theta = MLP([control_demension, 4 * control_demension, 2 * k * in_channel], bn=True)

    def forward(self, x, control_vector):
        n, _, _, _ = x.shape
        a_default = torch.ones(n, self.k * self.in_channel).cuda()
        a_default[:, self.k * self.in_channel // 2 : ] = torch.zeros(n, self.k * self.in_channel // 2).cuda()
        theta = self.Theta(control_vector)
        theta = 2 * torch.sigmoid(theta) - 1
        a = theta[:, 0 : self.k * self.in_channel] + a_default
        b = theta[:, self.k * self.in_channel : ] * 0.5
        a = rearrange(a, 'n ( k c ) -> n k c', k=self.k)
        b = rearrange(b, 'n ( k c ) -> n k c', k=self.k)
        # x (NCHW), a & b (N, k, C)
        x = einsum('nchw, nkc -> nchwk', x, a) + einsum('nchw, nkc -> nchwk', torch.ones_like(x).cuda(), b)
        return x.max(4)[0]


class Mobile(nn.Module):
    '''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
    def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
        super(Mobile, self).__init__()
        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.token_demension, self.kernel_size, self.stride, self.k = token_demension, kernel_size, stride, k

        if stride == 2:
            self.strided_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=2, padding=int(self.kernel_size // 2), groups=self.in_channel).cuda(),
                nn.BatchNorm2d(self.expand_size).cuda(),
                nn.ReLU6(inplace=True).cuda()
            )
            self.conv1 = nn.Conv2d(self.expand_size, self.in_channel, kernel_size=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.in_channel).cuda()
            self.ac1 = DynamicReLU(self.in_channel, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=1, padding=1, groups=self.in_channel).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()
        
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac1 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.expand_size, self.expand_size, kernel_size=3, stride=1, padding=1, groups=self.expand_size).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=1, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()

    def forward(self, x, first_token):
        if self.stride == 2:
            x = self.strided_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.ac1(x, first_token)
        x = self.bn2(self.conv2(x))
        x = self.ac2(x, first_token)
        return self.bn3(self.conv3(x))



class Mobile_advance(nn.Module):
    '''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
    def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
        super(Mobile_advance, self).__init__()
        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.token_demension, self.kernel_size, self.stride, self.k = token_demension, kernel_size, stride, k

        if stride == 2:
            self.strided_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=2, padding=int(self.kernel_size // 2), groups=1).cuda(),
                nn.BatchNorm2d(self.expand_size).cuda(),
                nn.ReLU6(inplace=True).cuda()
            )
            self.conv1 = nn.Conv2d(self.expand_size, self.in_channel, kernel_size=3, padding=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.in_channel).cuda()
            self.ac1 = DynamicReLU(self.in_channel, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=1, padding=1, groups=1).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=3, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()
        
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, padding=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac1 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.expand_size, self.expand_size, kernel_size=3, stride=1, padding=1, groups=1).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=3, stride=1, padding=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()

    def forward(self, x, first_token):
        if self.stride == 2:
            x = self.strided_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.ac1(x, first_token)
        x = self.bn2(self.conv2(x))
        x = self.ac2(x, first_token)
        return self.bn3(self.conv3(x))



class Mobile_advance2(nn.Module):
    '''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
    def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
        super(Mobile_advance2, self).__init__()
        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.token_demension, self.kernel_size, self.stride, self.k = token_demension, kernel_size, stride, k

        if stride == 2:
            self.strided_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=2, padding=int(self.kernel_size // 2), groups=1).cuda(),
                nn.BatchNorm2d(self.expand_size).cuda(),
                nn.ReLU6(inplace=True).cuda()
            )
            self.conv1 = nn.Conv2d(self.expand_size, self.in_channel, kernel_size=3, padding=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.in_channel).cuda()
            self.ac1 = DynamicReLU(self.in_channel, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, stride=1, padding=1, groups=1).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=3, stride=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()
        
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.expand_size, kernel_size=3, padding=1, stride=1).cuda()
            self.bn1 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac1 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()      
            self.conv2 = nn.Conv2d(self.expand_size, self.expand_size, kernel_size=3, stride=1, padding=1, groups=1).cuda()
            self.bn2 = nn.BatchNorm2d(self.expand_size).cuda()
            self.ac2 = DynamicReLU(self.expand_size, self.token_demension, k=self.k).cuda()          
            self.conv3 = nn.Conv2d(self.expand_size, self.out_channel, kernel_size=3, stride=1, padding=1).cuda()
            self.bn3 = nn.BatchNorm2d(self.out_channel).cuda()

    def forward(self, x, first_token):
        if self.stride == 2:
            x = self.strided_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.ac1(x, first_token)
        x = self.bn2(self.conv2(x) + x) 
        x = self.ac2(x, first_token) 
        return self.bn3(self.conv3(x))
    
    
    
class Mobile_Former(nn.Module):
    '''Local feature -> Global feature'''
    def __init__(self, d_model, in_channel):
        super(Mobile_Former, self).__init__()
        self.d_model, self.in_channel = d_model, in_channel

        self.project_Q = nn.Linear(self.d_model, self.in_channel).cuda()
        self.unproject = nn.Linear(self.in_channel, self.d_model).cuda()
 #       self.unproject1 = nn.Linear(16, self.d_model).cuda()
        self.eps = 1e-10
        self.shortcut = nn.Sequential().cuda()

    def forward(self, local_feature, x):
        _, c, _, _ = local_feature.shape
        local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')   # N, L, C local 10, 12544, 16
    
        project_Q = self.project_Q(x)   # N, M, C 10, 6, 1
        scores = torch.einsum('nmc , nlc -> nml', project_Q, local_feature) * (c ** -0.5) #10, 6, 12544
        scores_map = F.softmax(scores, dim=-1)  # each m to every l 10, 6, 12544
        fushion = torch.einsum('nml, nlc -> nmc', scores_map, local_feature) #10, 6, 16
        
        unproject = self.unproject(fushion) # N, m, d
 ##################################################3       #evaluate in debug mode
 ######       self.unproject1(fushion) #evaluate in debug mode
        return unproject + self.shortcut(x)



class Former(nn.Module):
    '''Post LayerNorm, no Res according to the paper.'''
    def __init__(self, head, d_model, expand_ratio=2):
        super(Former, self).__init__()
        self.d_model = d_model
        self.expand_ratio = expand_ratio
        self.eps = 1e-10
        self.head = head
        assert self.d_model % self.head == 0
        self.d_per_head = self.d_model // self.head

        self.QVK = MLP([self.d_model, self.d_model * 3], bn=False).cuda()
        self.Q_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.K_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.V_to_heads = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.heads_to_o = MLP([self.d_model, self.d_model], bn=False).cuda()
        self.norm = nn.LayerNorm(self.d_model).cuda()
        self.mlp = MLP([self.d_model, self.expand_ratio * self.d_model, self.d_model], bn=False).cuda()
        self.mlp_norm = nn.LayerNorm(self.d_model).cuda()

    def forward(self, x):
        QVK = self.QVK(x)
        Q = QVK[:, :, 0: self.d_model]
        Q = rearrange(self.Q_to_heads(Q), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        K = QVK[:, :, self.d_model: 2 * self.d_model]
        K = rearrange(self.K_to_heads(K), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        V = QVK[:, :, 2 * self.d_model: 3 * self.d_model]
        V = rearrange(self.V_to_heads(V), 'n m ( d h ) -> n m d h', h=self.head)   # (n, m, d/head, head)
        scores = torch.einsum('nqdh, nkdh -> nhqk', Q, K) / (np.sqrt(self.d_per_head) + self.eps)   # (n, h, q, k)
        scores_map = F.softmax(scores, dim=-1)  # (n, h, q, k)
        v_heads = torch.einsum('nkdh, nhqk -> nhqd', V, scores_map) #   (n, h, m, d_p) -> (n, m, h, d_p)
        v_heads = rearrange(v_heads, 'n h q d -> n q ( h d )')
        attout = self.heads_to_o(v_heads)
        attout = self.norm(attout)  #post LN
        attout = self.mlp(attout)
        attout = self.mlp_norm(attout)  # post LN
        return attout   # No res



class Former_Mobile(nn.Module):
    '''Global feature -> Local feature'''
    def __init__(self, d_model, in_channel):
        super(Former_Mobile, self).__init__()
        self.d_model, self.in_channel = d_model, in_channel
        
        self.project_KV = MLP([self.d_model, 2 * self.in_channel], bn=False).cuda()
        self.shortcut = nn.Sequential().cuda()
    
    def forward(self, x, global_feature):
        res = self.shortcut(x)
        n, c, h, w = x.shape
        project_kv = self.project_KV(global_feature)
        K = project_kv[:, :, 0 : c]  # (n, m, c)
        V = project_kv[:, :, c : ]   # (n, m, c)
        x = rearrange(x, 'n c h w -> n ( h w ) c') # (n, l, c) , l = h * w
        scores = torch.einsum('nqc, nkc -> nqk', x, K) # (n, l, m)
        scores_map = F.softmax(scores, dim=-1) # (n, l, m)
        v_agg = torch.einsum('nqk, nkc -> nqc', scores_map, V)  # (n, l, c)
        feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
        return feature + res



class MobileFormerBlock(nn.Module):
    '''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
    '''output local & global, if stride=2, then it is a downsample Block'''
    def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, expand_ratio=2):
        super(MobileFormerBlock, self).__init__()

        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.d_model, self.stride, self.k, self.head, self.expand_ratio = d_model, stride, k, head, expand_ratio

        self.mobile = Mobile(self.in_channel, self.expand_size, self.out_channel, self.d_model, kernel_size=3, stride=self.stride, k=self.k).cuda()
        self.former = Former(self.head, self.d_model, expand_ratio=self.expand_ratio).cuda()
        self.mobile_former = Mobile_Former(self.d_model, self.in_channel).cuda()
        self.former_mobile = Former_Mobile(self.d_model, self.out_channel).cuda()
    
    def forward(self, local_feature, global_feature):
        z_hidden = self.mobile_former(local_feature, global_feature)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile(local_feature, z_out[:, 0, :])
        x_out = self.former_mobile(x_hidden, z_out)
        return x_out, z_out

class MobileFormerBlock_advance(nn.Module):
    '''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
    '''output local & global, if stride=2, then it is a downsample Block'''
    def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, padding=1, expand_ratio=2):
        super(MobileFormerBlock_advance, self).__init__()

        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.d_model, self.stride, self.k, self.head, self.expand_ratio = d_model, stride, k, head, expand_ratio

        self.mobile_advance = Mobile_advance(self.in_channel, self.expand_size, self.out_channel, self.d_model, kernel_size=3, stride=self.stride, k=self.k).cuda()
        self.former = Former(self.head, self.d_model, expand_ratio=self.expand_ratio).cuda()
        self.mobile_former = Mobile_Former(self.d_model, self.in_channel).cuda()
        self.former_mobile = Former_Mobile(self.d_model, self.out_channel).cuda()
    
    def forward(self, local_feature, global_feature):
        z_hidden = self.mobile_former(local_feature, global_feature)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile_advance(local_feature, z_out[:, 0, :])
        x_out = self.former_mobile(x_hidden, z_out)
        return x_out, z_out



class MobileFormerBlock_advance2(nn.Module):
    '''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
    '''output local & global, if stride=2, then it is a downsample Block'''
    def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, padding=1, expand_ratio=2):
        super(MobileFormerBlock_advance2, self).__init__()

        self.in_channel, self.expand_size, self.out_channel = in_channel, expand_size, out_channel
        self.d_model, self.stride, self.k, self.head, self.expand_ratio = d_model, stride, k, head, expand_ratio

        self.mobile_advance = Mobile_advance2(self.in_channel, self.expand_size, self.out_channel, self.d_model, kernel_size=3, stride=self.stride, k=self.k).cuda()
        self.former = Former(self.head, self.d_model, expand_ratio=self.expand_ratio).cuda()
        self.mobile_former = Mobile_Former(self.d_model, self.in_channel).cuda()
        self.former_mobile = Former_Mobile(self.d_model, self.out_channel).cuda()
    
    def forward(self, local_feature, global_feature):
        z_hidden = self.mobile_former(local_feature, global_feature)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile_advance(local_feature, z_out[:, 0, :])
        x_out = self.former_mobile(x_hidden, z_out)
        return x_out, z_out



class MobileFormer(nn.Module):
    '''Resolution should larger than [2 ** (num_stages + 1) + 7]'''
    '''stem -> bneck-lite -> stages(strided at first block) -> up-project-1x1 -> avg-pool -> fc1 -> scores-fc'''
    def __init__(self, expand_sizes, out_channels=None, 
                       num_token=6, d_model=192, in_channel=3, bneck_exp=32, bneck_out=16, 
                       stem_out_channel=16, project_demension=1152, fc_demension=None, num_class=None):
        super(MobileFormer, self).__init__()

        parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_class)
        self.in_channel = in_channel
        self.stem_out_channel = stem_out_channel
        self.num_token, self.d_model = num_token, d_model
        self.num_stages = len(expand_sizes)
        self.bneck_exp = bneck_exp
        self.bneck_out = bneck_out
        self.inter_channel = bneck_out
        self.expand_sizes = expand_sizes
        self.out_channels = out_channels
        self.project_demension, self.fc_demension, self.num_class= project_demension, fc_demension, num_class
        
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channel, self.stem_out_channel, kernel_size=3, stride=2, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.blocks = []
        for num_stage in range(self.num_stages):
            num_blocks = len(self.expand_sizes[num_stage])
            for num_block in range(num_blocks):
                if num_block == 0:
                    self.blocks.append(
                        MobileFormerBlock(self.inter_channel, self.expand_sizes[num_stage][num_block], self.out_channels[num_stage][num_block], self.d_model, stride=2).cuda()
                    )
                    self.inter_channel = self.out_channels[num_stage][num_block]
                else:
                    self.blocks.append(
                        MobileFormerBlock(self.inter_channel, self.expand_sizes[num_stage][num_block], self.out_channels[num_stage][num_block], self.d_model, stride=1).cuda()
                    )
                    self.inter_channel = self.out_channels[num_stage][num_block]

        self.project = nn.Conv2d(self.inter_channel, self.project_demension, kernel_size=1, stride=1).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1).cuda()
        self.fc = MLP([self.project_demension + self.d_model, self.fc_demension], bn=True)
        self.scores = nn.Linear(self.fc_demension, self.num_class).cuda()

    def forward(self, x):
        n, _, _, _ = x.shape
        x = self.stem(x)
        x = self.bneck(x)
        tokens = repeat(self.tokens, 'm d -> n m d', n=n)
        for block in self.blocks:
            x, tokens = block(x, tokens)
        x = self.project(x)
        x = self.avgpool(x).squeeze()
        x = torch.cat([x, tokens[:, 0, :]], dim=-1)
        x = self.fc(x)
        return self.scores(x)



class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes,  dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                  edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            Norm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x_high, x_low = x
        stride_ratio = x_low.shape[2] / x_high.shape[2]
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        fusion_edge_feat = high_edge_feat + low_edge_feat

        # residual part
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        x_indices = maxpool_indices % W_h * stride_ratio
        y_indices = maxpool_indices // W_h * stride_ratio
        low_indices = x_indices + y_indices * W
        low_indices = low_indices.long()
        x_high = x_high + maxpool_grid*x_high
        flattened_high = x_high.flatten(start_dim=2)
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        feat_n, feat_c, feat_h, feat_w = high_features.shape
        high_features = high_features.view(feat_n, -1, feat_h*feat_w)
        low_features = low_features.view(feat_n, -1, feat_h*feat_w)
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        affinity = self.softmax(affinity)  # b, n, n
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        fusion_feature = high_features + low_features
        mp_b, mp_c, mp_h, mp_w = low_indices.shape
        low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H*W).scatter(2, low_edge_indices, fusion_edge_feat)
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        return final_features, edge_pred


#####start training#####
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 4) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

#ffc=t   skip=t           

class YNet_general(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=True,
                 cat_merge=True):
        super(YNet_general, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)
#local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
#cross attention  in1 in2 out1 out2
#feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
#enc4_out = enc4_2 + enc4_out
#enc_f_out = enc4_2_f + enc4_f_out
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            #cross attention  in1 in2 out1 out2 
            
            
           # enc4_f_cat =self.catLayer((enc4_f[0], enc4_f[1]))
            #enc4_out, enc4_f_out = cross_attention(enc4, enc4_f_cat)
            #enc4_in  = torch.cat((enc4_out, enc4_f_out), dim=1)
          
            #local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            #cross attention  in1 in2 out1 out2
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
             #cross attention  in1 in2 out1 out2
            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            #cross attention  in1 in2 out1 out2
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            #cross attention  in1 in2 out1 out2
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            #cross attention  in1 in2 out1 out2
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



class YNet_advance1(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance1, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance1._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance1._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance1._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance1._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance1._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance1._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance1._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance1._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance1._block(features * 12, features * 24, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance1._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance1._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance1._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance1._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance1._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance1._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance1._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance1._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance1._block((features * 8) * 2, features * 12, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance1._block((features * 5) * 2, features * 6, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 6, features * 3, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance1._block((features * 5), features * 3, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 3, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance1._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #10, 384, 14, 14

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #[10, 768, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #10, 512, 28, 28
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class YNet_advance2(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2._block(features * 12, features * 24, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2._block((features * 8) * 2, features * 12, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2._block((features * 5) * 2, features * 6, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 6, features * 3, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2._block((features * 5), features * 3, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 3, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #10, 384, 14, 14

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #[10, 768, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #10, 512, 28, 28
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)

class  MGR_Graph(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Graph, self).__init__()
        #self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        #self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))
        #self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        #self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
      #  self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
       # self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)
        
        
    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)
        #self.x1 = self.conv1_1(x)
        #out = self.x1
       # self.x1 = self.conv1_1(x)
        #out = self.x1
        #return out
        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        #self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        #self.g3= self.glou3(self.x3)
        #self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)
        self.x3 = self.conv3_1(x)
        #out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)
        out =  self.x3
        return out


class  MGR_Graph2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Graph2, self).__init__()
        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))
        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
       # self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        #self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        #self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        #self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        #self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        #self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
       # self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
       # self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
       # self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

      #  self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        #self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
      #  self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
       # self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)
        
        


    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)
        #self.x1 = self.conv1_1(x)
        #out = self.x1
        self.x1 = self.conv1_1(x)
        out = self.x1
        return out
        #self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        #self.g1 = self.glou1(self.x1)
      #  self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        #self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
       # self.g2 = self.glou2(self.x2)
       # self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        #self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        #self.g3= self.glou3(self.x3)
        #self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)
        #self.x3 = self.conv3_1(x)
        #out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)
      #  out =  self.x3
     #   return out


class YNet_advance_gcn(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, gcn=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance_gcn, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.gcn = gcn
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance_gcn._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_gcn._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_gcn._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_gcn._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Graph Convolution
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features, out_channels=features * 2)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 4)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            
    
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_gcn._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_gcn._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_gcn._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_gcn._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance_gcn._block(features * 16, features * 28, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_gcn._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_gcn._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_gcn._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_gcn._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_gcn._block((features * 8) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_gcn._block((features * 6) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_gcn._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_gcn._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 28, features * 16, kernel_size=2, stride=2  # 28.16
            )
            self.decoder4 = YNet_advance_gcn._block((features * 10) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_gcn._block((features * 6) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_gcn._block((features * 3) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_gcn._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
        
        #Graph
        if self.gcn:
           
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn) # ([10, 128, 14, 14])
            
              
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            g = torch.zeros_like(enc4_gcn_2) #10, 128, 14, 14])
            
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc4_gcn_2 = torch.reshape(enc4_gcn_2, (torch.numel(enc4_gcn_2), 1))   
   
            
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_gcn_2), 1) #[250880, 4]
            bottleneck = bottleneck.view_as(torch.cat((a, b, t, g), 1))  #[10, 512, 14, 14]

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_gcn_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #[[10, 512, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #10, 512, 28, 28
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
class YNet_advance2_gcn(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, gcn=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_gcn, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.gcn = gcn
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_gcn._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_gcn._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_gcn._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_gcn._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
       
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features, out_channels=features * 2)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 4)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_gcn._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_gcn._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_gcn._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_gcn._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_gcn._block(features * 16, features * 28, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_gcn._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_gcn._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_gcn._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_gcn._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_gcn._block((features * 8) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_adYNet_advance2_gcnvance_gcn._block((features * 6) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_gcn._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_gcn._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 28, features * 16, kernel_size=2, stride=2  # 28.16
            )
            self.decoder4 = YNet_advance2_gcn._block((features * 10) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_gcn._block((features * 6) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_gcn._block((features * 3) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_gcn._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted


    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
        
        #Graph
        if self.gcn:
           
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn) # ([10, 128, 14, 14])
            
              
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            g = torch.zeros_like(enc4_gcn_2) #10, 128, 14, 14])
            
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc4_gcn_2 = torch.reshape(enc4_gcn_2, (torch.numel(enc4_gcn_2), 1))   
   
            
            #bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_gcn_2), 1) #[250880, 4]
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_attention_2), 1) #[250880, 4]
            bottleneck = bottleneck.view_as(torch.cat((a, b, t, t), 1))  #[10, 512, 14, 14]

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #[[10, 512, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #10, 512, 28, 28
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

#cat
class YNet_advance2_cat(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat._block(features * 8, features * 16, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            s = torch.add(a, t)
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1)  
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))   

        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) 

        bottleneck = self.bottleneck(bottleneck)  

        dec4 = self.upconv4(bottleneck)  
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #10, 512, 28, 28
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )      

#double channels
class YNet_advance_double(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance_double, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        #double all channels
        self.encoder1 = YNet_advance_double._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_double._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_double._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_double._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features* 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features* 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_double._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_double._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_double._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_double._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance_double._block(features * 24, features * 48, name="bottleneck")  # 24, 48

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 48, features * 24, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance_double._block((features * 16) * 2, features * 24, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double._block((features * 10) * 2, features * 12, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double._block((features * 5) * 2, features * 6, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 6, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10, 256, 14, 14]
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14
            t = torch.zeros_like(enc4_attention_2) #[10, 256, 14, 14])

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #[10, 768, 14, 14])

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #([10, 1536, 14, 14])

        dec4 = self.upconv4(bottleneck) #10, 768, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1) #enc4_f 10 128 28 28

            dec4 = torch.cat((dec4, enc4_in), dim=1) #[10, 768, 28, 28])
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 1024, 28, 28]) enc4:[10, 256, 28, 28])
            dec4 = self.decoder4(dec4) #[10, 768, 28, 28])
            dec3 = self.upconv3(dec4) #[10, 384, 56, 56])
            dec3 = torch.cat((dec3, enc3), dim=1) #([10, 640, 56, 56])
            dec3 = self.decoder3(dec3) #[10, 384, 56, 56])
            dec2 = self.upconv2(dec3) #10, 192, 112, 112])
            dec2 = torch.cat((dec2, enc2), dim=1) #[10, 320, 112, 112])
            dec2 = self.decoder2(dec2) #([10, 192, 112, 112])
            dec1 = self.upconv1(dec2)#[10, 64, 224, 224])
            dec1 = torch.cat((dec1, enc1), dim=1) #[10, 128, 224, 224])

        dec1 = self.decoder1(dec1) #[10, 64, 224, 224])

        return self.softmax(self.conv(dec1)) #([10, 9, 224, 224])

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




#cat the double 
class YNet_advance2_cat_double(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_double._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14])
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s = torch.add(a, t) #10, 256, 14, 14])
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #[10, 1024, 14, 14]
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
        
        
#branching 2
        
class YNet_advance2_branch(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_branch, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_branch._block(features * 8, features * 16, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            
            
        
    
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch._block((features * 4) * 2, features * 4, name="dec3")
             
            
            
             
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch._block((features * 2) * 2, features * 2, name="dec2")
       
            
            
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch._block(features * 4, features, name="dec1")  # 2,3

            
            
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()


    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            #s = torch.add(a, t)
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            #enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottle1 = torch.cat((enc4_2, enc4_f2), 1) 
            bottle2 = torch.cat((enc4_f2, enc4_attention_2), 1) 
    
          
            bottle1 = bottle1.view_as(torch.cat((a, b), 1))   
            bottle2 = bottle2.view_as(torch.cat((b, t), 1))
        else:
             
            bottleneck = torch.cat((enc4_2, enc4_f2), 1) #[10, 512, 14, 14])

        bottle1 = self.bottleneck(bottle1) 
        bottle2 = self.bottleneck(bottle2)

        dec4_1 = self.upconv4(bottle1)  
        dec4_2 = self.upconv4(bottle2)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4_1 = torch.cat((dec4_1, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4_2 = torch.cat((dec4_2, enc4), dim=1)
            
            dec4_1 = self.decoder4(dec4_1)
            dec4_2 = self.decoder4(dec4_2)
            
            
            dec3_1 = self.upconv3(dec4_1)
            dec3_2 = self.upconv3(dec4_2)
            
            
            dec3_1 = torch.cat((dec3_1, enc3), dim=1)
            dec3_2 = torch.cat((dec3_2, enc3), dim=1)
            
            
            dec3_1 = self.decoder3(dec3_1)
            dec3_2 = self.decoder3(dec3_2)
            
            
            dec2_1 = self.upconv2(dec3_1)
            dec2_2 = self.upconv2(dec3_2)
            
            
            dec2_1 = torch.cat((dec2_1, enc2), dim=1)
            dec2_2 = torch.cat((dec2_2, enc2), dim=1)
            
            
            dec2_1 = self.decoder2(dec2_1)
            dec2_2 = self.decoder2(dec2_2)
            
            
            dec1_1 = self.upconv1(dec2_1)
            dec1_2 = self.upconv1(dec2_2)
            
            
            dec1_1 = torch.cat((dec1_1, enc1), dim=1)
            dec1_2 = torch.cat((dec1_2, enc1), dim=1)
            dec_connected = torch.cat((dec1_1, dec1_2), dim=1)
            
            

        dec_connected = self.decoder1(dec_connected)
         
         
        return self.softmax(self.conv(dec_connected))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )   
        
        
        
        
#cnn+fcc 先contact encoder 然后获得decoder
  #torch.cat(enc1, enc1_f), dim=1)) conv block
  
class YNet_advance2_combine(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_combine, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_combine._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_combine._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_combine._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_combine._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features, expand_size=int(features * 1.0), out_channel= features * 2, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 4 , expand_size=int(features * 2.0), out_channel=features * 4, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_combine._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_combine._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_combine._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_combine._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_combine._block(features * 8, features * 16, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_combine._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_combine._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_combine._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_combine._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_combine._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_combine._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_combine._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_combine._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_combine._block(features * 2, features, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_combine._block(features, features, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 10, features * 5, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_combine._block(features, features, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 7, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_combine._block(features, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            s = torch.add(a, t)
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))  #10, 384, 14, 14

        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #[10, 512, 14, 14])

        bottleneck = self.bottleneck(bottleneck) #10, 512, 14, 14])

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            
            dec4a = self.decoder4(dec1) #10, 512, 28, 28

            dec3a = self.decoder3(dec4a)
 
            dec2a = self.decoder2(dec3a)
             

        dec1a = self.decoder1(dec2a)

        return self.softmax(self.conv(dec1a))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        ) 

#
# 
# 
# 
# 
# 

############        
#add graph first before attention and two y nets
class YNet_advance2_branch_graph(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, gcn=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_branch_graph, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.gcn = gcn
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch_graph._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch_graph._block(features  * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch_graph._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch_graph._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features * 2)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 8)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 8, out_channels=features * 8)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch_graph._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch_graph._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch_graph._block(features * 4, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch_graph._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.bottleneck = YNet_advance2_branch_graph._block(features * 32, features * 64, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph._block((features * 4) * 2, features * 5, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph._block((features * 5), features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph._block(features * 2, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 64, features * 32, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch_graph._block((features * 20) * 2, features * 32, name="dec4")  # 8, 12
            
            
        
    
            self.upconv3 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph._block((features * 12) * 2, features * 16, name="dec3")
             
            
            
             
            self.upconv2 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph._block((features * 6) * 2, features * 8, name="dec2")
       
            
            
            self.upconv1 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph._block((features * 6) * 2, features * 2, name="dec1")  # 2,3

            
            
        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()


    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
        if self.gcn:
           
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn)    
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 128, 14, 14
            t = torch.zeros_like(enc4_attention_2) #10, 128, 14, 14
            g = torch.zeros_like(enc4_gcn_2)
            #s = torch.add(a, t)
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc4_gcn_2 = torch.reshape(enc4_gcn_2, (torch.numel(enc4_gcn_2), 1))
            #enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottle1 = torch.cat((enc4_2,  enc4_f2, enc4_gcn_2, enc4_attention_2), 1) 
            bottle2 = torch.cat((enc4_2, enc4_f2,enc4_gcn_2, enc4_attention_2), 1) 
    
          
            bottle1 = bottle1.view_as(torch.cat((a, b, g, t), 1))   
            bottle2 = bottle2.view_as(torch.cat((a, b, g, t), 1))
        else:
             
            bottleneck = torch.cat((enc4_2, enc4_f2), 1) #[10, 512, 14, 14])

        bottle1 = self.bottleneck(bottle1) 
        bottle2 = self.bottleneck(bottle2)

        dec4_1 = self.upconv4(bottle1)  
        dec4_2 = self.upconv4(bottle2)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4_1 = torch.cat((dec4_1, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4_2 = torch.cat((dec4_2, enc4), dim=1)
            
            dec4_1 = self.decoder4(dec4_1)
            dec4_2 = self.decoder4(dec4_2)
            
            
            dec3_1 = self.upconv3(dec4_1)
            dec3_2 = self.upconv3(dec4_2)
            
            
            dec3_1 = torch.cat((dec3_1, enc3), dim=1)
            dec3_2 = torch.cat((dec3_2, enc3), dim=1)
            
            
            dec3_1 = self.decoder3(dec3_1)
            dec3_2 = self.decoder3(dec3_2)
            
            
            dec2_1 = self.upconv2(dec3_1)
            dec2_2 = self.upconv2(dec3_2)
            
            
            dec2_1 = torch.cat((dec2_1, enc2), dim=1)
            dec2_2 = torch.cat((dec2_2, enc2), dim=1)
            
            
            dec2_1 = self.decoder2(dec2_1)
            dec2_2 = self.decoder2(dec2_2)
            
            
            dec1_1 = self.upconv1(dec2_1)
            dec1_2 = self.upconv1(dec2_2)
            
            
            dec1_1 = torch.cat((dec1_1, enc1), dim=1)
            dec1_2 = torch.cat((dec1_2, enc1), dim=1)
            dec_connected = torch.cat((dec1_1, dec1_2), dim=1)
            
            

        dec_connected = self.decoder1(dec_connected)
         
         
        return self.softmax(self.conv(dec_connected))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )   
        
        
##########
##all true skip connection
class YNet_advance2_branch_graph_true(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=False, attention=True, skip_ffc=True,
                 cat_merge=True):
        super(YNet_advance2_branch_graph_true, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch_graph_true._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch_graph_true._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch_graph_true._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch_graph_true._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch_graph_true._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch_graph_true._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch_graph_true._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch_graph_true._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_branch_graph_true._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph_true._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_true._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_true._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_true._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph_true._block((features * 16) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_true._block((features * 12) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_true._block((features * 6) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_true._block((features * 3) * 2, features * 2, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch_graph_true._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_true._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_true._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_true._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14])
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s = torch.add(a, t) #10, 256, 14, 14])
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #[10, 1024, 14, 14]
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 512, 14, 14])

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



###########cross attention
# ########################cross attention#######################
class YNet_advance2_branch_cross(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, gcn=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_branch_cross, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.gcn = gcn
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.joint_cross_attn1 = BidirectionalCrossAttention(
            dim = 512,  
            heads = 8,
            dim_head = 64,
            context_dim = 386
        )
        self.joint_cross_attn2 = BidirectionalCrossAttention(
            dim = 512,
            heads = 8,
            dim_head = 64,
            context_dim = 386
        )
        self.joint_cross_attn3 = BidirectionalCrossAttention(
            dim = 512,
            heads = 8,
            dim_head = 64,
            context_dim = 386
        )
        self.joint_cross_attn4 = BidirectionalCrossAttention(
            dim = 512,
            heads = 8,
            dim_head = 64,
            context_dim = 386
        )
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch_cross._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch_cross._block(features  * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch_cross._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch_cross._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features * 2)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 8)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 8, out_channels=features * 8)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch_cross._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch_cross._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch_cross._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch_cross._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.bottleneck = YNet_advance2_branch_cross._block(features * 16, features * 32, name="bottleneck")  # 8, 24


        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_cross._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_cross._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_cross._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_cross._block((features * 4) * 2, features * 5, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_cross._block((features * 5), features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_cross._block(features * 2, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch_cross._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            
            
        
    
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_cross._block((features * 8) * 2, features * 8, name="dec3")
             
            
            
             
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_cross._block((features * 4) * 2, features * 4, name="dec2")
       
            
            
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_cross._block((features * 4) * 2, features * 2, name="dec1")  # 2,3

            
            
        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()


    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
        if self.gcn:
           
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn)    
        #cross attention

       #cnn_fcc
        enc1_out, enc1_f_out = self.joint_cross_attn1(enc1, enc1_f)
        enc2_out, enc2_f_out = self.joint_cross_attn2(enc2, enc2_f)
        enc3_out, enc3_f_out = self.joint_cross_attn3(enc3, enc3_f)
        enc4_out, enc4_f_out = self.joint_cross_attn4(enc4, enc4_f)
        
        #connect
        enc2_out_2 = self.encoder2(self.pool1(enc1_out))
        enc3_out_2 = self.encoder3(self.pool2(enc2_out))
        enc4_out_2 = self.encoder4(self.pool3(enc3_out))
        enc4_out_2_2 = self.pool4(enc4_out_2)
 
        enc2_f_out_2 = self.encoder2_f(self.pool1_f(enc1_f_out))
        enc3_f_out_2 = self.encoder3_f(self.pool2_f(enc2_f_out))
        enc4_f_out_2 = self.encoder4_f(self.pool3_f(enc3_f_out))
        enc4_f_out_2_2 = self.pool4(enc4_f_out_2)
 
 
 
       #fcc_Attention
        enc1_f_out2, enc1_attention_out = self.joint_cross_attn1(enc1_f, enc1_attention)
        enc2_f_out2, enc2_attention_out = self.joint_cross_attn2(enc2_f, enc2_attention)
        enc3_f_out2, enc3_attention_out = self.joint_cross_attn3(enc3_f, enc3_attention)
        enc4_f_out2, enc4_attention_out = self.joint_cross_attn4(enc4_f, enc4_attention)
        enc4_f_out2_2 = self.pool4(enc4_f_out2) 
        enc4_attention_out_2 = self.pool4_f(enc4_attention_out)
        
        #connect
        enc2_f_out_2a = self.encoder2_f(self.pool1_f(enc1_f_out2))
        enc3_f_out_2a = self.encoder3_f(self.pool2_f(enc2_f_out2))
        enc4_f_out_2a = self.encoder4_f(self.pool3_f(enc3_f_out2))
        enc4_f_out_2_2a = self.pool4_f(enc4_f_out2)
 
        attention_out_2 = self.encoder2_f(self.pool1_f(enc1_f_out))
        attention_out_2 = self.encoder3_f(self.pool2_f(enc2_f_out))
        attention_out_2 = self.encoder4_f(self.pool3_f(enc3_f_out))
        attention_out_2_2 = self.pool4(enc4_f_out_2)
        
   
        #attention_grph
        enc1_attention_out2, enc1_gcn_out = self.joint_cross_attn1(enc1_attention, enc1_gcn)
        enc2_attention_out2, enc2_gcn_out = self.joint_cross_attn2(enc2_attention, enc2_gcn)
        enc3_attention_out2, enc3_gcn_out = self.joint_cross_attn3(enc3_attention, enc3_gcn)
        enc4_attention_out2, enc4_gcn_out = self.joint_cross_attn4(enc4_attention, enc4_gcn)
        enc4_attention_out2_2 = self.pool4(enc4_attention_out2) 
        enc4_gcn_out_2 = self.pool4_f(enc4_gcn_out)
        
        
        
        
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_out) #10， 128, 14, 14
            b = torch.zeros_like(enc4_f_out) #10, 128, 14, 14
            t = torch.zeros_like(enc1_attention_out) #10, 128, 14, 14
            g = torch.zeros_like(enc1_gcn_out)
            #s = torch.add(a, t)
            enc4_2 = enc4_2.view(torch.numel(enc4_out), 1) #250880, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f_out), 1)  #250880, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
            enc4_gcn_2 = torch.reshape(enc4_gcn_2, (torch.numel(enc4_gcn_2), 1))
            #enc_sum = torch.add(enc4_2, enc4_attention_2)
        
            #enc4_1_cross, enc4_f1_cross = joint_cross_attn1(enc4_1,  enc4_f1) 
            #bottle1 = torch.cat((enc4_1_cross,  enc4_f1_cross, enc4_1, enc4_f1), 1) 
            #enc4_2_cross, enc4_f2_cross = joint_cross_attn2(enc4_2,  enc4_f2)
            bottle2 = torch.cat((enc4_f2, enc4_attention_2), 1) 
    
          
            bottle1 = bottle1.view_as(torch.cat((a,  b), 1))   
            bottle2 = bottle2.view_as(torch.cat((b, t), 1))
        else:
             
            bottleneck = torch.cat((enc4_2, enc4_f2), 1) #[10, 512, 14, 14])

        bottle1 = self.bottleneck(bottle1) 
        bottle2 = self.bottleneck(bottle2)

        dec4_1 = self.upconv4(bottle1)  
        dec4_2 = self.upconv4(bottle2)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4_1 = torch.cat((dec4_1, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4_2 = torch.cat((dec4_2, enc4), dim=1)
            
            dec4_1 = self.decoder4(dec4_1)
            dec4_2 = self.decoder4(dec4_2)
            
            
            dec3_1 = self.upconv3(dec4_1)
            dec3_2 = self.upconv3(dec4_2)
            
            
            dec3_1 = torch.cat((dec3_1, enc3), dim=1)
            dec3_2 = torch.cat((dec3_2, enc3), dim=1)
            
            
            dec3_1 = self.decoder3(dec3_1)
            dec3_2 = self.decoder3(dec3_2)
            
            
            dec2_1 = self.upconv2(dec3_1)
            dec2_2 = self.upconv2(dec3_2)
            
            
            dec2_1 = torch.cat((dec2_1, enc2), dim=1)
            dec2_2 = torch.cat((dec2_2, enc2), dim=1)
            
            
            dec2_1 = self.decoder2(dec2_1)
            dec2_2 = self.decoder2(dec2_2)
            
            
            dec1_1 = self.upconv1(dec2_1)
            dec1_2 = self.upconv1(dec2_2)
            
            
            dec1_1 = torch.cat((dec1_1, enc1), dim=1)
            dec1_2 = torch.cat((dec1_2, enc1), dim=1)
            dec_connected = torch.cat((dec1_1, dec1_2), dim=1)
            
            

        dec_connected = self.decoder1(dec_connected)
         
         
        return self.softmax(self.conv(dec_connected))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )   




###########cross attention
##########cat double attention cross attention:
class YNet_advance2_cat_double_cs(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double_cs, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double_cs._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double_cs._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double_cs._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double_cs._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_double._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14])
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s = torch.add(a, t) #10, 256, 14, 14])
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #[10, 1024, 14, 14]
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
        
###########cross attention
######baseline
###########cross attention
######baseline
class YNet_general_cs(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_general_cs, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 128,  
                heads = 8,
                dim_head = 64,
                context_dim = 128
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general_cs._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general_cs._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general_cs._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general_cs._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general_cs._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general_cs._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general_cs._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general_cs._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general_cs._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f)) #[10, 128, 14, 14]
            enc4_f2 = self.pool4(enc4_f)
        #video = torch.randn(1, 4096, 512)
        #audio = torch.randn(1, 8192, 386) 
        #enc4_2 = rearrange(enc4_2, 'n c h w -> n ( h w ) c')
        #enc4_f2 = rearrange(enc4_f2, 'n c h w -> n ( h w ) c')
        #enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
        #enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
       # print(enc4_2.shape, enc4_f2.shape)
        #enc4_2 = torch.reshape(enc4_2, (torch.numel(enc4_2), 1))
        #enc4_f2 = torch.reshape(enc4_f2, (torch.numel(enc4_f2), 1))
        
    
        #enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
       # h = 14
        #enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
        #enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
        #enc4_2_out = enc4_2 + enc4_2_out
       # enc4_f2_out = enc4_f2 + enc4_f2_out
        #concat  + 调参数
        ###
        ###
        ##
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
             
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            h = 14
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)
            #enc4_new = torch.cat((enc4_2, enc4_2_out), 1)
            #enc4_f_new = torch.cat((enc4_f2, enc4_f2_out), 1)
#local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
#cross attention  in1 in2 out1 out2
#feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
#enc4_out = enc4_2 + enc4_out
#enc_f_out = enc4_2_f + enc4_f_out  

            bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))
        else:
            bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            #cross attention  in1 in2 out1 out2 
            
            
           # enc4_f_cat =self.catLayer((enc4_f[0], enc4_f[1]))
            #enc4_out, enc4_f_out = cross_attention(enc4, enc4_f_cat)
            #enc4_in  = torch.cat((enc4_out, enc4_f_out), dim=1)
          
            #local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            #cross attention  in1 in2 out1 out2
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
             #cross attention  in1 in2 out1 out2
            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            #cross attention  in1 in2 out1 out2
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            #cross attention  in1 in2 out1 out2
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            #cross attention  in1 in2 out1 out2
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



###concact cs
class YNet_general_cs_cat(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_general_cs_cat, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 128,  
                heads = 8,
                dim_head = 64,
                context_dim = 128
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general_cs_cat._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general_cs_cat._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general_cs_cat._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general_cs_cat._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general_cs_cat._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general_cs_cat._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general_cs_cat._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general_cs_cat._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general_cs_cat._block(features * 16, features * 32, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs_cat._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs_cat._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs_cat._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs_cat._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general_cs_cat._block((features * 10) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general_cs_cat._block((features * 6) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general_cs_cat._block((features * 3) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general_cs_cat._block(features * 3, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f)) #[10, 128, 14, 14]
            enc4_f2 = self.pool4(enc4_f)
        #video = torch.randn(1, 4096, 512)
        #audio = torch.randn(1, 8192, 386) 
        #enc4_2 = rearrange(enc4_2, 'n c h w -> n ( h w ) c')
        #enc4_f2 = rearrange(enc4_f2, 'n c h w -> n ( h w ) c')
        #enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
        #enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
       # print(enc4_2.shape, enc4_f2.shape)
        #enc4_2 = torch.reshape(enc4_2, (torch.numel(enc4_2), 1))
        #enc4_f2 = torch.reshape(enc4_f2, (torch.numel(enc4_f2), 1))
        
    
        #enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
       # h = 14
        #enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
        #enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
        #enc4_2_out = enc4_2 + enc4_2_out
       # enc4_f2_out = enc4_f2 + enc4_f2_out
        #concat  + 调参数
        ###
        ###
        ##
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128]
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128])
             
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # ((10, 196, 128]]) (10, 196, 128]))
            h = 14
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14])
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
            enc4_2_out = torch.cat((enc4_2, enc4_2_out), 1) #(10, 128, 14, 14]) [10, 128, 14, 14])
            enc4_f2_out = torch.cat((enc4_f2 , enc4_f2_out), 1) #([10, 128, 14, 14])  10, 128, 14, 14])
            a = torch.zeros_like(enc4_2_out)
            b = torch.zeros_like(enc4_f2_out)

           # enc4_2 = enc4_2.view(torch.numel(enc4_2_out), 1) #[250880, 1])
           # enc4_f2 = enc4_f2.view(torch.numel(enc4_f2_out), 1) #e([250880, 1])
            #enc4_new = torch.cat((enc4_2, enc4_2_out), 1)
            #enc4_f_new = torch.cat((enc4_f2, enc4_f2_out), 1)
#local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
#cross attention  in1 in2 out1 out2
#feature = rearrange(v_agg, 'n ( h w ) c -> n c h w', h=h)
#enc4_out = enc4_2 + enc4_out
#enc_f_out = enc4_2_f + enc4_f_out  

            bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1) #10, 256, 14, 14])
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1)) #ze([10, 256, 14, 14])
        else:
            bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)

        bottleneck = self.bottleneck(bottleneck) #e([10, 512, 14, 14])

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            #cross attention  in1 in2 out1 out2 
            
            
           # enc4_f_cat =self.catLayer((enc4_f[0], enc4_f[1]))
            #enc4_out, enc4_f_out = cross_attention(enc4, enc4_f_cat)
            #enc4_in  = torch.cat((enc4_out, enc4_f_out), dim=1)
          
            #local_feature = rearrange(local_feature, 'n c h w -> n ( h w ) c')
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            #cross attention  in1 in2 out1 out2
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
             #cross attention  in1 in2 out1 out2
            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            #cross attention  in1 in2 out1 out2
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            #cross attention  in1 in2 out1 out2
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            #cross attention  in1 in2 out1 out2
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




class YNet_advance2_cat_double_cross(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double_cross, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 256,  
                heads = 8,
                dim_head = 64,
                context_dim = 256
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double_cross._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double_cross._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double_cross._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double_cross._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double_cross._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double_cross._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double_cross._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double_cross._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_double_cross._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double_cross._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
           
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            enc4_f2_out_2, enc_attention_out = self.joint_cross_attn1(enc4_f2_token, enc4_attention_2_token) # (10, 196, 128) (10, 196, 128)
            
            h = 14
            
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
           
            #enc4_2_out = torch.cat((enc4_2, enc4_2_out), 1)
           # enc4_f2_out = torch.cat((enc4_f2 , enc4_f2_out), 1)
            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
            enc4_f2_out_2 = enc4_f2 + enc4_f2_out_2
            enc_attention_out = enc4_attention_2 + enc_attention_out
            
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14])
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s = torch.add(a, t) #10, 256, 14, 14])
            #enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            #enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
           # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            #enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            enc_sum_out = torch.add(enc4_2_out, enc_attention_out) #501760, 1])
            enc_f2_sum = torch.add(enc4_f2_out, enc4_f2_out_2) 
            bottleneck = torch.cat((enc_sum_out, enc_f2_sum), 1)  
            # bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
#######################################################################################################################        
class YNet_advance2_cat_double_cscat(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, gcn=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double_cscat, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.gcn = gcn
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 256,  
                heads = 8,
                dim_head = 64,
                context_dim = 256
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double_cscat._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double_cscat._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double_cscat._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double_cscat._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
    #Graph Convolution
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features * 2)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 8)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 8, out_channels=features * 8)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double_cscat._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double_cscat._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double_cscat._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double_cscat._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        #self.bottleneck = YNet_advance2_cat_double_cscat._block(features * 32, features *  64, name="bottleneck")  # 8, 24
        self.bottleneck = YNet_advance2_cat_double_cscat._block(features * 16, features *  32, name="bottleneck")
        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cscat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cscat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cscat._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cscat._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cscat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cscat._block((features * 8) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cscat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cscat._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double_cscat._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cscat._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cscat._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cscat._block((features * 3), features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted
#y_net_gen_cat_cs_channel
    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #e([10, 256, 14, 14])
          
     #Graph
        if self.gcn:
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn) # ([10, 128, 14, 14])
              
        #catmerge
            
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c')  #[10, 196, 256])
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c')  #10, 196, 256
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c')   #([10, 196, 256])
            enc4_gcn_2_token = rearrange(enc4_gcn_2, 'n c h w -> n ( h w ) c')   #10, 196, 128]
            
    # cnn+fcc
    #cc n+ att
    #cnn+graph
    
    
    #cnn+fcc  fcc+ att  att+graph
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token)  
    # 
            enc4_f2_out_2, enc_attention_out = self.joint_cross_attn1(enc4_f2_token, enc4_attention_2_token)             
    #      
            enc_attention_out_2, enc4_gcn_2_out = self.joint_cross_attn1(enc4_attention_2_token, enc4_gcn_2_token)  
            
            
            h = 14
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h)  
        # 
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
        #a 
            enc_attention_out_2 = rearrange(enc_attention_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc4_gcn_2_out = rearrange(enc4_gcn_2_out, 'n ( h w ) c -> n c h w', h=h)
                     #enc4_2_out = torch.cat((enc4_2, enc4_2_out), 1)
                     # enc4_f2_out = torch.cat((enc4_f2 , enc4_f2_out), 1)
            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
  # 
            enc4_f2_out_2 = enc4_f2 + enc4_f2_out_2
            enc_attention_out = enc4_attention_2 + enc_attention_out    
  # 
            enc_attention_out_2 = enc4_attention_2 + enc_attention_out_2
            enc4_gcn_2_out = enc4_gcn_2 + enc4_gcn_2_out
            
            
            #enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #250880, 1
            #enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #250880, 1
            #enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #250880, 1
           # enc_sum = torch.add(enc4_2, enc4_attention_2)
             
            a = torch.zeros_like(enc4_2_out)  
            b = torch.zeros_like(enc4_f2_out) 
            t = torch.zeros_like(enc_attention_out)  
            g = torch.zeros_like(enc4_gcn_2_out) 
            x = torch.zeros_like(enc4_f2_out_2)
            y = torch.zeros_like(enc_attention_out_2)
            z = torch.zeros_like(enc4_gcn_2_out)
                         #enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
                         #enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
                         # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
                         #enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
                         #enc_sum_out = torch.add(enc4_2_out, enc_attention_out) #501760, 1])
           # enc_cat1 = torch.add(enc4_2_out * 0.4, enc4_2_out_2 * 0.3)
           # enc_cat = torch.add(enc_cat1, enc4_ 2_out_2_2 * 0.3)  
            cat_sum = torch.add(enc4_2_out,  enc_attention_out_2)
            cat_sum1 = torch.add(cat_sum, enc4_gcn_2_out)
            s = torch.add(a, x)
            s1 = torch.add(s, z)
            bottleneck = torch.cat((cat_sum1, enc4_f2_out), 1)  
                        # bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)
            bottleneck = bottleneck.view_as(torch.cat((s, b), 1))   
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck)  

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4_2 = enc4_attention_2
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
###add attention 
class YNet_advance2_cat_double_cross_cat(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double_cross_cat, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 512,  
                heads = 8,
                dim_head = 64,
                context_dim = 512
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double_cross_cat._block(in_channels, features * 4, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double_cross_cat._block(features * 4, features * 8, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double_cross_cat._block(features * 8, features * 16, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double_cross_cat._block(features * 16, features * 16, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 4, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 8, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 16, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 4, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 1.0), out_channel= features * 8, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 16 , expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double_cross_cat._block(in_channels, features * 4, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double_cross_cat._block(features * 4, features * 8, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double_cross_cat._block(features * 8, features * 16, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double_cross_cat._block(features * 16, features * 16, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_double_cross_cat._block(features * 32, features * 64, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 64, features * 32, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 24) * 2, features * 32, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 16) * 2, features * 16, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block((features * 4) * 2, features * 4, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 4, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
           
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            enc4_f2_out_2, enc_attention_out = self.joint_cross_attn1(enc4_f2_token, enc4_attention_2_token) # (10, 196, 128) (10, 196, 128)
            
            h = 14
            
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
           
            #enc4_2_out = torch.cat((enc4_2, enc4_2_out), 1)
           # enc4_f2_out = torch.cat((enc4_f2 , enc4_f2_out), 1)
            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
            enc4_f2_out_2 = enc4_f2 + enc4_f2_out_2
            enc_attention_out = enc4_attention_2 + enc_attention_out
            
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2)  
            b2 = torch.zeros_like(enc4_f2_out_2)
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s1 = torch.add(a, t) 
            #s2 = torch.add(b , b2)
            #enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            #enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
           # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            #enc_sum_out = torch.add(enc4_2_out, enc_attention_out) #501760, 1])
            #enc_f2_sum = torch.add(enc4_f2_out, enc4_f2_out_2) 
            bottleneck = torch.cat((enc_sum, enc4_f2_out_2), 1)  
            # bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)
            bottleneck = bottleneck.view_as(torch.cat((s1, b2), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        


#y_net_two_cs
class YNet_advance2_branch_graph_cs(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True,  skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_branch_graph_cs, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc

        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 256,  
                heads = 8,
                dim_head = 64,
                context_dim = 256
            )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch_graph_cs._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch_graph_cs._block(features  * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch_graph_cs._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch_graph_cs._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch_graph_cs._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch_graph_cs._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch_graph_cs._block(features * 4, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch_graph_cs._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.bottleneck = YNet_advance2_branch_graph_cs._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph_cs._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_cs._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_cs._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_cs._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_graph_cs._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_cs._block((features * 4) * 2, features * 5, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_cs._block((features * 5), features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_cs._block(features * 2, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch_graph_cs._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            
            
        
    
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_graph_cs._block((features * 8) * 2, features * 8, name="dec3")
             
            
            
             
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_graph_cs._block((features * 4) * 2, features * 4, name="dec2")
       
            
            
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_graph_cs._block((features * 3) * 2, features * 2, name="dec1")  # 2,3

            
            
        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()


    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14

        #catmerge
        if self.cat_merge:

            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c')  #[10, 196, 256])
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c')  #10, 196, 256
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c')   #([10, 196, 256])
 
            

            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token)  
    # 
            enc4_f2_out_2, enc_attention_out = self.joint_cross_attn1(enc4_f2_token, enc4_attention_2_token)           

            h = 14
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h)  
        # 
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)

            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
  # 
            enc4_f2_out_2 = enc4_f2 + enc4_f2_out_2
            enc_attention_out = enc4_attention_2 + enc_attention_out    

            a = torch.zeros_like(enc4_2_out)  
            b = torch.zeros_like(enc4_f2_out) 
            t = torch.zeros_like(enc_attention_out)  
            x = torch.zeros_like(enc4_f2_out_2)


            cat_sum1 = torch.add(enc4_2_out,  enc_attention_out)
            cat_sum2 = torch.add(enc4_f2_out, enc4_f2_out_2)
            s1 = torch.add(a, t)
            s2 = torch.add(b, x)
        
            bottle1 = torch.cat((cat_sum1, cat_sum2), 1) 
            bottle2 = torch.cat((cat_sum1, cat_sum2), 1) 
    
          
            bottle1 = bottle1.view_as(torch.cat((s1, s2), 1))   
            bottle2 = bottle2.view_as(torch.cat((s1, s2), 1))    

        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) 

        bottle1 = self.bottleneck(bottle1) 
        bottle2 = self.bottleneck(bottle2)

        dec4_1 = self.upconv4(bottle1)  
        dec4_2 = self.upconv4(bottle2)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4_1 = torch.cat((dec4_1, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4_2 = torch.cat((dec4_2, enc4), dim=1)
            
            dec4_1 = self.decoder4(dec4_1)
            dec4_2 = self.decoder4(dec4_2)
            
            
            dec3_1 = self.upconv3(dec4_1)
            dec3_2 = self.upconv3(dec4_2)
            
            
            dec3_1 = torch.cat((dec3_1, enc3), dim=1)
            dec3_2 = torch.cat((dec3_2, enc3), dim=1)
            
            
            dec3_1 = self.decoder3(dec3_1)
            dec3_2 = self.decoder3(dec3_2)
            
            
            dec2_1 = self.upconv2(dec3_1)
            dec2_2 = self.upconv2(dec3_2)
            
            
            dec2_1 = torch.cat((dec2_1, enc2), dim=1)
            dec2_2 = torch.cat((dec2_2, enc2), dim=1)
            
            
            dec2_1 = self.decoder2(dec2_1)
            dec2_2 = self.decoder2(dec2_2)
            
            
            dec1_1 = self.upconv1(dec2_1)
            dec1_2 = self.upconv1(dec2_2)
            
            
            dec1_1 = torch.cat((dec1_1, enc1), dim=1)
            dec1_2 = torch.cat((dec1_2, enc1), dim=1)
            dec_connected = torch.cat((dec1_1, dec1_2), dim=1)
            
            

        dec_connected = self.decoder1(dec_connected)
         
         
        return self.softmax(self.conv(dec_connected))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )   
        
            
###add attention 
class YNet_advance2_cat_double_cross_cat(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_double_cross_cat, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        self.joint_cross_attn1 = BidirectionalCrossAttention(
                dim = 512,  
                heads = 8,
                dim_head = 64,
                context_dim = 512
            )
 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_double_cross_cat._block(in_channels, features * 4, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_double_cross_cat._block(features * 4, features * 8, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_double_cross_cat._block(features * 8, features * 16, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_double_cross_cat._block(features * 16, features * 16, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 4, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 8, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 16, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 4, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 1.0), out_channel= features * 8, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 16 , expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_double_cross_cat._block(in_channels, features * 4, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_double_cross_cat._block(features * 4, features * 8, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_double_cross_cat._block(features * 8, features * 16, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_double_cross_cat._block(features * 16, features * 16, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_double_cross_cat._block(features * 32, features * 64, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 64, features * 32, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_cat_double_cross_cat._block((features * 24) * 2, features * 32, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_double_cross_cat._block((features * 16) * 2, features * 16, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_double_cross_cat._block((features * 8) * 2, features * 8, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_double_cross_cat._block((features * 4) * 2, features * 4, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 4, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c') # (10, 196, 128)
           
            enc4_2_out, enc4_f2_out = self.joint_cross_attn1(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            enc4_f2_out_2, enc_attention_out = self.joint_cross_attn1(enc4_f2_token, enc4_attention_2_token) # (10, 196, 128) (10, 196, 128)
            
            h = 14
            
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[10, 128, 14, 14] [10, 128, 14, 14])
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
           
            #enc4_2_out = torch.cat((enc4_2, enc4_2_out), 1)
           # enc4_f2_out = torch.cat((enc4_f2 , enc4_f2_out), 1)
            enc4_2_out = enc4_2 + enc4_2_out
            enc4_f2_out = enc4_f2 + enc4_f2_out
            enc4_f2_out_2 = enc4_f2 + enc4_f2_out_2
            enc_attention_out = enc4_attention_2 + enc_attention_out
            
            a = torch.zeros_like(enc4_2) #[10, 256, 14, 14
            b = torch.zeros_like(enc4_f2)  
            b2 = torch.zeros_like(enc4_f2_out_2)
            t = torch.zeros_like(enc4_attention_2) #10, 256, 14, 14])
            s1 = torch.add(a, t) 
            #s2 = torch.add(b , b2)
            #enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1]
            #enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1])
           # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1])
            enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
            #enc_sum_out = torch.add(enc4_2_out, enc_attention_out) #501760, 1])
            #enc_f2_sum = torch.add(enc4_f2_out, enc4_f2_out_2) 
            bottleneck = torch.cat((enc_sum, enc4_f2_out_2), 1)  
            # bottleneck = torch.cat((enc4_2_out, enc4_f2_out), 1)
            bottleneck = bottleneck.view_as(torch.cat((s1, b2), 1))  #[10, 1024, 14, 14]
        else:
            enc_sum = torch.add(enc4_2, enc4_attention_2)
            bottleneck = torch.cat((enc_sum, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #[10, 384, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) #110, 768, 28, 28]
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
 
 
        
#UNET VIT
########

class UNETR2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int],
        feature_size: int = 32,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=2
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])
    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in) #10, 768, 14, 14]
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
    


class YNet_advance_double_unetr(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance_double_unetr, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.hidden_size = 768
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip((224, 224), (14,14)))
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.vit = ViT(
            in_channels=in_channels,
            img_size=(224,224),
            patch_size=(14,14),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            pos_embed="perceptron",
            classification=False,
            dropout_rate=0.0,
            spatial_dims=2
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        #double all channels
        self.encoder1 = YNet_advance_double_unetr._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_double_unetr._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_double_unetr._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_double_unetr._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features* 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features* 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = UNETR2D(in_channels=1, out_channels=9, img_size=(224,224))
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = UNETR2D(in_channels=1, out_channels=9, img_size=(224,224))
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = UNETR2D(in_channels=1, out_channels=9, img_size=(224,224))
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = UNETR2D(in_channels=1, out_channels=9, img_size=(224,224))
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_double_unetr._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_double_unetr._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_double_unetr._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_double_unetr._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance_double_unetr._block(features * 24, features * 48, name="bottleneck")  # 24, 48

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_unetr._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_unetr._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_unetr._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_unetr._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_unetr._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_unetr._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_unetr._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_unetr._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 48, features * 24, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance_double_unetr._block((features * 16) * 2, features * 24, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_unetr._block((features * 10) * 2, features * 12, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_unetr._block((features * 5) * 2, features * 6, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 6, features * 2, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_unetr._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()
        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            xtoken, hidden_states_out = self.vit(x.reshape(-1, 1, 224, 224))
             
            enc1_attention = self.encoder1_attention(x) #10, 32, 224, 224
            x2 = hidden_states_out[3]
            enc2_attention = self.encoder2_attention(self.proj_feat(x2, self.hidden_size, self.feat_size)) 
            x3 = hidden_states_out[6]
            enc3_attention = self.encoder3_attention(self.proj_feat(x3, self.hidden_size, self.feat_size))  
            x4 = hidden_states_out[9]
            enc4_attention = self.encoder4_attention(self.proj_feat(x4, self.hidden_size, self.feat_size)) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10, 256, 14, 14]
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14
            t = torch.zeros_like(enc4_attention_2) #[10, 256, 14, 14])

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #[10, 768, 14, 14])

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #([10, 1536, 14, 14])

        dec4 = self.upconv4(bottleneck) #10, 768, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1) #enc4_f 10 128 28 28
            
            
            
            
            dec4 = torch.cat((dec4, enc4_in), dim=1) #[10, 768, 28, 28])
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, efnc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1) #dec4:[10, 1024, 28, 28]) enc4:[10, 256, 28, 28])
            dec4 = self.decoder4(dec4) #[10, 768, 28, 28])
            dec3 = self.upconv3(dec4) #[10, 384, 56, 56])
            dec3 = torch.cat((dec3, enc3), dim=1) #([10, 640, 56, 56])
            dec3 = self.decoder3(dec3) #[10, 384, 56, 56])
            dec2 = self.upconv2(dec3) #10, 192, 112, 112])
            dec2 = torch.cat((dec2, enc2), dim=1) #[10, 320, 112, 112])
            dec2 = self.decoder2(dec2) #([10, 192, 112, 112])
            dec1 = self.upconv1(dec2)#[10, 64, 224, 224])
            dec1 = torch.cat((dec1, enc1), dim=1) #[10, 128, 224, 224])

        dec1 = self.decoder1(dec1) #[10, 64, 224, 224])

        return self.softmax(self.conv(dec1)) #([10, 9, 224, 224])

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


#att layer att layer four times
#add attention layers
class YNet_advance_double_add_att_four(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True, channels=[256, 512, 1024, 2048]):
        super(YNet_advance_double_add_att,self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192

        
        ####old
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        #double all channels
        self.encoder1 = YNet_advance_double_add_att._block(in_channels, features * 4, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_double_add_att._block(features * 4, features * 8, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_double_add_att._block(features * 8, features * 16, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_double_add_att._block(features * 16, features * 16, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features* 4, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features* 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 8, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 16, features * 16, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 4, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 1.0), out_channel= features * 8, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 16, expand_size=int(features * 2.0), out_channel=features * 16, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_double_add_att._block(in_channels, features * 4, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_double_add_att._block(features * 4, features * 8, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_double_add_att._block(features * 8, features * 16, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_double_add_att._block(features * 16, features * 16, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_advance_double_add_att._block(features * 48, features * 96, name="bottleneck")  # 24, 48

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 96, features * 48, kernel_size=2, stride=2  # 24,12
            )
            self.Att4 = Attention_block(features * 48, features * 16, features * 48 // 2)           
        
                  
                  
                  
                  
            self.decoder4 = YNet_advance_double_add_att._block((features * 32) * 2, features * 48, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 48, features * 24, kernel_size=2, stride=2
            )
            
            
            
            
            
            self.Att3 = Attention_block(features * 24, features * 16, features * 24 // 2)

            self.decoder3 = YNet_advance_double_add_att._block((features * 20) * 2, features * 24, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2
            )
            
            
            
            
            self.Att2 = Attention_block(features * 12, features * 8, features * 12 // 2)
     
            self.decoder2 = YNet_advance_double_add_att._block((features * 10) * 2, features * 12, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            
            
            
            
            
            
            self.Att1 = Attention_block(features * 6, features * 4, features * 6 // 2)
            self.decoder1 = YNet_advance_double_add_att._block((features * 5) * 2, features * 4, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 4, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224])
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112])

        enc3 = self.encoder3(self.pool2(enc2)) #[[10, 256, 56, 56])
        enc4 = self.encoder4(self.pool3(enc3))  #[[10, 256, 28, 28])
        enc4_2 = self.pool4(enc4) #[10, 256, 14, 14])]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10, 256, 14, 14]
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14
            t = torch.zeros_like(enc4_attention_2) #[10, 256, 14, 14])

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #[10, 768, 14, 14])

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #([10, 1536, 14, 14])
###***
        dec4 = self.upconv4(bottleneck) #10, 768, 28, 28]

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1) #enc4_f 10 128 28 28

            dec4 = torch.cat((dec4, enc4_in), dim=1) #[10, 768, 28, 28])
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            enc4 = self.Att4(dec4,enc4)
            dec4 = torch.cat((enc4, dec4), dim=1) #
            dec4 = self.decoder4(dec4) # 
            dec3 = self.upconv3(dec4) #[ 
            enc3 = self.Att3(dec3,enc3)
            dec3 = torch.cat((enc3, dec3), dim=1) #( 
            dec3 = self.decoder3(dec3) #[ 
            dec2 = self.upconv2(dec3) #1 
            enc2 = self.Att2(dec2,enc2)
            dec2 = torch.cat((enc2, dec2), dim=1) # 
            dec2 = self.decoder2(dec2) # 
            dec1 = self.upconv1(dec2)#[ 
            enc1 = self.Att1(dec1,enc1)
            dec1 = torch.cat((enc1, dec1), dim=1) # 

        dec1 = self.decoder1(dec1) # 

        return self.softmax(self.conv(dec1)) # 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


#add attention layers with gcn
class YNet_advance_double_add_att_gcn(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True, gcn=True):
        super(YNet_advance_double_add_att,self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.gcn = gcn
        
        ####old
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        
        features = init_features
        ############### Regular ##################################
        #double all channels
        self.encoder1 = YNet_advance_double_add_att._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_double_add_att._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_double_add_att._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_double_add_att._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features* 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features* 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)
        if gcn:
            self.encoder1_gcn =  MGR_Graph(in_channels=in_channels * 16, out_channels=features * 2)
            self.pool1_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_gcn =  MGR_Graph(in_channels=features * 2, out_channels=features * 4)
            self.pool2_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_gcn =  MGR_Graph(in_channels=features * 4, out_channels=features * 8)
            self.pool3_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_gcn =  MGR_Graph(in_channels=features * 8, out_channels=features * 8)
            self.pool4_gcn = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_double_add_att._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_double_add_att._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_double_add_att._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_double_add_att._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_advance_double_add_att._block(features * 32, features * 64, name="bottleneck")  # 24, 48

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 64, features * 32, kernel_size=2, stride=2  # 24,12
            )
            self.Att4 = Attention_block(features * 32, features * 8, features * 32 // 2)           
        
                  
            self.decoder4 = YNet_advance_double_add_att._block((features * 20) * 2, features * 32, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 32, features * 16, kernel_size=2, stride=2
            )
            
            self.Att3 = Attention_block(features * 16, features * 8, features * 16 // 2)

            self.decoder3 = YNet_advance_double_add_att._block((features * 12) * 2, features * 16, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            
            self.Att2 = Attention_block(features * 8, features * 4, features * 8 // 2)
     
            self.decoder2 = YNet_advance_double_add_att._block((features * 6) * 2, features * 8, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            
            
            self.Att1 = Attention_block(features * 4, features * 2, features * 4 // 2)
            self.decoder1 = YNet_advance_double_add_att._block((features * 3) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224])
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112])

        enc3 = self.encoder3(self.pool2(enc2)) #[[10, 256, 56, 56])
        enc4 = self.encoder4(self.pool3(enc3))  #[[10, 256, 28, 28])
        enc4_2 = self.pool4(enc4) #[10, 256, 14, 14])]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
          
        #Graph
        if self.gcn:
           
            batch = x.shape[0] #[10, 16, 224, 224])
            enc1_gcn = self.encoder1_gcn(x)  # [10, 32, 224, 224])
            enc2_gcn = self.encoder2_gcn(self.pool1_gcn(enc1_gcn)) # [10, 64, 112, 112])
            enc3_gcn = self.encoder3_gcn(self.pool2_gcn(enc2_gcn)) # [10, 128, 56, 56])
            enc4_gcn = self.encoder4_gcn(self.pool3_gcn(enc3_gcn))  # ([10, 128, 28, 28])
            enc4_gcn_2 = self.pool4_gcn(enc4_gcn) # ([10, 128, 14, 14])        
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10, 256, 14, 14]
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14
            t = torch.zeros_like(enc4_attention_2) #[10, 256, 14, 14])
            g = torch.zeros_like(enc4_gcn_2) #10, 128, 14, 14])
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1
            enc4_gcn_2 = torch.reshape(enc4_gcn_2, (torch.numel(enc4_gcn_2), 1))
            
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_gcn_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t, g), 1))  #[10, 768, 14, 14])

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2, enc4_gcn_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #([10, 1536, 14, 14])

        dec4 = self.upconv4(bottleneck) #10, 768, 28, 28]

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1) #enc4_f 10 128 28 28

            dec4 = torch.cat((dec4, enc4_in), dim=1) #[10, 768, 28, 28])
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            #enc4 = self.Att4(dec4,enc4)
        # cross attention birection (dec4, enc4)
        #+gan (生成) 
        #few shot 或者gan 学习
            dec4 = torch.cat((enc4, dec4), dim=1) #
            dec4 = self.decoder4(dec4) # 
            dec3 = self.upconv3(dec4) #[ 
            enc3 = self.Att3(dec3,enc3)
            dec3 = torch.cat((enc3, dec3), dim=1) #( 
            dec3 = self.decoder3(dec3) #[ 
            dec2 = self.upconv2(dec3) #1 
            enc2 = self.Att2(dec2,enc2)
            dec2 = torch.cat((enc2, dec2), dim=1) # 
            dec2 = self.decoder2(dec2) # 
            dec1 = self.upconv1(dec2)#[ 
            enc1 = self.Att1(dec1,enc1)
            dec1 = torch.cat((enc1, dec1), dim=1) # 

        dec1 = self.decoder1(dec1) # 

        return self.softmax(self.conv(dec1)) # 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )





class YNet_advance_double_add_att_cross(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance_double_add_att_cross,self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.joint_cross_attn4 = BidirectionalCrossAttention(
            dim = 768,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )   
        self.joint_cross_attn3 = BidirectionalCrossAttention(
            dim = 384,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )  
        self.joint_cross_attn2 = BidirectionalCrossAttention(
            dim = 192,  
            heads = 8,
            dim_head = 64,
            context_dim = 128
        )  
        self.joint_cross_attn1 = BidirectionalCrossAttention(
            dim = 96,  
            heads = 8,
            dim_head = 64,
            context_dim = 64
        )      
        ####old
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        
        features = init_features
        ############### Regular ##################################
        #double all channels
        self.encoder1 = YNet_advance_double_add_att_cross._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance_double_add_att_cross._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance_double_add_att_cross._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance_double_add_att_cross._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features* 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features* 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance_double_add_att_cross._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance_double_add_att_cross._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance_double_add_att_cross._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance_double_add_att_cross._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_advance_double_add_att_cross._block(features * 24, features * 48, name="bottleneck")  # 24, 48

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att_cross._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att_cross._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att_cross._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance_double_add_att_cross._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance_double_add_att_cross._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance_double_add_att_cross._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance_double_add_att_cross._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 48, features * 24, kernel_size=2, stride=2  # 24,12
            )
            self.Att4 = Attention_block(features * 32, features * 8, features * 32 // 2)           
        
                  
            self.decoder4 = YNet_advance_double_add_att_cross._block((features * 16) * 2, features * 24, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2
            )
            
            self.Att3 = Attention_block(features * 16, features * 8, features * 16 // 2)

            self.decoder3 = YNet_advance_double_add_att_cross._block((features * 10) * 2, features * 12, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            
            self.Att2 = Attention_block(features * 6, features * 4, features * 6 // 2)
     
            self.decoder2 = YNet_advance_double_add_att_cross._block((features * 5) * 2, features * 6, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 6, features * 3, kernel_size=2, stride=2
            )
            
            
            self.Att1 = Attention_block(features * 3, features * 2, features * 3 // 2)
            self.decoder1 = YNet_advance_double_add_att_cross._block((features * 5), features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0] #10, 16, 224, 224])
        enc1 = self.encoder1(x)  #[10, 64, 224, 224])
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112])

        enc3 = self.encoder3(self.pool2(enc2)) #[[10, 256, 56, 56])
        enc4 = self.encoder4(self.pool3(enc3))  #[[10, 256, 28, 28])
        enc4_2 = self.pool4(enc4) #[10, 256, 14, 14])]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
          
       
        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2) #10, 256, 14, 14]
            b = torch.zeros_like(enc4_f2) #10, 256, 14, 14
            t = torch.zeros_like(enc4_attention_2) #[10, 256, 14, 14])
            
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1) #501760, 1
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)  #501760, 1
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1
             
            
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) #250880, 3
            bottleneck = bottleneck.view_as(torch.cat((a, b, t), 1))  #[10, 768, 14, 14])

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottleneck = self.bottleneck(bottleneck) #([10, 1536, 14, 14])
        dec4 = self.upconv4(bottleneck) #10, 768, 28, 28]

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1) #enc4_f 10 128 28 28

            dec4 = torch.cat((dec4, enc4_in), dim=1) #[10, 768, 28, 28])
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            #enc4 = self.Att4(dec4,enc4)
        # cross attention birection (dec4, enc4)
        #+gan (生成) 
        #few shot 或者gan 学习
            
            h = 28
            enc4_token = rearrange(enc4, 'n c h w -> n ( h w ) c') # [ 256, 28, 28])
            dec4_token = rearrange(dec4, 'n c h w -> n ( h w ) c') # ([[ 768, 28, 28])
        
            dec4_out, enc4_out = self.joint_cross_attn4(dec4_token, enc4_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc4_out = rearrange(enc4_out, 'n ( h w ) c -> n c h w', h=h) #[10, 256, 28, 28])
            dec4_out = rearrange(dec4_out, 'n ( h w ) c -> n c h w', h=h) #(10, 768, 28, 28]
            
            enc4_out = enc4 + enc4_out
            dec4_out = dec4 + dec4_out
            
            
            dec4 = torch.cat((enc4_out, dec4_out), dim=1) #[768, 28, 28
            dec4 = self.decoder4(dec4) # 
            dec3 = self.upconv3(dec4) #[[5, 384, 56, 56]) 
            
            h = 56
            enc3_token = rearrange(enc3, 'n c h w -> n ( h w ) c') # 256, 56, 56])
            dec3_token = rearrange(dec3, 'n c h w -> n ( h w ) c') # 384, 56, 56])
        
            dec3_out, enc3_out = self.joint_cross_attn3(dec3_token, enc3_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc3_out = rearrange(enc3_out, 'n ( h w ) c -> n c h w', h=h) # 256, 28, 112])
            dec3_out = rearrange(dec3_out, 'n ( h w ) c -> n c h w', h=h) #384, 28, 112])
            
            enc3_out = enc3 + enc3_out
            dec3_out = dec3 + dec3_out
            
            
            
            dec3 = torch.cat((enc3_out, dec3_out), dim=1) #( 
            dec3 = self.decoder3(dec3) #[ 
            dec2 = self.upconv2(dec3) #1 
            
            #h = 112
           # enc2_token = rearrange(enc2, 'n c h w -> n ( h w ) c') #5, 128, 112, 112])
           # dec2_token = rearrange(dec2, 'n c h w -> n ( h w ) c') # 5, 192, 112, 112])
        
           # dec2_out, enc2_out = self.joint_cross_attn2(dec2_token, enc2_token) #
            
           # enc2_out = rearrange(enc2_out, 'n ( h w ) c -> n c h w', h=h) # 
            #dec2_out = rearrange(dec2_out, 'n ( h w ) c -> n c h w', h=h) # 
            
           # enc2_out = enc2 + enc2_out
            #dec2_out = dec2 + dec2_out
            
            
            enc2 = self.Att2(dec2,enc2)
            dec2 = torch.cat((enc2, dec2), dim=1) # 
            dec2 = self.decoder2(dec2) # 
            dec1 = self.upconv1(dec2)#[ 
           
          #  h = 224
           # enc1_token = rearrange(enc1, 'n c h w -> n ( h w ) c') # [10, 781, 156])
           # dec1_token = rearrange(dec1, 'n c h w -> n ( h w ) c') # ([10, 781, 768])
        
          #  dec1_out, enc1_out = self.joint_cross_attn1(dec1_token, enc1_token) #[[10, 768, 18, 18])) [[10, 156, 18, 18])
            
           # enc1_out = rearrange(enc1_out, 'n ( h w ) c -> n c h w', h=h) #[10, 156, 18, 18])
           # dec1_out = rearrange(dec1_out, 'n ( h w ) c -> n c h w', h=h) #(10, 768, 18, 18]
            
            #enc1_out = enc1 + enc1_out
            #dec1_out = dec1 + dec1_out
            enc1 = self.Att1(dec1,enc1)
            dec1 = torch.cat((enc1, dec1), dim=1) # 

        dec1 = self.decoder1(dec1) # 

        return self.softmax(self.conv(dec1)) # 
    
    

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )






class YNet_advance2_branch_layer(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True,  skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_branch_layer, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc

        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        
        self.joint_cross_attn4 = BidirectionalCrossAttention(
            dim = 768,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )   
        self.joint_cross_attn3 = BidirectionalCrossAttention(
            dim = 640,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )  
        self.joint_cross_attn2 = BidirectionalCrossAttention(
            dim = 192,  
            heads = 8,
            dim_head = 64,
            context_dim = 128
        )  
        self.joint_cross_attn1 = BidirectionalCrossAttention(
            dim = 96,  
            heads = 8,
            dim_head = 64,
            context_dim = 64
        )      
 
 
 
 
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_branch_layer._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_branch_layer._block(features  * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_branch_layer._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_branch_layer._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_branch_layer._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_branch_layer._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_branch_layer._block(features * 4, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_branch_layer._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.bottleneck = YNet_advance2_branch_layer._block(features * 24, features * 48, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_layer._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_layer._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_layer._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_layer._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_branch_layer._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_layer._block((features * 4) * 2, features * 5, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_branch_layer._block((features * 5), features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_branch_layer._block(features * 2, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 48, features * 24, kernel_size=2, stride=2  # 24,12
            )
            self.decoder4 = YNet_advance2_branch_layer._block((features * 16) * 2, features * 24, name="dec4")  # 8, 12
            
            
        
    
            self.upconv3 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_branch_layer._block((features * 14) * 2, features * 12, name="dec3")
             
            
            
             
            self.upconv2 = nn.ConvTranspose2d(
                features * 12, features * 6, kernel_size=2, stride=2
            )
            self.Att2 = Attention_block(features * 6, features * 4, features * 6 // 2)
            self.decoder2 = YNet_advance2_branch_layer._block((features * 5) * 2, features * 6, name="dec2")
       
            
            
            self.upconv1 = nn.ConvTranspose2d(
                features * 6, features * 2, kernel_size=2, stride=2
            )
            self.Att1 = Attention_block(features * 2, features * 2, features * 2 // 2)
            self.decoder1 = YNet_advance2_branch_layer._block((features * 4) * 2, features * 2, name="dec1")  # 2,3

            
            
        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()


    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 32, 224, 224]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 64, 112, 112]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 128, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #[10, 128, 14, 14]
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14

        #catmerge
        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)
            t = torch.zeros_like(enc4_attention_2) 
            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1) 
            enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))
            bottle1 = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 
            bottle2 = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 
    
          
            bottle1 = bottle1.view_as(torch.cat((a, b, t), 1))   
            bottle2 = bottle2.view_as(torch.cat((a, b, t), 1))    

        else:

            bottleneck = torch.cat((enc4_2, enc4_f2, enc4_attention_2), 1) 

        bottle1 = self.bottleneck(bottle1) 
        bottle2 = self.bottleneck(bottle2)

        dec4_1 = self.upconv4(bottle1)  
        dec4_2 = self.upconv4(bottle2)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            
            
            h = 28
            enc4_token = rearrange(enc4, 'n c h w -> n ( h w ) c') # [ 256, 28, 28])
            dec4_token = rearrange(dec4_1, 'n c h w -> n ( h w ) c') # ([[ 768, 28, 28])
        
            dec4_out, enc4_out = self.joint_cross_attn4(dec4_token, enc4_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc4_out = rearrange(enc4_out, 'n ( h w ) c -> n c h w', h=h) #[10, 256, 28, 28])
            dec4_out = rearrange(dec4_out, 'n ( h w ) c -> n c h w', h=h) #(10, 768, 28, 28]
            
            enc4_out = enc4 + enc4_out
            dec4_out = dec4_1 + dec4_out
            dec4_1 = torch.cat((enc4_out, dec4_out), dim=1) #dec4:[10, 512, 28, 28])
            dec4_2 = torch.cat((enc4_out, dec4_out), dim=1)
            
            dec4_1 = self.decoder4(dec4_1)
            dec4_2 = self.decoder4(dec4_2)
            
            
            dec3_1 = self.upconv3(dec4_1)
            dec3_2 = self.upconv3(dec4_2)
            
            
            dec3_1 = torch.cat((dec3_1, enc3), dim=1)
            dec3_2 = torch.cat((dec3_2, enc3), dim=1)
            
            h = 56
            enc3_token = rearrange(enc3, 'n c h w -> n ( h w ) c') # 256, 56, 56])
            dec3_token = rearrange(dec3_1, 'n c h w -> n ( h w ) c') # 384, 56, 56])
        
            dec3_out, enc3_out = self.joint_cross_attn3(dec3_token, enc3_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc3_out = rearrange(enc3_out, 'n ( h w ) c -> n c h w', h=h) # 256, 28, 112])
            dec3_out = rearrange(dec3_out, 'n ( h w ) c -> n c h w', h=h) #384, 28, 112])
            
            enc3_out = enc3 + enc3_out
            dec3_out = dec3_1 + dec3_out
            dec3_1 = torch.cat((enc3_out, dec3_out), dim=1)
            dec3_2 = torch.cat((enc3_out, dec3_out), dim=1)
            dec3_1 = self.decoder3(dec3_1)
            dec3_2 = self.decoder3(dec3_2)
            
             
            dec2_1 = self.upconv2(dec3_1)
            dec2_2 = self.upconv2(dec3_2)
            
            enc2_1 = self.Att2(dec2_1,enc2)
            enc2_2 = self.Att2(dec2_2,enc2)
            dec2_1 = torch.cat((dec2_1, enc2_1), dim=1)
            dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
            
            
            dec2_1 = self.decoder2(dec2_1)
            dec2_2 = self.decoder2(dec2_2)
            
            
            dec1_1 = self.upconv1(dec2_1)
            dec1_2 = self.upconv1(dec2_2)
            
            enc1_1 = self.Att1(dec1_1,enc1)
            enc1_2 = self.Att1(dec1_2,enc1)
            dec1_1 = torch.cat((dec1_1, enc1_1), dim=1)
            dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
            dec_connected = torch.cat((dec1_1, dec1_2), dim=1)
            
            

        dec_connected = self.decoder1(dec_connected)
         
         
        return self.softmax(self.conv(dec_connected))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )   
        
        
        
        
###add luyaer layer 
class YNet_advance2_cat_add_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_add_layer, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192

        self.joint_cross_attntemp = BidirectionalCrossAttention(
                dim = 256,  
                heads = 8,
                dim_head = 64,
                context_dim = 256
            )
        self.joint_cross_attn4 = BidirectionalCrossAttention(
            dim = 512,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )   
        self.joint_cross_attn3 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )  
        self.joint_cross_attn2 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 128
        )  
        self.joint_cross_attn1 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 64
        )  
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()

 
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_add_layer._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_add_layer._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_add_layer._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_add_layer._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_add_layer._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_add_layer._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_add_layer._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_add_layer._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_add_layer._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_add_layer._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_add_layer._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_add_layer._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_add_layer._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_add_layer._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_add_layer._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_add_layer._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_add_layer._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                
                features * 32, features *  16, kernel_size=2, stride=2  # 24,12
            )
            
            
           # self.Att4 = Attention_block(features * 32, features * 32, features * 32 // 2)           
        
        
        
            self.decoder4 = YNet_advance2_cat_add_layer._block((features * 12) * 2, features * 16, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            
            #self.Att3 = Attention_block(features * 16, features * 16, features * 16 // 2)
            
            
            self.decoder3 = YNet_advance2_cat_add_layer._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            
            self.Att2 = Attention_block(features * 4, features * 4, features * 4 // 2)
            
            
            self.decoder2 = YNet_advance2_cat_add_layer._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            
            self.Att1 = Attention_block(features * 2, features * 2, features * 2 // 2)
            
            
            self.decoder1 = YNet_advance2_cat_add_layer._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
           
            enc4_2_out, enc4_f2_out = self.joint_cross_attntemp(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            enc_attention_out, enc4_f2_out_2 = self.joint_cross_attntemp(enc4_attention_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            
            h = 14  
            
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[6, 256, 14, 14])
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
            
            enc4_2_out =  enc4_2_out + enc4_2
            enc4_f2_out = enc4_f2_out + enc4_f2
            enc4_f2_out_2 = enc4_f2_out_2 +enc4_f2
            enc_attention_out = enc_attention_out + enc4_attention_2
            #concatenate channel axis
            enc4_cs_cat = torch.add(enc4_2_out, enc_attention_out) #[6, 512, 14, 14])
            enc4_attention_cat = torch.add(enc4_f2_out, enc4_f2_out_2) #6, 512, 14, 14])
            
            a = torch.zeros_like(enc4_cs_cat) #[([6, 512, 14, 14])
            b = torch.zeros_like(enc4_attention_cat)  #6, 512, 14, 14])
            #b2 = torch.zeros_like(enc4_f2_out_2)
            #t = torch.zeros_like(enc_attention_out) #10, 256, 14, 14])
            #s1 = torch.add(a, t) 
            
        
           # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1
                   
            #enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
           # s = torch.add(a, t)
            bottleneck = torch.cat((enc4_cs_cat, enc4_attention_cat), 1)  
            
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))  #[10, 1024, 14, 14]
        else:
            
            bottleneck = torch.cat((enc4_2, enc4_f2), 1) #

        bottleneck = self.bottleneck(bottleneck) #[[10, 1024, 14, 14]

        dec4 = self.upconv4(bottleneck) #10, 1024, 28, 28])

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            
            h = 28
            enc4_token = rearrange(enc4, 'n c h w -> n ( h w ) c') # [ 256, 28, 28])
            dec4_token = rearrange(dec4, 'n c h w -> n ( h w ) c') # ([[ 768, 28, 28])
        
            dec4_out, enc4_out = self.joint_cross_attn4(dec4_token, enc4_token) #[10, 784, 1024]) [10, 784, 256])
            
            enc4_out = rearrange(enc4_out, 'n ( h w ) c -> n c h w', h=h) #[10, 256, 28, 28])
            dec4_out = rearrange(dec4_out, 'n ( h w ) c -> n c h w', h=h) #(10, 768, 28, 28]
            
            enc4_out = enc4 + enc4_out
            dec4_out = dec4 + dec4_out
            
            
            dec4 = torch.cat((enc4_out, dec4_out), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) # 
            dec3 = self.upconv3(dec4)
            
            h = 56
            enc3_token = rearrange(enc3, 'n c h w -> n ( h w ) c') # 256, 56, 56])
            dec3_token = rearrange(dec3, 'n c h w -> n ( h w ) c') # 384, 56, 56])
        
            dec3_out, enc3_out = self.joint_cross_attn3(dec3_token, enc3_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc3_out = rearrange(enc3_out, 'n ( h w ) c -> n c h w', h=h) # 256, 28, 112])
            dec3_out = rearrange(dec3_out, 'n ( h w ) c -> n c h w', h=h) #384, 28, 112])
            
            enc3_out = enc3 + enc3_out
            dec3_out = dec3 + dec3_out
            
            
            
            dec3 = torch.cat((enc3_out, dec3_out), dim=1) #( 
            dec3 = self.decoder3(dec3) #[ 
            dec2 = self.upconv2(dec3)
            
           # h = 112
            #enc2_token = rearrange(enc2, 'n c h w -> n ( h w ) c') #5, 128, 112, 112])
            #dec2_token = rearrange(dec2, 'n c h w -> n ( h w ) c') # 5, 192, 112, 112])
        
            #dec2_out, enc2_out = self.joint_cross_attn2(dec2_token, enc2_token) #
            
            #enc2_out = rearrange(enc2_out, 'n ( h w ) c -> n c h w', h=h) # 
            #dec2_out = rearrange(dec2_out, 'n ( h w ) c -> n c h w', h=h) # 
            
            #enc2_out = enc2 + enc2_out
            #dec2_out = dec2 + dec2_out
            
            
            enc2 = self.Att2(dec2,enc2)
            dec2 = torch.cat((enc2, dec2), dim=1) # 
            dec2 = self.decoder2(dec2) # 
            dec1 = self.upconv1(dec2)#[ 
           
           
            enc1 = self.Att1(dec1,enc1)
            dec1 = torch.cat((enc1, dec1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
 
#mask features
#backbone = swin_transformer(img)
#替换成 upconv1 ：features 
#features
#定义
pixel_decoder_config = {}
pixel_decoder_config['input_shape'] = {}
pixel_decoder_config['input_shape']['res2'] = ShapeSpec(channels=96, height=None, width=None, stride=4)
pixel_decoder_config['input_shape']['res3'] = ShapeSpec(channels=192, height=None, width=None, stride=8)
pixel_decoder_config['input_shape']['res4'] = ShapeSpec(channels=384, height=None, width=None, stride=16)
pixel_decoder_config['input_shape']['res5'] = ShapeSpec(channels=768, height=None, width=None, stride=32)

pixel_decoder_config['transformer_dropout'] = 0.0
pixel_decoder_config['transformer_nheads'] = 8
pixel_decoder_config['transformer_dim_feedforward'] = 1024
pixel_decoder_config['transformer_enc_layers'] = 6
pixel_decoder_config['conv_dims'] = 256
pixel_decoder_config['mask_dim'] = 256
pixel_decoder_config['norm'] = 'GN'
pixel_decoder_config['transformer_in_features'] = ['res3', 'res4', 'res5']
pixel_decoder_config['common_stride'] = 4

class Masked_attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mh_attention = nn.MultiheadAttention(embed_dim = model_dim, num_heads = num_heads)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, query, value, key_pos, attn_mask):
        key = value + key_pos
        
        out = self.mh_attention(
            query = query,
            key = key,
            value = value,
            attn_mask = attn_mask
        )[0]
        
        return self.norm(out + query)
    
class Self_attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mh_attention = nn.MultiheadAttention(embed_dim = model_dim, num_heads = num_heads)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, query):        
        out = self.mh_attention(
            query = query,
            key = query,
            value = query
        )[0]
        
        return self.norm(out + query)
    
class FFN(nn.Module):
    def __init__(self, model_dim, inter_dim):
        super(FFN, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, model_dim)
        )
        
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.ffn(x) + x
        return self.norm(x)
    
class MLP2(nn.Module):
    def __init__(self, model_dim = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)
    
class Transformer_decoder_block(nn.Module):
    def __init__(self, model_dim = 256, num_heads = 8):
        super().__init__()
        
        self.masked_attention = Masked_attention(model_dim, num_heads)
        self.self_attention = Self_attention(model_dim, num_heads)
        self.ffn = FFN(model_dim, 2*model_dim)
        
    def forward(self, query, value, key_pos, attn_mask):
        query = self.masked_attention(query, value, key_pos, attn_mask)
        out = self.self_attention(query)
        out = self.ffn(out)
        
        return out
    
class Transformer_decoder(nn.Module):
    def __init__(self, n_class = 9, L = 3, num_query = 100, num_features = 3, model_dim = 256, num_heads = 8):
        super().__init__()
        
        self.num_features = num_features
        self.num_heads = num_heads
        self.transformer_block = nn.ModuleList([Transformer_decoder_block(model_dim=model_dim, num_heads=num_heads) for _ in range(L * 3)])
        self.query = nn.Parameter(torch.rand(num_query, 1, model_dim)) #[100, 1, 256])
        
        self.from_features_linear = nn.ModuleList([nn.Conv2d(model_dim, model_dim, kernel_size=1) for _ in range(num_features)])
        self.from_features_bias = nn.ModuleList([nn.Embedding(1, model_dim) for _ in range(num_features)])
        self.pos_emb = PositionEmbeddingSine(model_dim // 2, normalize=True)
        
        self.decoder_norm = nn.LayerNorm(model_dim)
        self.classfication_module = nn.Linear(model_dim, n_class)
        self.segmentation_module = MLP2(model_dim)
        
    def forward_prediction_heads(self, mask_embed, pix_emb, decoder_layer_size=None):
        mask_embed = self.decoder_norm(mask_embed)
        mask_embed = mask_embed.transpose(0, 1) # b, 100, 256
        outputs_class = self.classfication_module(mask_embed)
        mask_embed = self.segmentation_module(mask_embed)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, pix_emb)
        
        if decoder_layer_size is not None:
            attn_mask = F.interpolate(outputs_mask, size=decoder_layer_size, mode="bilinear", align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool() # head 수 만큼 복사한다. bool 형으로 넣어야 한다. True인 곳이 무시할 픽셀
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None
        return outputs_class, outputs_mask, attn_mask

        
    def forward(self, features, mask_features):
        query = self.query.expand(self.query.shape[0], features[0].shape[0], self.query.shape[2]) #2, 256, 16, 16]; 2, 256, 132,322, 256, ,64, 64
        
        predictions_class = []
        predictions_mask = []
        
        for i in range(self.num_features):
            b, c, h, w = features[i].shape
                                
            kv = self.from_features_linear[i](features[i])  + self.from_features_bias[i].weight[:, :, None, None]
            kv = rearrange(kv, 'b c h w-> (h w) b c')
            
            key_pos = self.pos_emb(b, h, w, features[i].device, None)
            key_pos = rearrange(key_pos, 'b c h w -> (h w) b c')
            
            for j in range(3):
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(query, mask_features, decoder_layer_size=(h, w)) #2, 256, 128, 128])
                # axial training을 위해 중간 결과를 저장한다.
                predictions_class.append(outputs_class) #[2, 100, 11]) 2, 100, 11])
                predictions_mask.append(outputs_mask) #2, 100, 128, 128]) [2, 100, 128, 128])
                
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # 중간 추출된 mask가 아무것도 가리키지 않을 경우 global context attention으로 처리한다.
                query = self.transformer_block[i * 3 + j](query, kv, key_pos, attn_mask)
                
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(query, mask_features, decoder_layer_size=None)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
                
        out = {
            'pred_logits': predictions_class[-1], #[2, 100, 11])
            'pred_masks': predictions_mask[-1], #([2, 100, 128, 128])
            'aux_outputs': {
                'pred_logits' : predictions_class, #10
                'pred_masks': predictions_mask, #10
            }
        }
        return out

transformer_decoder_config = {}
transformer_decoder_config['n_class'] = 9
transformer_decoder_config['L'] = 3
transformer_decoder_config['num_query'] = 100
transformer_decoder_config['num_features'] = 3
transformer_decoder_config['model_dim'] = 256
transformer_decoder_config['num_heads'] = 8
class HungarianMatcher(nn.Module):
    def __init__(self, n_sample = 112 * 112, w_class: float = 1, w_ce: float = 1, w_dice: float = 1):
        super().__init__()
        self.n_sample = n_sample
        self.w_class = w_class
        self.w_ce = w_ce
        self.w_dice = w_dice
        
    @torch.no_grad()
    def dice_cost(self, predict, target):
        # predict : b * n_queries, n_sample_points
        # target : b * n_obj, n_sample_points
        numerator = 2 * (predict[:, None, :] * target[None, :, :]).sum(-1)
        denominator = predict.sum(-1)[:, None] + target.sum(-1)[None, :]
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        return cost_dice
    
    @torch.no_grad()
    def ce_cost(self, predict, target):
        # predict : b * n_queries, n_sample_points
        # target : b * n_obj, n_sample_points
        predict = predict[:, None, :].expand((predict.shape[0], target.shape[0], predict.shape[1]))
        target = target[None, :, :].expand((predict.shape[0], target.shape[0], target.shape[1]))
        ce = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        
        return ce.mean(-1)
        
    @torch.no_grad()
    def forward(self, out, targets):
        pred_logits = out["pred_logits"] # b, n, class + 1
        pred_masks = out["pred_masks"] # b, n, h, w
        target_logits = targets["labels"] # [ m_i for i in b]
        target_masks = targets["masks"] # [ m_i, h, w for i in b]
        bs, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes] 2, 100, 10]
        tgt_ids = torch.cat([v for v in target_logits]) # [batch_size * num_obj]
        
        
        out_mask = pred_masks.flatten(0, 1).unsqueeze(1)  # [batch_size * num_queries, 1, h, w]
        tgt_mask = torch.cat([v for v in target_masks]).unsqueeze(1) # [batch_size * num_obj, 1, h, w]
        grid = torch.rand((1, 1, self.n_sample, 2), device=out_mask.device) * 2 - 1
        out_grid = grid.expand(out_mask.shape[0], *grid.shape[1:])
        out_mask = F.grid_sample(out_mask, out_grid, mode='nearest', align_corners=False).squeeze()  # [batch_size * num_queries, n_sample_points]
        tgt_grid = grid.expand(tgt_mask.shape[0], *grid.shape[1:])
        tgt_mask = F.grid_sample(tgt_mask, tgt_grid, mode='nearest' , align_corners=False).squeeze()  # [batch_size * num_obj, n_sample_points]

        # cost :
        #     row : pred_querys
        #     col : target_obj
        cost_class = -out_prob[:, tgt_ids]                   # [batch_size * num_queries, batch_size * num_obj]
        cost_dice = self.dice_cost(out_mask, tgt_mask)       # [batch_size * num_queries, batch_size * num_obj] 
        cost_ce = self.ce_cost(out_mask, tgt_mask)           # [batch_size * num_queries, batch_size * num_obj]
        
        # Final cost matrix
        C = self.w_dice * cost_dice + self.w_class * cost_class + self.w_ce * cost_ce
        C = C.view(bs, num_queries, -1).cpu() # [batch_size, num_queries, batch_size * num_obj] [2, 100, 19])
      
        sizes = [len(v) for v in target_masks]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        result = []
        for i, j in indices:
            i = torch.as_tensor(i, dtype=torch.int64, device=device)
            j = torch.as_tensor(j, dtype=torch.int64, device=device)
            result.append(i[j])
        return result

 
matcher_config = {}
matcher_config['n_sample'] = 112 * 112
matcher_config['w_class'] = 1.0
matcher_config['w_ce'] = 20.0
matcher_config['w_dice'] = 1.0
class Maskformer_loss(nn.Module):
    def __init__(self, n_sample = 112 * 112, w_ce = 1., w_dice = 1., w_class = 1., w_noobj = 1., oversample_ratio = 3.0, importance_sample_ratio = 0.75):
        super(Maskformer_loss, self).__init__()
        self.n_sample = n_sample
        self.w_class = w_class
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_noobj = w_noobj
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
    def class_loss(self, pred_logits, target_logits, match_indexs):
        device = pred_logits.device
        target_labels = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=device)
        cost_no_obj = torch.ones(pred_logits.shape[2], device=device)
        cost_no_obj[0] *= self.w_noobj
        
        for i, match_index in enumerate(match_indexs):
            target_labels[i, match_index] = target_logits[i]
        
        class_loss = F.cross_entropy(pred_logits.flatten(0, 1), target_labels.flatten(0, 1), cost_no_obj)
        return class_loss
        
    def ce_loss(self, predict, target, gamma = 2.0, alpha = 0.25):
        # predict : b * n_queries, h * w
        # target : b * n_obj, h * w
        ce = F.binary_cross_entropy_with_logits(predict, target, reduction='none')

        return ce.mean()
        
    def dice_loss(self, predict, target):
        numerator = 2 * (predict * target).sum(-1)
        denominator = predict.sum(-1) + target.sum(-1)
        loss_dice = 1 - (numerator + 1) / (denominator + 1)
        return loss_dice.mean()
    
    def calculate_uncertainty(self, logits):
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))

    def forward(self, out, targets, match_indexs):
        pred_logits = out["pred_logits"] # b, n, class + 1;  2, 100, 11
        pred_masks = out["pred_masks"] # b, n, h, w; 2, 100, 128, 128
        target_logits = targets["labels"] # [ m_i for i in b]
        target_boxes = targets["masks"] # [ m_i, h, w for i in b]
        
        tgt_mask = torch.cat([v for v in target_boxes]).unsqueeze(1) # [batch_size * num_obj, 1, h, w]
        out_mask = pred_masks  # [batch_size, num_queries, h, w]
        out_mask = torch.cat([out_mask[i, match_index, :] for i, match_index in enumerate(match_indexs)]).unsqueeze(1)  # [batch_size * num_obj, 1, h, w]
        
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                out_mask,
                lambda logits: self.calculate_uncertainty(logits),
                self.n_sample,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).unsqueeze(1)
            
            tgt_mask = F.grid_sample(tgt_mask, point_coords, mode='nearest', align_corners=False).squeeze(1) # [batch_size * num_queries, n_sample_points]
        out_mask = F.grid_sample(out_mask, point_coords, mode='nearest', align_corners=False).squeeze(1) # [batch_size * num_queries, n_sample_points]

        class_loss = self.class_loss(pred_logits, target_logits, match_indexs) * self.w_class
        ce_loss = self.ce_loss(out_mask, tgt_mask) * self.w_ce
       # dice_loss = self.dice_loss(out_mask, tgt_mask) * self.w_dice
        dice_loss = self.dice_loss(out_mask, tgt_mask)  
        
        return class_loss + dice_loss


loss_config = {}
loss_config['n_sample'] = 112 * 112
loss_config['w_class'] = 1.0
loss_config['w_ce'] = 20.0
loss_config['w_dice'] = 1.0
loss_config['w_noobj'] = 0.1
loss_config['oversample_ratio'] = 3.0
loss_config['importance_sample_ratio'] = 0.75
#############################################
############################################# 
#############################################
##############################################
#############################################
#***

    #***
#mask2former
class YNet_advance2_cat_layer_mp2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, attention=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_advance2_cat_layer_mp2, self).__init__()
        self.in_channels = in_channels
        self.stem_out_channel = 16
        self.ffc = ffc
        self.attention = attention
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        self.bneck_exp = 32
        self.bneck_out = 16
        self.num_token = 6
        self.d_model = 192

        self.joint_cross_attntemp = BidirectionalCrossAttention(
                dim = 256,  
                heads = 8,
                dim_head = 64,
                context_dim = 256
            )
        self.joint_cross_attn4 = BidirectionalCrossAttention(
            dim = 512,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )   
        self.joint_cross_attn3 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 256
        )  
        self.joint_cross_attn2 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 128
        )  
        self.joint_cross_attn1 = BidirectionalCrossAttention(
            dim = 256,  
            heads = 8,
            dim_head = 64,
            context_dim = 64
        )  
        self.tokens = nn.Parameter(torch.randn(self.num_token, self.d_model), requires_grad=True).cuda()
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_out_channel, kernel_size=3, stride=1, padding=1).cuda(),
            nn.BatchNorm2d(self.stem_out_channel).cuda(),
            nn.ReLU6(inplace=True).cuda()
        )
        self.bneck = BottleneckLite(self.stem_out_channel, self.bneck_exp, self.bneck_out, kernel_size=3, stride=1, padding=1).cuda()
       
            
        self.pixel_decoder = MSDeformAttnPixelDecoder(
                input_shape = pixel_decoder_config['input_shape'], 
                transformer_dropout = pixel_decoder_config['transformer_dropout'],
                transformer_nheads = pixel_decoder_config['transformer_nheads'],
                transformer_dim_feedforward = pixel_decoder_config['transformer_dim_feedforward'],
                transformer_enc_layers = pixel_decoder_config['transformer_enc_layers'],
                conv_dim = pixel_decoder_config['conv_dims'],
                mask_dim = pixel_decoder_config['mask_dim'],
                norm = pixel_decoder_config['norm'],
                transformer_in_features = pixel_decoder_config['transformer_in_features'],
                common_stride = pixel_decoder_config['common_stride'],
            ).cuda()            
        self.transformer_decoder = Transformer_decoder(
                n_class = transformer_decoder_config['n_class'] + 1, 
                L = transformer_decoder_config['L'], 
                num_query = transformer_decoder_config['num_query'], 
                num_features = transformer_decoder_config['num_features'], 
                model_dim = transformer_decoder_config['model_dim'], 
                num_heads = transformer_decoder_config['num_heads']
            ).cuda()
        
        
        self.matcher = HungarianMatcher(
                n_sample = matcher_config['n_sample'],
                w_class = matcher_config['w_class'],
                w_ce = matcher_config['w_ce'],
                w_dice = matcher_config['w_dice']
            ).cuda()
        
        self.Loss = Maskformer_loss(
                n_sample = loss_config['n_sample'] , 
                w_ce = loss_config['w_class'] , 
                w_dice = loss_config['w_ce'] , 
                w_class = loss_config['w_dice'], 
                w_noobj = loss_config['w_noobj'], 
                oversample_ratio = loss_config['oversample_ratio'], 
                importance_sample_ratio = loss_config['importance_sample_ratio']
            ).cuda()
    #开卷
        self.conv1_seg = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_seg = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1)
 
    
    #load target
      
        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_advance2_cat_layer_mp2._block(in_channels, features * 2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_advance2_cat_layer_mp2._block(features * 2, features * 4, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_advance2_cat_layer_mp2._block(features * 4, features * 8, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_advance2_cat_layer_mp2._block(features * 8, features * 8, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features * 2, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 8, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        #attention
        if attention:
            ################ Attention #######################################
            self.encoder1_attention = MobileFormerBlock_advance(16, expand_size=int(features * 0.5), out_channel=features * 2, d_model=192)
            self.pool1_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_attention = MobileFormerBlock_advance(features * 2, expand_size=int(features * 1.0), out_channel= features * 4, d_model=192)
            self.pool2_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_attention = MobileFormerBlock_advance(features * 4, expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool3_attention = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_attention = MobileFormerBlock_advance(features * 8 , expand_size=int(features * 2.0), out_channel=features * 8, d_model=192)
            self.pool4_attention = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_advance2_cat_layer_mp2._block(in_channels, features * 2, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_advance2_cat_layer_mp2._block(features * 2, features * 4, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_advance2_cat_layer_mp2._block(features * 4, features * 8, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_advance2_cat_layer_mp2._block(features * 8, features * 8, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = YNet_advance2_cat_layer_mp2._block(features * 16, features * 32, name="bottleneck")  # 8, 24

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 24, features * 12, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_layer_mp2._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 12, features * 8, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_layer_mp2._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_layer_mp2._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_layer_mp2._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_advance2_cat_layer_mp2._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_advance2_cat_layer_mp2._block((features * 5) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 5, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_advance2_cat_layer_mp2._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_advance2_cat_layer_mp2._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                
                features * 32, features *  16, kernel_size=2, stride=2  # 24,12
            )
           # self.Att4 = Attention_block(features * 32, features * 32, features * 32 // 2)           
            self.decoder4 = YNet_advance2_cat_layer_mp2._block((features * 12) * 2, features * 16, name="dec4")  # 8, 1                      
            self.upconv3 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )         
            #self.Att3 = Attention_block(features * 16, features * 16, features * 16 // 2)                  
            self.decoder3 = YNet_advance2_cat_layer_mp2._block((features * 8) * 2, features * 8, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )           
            self.Att2 = Attention_block(features * 4, features * 4, features * 4 // 2)                     
            self.decoder2 = YNet_advance2_cat_layer_mp2._block((features * 4) * 2, features * 4, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )           
            self.Att1 = Attention_block(features * 2, features * 2, features * 2 // 2)                      
            self.decoder1 = YNet_advance2_cat_layer_mp2._block((features * 2) * 2, features * 2, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features * 2, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted
    
#接口
    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)  #[10, 64, 224, 224]]
        enc2 = self.encoder2(self.pool1(enc1)) #[10, 128, 112, 112]]

        enc3 = self.encoder3(self.pool2(enc2)) #[10, 256, 56, 56]

        enc4 = self.encoder4(self.pool3(enc3))  #[10, 128, 28, 28]
        enc4_2 = self.pool4(enc4) #501760, 1
        
        if self.ffc:
            enc1_f = self.encoder1_f(x)  #tensor0: 10, 16, 224, 224; tensor1: 10, 16, 224, 224
            enc1_l, enc1_g = enc1_f #enc1_l: 10, 16, 224, 224; enc1_g: 10, 16, 224, 224
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f #enc2_l: 10, 32, 112, 112; enc2_g: 10, 32, 112, 112
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f #enc3_l: 10, 64, 56, 56; enc3_g: 10, 64, 56, 56
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f #enc4_l: 10, 128, 28, 28; enc4_g: 10, 128, 28, 28
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
        
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.attention:
            n, _, _, _ = x.shape
            x = self.stem(x)
            x = self.bneck(x)
            tokens = repeat(self.tokens, 'm d -> n m d', n=n)
            enc1_attention, tokens = self.encoder1_attention(x, tokens) #10, 32, 224, 224
            enc2_attention, tokens = self.encoder2_attention(self.pool1_attention(enc1_attention), tokens) #10, 64, 112, 112
            enc3_attention, tokens = self.encoder3_attention(self.pool2_attention(enc2_attention), tokens) #10, 128, 56, 56
            enc4_attention, tokens = self.encoder4_attention(self.pool3_attention(enc3_attention), tokens) #10, 128, 28, 28
            enc4_attention_2 = self.pool4_attention(enc4_attention) #10, 128, 14, 14
            
        #catmerge
        if self.cat_merge:
            enc4_2_token = rearrange(enc4_2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
            enc4_f2_token = rearrange(enc4_f2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
            enc4_attention_2_token = rearrange(enc4_attention_2, 'n c h w -> n ( h w ) c') # 6, 196, 256])
           
            enc4_2_out, enc4_f2_out = self.joint_cross_attntemp(enc4_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            enc_attention_out, enc4_f2_out_2 = self.joint_cross_attntemp(enc4_attention_2_token, enc4_f2_token) # (10, 196, 128) (10, 196, 128)
            
            h = 14  
            
            enc4_2_out = rearrange(enc4_2_out, 'n ( h w ) c -> n c h w', h=h)
            enc4_f2_out = rearrange(enc4_f2_out, 'n ( h w ) c -> n c h w', h=h) #[6, 256, 14, 14])
            enc4_f2_out_2 = rearrange(enc4_f2_out_2, 'n ( h w ) c -> n c h w', h=h)
            enc_attention_out = rearrange(enc_attention_out, 'n ( h w ) c -> n c h w', h=h)
            
            enc4_2_out =  enc4_2_out + enc4_2
            enc4_f2_out = enc4_f2_out + enc4_f2
            enc4_f2_out_2 = enc4_f2_out_2 +enc4_f2
            enc_attention_out = enc_attention_out + enc4_attention_2
            #concatenate channel axis
            enc4_cs_cat = torch.add(enc4_2_out, enc_attention_out) #[6, 512, 14, 14])
            enc4_attention_cat = torch.add(enc4_f2_out, enc4_f2_out_2) #6, 512, 14, 14])
            
            a = torch.zeros_like(enc4_cs_cat) #[([6, 512, 14, 14])
            b = torch.zeros_like(enc4_attention_cat)  #6, 512, 14, 14])
            #b2 = torch.zeros_like(enc4_f2_out_2)
            #t = torch.zeros_like(enc_attention_out) #10, 256, 14, 14])
            #s1 = torch.add(a, t) 
            
        
           # enc4_attention_2 = torch.reshape(enc4_attention_2, (torch.numel(enc4_attention_2), 1))  #501760, 1
                   
            #enc_sum = torch.add(enc4_2, enc4_attention_2) #501760, 1])
           # s = torch.add(a, t)
            bottleneck = torch.cat((enc4_cs_cat, enc4_attention_cat), 1)  
            
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))  #[7, 512, 14, 14])
        else:
            
            bottleneck = torch.cat((enc4_2, enc4_f2), 1) #
#1： bottlenectk:  [1024, 14, 14]
#2: dec4: 512, 28, 28])
#3 dec3: 256, 56, 56])
#4 dec2: 128, 112, 112
#mask2former:      #([ 96, 128, 128])
                    # 192, 64, 64])
                     #[ 384, 32, 32])
                     #[ 768, 16, 16])
    #cnn 变形 conv2d.pending 28->32
        bottleneck = self.bottleneck(bottleneck) #[2, 1024, 14, 14])
        
        
        
        
    #变形1   
        self.conv_bottleneck_to_res5 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0).cuda()
        bottleneck_adjusted = self.conv_bottleneck_to_res5(bottleneck)

        self.upsample_bottleneck_to_res5 = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True)
        bottleneck_adjusted_upsampled = self.upsample_bottleneck_to_res5(bottleneck_adjusted)    
    #变形1   
    #变形2         
        dec4 = self.upconv4(bottleneck) #10, 1024, 28, 28])
        self.conv_dec4_to_res4 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0).cuda()
        dec4_adjusted = self.conv_dec4_to_res4(dec4)
        self.upsample_dec4_to_res4 = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)
        dec4_adjusted_upsampled = self.upsample_dec4_to_res4(dec4_adjusted)
    #变形2
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            
            h = 28
            enc4_token = rearrange(enc4, 'n c h w -> n ( h w ) c') # [ 256, 28, 28])
            dec4_token = rearrange(dec4, 'n c h w -> n ( h w ) c') # ([[ 768, 28, 28])
        
            dec4_out, enc4_out = self.joint_cross_attn4(dec4_token, enc4_token) #[10, 784, 1024]) [10, 784, 256])
            
            enc4_out = rearrange(enc4_out, 'n ( h w ) c -> n c h w', h=h) #[10, 256, 28, 28])
            dec4_out = rearrange(dec4_out, 'n ( h w ) c -> n c h w', h=h) #(10, 768, 28, 28]
            
            enc4_out = enc4 + enc4_out
            dec4_out = dec4 + dec4_out
            
            
            dec4 = torch.cat((enc4_out, dec4_out), dim=1) #dec4:[10, 512, 28, 28])
            dec4 = self.decoder4(dec4) 
        #变形3
            dec3 = self.upconv3(dec4)
            self.conv_dec3_to_res3 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0).cuda()
            dec3_adjusted = self.conv_dec3_to_res3(dec3)
            self.upsample_dec3_to_res3 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)
            dec3_adjusted_upsampled = self.upsample_dec3_to_res3(dec3_adjusted)
        #变形3   
            h = 56
            enc3_token = rearrange(enc3, 'n c h w -> n ( h w ) c') # 256, 56, 56])
            dec3_token = rearrange(dec3, 'n c h w -> n ( h w ) c') # 384, 56, 56])
        
            dec3_out, enc3_out = self.joint_cross_attn3(dec3_token, enc3_token) #[[10, 768, 28, 28])) [[10, 256, 28, 28])
            
            enc3_out = rearrange(enc3_out, 'n ( h w ) c -> n c h w', h=h) # 256, 28, 112])
            dec3_out = rearrange(dec3_out, 'n ( h w ) c -> n c h w', h=h) #384, 28, 112])
            
            enc3_out = enc3 + enc3_out
            dec3_out = dec3 + dec3_out
            
            
            
            dec3 = torch.cat((enc3_out, dec3_out), dim=1) #( 
            dec3 = self.decoder3(dec3) #[ 
        #变形4
            dec2 = self.upconv2(dec3)
            self.conv_dec2_to_res2 = nn.Conv2d(128, 96, kernel_size=1, stride=1, padding=0).cuda()
            dec2_adjusted = self.conv_dec2_to_res2(dec2)
            self.upsample_dec2_to_res2 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        
            
            dec2_adjusted_upsampled = self.upsample_dec2_to_res2(dec2_adjusted)
         #变形4 
           # h = 112
            #enc2_token = rearrange(enc2, 'n c h w -> n ( h w ) c') #5, 128, 112, 112])
            #dec2_token = rearrange(dec2, 'n c h w -> n ( h w ) c') # 5, 192, 112, 112])
        
            #dec2_out, enc2_out = self.joint_cross_attn2(dec2_token, enc2_token) #
            
            #enc2_out = rearrange(enc2_out, 'n ( h w ) c -> n c h w', h=h) # 
            #dec2_out = rearrange(dec2_out, 'n ( h w ) c -> n c h w', h=h) # 
            
            #enc2_out = enc2 + enc2_out
            #dec2_out = dec2 + dec2_out
            
            
            enc2 = self.Att2(dec2,enc2)
            dec2 = torch.cat((enc2, dec2), dim=1) # 
            dec2 = self.decoder2(dec2) # 
            dec1 = self.upconv1(dec2)#[ 
           
            
           
            enc1 = self.Att1(dec1,enc1)
            dec1 = torch.cat((enc1, dec1), dim=1)
#结合
 
            self._out_feature_channels = {
                'res2': dec2_adjusted_upsampled,
                'res3': dec3_adjusted_upsampled,
                'res4': dec4_adjusted_upsampled,
                'res5': bottleneck_adjusted_upsampled,}
            
            
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(self._out_feature_channels)
        
            transformer_decoder_output = self.transformer_decoder(multi_scale_features, mask_features)
#matchers 
            
            #match_indexs = self.matcher(transformer_decoder_output, target)
      
            #maskloss = self.Loss(transformer_decoder_output, target, match_indexs)
          
            #masklogits = transformer_decoder_output['pred_masks'] #2, 100, 10
            mask_cls_results = transformer_decoder_output['pred_logits'] #2, 100, 9]
            mask_pred_results = transformer_decoder_output["pred_masks"] #2, 100, 128, 128])

            
            mask_cls_results = F.softmax(mask_cls_results, dim=-1)[...,1:]  #[2, 100, 9])
            mask_pred_results = mask_pred_results.sigmoid()  #[2, 100, 128, 128]
            semseg = torch.einsum("bqc,bqhw->bchw", mask_cls_results, mask_pred_results)      #2, 9, 128, 128])
            #pred_mask = semseg #([2, 9, 128, 128])
            transform = T.Resize((224,224))
            semseg_news = transform(semseg)
#上采样卷积 
            semseg_up = self.conv1_seg(semseg) #[2, 64, 128, 128])
            
            upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
            semseg_upsampled = upsample(semseg_up) #([2, 64, 224, 224])

            
            
            semseg_new =self.conv2_seg(semseg_upsampled) #orch.Size([2, 9, 224, 224])
 
        
        
        dec1 = self.decoder1(dec1) #[2, 64, 224, 224])
    #合并
    # add: logit + ynet logits
    #loss 
    #class_loss = self.class_loss(pred_logits, target_logits, match_indexs) * self.w_class
        #log_sum = dec1 + masklogits

        final = {}
        final['original'] = self.softmax(self.conv(dec1)) #2, 9, 224, 224])
        final['pred_mask'] = semseg_news #[2, 9, 224, 224])
       
        return final

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
 
 