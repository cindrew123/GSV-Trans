 # add
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import copy
import logging
from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from . import test_Fast_vit_configs_mine as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from torch.nn.modules.utils import _pair
from scipy import ndimage
# add end
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=1000000)
from torch.autograd import Variable
from .adapt_eye_atten import Eye_Center
from .adapt_Fast_Class_Activation_Mapping_show import Class_Activation_Mapping, Show_Avg_CAM_Mapping

'''
Cam_Centre1 = [64,64]
Cam_Centre2 = [32,32]
Cam_Centre3 = [16,16]
Cam_Centre4 = [8,8]
Cam_Centre5 = [4,4]
'''


cam_input_layers = 6
cam_rate = 1
atlr = 64 # attn_to_last_rate

def swish(x):
    return x * torch.sigmoid(x)
eye_center = "constant_center"
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Multi_Transformer_test(nn.Module):
    def __init__(self, config, depth, img_size=512, num_classes=6, zero_head=False, vis=False):
        super(Multi_Transformer_test, self).__init__()



    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = torch.zeros([x.shape[0],24, 32, 32]).cuda() # [2, 24, 32, 32]
        camone = torch.zeros([x.shape[0], 512, 512]).cuda()
        camsix = torch.zeros([x.shape[0],6, 512, 512]).cuda()
        return logits,camone, camsix



class Multi_Transformer(nn.Module):
    def __init__(self, config, depth, img_size=512, num_classes=6, zero_head=False, vis=False):
        super(Multi_Transformer, self).__init__()
        self.transformer = Transformer(config, depth, img_size, vis)
        self.config = config



    def forward(self, x,complex_x):

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        #print (x.shape)
        x,camone, camsix = self.transformer(x,complex_x)  # (B, n_patch, hidden) x = [2, 24, 32, 32]
        return x,camone, camsix

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4. 4 pixel
        in_chans (int): Number of input image channels. Default: 3. RGB
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, config, img_size=512, in_chans=3,  norm_layer=None):
        super(PatchEmbed, self).__init__()
        # parameters
        embed_dim = config.embed_dim
        patch_size = config.patch_size

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # [128,128]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution  # [128,128]
        self.num_patches = patches_resolution[0] * patches_resolution[1] #128*128

        self.in_chans = in_chans  # 3
        self.embed_dim = embed_dim   #in_chans:3 out_chans:96,out_chans is embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape  #batch channel height width,[2, 3, 512, 512]
        # shape : B, C, H, W --> B, C, H/4, W/4 --> B, C, (H/4)*(W/4)--> B, (H/4)*(W/4), c
        x = self.proj(x).flatten(2).transpose(1, 2)  # x [2, 16384, 96],self.proj(x)   [2, 96, 128, 128],x = x.flatten(2)   [2, 96, 16384]
        if self.norm is not None:
            x = self.norm(x)      
        return x


class Channel_CAM(nn.Module): # not use
    def __init__(self):
        super(Channel_CAM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.Transpose = np.transpose
        self.Add = np.add
        self.Dot = np.dot
        self.Multiply = np.multiply
        #self.Batch = nn.BatchNorm2d(C,affine=affine_par)

    def forward(self, x): # input [2, 128, 128, 96]
        x = x.permute(0, 3, 1, 2).contiguous()  #[2, 96, 128, 128]
        Batch = nn.BatchNorm2d(x.shape[1],affine=affine_par).cuda()   ############
        x = Batch(x)
        nn.functional.adaptive_avg_pool2d(x, (1, 1))
        xcam1 = self.relu(x)
        xcam1 = self.sigmoid(xcam1)
        print (xcam1.shape)
        
        Downchannel = nn.Conv2d(xcam1.shape[1],16,kernel_size=1,stride=1,bias=False).cuda()  #########
        xcam1 = Downchannel(xcam1)
        xcam1 = xcam1.data.cpu().numpy()
        if xcam1.shape[0] ==  2:
            xcam1a = xcam1[0,...].reshape(16,xcam1.shape[2] * xcam1.shape[3] )
            xcam1b = xcam1[1,...].reshape(16,xcam1.shape[2] * xcam1.shape[3] )
            xcam2a = self.Transpose(xcam1a)
            xcam2b = self.Transpose(xcam1b)
            xcam3a = self.Dot(xcam1a, xcam2a)
            xcam3b = self.Dot(xcam1b, xcam2b)
            xcam4a = self.Dot(xcam3a, xcam1a)
            xcam4b = self.Dot(xcam3a, xcam1b)
            xcam4a = xcam4a.reshape(16,xcam1.shape[2] , xcam1.shape[3])
            xcam4b = xcam4b.reshape(16,xcam1.shape[2] , xcam1.shape[3])
            xcam4 = xcam1
            xcam4[0,...] = torch.from_numpy(xcam4a).cuda().data.cpu().numpy()
            xcam4[1,...] = torch.from_numpy(xcam4b).cuda().data.cpu().numpy()
            xcam4 =  torch.from_numpy(xcam4).cuda()    # [2, 16, 128, 128]
        else:
            xcam1 = xcam1.reshape(16,xcam1.shape[2] * xcam1.shape[3])
            xcam2 = self.Transpose(xcam1)
            xcam3 = self.Dot(xcam1, xcam2)
            xcam4 = self.Dot(xcam3, xcam1)
            xcam4 = xcam4.reshape(16,x.shape[2] , x.shape[3])
            xcam4 = torch.from_numpy(xcam4).cuda().data.cpu().numpy()
            xcam4 = (torch.tensor(xcam4)).unsqueeze(0) # [1, 16, 128, 128]
        
        upchannel = nn.Conv2d(16,x.shape[1],kernel_size=1,stride=1,bias=False).cuda()        #######
        xcam1 = upchannel(xcam4.cuda())  
        xcam1 = torch.cat((x,xcam1),1)
        Downchannel = nn.Conv2d(xcam1.shape[1],x.shape[1],kernel_size=1,stride=1,bias=False).cuda() #######
        x = Downchannel(xcam1)    # [2, 96, 128, 128]
        return x 
        


class Transformer(nn.Module):
    def __init__(self, config, depth, img_size, vis, drop_rate=0., norm_layer=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.TransBlock = TransBlock(config, depth, img_size, vis)
        self.blocks = nn.ModuleList([
            self.TransBlock])

    def forward(self, x, complex_x):
        for blk in self.blocks:
            x,camone, camsix = blk(x, complex_x)
        return x,camone, camsix

class TransBlock(nn.Module):
    def __init__(self, config, depth, img_size, vis,drop_rate=0.,norm_layer=nn.LayerNorm):
        super(TransBlock, self).__init__()
        #self.encoder = Encoder(config, vis)
        self.depth =  depth
        H = int(config.embed_dim)
        self.encoder_norm = LayerNorm(H, eps=1e-6)
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=2, bias=False, padding = 1, dilation = 1)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=4, bias=False, padding = 1, dilation = 1)
        self.conv2 = nn.Conv2d(H, int(H/2), kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(H, int(H/2), kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(int(H/3*2), int(H/3), kernel_size=1, stride=1, bias=False)
        self.patchembed = PatchEmbed(config)
        num_patches = self.patchembed.num_patches
        embed_dim = self.patchembed.embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        self.eyeblock1 = Eyetransblock1(config)
        self.eyeblock12 = Eyetransblock12(config)
        self.eyeblock13 = Eyetransblock13(config)
        self.eyeblock2 = Eyetransblock2(config)
        self.eyeblock22 = Eyetransblock22(config)
        self.eyeblock23 = Eyetransblock23(config)
        self.eyeblock3 = Eyetransblock3(config)
        self.eyeblock33 = Eyetransblock33(config)
        self.eyeblock4 = Eyetransblock4(config)
        self.eyeblock14 = Eyetransblock14(config)
        self.eyeblock24 = Eyetransblock24(config)
        self.eyeblock34 = Eyetransblock34(config)
        self.eyeblock44 = Eyetransblock44(config)
        self.eyeblock5 = Eyetransblock5(config)
        
        
        self.patchmerg12 = PatchMerging12(config)
        self.patchmerg13 = PatchMerging13(config)
        self.patchmerg22 = PatchMerging22(config)
        self.patchmerg23 = PatchMerging23(config)
        self.patchmerg3 = PatchMerging3(config)
        self.patchmerg33 = PatchMerging33(config)
        self.patchmerg4 = PatchMerging4(config)
        self.patchmerg14 = PatchMerging14(config)
        self.patchmerg24 = PatchMerging24(config)
        self.patchmerg34 = PatchMerging34(config)
        self.patchmerg44 = PatchMerging44(config)
        self.patchmerg5 = PatchMerging5(config)
        
        self.Transout_fuse_depth4 = Transout_fuse_depth4(config)
        self.Transout_fuse_depth3 = Transout_fuse_depth3(config)
        self.Cam_Fuse_Depth4 = CAM_Fuse_depth4(config)
        self.Cam_Fuse_Depth3 = CAM_Fuse_depth3(config)
        # self add 
        patch = config.patch_size
        self.norm = norm_layer(int(config.embed_dim))
        self.norm1 = norm_layer(2*int(config.embed_dim))
        self.norm2 = norm_layer(4*int(config.embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, int(256*256/int(patch*patch)), self.patchembed.embed_dim))
        self.position_embeddings2 = nn.Parameter(torch.zeros(1, int(256*256/int(4*patch*patch)), self.patchembed.embed_dim))
        self.norm22 = norm_layer(2*int(config.embed_dim))
        self.norm23 = norm_layer(4*int(config.embed_dim))
        self.norm3 = norm_layer(1*int(config.embed_dim))
        self.norm33 = norm_layer(2*int(config.embed_dim))
        self.norm4 = norm_layer(1*int(config.embed_dim))
        self.norm14 = norm_layer(8*int(config.embed_dim))
        self.norm24 = norm_layer(8*int(config.embed_dim))
        self.norm34 = norm_layer(4*int(config.embed_dim))
        self.norm44 = norm_layer(2*int(config.embed_dim))
        self.norm5 = norm_layer(1*int(config.embed_dim))


    def forward(self, x,complex_x):
        depth =  self.depth

        # x torch.Size([2, 3, 512, 512])
        # 1row 1column  --> x1: [2,  3, 512, 512],output 
        x1 = self.conv0(x)  # [2, 3, 256, 256]  
        x1= self.patchembed(x1)       # x [2, 256, 24]
        x1 = x1 + self.position_embeddings # position_embeddings [1, 16384, 96] after add position_embeddings, x1 [2, 16384, 96]
        x1 = self.pos_drop(x1)        # [2, 16384, 96]
        x1 = self.norm(x1)            # [2, 16384, 96]
        x1, cam1 = self.eyeblock1(x1, x1.shape[0], x1.shape[2], x1.shape[1],complex_x)  # output [2, 16384, 96]
        # 2row 1column --> x2: [2, 3, 256, 256], output [2, 64, 64, 96]
        x2 = self.conv1(x)            # [2, 3, 256, 256]
        x2= self.patchembed(x2)       # [2, 4096, 96]
        x2 = x2 + self.position_embeddings2 # [2, 4096, 96]
        x2 = self.pos_drop(x2)        # [2, 4096, 96]
        x2 = self.norm(x2)            # [2, 4096, 96]
        x2, cam2 = self.eyeblock2(x2, x2.shape[0], x2.shape[2], x2.shape[1],complex_x)      # output [2, 4096, 96]

        # 1row 2column  --> input [2, 16384, 96],output [2, 4096, 192]
        x12 = x1
        x12= self.patchmerg12(x12, x12.shape[0], x12.shape[2], x12.shape[1]) # [2, 4096, 192]
        x12 = self.pos_drop(x12)        #[2, 4096, 192]
        #norm = nn.LayerNorm(x12.shape[2]).cuda() 
        x12 = self.norm1(x12)            # [2, 4096, 192]
        x12, cam12 = self.eyeblock12(x12, x12.shape[0], x12.shape[2], x12.shape[1],complex_x)
        # 2row 2column  --> input [2, 4096, 96], output [2, 1024, 192]
        x22 = x2
        x22 = self.patchmerg22(x22, x22.shape[0], x22.shape[2], x22.shape[1])  # [2, 1024, 192]
        x22 = self.pos_drop(x22)  # [2, 1024, 192]
        x22 = self.norm22(x22)  # [2, 1024, 192]
        x22, cam22 = self.eyeblock22(x22, x22.shape[0], x22.shape[2], x22.shape[1],complex_x)  # [2, 1024, 192]
        # 3row 2column--> input [2, 4096, 96], output [2, 1024, 96]  3row has no 1 column
        x3 = x2.view(x2.shape[0], int(math.sqrt(x2.shape[1])), int(math.sqrt(x2.shape[1])),
                     x2.shape[2])  # x2[2, 4096, 96] -- >[2,64,64,96]
        x3 = x3.permute(0, 3, 1, 2).contiguous()  # [2, 96, 64, 64]
        x3 = self.conv2(x3)  # [2, 48, 64, 64]
        x3 = x3.view(x3.shape[0], int(x3.shape[2]) * int(x3.shape[3]), x3.shape[1])  # [2, 4096, 48]
        x3 = self.patchmerg3(x3, x3.shape[0], x3.shape[2], x3.shape[1])  # [2, 1024, 96]
        x3 = self.pos_drop(x3)  # [2, 1024, 96]
        x3 = self.norm3(x3)  # [2, 1024, 96]
        x3, cam3 = self.eyeblock3(x3, x3.shape[0], x3.shape[2], x3.shape[1],complex_x)  # [2, 1024, 96]

        # 1row 3column  --> input [2, 4096, 192],output [2, 1024, 384]
        x13 = x12
        x13= self.patchmerg13(x13, x13.shape[0], x13.shape[2], x13.shape[1]) # [2, 1024, 384]
        x13 = self.pos_drop(x13)        # [2, 1024, 384]
        x13 = self.norm2(x13)            # [2, 1024, 384]
        x13, cam13 = self.eyeblock13(x13, x13.shape[0], x13.shape[2], x13.shape[1],complex_x) #[2, 1024, 384]
        # 2row 3column  --> input [2, 1024, 192],output [2, 256, 384]
        x23 = x22
        x23= self.patchmerg23(x23, x23.shape[0], x23.shape[2], x23.shape[1]) # [2, 256, 384]
        x23 = self.pos_drop(x23)        # [2, 256, 384]
        x23 = self.norm23(x23)                 # [2, 256, 384]
        x23, cam23 = self.eyeblock23(x23, x23.shape[0], x23.shape[2], x23.shape[1],complex_x) #[2, 256, 384]
        # 3row 3column  --> input [2, 1024, 96],output [2, 256, 192]
        x33= self.patchmerg33(x3, x3.shape[0], x3.shape[2], x3.shape[1]) # [2, 256, 192]
        x33 = self.pos_drop(x33)        # [2, 256, 192]
        x33 = self.norm33(x33)                 # [2, 256, 192]
        x33, cam33 = self.eyeblock33(x33, x33.shape[0], x33.shape[2], x33.shape[1],complex_x) #[2, 256, 192]
        # fourth row --> 64
        # 4row 3column--> input [2, 1024, 96], output [2, 1024, 96]  4row has no 1 2 column
        x4 = x3.view(x3.shape[0], int(math.sqrt(x3.shape[1])),int(math.sqrt(x3.shape[1])),x3.shape[2]) # x4[2, 32, 32, 96]
        x4 = x4.permute(0, 3, 1, 2).contiguous()  # [2, 96, 32, 32]
        x4 = self.conv3(x4)                       # [2, 48, 32, 32]
        x4 = x4.view(x4.shape[0], int(x4.shape[2])*int(x4.shape[3]), x4.shape[1]) 
        x4= self.patchmerg4(x4, x4.shape[0], x4.shape[2], x4.shape[1])             # [2, 256, 96]
        x4= self.pos_drop(x4)        # [2, 256, 96]
        x4 = self.norm4(x4)                # [2, 256, 96]
        x4, cam4 = self.eyeblock4(x4, x4.shape[0], x4.shape[2], x4.shape[1],complex_x)  # [2, 256, 96]

        # extra column   --> x
        # 1row 4column  --> input [2, 1024, 384],output [2, 256, 768]
        if int(depth) >= 4:
            x14 = x13
            x14= self.patchmerg14(x14, x14.shape[0], x14.shape[2], x14.shape[1]) # [2, 256, 768]
            x14 = self.pos_drop(x14)        # [2, 256, 768]
            x14 = self.norm14(x14)                 # [2, 256, 768]
            x14, cam14 = self.eyeblock14(x14, x14.shape[0], x14.shape[2], x14.shape[1],complex_x) #[2, 256, 768]
            # 2row 4column  --> input [2, 256, 384],output [2, 64, 768]
            x24 = x23
            x24= self.patchmerg24(x24, x24.shape[0], x24.shape[2], x24.shape[1]) # [2, 64, 768]
            x24 = self.pos_drop(x24)        # [2, 64, 768]
            x24 = self.norm24(x24)                 # [2, 64, 768]
            x24, cam24 = self.eyeblock24(x24, x24.shape[0], x24.shape[2], x24.shape[1],complex_x) #[2, 64, 768]
             # 3row 4column  --> input [2, 256, 192],output [2, 64, 384]
            x34 = x33
            x34= self.patchmerg34(x34, x34.shape[0], x34.shape[2], x34.shape[1]) # [2, 64, 384]
            x34 = self.pos_drop(x34)        # [2, 64, 384]
            x34 = self.norm34(x34)                 # [2, 64, 384]
            x34, cam34 = self.eyeblock34(x34, x34.shape[0], x34.shape[2], x34.shape[1],complex_x) #[2, 64, 384]
            # 4row 4column  --> input [2, 1024, 96],output [2, 64, 192]
            x44 = x4
            x44= self.patchmerg44(x44, x44.shape[0], x44.shape[2], x44.shape[1]) # [2, 64, 192]
            x44 = self.pos_drop(x44)        # [2, 64, 192]
            x44 = self.norm44(x44)                 # [2, 64, 192]
            x44, cam44 = self.eyeblock44(x44, x44.shape[0], x44.shape[2], x44.shape[1],complex_x) #[2, 64, 192]
            # 5th row
            # 5row 4column--> input [2, 1024, 96], output [2, 64, 96]  5row has no 1 2 3 column
            x5 = x4.view(x4.shape[0], int(math.sqrt(x4.shape[1])),int(math.sqrt(x4.shape[1])),x4.shape[2]) # x5[2, 16, 16, 96]
            x5 = x5.permute(0, 3, 1, 2).contiguous()  # [2, 96, 16, 16]
            x5 = self.conv3(x5)                       # [2, 48, 16, 16]
            x5 = x5.view(x5.shape[0], int(x5.shape[2])*int(x5.shape[3]), x5.shape[1]) # [2, 256, 48]
            x5= self.patchmerg5(x5, x5.shape[0], x5.shape[2], x5.shape[1])             # [2, 64, 96]
            x5= self.pos_drop(x5)        # [2, 64, 96]
            x5 = self.norm5(x5)
            x5, cam5 = self.eyeblock5(x5, x5.shape[0], x5.shape[2], x5.shape[1],complex_x)  # [2, 64, 96]
        
            # fuse
            transout = self.Transout_fuse_depth4(x1, x12, x13, x14, x2, x22, x23, x24, x3, x33, x34, x4, x44, x5) # [2, 96, 8, 8]
            fusecamone, fusecamsix= self.Cam_Fuse_Depth4(cam1, cam12, cam13, cam14, cam2, cam22, cam23, cam24, cam3, cam33, cam34, cam4, cam44, cam5) # CAM_Fuse
        else: # only depth = 3
            transout = self.Transout_fuse_depth3(x1, x12, x13, x2, x22, x23, x3, x33, x4)  # [2, 96, 8, 8]
            fusecamone, fusecamsix = self.Cam_Fuse_Depth3(cam1, cam12, cam13, cam2, cam22, cam23, cam3, cam33, cam4)  # CAM_Fuse

        return transout,fusecamone, fusecamsix


class CAM_Fuse_depth3(nn.Module):  # -->[1 or 2, 6, 512, 512]
    def __init__(self, config):
        super(CAM_Fuse_depth3, self).__init__()
        if config.patch_size == 8:
            C = 4
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=C/2)
        elif config.patch_size == 4:
            C = 1
            self.up1 =  nn.AvgPool2d(2, stride=2)
        elif config.patch_size == 16:
            C = 1
            self.up1 =  nn.AvgPool2d(2, stride=2)
            
        self.up12 = nn.UpsamplingBilinear2d(scale_factor=2*C)
        self.up13 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up14 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2*C)
        self.up22 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up23 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up24 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up33 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up34 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up44 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=128*C)            
        self.return_avgCAM = Show_Avg_CAM_Mapping()

    def forward(self,cam1, cam12, cam13, cam2, cam22, cam23, cam3, cam33, cam4):  #
        cam1 = self.up1(cam1)
        cam12 = self.up12(cam12)
        cam13 = self.up13(cam13)
        cam2 = self.up2(cam2)
        cam22 = self.up22(cam22)
        cam23 = self.up23(cam23)
        cam3 = self.up3(cam3)
        cam33 = self.up33(cam33)
        cam4 = self.up4(cam4)


        cam_all = torch.stack((cam1, cam12, cam13, cam2, cam22, cam23,  cam3, cam33,  cam4)) # [9, 2, 6, 128, 128]
        cam_all_one = torch.mean(cam_all, dim=2)  # [14, 2, 512, 512]
        cam_all_one = cam_all_one.sum(0)  # # [2, 512, 512]
        cam_all_six = torch.mean(cam_all, dim=0)  # [2, 6, 512, 512]

        camout = self.return_avgCAM(cam_all_one, "multi")  # [2, 512, 512]
        return cam_all_one, cam_all_six

class CAM_Fuse_depth4(nn.Module): # -->[1 or 2, 6, 512, 512]
    def __init__(self,config):
        super(CAM_Fuse_depth4, self).__init__()
        if config.patch_size == 8:
            C = 4
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=C/2)
        elif config.patch_size == 4:
            C = 1
            self.up1 =  nn.AvgPool2d(2, stride=2)
        elif config.patch_size == 16:
            C = 1
            self.up1 =  nn.AvgPool2d(2, stride=2)
            
        self.up12 = nn.UpsamplingBilinear2d(scale_factor=2*C)
        self.up13 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up14 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2*C)
        self.up22 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up23 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up24 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=8*C)
        self.up33 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up34 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=32*C)
        self.up44 = nn.UpsamplingBilinear2d(scale_factor=128*C)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=128*C)            
        self.return_avgCAM = Show_Avg_CAM_Mapping()
        
        
    def forward(self, cam1, cam12, cam13, cam14, cam2, cam22, cam23, cam24, cam3, cam33, cam34, cam4, cam44, cam5): #
        
        cam1 = self.up1(cam1)
        cam12 = self.up12(cam12)
        cam13 = self.up13(cam13)
        cam14 = self.up14(cam14)
        cam2 = self.up2(cam2)
        cam22 = self.up22(cam22)
        cam23 = self.up23(cam23)
        cam24 = self.up24(cam24)
        cam3 = self.up3(cam3)
        cam33 = self.up33(cam33)
        cam34 = self.up34(cam34)
        cam4 = self.up4(cam4)
        cam44 = self.up44(cam44)
        cam5 = self.up5(cam5)    # [1or2, 6, 512, 512]


        cam_all = torch.stack((cam1, cam12, cam13, cam14, cam2, cam22, cam23, cam24, cam3, cam33, cam34, cam4, cam44, cam5))
        cam_all_one = torch.mean(cam_all, dim=2)  # [14, 2, 512, 512]
        cam_all_one = cam_all_one.sum(0) # # [2, 512, 512]
        cam_all_six = torch.mean(cam_all, dim=0) # [2, 6, 512, 512]
        #cam_all = cam1 + cam12 + cam13 + cam14 + cam2 + cam22 + cam23 + cam24 + cam3 + cam33 + cam34 + cam4 + cam44 + cam5 
        #print("cam_all",cam_all.shape)
        #cam_avg = (1/14) * cam_all
        camout = self.return_avgCAM(cam_all_one, "multi") # [2, 512, 512]
        #print("cam_avg",cam_all.shape)
        return cam_all_one,cam_all_six
        
class Transout_fuse_depth3(nn.Module):     
    def __init__(self,config):
        super(Transout_fuse_depth3, self).__init__()
        C = config.embed_dim
        self.down1 = nn.Conv2d(C,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down12 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down13 = nn.Conv2d(C*4,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down2 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down22 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down23 = nn.Conv2d(C*4,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down3 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down33 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down4 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        
        self.up12 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up13 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up22 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up23 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up33 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        
        self.down = nn.Conv2d(C*9,C,kernel_size=3,stride=1,padding = 1,bias=False)
 
    def forward(self, x1, x12, x13, x2, x22, x23, x3, x33, x4):
        '''
        print (x1.shape)  # [2, 1024, 24]
        print (x12.shape) # [2, 256, 48]
        print (x13.shape) # [2, 64, 96]
        print (x2.shape)  # [2, 256, 24]
        print (x22.shape) # [2, 64, 48]
        print (x23.shape) # [2, 16, 96]
        print (x3.shape)  # [2, 64, 24]
        print (x33.shape) # [2, 16, 48]
        print (x4.shape)  # [2, 16, 24]
        '''
    
    
        transout1 = x1.permute(0, 2, 1).contiguous()    
        transout12 = x12.permute(0, 2, 1).contiguous()    
        transout13 = x13.permute(0, 2, 1).contiguous()    
        transout2 = x2.permute(0, 2, 1).contiguous()    
        transout22 = x22.permute(0, 2, 1).contiguous() 
        transout23 = x23.permute(0, 2, 1).contiguous() 
        transout3 = x3.permute(0, 2, 1).contiguous() 
        transout33 = x33.permute(0, 2, 1).contiguous() 
        transout4 = x4.permute(0, 2, 1).contiguous() 

        
        transout1 = transout1.view(transout1.shape[0], transout1.shape[1], int(math.sqrt(transout1.shape[2])), int(math.sqrt(transout1.shape[2])))       # [2, 24, 32, 32]
        transout12 = transout12.view(transout12.shape[0], transout12.shape[1], int(math.sqrt(transout12.shape[2])), int(math.sqrt(transout12.shape[2]))) # [2, 48, 16, 16]
        transout13 = transout13.view(transout13.shape[0], transout13.shape[1], int(math.sqrt(transout13.shape[2])), int(math.sqrt(transout13.shape[2]))) # [2, 96, 8, 8]
        transout2 = transout2.view(transout13.shape[0], transout2.shape[1], int(math.sqrt(transout2.shape[2])), int(math.sqrt(transout2.shape[2])))      # [2, 24, 16, 16]
        transout22 = transout22.view(transout13.shape[0], transout22.shape[1], int(math.sqrt(transout22.shape[2])), int(math.sqrt(transout22.shape[2]))) # [2, 48, 8, 8]
        transout23 = transout23.view(transout23.shape[0], transout23.shape[1], int(math.sqrt(transout23.shape[2])), int(math.sqrt(transout23.shape[2]))) # [2, 96, 4, 4]
        transout3 = transout3.view(transout3.shape[0], transout3.shape[1], int(math.sqrt(transout3.shape[2])), int(math.sqrt(transout3.shape[2])))       # [2, 24, 8, 8]
        transout33 = transout33.view(transout33.shape[0], transout33.shape[1], int(math.sqrt(transout33.shape[2])), int(math.sqrt(transout33.shape[2]))) # [2, 48, 4, 4]
        transout4 = transout4.view(transout4.shape[0], transout4.shape[1], int(math.sqrt(transout4.shape[2])), int(math.sqrt(transout4.shape[2])))       # [2, 24, 4, 4]


        transout1 = self.down1(transout1) # [2, 96, 8, 8]
        transout12 = self.down12(transout12) # [2, 96, 8, 8]
        transout13 = self.down13(transout13) # [2, 96, 8, 8]
        transout2 = self.down2(transout2) # [2, 96, 8, 8]
        transout22 = self.down22(transout22) # [2, 96, 8, 8]
        transout23 = self.down23(transout23) # [2, 96, 8, 8]
        transout3 = self.down3(transout3) # [2, 96, 8, 8]
        transout33 = self.down33(transout33) # [2, 96, 8, 8]        
        transout4 = self.down4(transout4) # [2, 96, 8, 8]  
              
        transout12 = self.up12(transout12) # [2, 96, 8, 8]
        transout13 = self.up13(transout13) # [2, 96, 8, 8]
        transout2 = self.up2(transout2) # [2, 96, 8, 8]
        transout22 = self.up22(transout22) # [2, 96, 8, 8]
        transout23 = self.up23(transout23) # [2, 96, 8, 8]
        transout3 = self.up3(transout3) # [2, 96, 8, 8]
        transout33 = self.up33(transout33) # [2, 96, 8, 8]        
        transout4 = self.up4(transout4) # [2, 96, 8, 8]     
        '''
        print (transout1.shape)  # [2, 1024, 24]
        print (transout12.shape) # [2, 256, 48]
        print (transout13.shape) # [2, 64, 96]
        print (transout2.shape)  # [2, 256, 24]
        print (transout22.shape) # [2, 64, 48]
        print (transout23.shape) # [2, 16, 96]
        print (transout3.shape)  # [2, 64, 24]
        print (transout33.shape) # [2, 16, 48]
        print (transout4.shape)  # [2, 16, 24]     
        '''
        transout = torch.cat([transout1, transout12, transout13, transout2,transout22, transout23, transout3, transout33, transout4], 1) # [2, 480, 8, 8]
        transout = self.down(transout)  # [2, 24, 32, 32] or 32-->64
        return transout
        
class Transout_fuse_depth4(nn.Module):     
    def __init__(self,config):
        super(Transout_fuse_depth4, self).__init__()
        C = config.embed_dim
        self.down1 = nn.Conv2d(C,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down12 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down13 = nn.Conv2d(C*4,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down2 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down22 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down23 = nn.Conv2d(C*4,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down3 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down33 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down4 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down14 = nn.Conv2d(C*8,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down24 = nn.Conv2d(C*8,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down34 = nn.Conv2d(C*4,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down44 = nn.Conv2d(C*2,C,kernel_size=3,stride=1,padding = 1,bias=False)
        self.down5 = nn.Conv2d(C*1,C,kernel_size=3,stride=1,padding = 1,bias=False)
        
        self.up12 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up13 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up22 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up23 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up33 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up14 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up24 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up34 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up44 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=16)
        
        self.down = nn.Conv2d(C*14,C,kernel_size=3,stride=1,padding = 1,bias=False)
 
    def forward(self, x1, x12, x13, x14, x2, x22, x23, x24, x3, x33, x34, x4, x44, x5):
        '''
        print (x1.shape)  # [2, 1024, 24]
        print (x12.shape) # [2, 256, 48]
        print (x13.shape) # [2, 64, 96]
        print (x2.shape)  # [2, 256, 24]
        print (x22.shape) # [2, 64, 48]
        print (x23.shape) # [2, 16, 96]
        print (x3.shape)  # [2, 64, 24]
        print (x33.shape) # [2, 16, 48]
        print (x4.shape)  # [2, 16, 24]
        print (x14.shape) # [2, 64, 192]
        print (x24.shape)  # [2, 16, 192]
        print (x34.shape) # [2, 16, 96]
        print (x44.shape)  # [2, 16, 48]
        print (x5.shape)  # [2, 16, 24
        '''
    
    
        transout1 = x1.permute(0, 2, 1).contiguous()    
        transout12 = x12.permute(0, 2, 1).contiguous()    
        transout13 = x13.permute(0, 2, 1).contiguous()    
        transout2 = x2.permute(0, 2, 1).contiguous()    
        transout22 = x22.permute(0, 2, 1).contiguous() 
        transout23 = x23.permute(0, 2, 1).contiguous() 
        transout3 = x3.permute(0, 2, 1).contiguous() 
        transout33 = x33.permute(0, 2, 1).contiguous() 
        transout4 = x4.permute(0, 2, 1).contiguous() 
        transout14 = x14.permute(0, 2, 1).contiguous() 
        transout24 = x24.permute(0, 2, 1).contiguous() 
        transout34 = x34.permute(0, 2, 1).contiguous() 
        transout44 = x44.permute(0, 2, 1).contiguous() 
        transout5 = x5.permute(0, 2, 1).contiguous() 
        
        transout1 = transout1.view(transout1.shape[0], transout1.shape[1], int(math.sqrt(transout1.shape[2])), int(math.sqrt(transout1.shape[2])))       # [2, 24, 32, 32]
        transout12 = transout12.view(transout12.shape[0], transout12.shape[1], int(math.sqrt(transout12.shape[2])), int(math.sqrt(transout12.shape[2]))) # [2, 48, 16, 16]
        transout13 = transout13.view(transout13.shape[0], transout13.shape[1], int(math.sqrt(transout13.shape[2])), int(math.sqrt(transout13.shape[2]))) # [2, 96, 8, 8]
        transout2 = transout2.view(transout13.shape[0], transout2.shape[1], int(math.sqrt(transout2.shape[2])), int(math.sqrt(transout2.shape[2])))      # [2, 24, 16, 16]
        transout22 = transout22.view(transout13.shape[0], transout22.shape[1], int(math.sqrt(transout22.shape[2])), int(math.sqrt(transout22.shape[2]))) # [2, 48, 8, 8]
        transout23 = transout23.view(transout23.shape[0], transout23.shape[1], int(math.sqrt(transout23.shape[2])), int(math.sqrt(transout23.shape[2]))) # [2, 96, 4, 4]
        transout3 = transout3.view(transout3.shape[0], transout3.shape[1], int(math.sqrt(transout3.shape[2])), int(math.sqrt(transout3.shape[2])))       # [2, 24, 8, 8]
        transout33 = transout33.view(transout33.shape[0], transout33.shape[1], int(math.sqrt(transout33.shape[2])), int(math.sqrt(transout33.shape[2]))) # [2, 48, 4, 4]
        transout4 = transout4.view(transout4.shape[0], transout4.shape[1], int(math.sqrt(transout4.shape[2])), int(math.sqrt(transout4.shape[2])))       # [2, 24, 4, 4]
        transout14 = transout14.view(transout14.shape[0], transout14.shape[1], int(math.sqrt(transout14.shape[2])), int(math.sqrt(transout14.shape[2]))) # [2, 48, 8, 8]
        transout24 = transout24.view(transout24.shape[0], transout24.shape[1], int(math.sqrt(transout24.shape[2])), int(math.sqrt(transout24.shape[2]))) # [2, 96, 4, 4]
        transout34 = transout34.view(transout34.shape[0], transout34.shape[1], int(math.sqrt(transout34.shape[2])), int(math.sqrt(transout34.shape[2])))       # [2, 24, 8, 8]
        transout44 = transout44.view(transout33.shape[0], transout44.shape[1], int(math.sqrt(transout44.shape[2])), int(math.sqrt(transout44.shape[2]))) # [2, 48, 4, 4]
        transout5 = transout5.view(transout5.shape[0], transout5.shape[1], int(math.sqrt(transout5.shape[2])), int(math.sqrt(transout5.shape[2])))       # [2, 24, 4, 4]

        transout1 = self.down1(transout1) # [2, 96, 8, 8]
        transout12 = self.down12(transout12) # [2, 96, 8, 8]
        transout13 = self.down13(transout13) # [2, 96, 8, 8]
        transout2 = self.down2(transout2) # [2, 96, 8, 8]
        transout22 = self.down22(transout22) # [2, 96, 8, 8]
        transout23 = self.down23(transout23) # [2, 96, 8, 8]
        transout3 = self.down3(transout3) # [2, 96, 8, 8]
        transout33 = self.down33(transout33) # [2, 96, 8, 8]        
        transout4 = self.down4(transout4) # [2, 96, 8, 8]  
        transout14 = self.down14(transout14) # [2, 96, 8, 8]
        transout24 = self.down14(transout24) # [2, 96, 8, 8]
        transout34 = self.down34(transout34) # [2, 96, 8, 8]
        transout44 = self.down44(transout44) # [2, 96, 8, 8]        
        transout5 = self.down5(transout5) # [2, 96, 8, 8]  
        '''
        print (transout1.shape)  # [2, 24, 64, 64]
        print (transout12.shape) # [2, 24, 32, 32]
        print (transout13.shape) # [2, 24, 16, 16]
        print (transout2.shape)  # [2, 24, 32, 32]
        print (transout22.shape) # [2, 24, 16, 16]
        print (transout23.shape) # [2, 24, 8, 8]
        print (transout3.shape)  # [2, 24, 16, 16]
        print (transout33.shape) # [2, 24, 8, 8]
        print (transout4.shape)  # [2, 24, 8, 8]
        print (transout14.shape) # [2, 24, 8, 8]
        print (transout24.shape) # [2, 24, 4, 4]
        print (transout34.shape) # [2, 24, 4, 4]
        print (transout44.shape) # [2, 24, 4, 4]
        print (transout5.shape)  # [2, 24, 4, 4]     
        '''
        transout12 = self.up12(transout12) # [2, 96, 8, 8]
        transout13 = self.up13(transout13) # [2, 96, 8, 8]
        transout2 = self.up2(transout2) # [2, 96, 8, 8]
        transout22 = self.up22(transout22) # [2, 96, 8, 8]
        transout23 = self.up23(transout23) # [2, 96, 8, 8]
        transout3 = self.up3(transout3) # [2, 96, 8, 8]
        transout33 = self.up33(transout33) # [2, 96, 8, 8]        
        transout4 = self.up4(transout4) # [2, 96, 8, 8]     
        transout14 = self.up14(transout14) # [2, 96, 8, 8]
        transout24 = self.up24(transout24) # [2, 96, 8, 8]
        transout34 = self.up34(transout34) # [2, 96, 8, 8]
        transout44 = self.up44(transout44) # [2, 96, 8, 8]        
        transout5 = self.up5(transout5) # [2, 96, 8, 8]   
        '''
        print (transout1.shape)  # [2, 1024, 24]
        print (transout12.shape) # [2, 256, 48]
        print (transout13.shape) # [2, 64, 96]
        print (transout2.shape)  # [2, 256, 24]
        print (transout22.shape) # [2, 64, 48]
        print (transout23.shape) # [2, 16, 96]
        print (transout3.shape)  # [2, 64, 24]
        print (transout33.shape) # [2, 16, 48]
        print (transout4.shape)  # [2, 16, 24]     
        '''
        transout = torch.cat([transout1, transout12, transout13, transout2,transout22, transout23, transout3, transout33, transout4, transout14, transout24, transout34, transout44, transout5], 1) # [2, 480, 8, 8]
        transout = self.down(transout)  # [2, 24, 32, 32]

        return transout





class Eyetransblock1(nn.Module):     # INPUT [2, 16384, 96]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock1, self).__init__()
        self.dim = config.embed_dim
        C = int(1*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C,config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre1
        global Cam_correction
        Cam_Centre1 = [64,64]
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]
        # eye
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre1,complex_x) # [2, 128, 128, 96]
        Cam_correction = torch.zeros([attn.shape[0], 6, attn.shape[2], attn.shape[3]]).cuda()
        cam, cam_centre= self.return_CAM(attn,1, Cam_correction) # [2, 6, 128, 128] 
        
        Cam_correction = cam
        Cam_Centre1 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x,cam

class Eyetransblock12(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock12, self).__init__()
        self.dim = config.embed_dim
        C = int(2*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C,config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre1
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre1[0] = int(Cam_Centre1[0])/atlr
        Cam_Centre1[1] = int(Cam_Centre1[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre1,complex_x) # [2, 64, 64, 192]
        cam, cam_centre = self.return_CAM(attn,12,Cam_correction)
        Cam_correction = cam
        Cam_Centre1 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam

class Eyetransblock13(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock13, self).__init__()
        self.dim = config.embed_dim
        C = int(4*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre1
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre1[0] = int(Cam_Centre1[0])/atlr
        Cam_Centre1[1] = int(Cam_Centre1[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre1,complex_x) # [2, 32, 32, 384]
        cam, cam_centre = self.return_CAM(attn,13,Cam_correction)
        Cam_correction = cam
        Cam_Centre1 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
        
class Eyetransblock2(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Eyetransblock2, self).__init__()
        self.dim = config.embed_dim
        C = int(1*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
        # cam cam_input_layers
        self.down_cam = nn.Conv2d(cam_input_layers, cam_input_layers, kernel_size = 3, stride = 2,padding=1)
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre2
        global Cam_correction
        Cam_Centre2 = [32,32]
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre2,complex_x) # [2, 64, 64, 96]
        cam, cam_centre = self.return_CAM(attn,2,Cam_correction)
        Cam_correction = cam
        Cam_Centre2 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)

        x = x + self.drop_path(x_mlp) # [2, 16384, 96]
        return x,cam

class Eyetransblock22(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock22, self).__init__()
        self.dim = config.embed_dim
        C = int(2*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre2
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre2[0] = int(Cam_Centre2[0])/atlr
        Cam_Centre2[1] = int(Cam_Centre2[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre2,complex_x) # [2, 32, 32, 192]
        cam, cam_centre = self.return_CAM(attn,22,Cam_correction)
        Cam_correction = cam
        Cam_Centre2 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
        
class Eyetransblock23(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock23, self).__init__()
        self.dim = config.embed_dim
        C = int(4*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre2
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre2[0] = int(Cam_Centre2[0])/atlr
        Cam_Centre2[1] = int(Cam_Centre2[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre2,complex_x) # [2, 16, 16, 384]
        cam, cam_centre = self.return_CAM(attn,23,Cam_correction)
        Cam_correction = cam
        Cam_Centre2 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
        
class Eyetransblock3(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock3, self).__init__()
        self.dim = config.embed_dim
        C = int(1*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre3
        global Cam_correction
        Cam_Centre3 = [16,16]
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre3,complex_x) # [2, 32, 32, 96]
        cam, cam_centre = self.return_CAM(attn,3, Cam_correction)
        Cam_correction = cam
        Cam_Centre3 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
        
class Eyetransblock33(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock33, self).__init__()
        self.dim = config.embed_dim
        C = int(2*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre3
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre3[0] = int(Cam_Centre3[0])/atlr
        Cam_Centre3[1] = int(Cam_Centre3[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre3,complex_x) # [2, 16, 16, 192]
        cam, cam_centre = self.return_CAM(attn,33,Cam_correction)
        Cam_correction = cam
        Cam_Centre3 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
        
class Eyetransblock4(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, dim = 96,input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock4, self).__init__()
        self.dim = config.embed_dim
        C = int(1*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre4
        global Cam_correction
        Cam_Centre4 = [8,8]
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre4,complex_x) #[2, 16, 16, 96]
        cam, cam_centre = self.return_CAM(attn,4,Cam_correction)
        Cam_correction = cam
        Cam_Centre4 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
class Eyetransblock14(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock14, self).__init__()
        self.dim = config.embed_dim
        C = int(8*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre1
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre1[0] = int(Cam_Centre1[0])/atlr
        Cam_Centre1[1] = int(Cam_Centre1[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre1,complex_x) # [2, 16, 16, 768]
        cam, cam_centre = self.return_CAM(attn,14,Cam_correction)
        Cam_correction = cam
        Cam_Centre1 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
class Eyetransblock24(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock24, self).__init__()
        self.dim = config.embed_dim
        C = int(8*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre2
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre2[0] = int(Cam_Centre2[0])/atlr
        Cam_Centre2[1] = int(Cam_Centre2[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre2,complex_x) # [2, 8, 8, 768]
        cam, cam_centre = self.return_CAM(attn,24,Cam_correction)
        Cam_correction = cam
        Cam_Centre2 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
class Eyetransblock34(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock34, self).__init__()
        self.dim = config.embed_dim
        C = int(4*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre3
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre3[0] = int(Cam_Centre3[0])/atlr
        Cam_Centre3[1] = int(Cam_Centre3[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre3,complex_x) # [2, 8, 8, 384]
        cam, cam_centre = self.return_CAM(attn,34,Cam_correction)
        Cam_correction = cam
        Cam_Centre3 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
class Eyetransblock44(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock44, self).__init__()
        self.dim = config.embed_dim
        C = int(2*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre4
        global Cam_correction
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        Cam_Centre4[0] = int(Cam_Centre4[0])/atlr
        Cam_Centre4[1] = int(Cam_Centre4[1])/atlr
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre4,complex_x) # [2, 8, 8, 192]
        cam, cam_centre = self.return_CAM(attn,44,Cam_correction)
        Cam_correction = cam
        Cam_Centre4 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         
        return x ,cam
class Eyetransblock5(nn.Module):     # INPUT [2, 4096, 192]
    def __init__(self, config, input_resolution = 512*512,drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super(Eyetransblock5, self).__init__()
        self.dim = config.embed_dim
        C = int(1*(self.dim))
        self.input_resolution = input_resolution
        self.num_heads = config.transformer.num_heads
        self.Eye_Center = Eye_Center(C, config)
        self.Channel_CAM = Channel_CAM()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm =  nn.LayerNorm(C)
        self.conv1 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=1,dilation=1)
        self.conv2 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=3,dilation=3)
        self.conv3 = nn.Conv2d(C,int(C/3),kernel_size = 3 , stride = 1,padding=5,dilation=5)
        self.down = nn.Conv2d(C*2,C,kernel_size = 3 , stride = 1,padding=1)
        # MLP
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["drop_rate"])
        self.fc1 = Linear(C, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"],C)
        self.return_CAM = Class_Activation_Mapping()
    def forward(self, x, B, C, L,complex_x):
        global Cam_Centre5
        global Cam_correction
        Cam_Centre5 = [4,4]
        x_0 = self.norm(x) #[2, 16384, 96]
        x_0 = x_0.view(x_0.shape[0], int(math.sqrt(x_0.shape[1])),int(math.sqrt(x_0.shape[1])),x_0.shape[2]) #[2, 128, 128, 96]

        # eye
        x_eye, attn = self.Eye_Center(x_0, Cam_Centre5,complex_x) # [2, 8, 8, 96]
        cam, cam_centre = self.return_CAM(attn,5,Cam_correction)
        Cam_correction = cam
        Cam_Centre5 = cam_centre
        x_eye = x_eye.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        # channel 
        #x_cam = self.Channel_CAM(x_0)
        # conv
        x_0 = x_0.permute(0, 3, 1, 2).contiguous() # [2, 96, 128, 128]
        x_1 = self.conv1(x_0)         # [2, 32, 128, 128]
        x_2 = self.conv2(x_0)         # [2, 32, 128, 128]
        x_3 = self.conv3(x_0)         # [2, 32, 128, 128]
        x_conv = torch.cat((x_1,x_2,x_3),1) #  concat [2, 96, 128, 128])
        #fuse
        #x_all = torch.cat( (x_eye, x_cam, x_conv) ,1) # [2, 288, 128, 128]
        x_all = torch.cat( (x_eye, x_conv) ,1)
        x_all = self.down(x_all)      # [2, 96, 128, 128]
        x_all = x_all.permute(0, 2, 3, 1).contiguous()  # [2, 128, 128 96]
        x_all = x_all.view(x_all.shape[0], int(x_all.shape[1]*x_all.shape[2]),x_all.shape[3]) # [2, 16384, 96]
        x = x + self.drop_path(x_all)  # after,add eye move and channle attention [2, 16384, 96]
        x = self.norm(x) # [2, 16384, 96]
        #MLP
        x_mlp = self.fc1(x)
        x_mlp = self.act_fn(x_mlp)
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)
        
        x = x + self.drop_path(x_mlp) # [2, 16384, 96]         

        return x ,cam

class PatchMerging12(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging12, self).__init__()
        C = int(2*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging13(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging13, self).__init__()
        C = int(4*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging22(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging22, self).__init__()
        C = int(2*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging23(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging23, self).__init__()
        C = int(4*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging3(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging3, self).__init__()
        C = int(1*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging33(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging33, self).__init__()
        C = int(2*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging4(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging4, self).__init__()
        C = int(1*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging14(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging14, self).__init__()
        C = int(8*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging24(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging24, self).__init__()
        C = int(8*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging34(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging34, self).__init__()
        C = int(4*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging44(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging44, self).__init__()
        C = int(2*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x
class PatchMerging5(nn.Module):     # [2, 128, 128, 96] --> [2, (128/2)* (128/2), 96*4/2]
    def __init__(self, config):
        super(PatchMerging5, self).__init__()
        C = int(1*(config.embed_dim))
        self.reduction = nn.Linear(4 * int(config.embed_dim), 2 * int(config.embed_dim), bias=False)
        self.norm = nn.LayerNorm(2*C)
        self.reduction = nn.Linear(2*C, C, bias=False)
 
    def forward(self, x, B, C, L):
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C) #[2, 128, 128, 96]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C [2, 64, 64, 96]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C x = [2, 64, 64, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C  [2, 4096, 384]
        x = self.norm(x)          # [2, 4096, 384]
        x = self.reduction(x)     # [2, 4096, 192]
        return x


def VisTrans(config_vit,img_size, num_classes,is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = VisionTransformer(config_vit,img_size, num_classes)
    return model

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
    'basical-transblock': configs.basical_transblock(),
}
