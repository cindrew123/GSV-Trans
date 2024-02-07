
import torch.nn as nn
import torch
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
np.set_printoptions(threshold=100000000)
torch.set_printoptions(threshold=100000000)
#eye_choose = "constant_no_zero"
eye_choose = "move_no_zero"
#eye_choose = "boundary_no_zero"
#eye_choose = "constant_center"
#eye_choose = "overflow_center"
#eye_choose = "boundary_center"
'''
class Eye_Center(nn.Module):
    def __init__(self, C, config):
        super(Eye_Center, self).__init__()
        size = [2,4,6,8]
        if eye_choose == "constant_center":
            self.eye_cent = Constant(C, config)
            self.eye_attention = Eye_Attention_Constant(C, config)
            self.padding_zero = "yes"
        if eye_choose == "overflow_center":
            self.eye_cent = Overflow(C, config)
            self.eye_attention = Eye_Attention_Overflow(C, config)
            self.padding_zero = "yes"
        if eye_choose == "boundary_center":
            self.eye_cent = Boundary(C,config)
            self.eye_attention = Eye_Attention_Boundary(C, config)
            self.padding_zero = "yes"
        if eye_choose == "constant_no_zero":
            self.eye_cent = Constant_no_zero(C,config) 
            self.padding_zero = "no"
        if eye_choose == "move_no_zero":
            self.eye_cent = move_no_zero(C,config)
            self.padding_zero = "no"



    def forward(self, x, Cam_Centre): # input [2, 128,128, 96]
        if self.padding_zero == "yes":
            xh2, xh4, xh6, xh8, x_0 = self.eye_cent(x, Cam_Centre)
            x_atten = self.eye_attention(x_0, xh2, xh4, xh6, xh8, Cam_Centre)
        else:
            x_atten = self.eye_cent(x, Cam_Centre)
        return x_atten
'''
class Eye_Center(nn.Module):
    def __init__(self, C, config):
        super(Eye_Center, self).__init__()
        size = [2,4,6,8]
        self.eye_cent_con = Constant_no_zero(C,config) 
        self.eye_cent_mov = move_no_zero(C,config)




    def forward(self, x, Cam_Centre,complex_x): # input [2, 128,128, 96]
        if complex_x == "constant_no_zero":
            x_atten = self.eye_cent_con(x, Cam_Centre)
        else:
            x_atten = self.eye_cent_mov(x, Cam_Centre)
        return x_atten
class move_no_zero(nn.Module):  # input [2, 32, 32, 768]
    def __init__(self, C, config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(move_no_zero, self).__init__()
        self.num_heads = config.transformer.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = config.eyedim // config.transformer.num_heads
        self.bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.atten_shift = Atten_Shift(C, config)
        self.get_attn_weight = Get_Weight(C)
        self.rate = config.eye_moverate

    def make_square(self, B):
        if B.shape[2] < B.shape[3]:
            BT = B.transpose(2, 3)  # [2, 6, 4, 12]-->[2, 6, 12, 4]
            square = torch.matmul(BT, B)  # [2, 6, 12, 12]
        else:
            BT = B.transpose(2, 3)
            square = torch.matmul(B, BT)
        return square

    def make_mask(self, square_B1, B):
        if B.shape[2] < B.shape[3]:
            window_size = B.shape[2]
        else:
            window_size = B.shape[3]
        attn_mask = torch.zeros(1, 1, square_B1.shape[2], square_B1.shape[3])  # [2, 6, 12, 12]
        H = square_B1.shape[2]
        for i in range(H):
            for j in range(H):  # H = W,because it is a square
                k_i = (i // window_size)
                k_j = (j // window_size)
                if abs(k_i - k_j) > 1:
                    attn_mask[:, :, i, j] = -500
        # print (attn_mask)
        return attn_mask

    def forward(self, x, Cam_Centre):
        attn_weight = self.get_attn_weight(x, self.num_heads, self.bias, self.scale)
        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 32, 32, 6] -- >  [2, 6, 32, 32]
        h = int((x.shape[2]) / 4)
        x_center = int(Cam_Centre[0])
        y_center = int(Cam_Centre[1])

        # Reduce size : h<2 then dont reduce, if want to reduce the shape, h>2
        if h > 2:  # [2, 6, 32, 32] -- >[2, 6, 16, 16]
            x0 = x
            avgpool = nn.AvgPool2d(2, stride=2)
            x0 = avgpool(x0)
            h0 = int((x0.shape[2]) / 4)
            ruduce = "ture"
        else:  # if feature map is too small ,it can not be smaller
            x0 = x
            h0 = h
            ruduce = "false"

        # Decide where to focus; D keep the resolution,D use x to cut, B and C use x0 to cut
        if x_center < 2*h:
            if y_center < 2*h:
                move_mode = "up_left"
            else:
                move_mode = "down_left"
        else:
            if y_center < 2*h:
                move_mode = "up_right"
            else:
                move_mode = "down_right"
        # cut
        if move_mode == "up_left":
            D = x[:, :, 0:3*h, 0:3*h].contiguous()
            B = x0[:, :, 3*h0:, :].contiguous()
            C = x0[:, :, 0:3*h0, 3*h0:].contiguous()
        elif move_mode == "up_right":
            D = x[:, :, h:, 0:3*h].contiguous()
            B = x0[:, :, 0:h0, :].contiguous()
            C = x0[:, :, h0:, 3*h0:].contiguous()
        elif move_mode == "down_right":
            D = x[:, :, h:, h:].contiguous()
            B = x0[:, :, 0:h0, :].contiguous()
            C = x0[:, :, h0:, 0:h0].contiguous()
        elif move_mode == "down_left":
            D = x[:, :, 0:3*h, 0:3*h].contiguous()
            B = x0[:, :, 3*h0:, :].contiguous()
            C = x0[:, :, 0:3*h0, 0:h0].contiguous()

        #print (D.shape, B.shape, C.shape) # torch.Size([2, 6, 24, 24]) torch.Size([2, 6, 4, 16]) torch.Size([2, 6, 12, 4])
        # B X transpose(B) [2, 6, 4, 12] -- > [2, 12, 12, 6]
        square_B = self.make_square(B)
        square_C = self.make_square(C)
        square_D = D
        #print (square_B.shape, square_C.shape) #torch.Size([2, 6, 16, 16]) torch.Size([2, 6, 12, 12])
        mask_B = self.make_mask(square_B, B)
        mask_C = self.make_mask(square_C, C)
        mask_D = torch.zeros(1, 1, D.shape[2], D.shape[3])
        # Rectangular as qkv
        qkv_B = B.permute(0, 2, 3, 1).contiguous()  # [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_B = qkv_B.view(qkv_B.shape[0], qkv_B.shape[1] * qkv_B.shape[2], qkv_B.shape[3])  # [2, 12, 4, 6] -- > [2, 48, 6]
        qkv_C = C.permute(0, 2, 3, 1).contiguous()  # [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_C = qkv_C.view(qkv_C.shape[0], qkv_C.shape[1] * qkv_C.shape[2], qkv_C.shape[3])
        qkv_D = D.permute(0, 2, 3, 1).contiguous()
        qkv_D = qkv_D.view(qkv_D.shape[0], qkv_D.shape[1] * qkv_D.shape[2], qkv_D.shape[3])
        # calculate attn (attn+mask)
        attn_B = self.atten_shift(qkv_B, B.shape[2], mask_B, self.num_heads, self.bias, self.scale, ruduce)
        attn_C = self.atten_shift(qkv_C, C.shape[2], mask_C, self.num_heads, self.bias, self.scale, ruduce)
        attn_D = self.atten_shift(qkv_D, D.shape[2], mask_D, self.num_heads, self.bias, self.scale, ruduce="false")
        #print (attn_D.shape, attn_B.shape, attn_C.shape) #torch.Size([2, 6, 24, 24]) torch.Size([2, 6, 8, 32]) torch.Size([2, 6, 24, 8]
        # reverse the cut
        attn_all = torch.zeros(x.shape).cuda()
        h0 = h # in self.atten_shift, already restored shape size
        if move_mode == "up_left":
            attn_all[:, :, 0:3*h, 0:3*h] = attn_D
            attn_all[:, :, 3*h0:, :] = attn_B
            attn_all[:, :, 0:3*h0, 3*h0:] = attn_C
        elif move_mode == "up_right":
            attn_all[:, :, h:, 0:3*h] = attn_D
            attn_all[:, :, 0:h0, :] = attn_B
            attn_all[:, :, h0:, 3*h0:] = attn_C
        elif move_mode == "down_right":
            attn_all[:, :, h:, h:] = attn_D
            attn_all[:, :, 0:h0, :] = attn_B
            attn_all[:, :, h0:, 0:h0] = attn_C
        elif move_mode == "down_left":
            attn_all[:, :, 0:3*h, 0:3*h] = attn_D
            attn_all[:, :, 3*h0:, :] = attn_B
            attn_all[:, :, 0:3*h0, 0:h0] = attn_C
        attn_all = attn_all.permute(0, 2, 3, 1).contiguous()
        return attn_all, attn_weight




class Constant_no_zero(nn.Module):  # input [2, 32, 32, 768]
    def __init__(self, C,config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Constant_no_zero, self).__init__()
        self.num_heads = config.transformer.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = config.eyedim // config.transformer.num_heads
        self.bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.atten_shift = Atten_Shift(C,config)
        self.get_attn_weight = Get_Weight(C) 
        self.rate = config.eye_moverate
    
    def make_square(self, B):
        if B.shape[2] < B.shape[3]:
            BT = B.transpose(2,3) #[2, 6, 4, 12]-->[2, 6, 12, 4]
            square = torch.matmul(BT, B) #[2, 6, 12, 12]
        else:
            BT = B.transpose(2,3)
            square = torch.matmul(B, BT)
        return square
        
    def make_mask(self, square_B1,B):
        if B.shape[2] < B.shape[3]:  
            window_size = B.shape[2]
        else:
            window_size = B.shape[3]
        attn_mask = torch.zeros(1,1, square_B1.shape[2],square_B1.shape[3]) # [2, 6, 12, 12]
        H = square_B1.shape[2]
        for i in range(H):
            for j in range(H): # H = W,because it is a square
                k_i = (i//window_size)
                k_j = (j//window_size)
                if abs(k_i-k_j) >1:
                        attn_mask[:, :, i, j] = -500
        #print (attn_mask)
        return attn_mask

    def forward(self, x, Cam_Centre): 
        attn_weight = self.get_attn_weight(x, self.num_heads, self.bias, self.scale)
        x = x.permute(0,3,1,2).contiguous() # [2, 32, 32, 6] -- >  [2, 6, 32, 32]
        h = int((x.shape[2]) / 4) 
        D = x[:, :, h:3*h, h:3*h].contiguous()
        # Reduce size : h<2 then dont reduce, if want to reduce the shape, h>2
        if h > 2:  # [2, 6, 32, 32] -- >[2, 6, 16, 16]
            x0 = x
            avgpool = nn.AvgPool2d(2, stride=2)
            x0 = avgpool(x0)
            h0 = int((x0.shape[2]) / 4)
            ruduce = "ture"
        else:  # if feature map is too small ,it can not be smaller
            x0 = x
            h0 = h
            ruduce = "false"
        # cut
        B1 = x0[:,:,0:h0, 0:3*h0].contiguous()
        B2 = x0[:,:,3*h0:, h0:].contiguous()
        #print (B1.shape, B2.shape) # [2, 6, 4, 12]
        C1 = x0[:,:,h0:, 0:h0].contiguous()
        C2 = x0[:,:,0:3*h0, 3*h0:].contiguous()
        #print (C1.shape, C2.shape) # [2, 6, 12, 4]
        # B X transpose(B) [2, 6, 4, 12] -- > [2, 12, 12, 6]
        square_B1 = self.make_square(B1)
        square_B2 = self.make_square(B2)
        square_C1 = self.make_square(C1)
        square_C2 = self.make_square(C2)
        square_D = D
        #print (square_B1.shape, square_B2.shape, square_C1.shape, square_C2.shape) # [2, 6, 12, 12]
        mask_B1 = self.make_mask(square_B1,B1)
        mask_B2 = self.make_mask(square_B2,B2)
        mask_C1 = self.make_mask(square_C1,C1)
        mask_C2 = self.make_mask(square_C2,C2)
        mask_D = torch.zeros(1, 1, D.shape[2], D.shape[3])
        # Rectangular as qkv 
        qkv_B1 = B1.permute(0,2,3,1).contiguous()# [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_B1 = qkv_B1.view(qkv_B1.shape[0], qkv_B1.shape[1] * qkv_B1.shape[2], qkv_B1.shape[3]) # [2, 12, 4, 6] -- > [2, 48, 6]
        qkv_C1 = C1.permute(0,2,3,1).contiguous()# [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_C1 = qkv_C1.view(qkv_C1.shape[0], qkv_C1.shape[1] * qkv_C1.shape[2], qkv_C1.shape[3])
        
        qkv_C2 = C2.permute(0,2,3,1).contiguous()# [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_C2 = qkv_C2.view(qkv_C2.shape[0], qkv_C2.shape[1] * qkv_C2.shape[2], qkv_C2.shape[3])
        
        qkv_B2 = B2.permute(0,2,3,1).contiguous()# [2, 6, 4, 12]--> [2, 12, 4, 6]
        qkv_B2 = qkv_B2.view(qkv_B2.shape[0], qkv_B2.shape[1] * qkv_B2.shape[2], qkv_B2.shape[3])
        
        qkv_D = D.permute(0,2,3,1).contiguous()
        qkv_D = qkv_D.view(qkv_D.shape[0], qkv_D.shape[1] * qkv_D.shape[2], qkv_D.shape[3])
        # calculate attn (attn+mask)
        attn_B1 = self.atten_shift(qkv_B1, B1.shape[2], mask_B1, self.num_heads, self.bias, self.scale, ruduce)
        attn_B2 = self.atten_shift(qkv_B2, B2.shape[2], mask_B2, self.num_heads, self.bias, self.scale, ruduce)
        attn_C1 = self.atten_shift(qkv_C1, C1.shape[2], mask_C1, self.num_heads, self.bias, self.scale, ruduce)
        attn_C2 = self.atten_shift(qkv_C2, C2.shape[2], mask_C2, self.num_heads, self.bias, self.scale, ruduce)
        attn_D = self.atten_shift(qkv_D, D.shape[2], mask_D, self.num_heads, self.bias, self.scale, ruduce= "false")
        #print (attn_B1.shape, attn_B2.shape, attn_C1.shape, attn_C2.shape , attn_D.shape) #[2, 6, 4, 12] [2, 6, 4, 12] [2, 6, 12, 4] [2, 6, 12, 4] [2, 6, 16, 16]

        # reverse the cut
        attn_all = torch.zeros(x.shape).cuda()
        h0 = h
        attn_all[:,:,0:h0, 0:3*h0] = attn_B1
        attn_all[:,:,3*h0:, h0:] = attn_B2
        attn_all[:,:,h0:, 0:h0] = attn_C1
        attn_all[:,:,0:3*h0, 3*h0:] = attn_C2
        attn_all[:, :, h:3*h, h:3*h] = attn_D
        attn_all = attn_all.permute(0, 2, 3, 1).contiguous()
        return attn_all, attn_weight


class Atten_Shift(nn.Module):
    def __init__(self, C, config, attn_drop=0., proj_drop=0.):
        super(Atten_Shift, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)
        self.num_heads = config.transformer.num_heads
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def norm_Zscore(self, x):
        x = x.cpu().detach().numpy()  # (2, 8, 8, 768)
        stand = StandardScaler()
        if x.shape[0] == 1:
            x_stand = x.reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand = stand.fit_transform(x_stand)
            out = x_stand.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            out = torch.from_numpy(out)
        else:
            x_stand1 = x[0, :].reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand2 = x[1, :].reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand1 = stand.fit_transform(x_stand1)
            x_stand2 = stand.fit_transform(x_stand2)
            out1 = x_stand1.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out2 = x_stand2.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out1 = torch.from_numpy(out1)
            out2 = torch.from_numpy(out2)
            out = torch.cat([out1, out2], dim=0)
        return out.cuda()
        
    def forward(self, x, h, mask, num_heads, bias, scale, ruduce):
        #print (x.shape) # [2, 48, 6]
        
        #x = self.Qkv(x) # [2, 48, 18]

        qkv = self.Qkv(x).reshape(x.shape[0], x.shape[1], 3, num_heads, x.shape[2] // num_heads).permute(2, 0, 3, 1,4)  # [2, 144, 6] -- >[3, 2, 3, 144, 2]
        #print (qkv.shape) # [2, 48, 6] -- >[3, 2, 3, 48, 2]
        q, k, v = qkv[0], qkv[1], qkv[2]  # q[2, 3, 144, 2], k [2, 3, 144, 2] , v [2, 3, 144, 2]
        #print (q.shape,k.shape,v.shape)
        q = q * scale
        if q.shape[0] == 1:
            attn = (q @ k.transpose(-2, -1))  # matrix multiplication  attn[2, 6, 1024, 1024]
            # relative_position_bias

            # attn+mask

            attn = self.softmax(attn)
            x = self.attn_drop(x)
            x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2]).contiguous()  # [2, 1024, 96]
        else:
            attn = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[2])).cuda()
            attn0 = (q[0, ...] @ k[0, ...].transpose(-2, -1))
            attn1 = (q[1, ...] @ k[1, ...].transpose(-2, -1))
            attn[0, ...] = self.softmax(attn0)
            attn[1, ...] = self.softmax(attn1)
            x = self.attn_drop(x)
            X = v
            X[0, ...] = (attn[0, ...] @ v[0, ...])
            X[1, ...] = (attn[1, ...] @ v[1, ...])
            x = X.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2]).contiguous()
        
        x = self.proj(x)
        x = self.proj_drop(x)
        #print (x.shape)
        x = x.view(x.shape[0], h, int((x.shape[1])/h), x.shape[2])
        x = x.permute(0, 3, 1, 2).contiguous()
        # reverse the shape
        if ruduce == "ture":
            x = self.up(x)
        
        # norm 
        x = self.norm_Zscore(x)
        
        return x

class Boundary(nn.Module):  # input [2, 128,128, 96] --size[32,64,96,128]
    def __init__(self, C, size=[2, 4, 6, 8]):
        super(Boundary, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)

    def forward(self, x, Cam_Centre):
        x_0 = x
        if x.shape[2] == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.up(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x_center = 2*int(Cam_Centre[0]) 
            y_center = 2*int(Cam_Centre[1])
        else:
            x_center = int(Cam_Centre[0]) 
            y_center = int(Cam_Centre[1])
        r = int((x.shape[1]) / 8)  # rate = 16/8 = 2
        x_add0 = x
        # Take out xh8, xh6, xh4, xh2
        # find xh2 center
        if x_center < r:
            xh2_xc = r
        elif x_center > 7*r:
            xh2_xc = 7 * r
        else:
            xh2_xc = x_center

        if y_center < r:
            xh2_yc = r
        elif y_center > 7*r:
            xh2_xc = 7 * r
        else:
            xh2_xc = y_center

        xh2 = x[:, xh2_xc - r:xh2_xc + r, xh2_xc - r:xh2_xc + r, :].contiguous()
        x_add0[:, xh2_xc - r:xh2_xc + r, xh2_xc - r:xh2_xc + r, :] = 0

        # find xh4 center
        if x_center < 2*r:
            xh4_xc = 2*r
        elif x_center > 6*r:
            xh4_xc = 6 * r
        else:
            xh4_xc = x_center

        if y_center < 2*r:
            xh4_yc = 2*r
        elif y_center > 6*r:
            xh4_yc = 6 * r
        else:
            xh4_yc = y_center

        xh4 = x_add0[:, xh4_xc - 2 * r:xh4_xc + 2 * r, xh4_yc - 2 * r:xh4_yc + 2 * r, :].contiguous()
        x_add0[:, xh4_xc - 2 * r:xh4_xc + 2 * r, xh4_yc - 2 * r:xh4_yc + 2 * r, :] = 0

        # find xh6 center
        if x_center < 3*r:
            xh6_xc = 3*r
        elif x_center > 5*r:
            xh6_xc = 5 * r
        else:
            xh6_xc = x_center

        if y_center < 3*r:
            xh6_yc = 3*r
        elif y_center > 5*r:
            xh6_yc = 5 * r
        else:
            xh6_yc = y_center

        xh6 = x_add0[:, xh6_xc - 3 * r:xh6_xc + 3 * r, xh6_yc - 3 * r:xh6_yc + 3 * r, :].contiguous()
        x_add0[:, xh4_xc - 2 * r:xh4_xc + 2 * r, xh4_yc - 2 * r:xh4_yc + 2 * r, :] = 0

        xh8 = x_add0

        xh2 = xh2.view(xh2.shape[0], xh2.shape[1] * xh2.shape[2], xh2.shape[3])  # [2, 16, 768]
        xh4 = xh4.view(xh4.shape[0], xh4.shape[1] * xh4.shape[2], xh4.shape[3])  # [2, 64, 768]
        xh6 = xh6.view(xh6.shape[0], xh6.shape[1] * xh6.shape[2], xh6.shape[3])  # [2, 144, 768]
        xh8 = xh8.view(xh8.shape[0], xh8.shape[1] * xh8.shape[2], xh8.shape[3])  # [2, 256, 768]
        return xh2, xh4, xh6, xh8, x_0 


class Eye_Attention_Boundary(nn.Module):
    def __init__(self, C, config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Eye_Attention_Boundary, self).__init__()
        self.num_heads = config.transformer.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = config.eyedim // config.transformer.num_heads
        self.bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.atten = Atten(C)
        self.rate = config.eye_moverate
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.get_attn_weight = Get_Weight(C) 
        Kernel = 1
        if Kernel == 1:
            Pad = 0
        else:
            Pad = 1

        self.conv96 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=Kernel, stride=1, padding=Pad, bias=False)
        self.conv192 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=Kernel, stride=1, padding=Pad,
                                 bias=False)
        self.conv384 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=Kernel, stride=1, padding=Pad,
                                 bias=False)
        self.conv768 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=Kernel, stride=1, padding=Pad,
                                 bias=False)

    def norm_maxmin(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return x

    def norm_Zscore(self, x):
        x = x.cpu().detach().numpy()  # (2, 8, 8, 768)
        stand = StandardScaler()
        if x.shape[0] == 1:
            x_stand = x.reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand = stand.fit_transform(x_stand)
            out = x_stand.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            out = torch.from_numpy(out)
        else:
            x_stand1 = x[0, :].reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand2 = x[1, :].reshape(x.shape[1] * x.shape[2], x.shape[3])
            x_stand1 = stand.fit_transform(x_stand1)
            x_stand2 = stand.fit_transform(x_stand2)
            out1 = x_stand1.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out2 = x_stand2.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out1 = torch.from_numpy(out1)
            out2 = torch.from_numpy(out2)
            out = torch.cat([out1, out2], dim=0)
        return out.cuda()

    def forward(self, x0, xh2, xh4, xh6, xh8, Cam_Centre):
        '''
        print("xh2 = ",xh2.shape)
        print("xh4 = ",xh4.shape)
        print("xh6 = ",xh6.shape)
        print("xh8 = ",xh8.shape)
        '''
        attn_weight = self.get_attn_weight(x0, self.num_heads, self.bias, self.scale)
        xh2 = self.atten(xh2, self.num_heads, self.bias, self.scale)  # [2, 256, 96]   16
        xh4 = self.atten(xh4, self.num_heads, self.bias, self.scale)  # [2, 1024, 96]  32
        xh6 = xh6.reshape(xh6.shape[0], int(math.sqrt(xh6.shape[1])), int(math.sqrt(xh6.shape[1])), xh6.shape[2])
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh6 = avgpool(xh6)
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        xh6 = xh6.reshape(xh6.shape[0], int((xh6.shape[1]) * (xh6.shape[2])), xh6.shape[3])
        xh6 = self.atten(xh6, self.num_heads, self.bias, self.scale)  # [2, 4096, 96]  64   [2, 32, 32, 96]
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        xh6 = self.up(xh6)  # [2, 96, 64, 64]
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        xh8 = xh8.reshape(xh8.shape[0], int(math.sqrt(xh8.shape[1])), int(math.sqrt(xh8.shape[1])), xh8.shape[2])
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh8 = avgpool(xh8)
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()
        xh8 = xh8.reshape(xh8.shape[0], int((xh8.shape[1]) * (xh8.shape[2])), xh8.shape[3])
        xh8 = self.atten(xh8, self.num_heads, self.bias, self.scale)  # [2, 4096, 96]  64   [2, 32, 32, 96]
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        xh8 = self.up(xh8)  # [2, 96, 64, 64]
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()

        # norm
        xh2 = self.norm_Zscore(xh2)
        xh4 = self.norm_Zscore(xh4)
        xh6 = self.norm_Zscore(xh6)
        xh8 = self.norm_Zscore(xh8)

        # Give the value of xh2 to xh4
        x_center = int(Cam_Centre[0])  # # first is the row (hang) index, second is column (lie) index
        y_center = int(Cam_Centre[1])
        r = int((xh8.shape[1]) / 8)
        # find xh6 center
        if x_center < 3*r:
            xh6_xc = 3*r
        elif x_center > 5*r:
            xh6_xc = 5 * r
        else:
            xh6_xc = x_center

        if y_center < 3*r:
            xh6_yc = 3*r
        elif y_center > 5*r:
            xh6_yc = 5 * r
        else:
            xh6_yc = y_center

        xh8[:, xh6_xc - 3 * r:xh6_xc + 3 * r, xh6_yc - 3 * r:xh6_yc + 3 * r, :] = xh6

        # find xh4 center
        if x_center < 2*r:
            xh4_xc = 2*r
        elif x_center > 6*r:
            xh4_xc = 6 * r
        else:
            xh4_xc = x_center

        if y_center < 2*r:
            xh4_yc = 2*r
        elif y_center > 6*r:
            xh4_yc = 6 * r
        else:
            xh4_yc = y_center

        xh8[:, xh4_xc - 2 * r:xh4_xc + 2 * r, xh4_yc - 2 * r:xh4_yc + 2 * r, :] = xh4

        # find xh2 center
        if x_center < r:
            xh2_xc = r
        elif x_center > 7*r:
            xh2_xc = 7 * r
        else:
            xh2_xc = x_center

        if y_center < r:
            xh2_yc = r
        elif y_center > 7*r:
            xh2_xc = 7 * r
        else:
            xh2_xc = y_center

        xh8[:, xh2_xc - r:xh2_xc + r, xh2_xc - r:xh2_xc + r, :] = xh2

        # Filling edge
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        if xh8.shape[1] == 96:
            xh8 = self.conv96(xh8)
        elif xh8.shape[1] == 192:
            cam_class = self.conv192(xh8)
        elif xh8.shape[1] == 384:
            xh8 = self.conv384(xh8)
        elif xh8.shape[1] == 768:
            xh8 = self.conv768(xh8)
        if x0.shape[2] == 4:
            avgpool = nn.AvgPool2d(2, stride=2)
            xh8 = avgpool(xh8)            
        
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()

        return xh8, attn_weight

class Overflow(nn.Module):  # input [2, 128,128, 96] --size[32,64,96,128]
    def __init__(self, C, size=[2, 4, 6, 8]):
        super(Overflow, self).__init__()
        #self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x, Cam_Centre):
        x_0 = x
        if x.shape[2] == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.up(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x_center = 2*int(Cam_Centre[0]) 
            y_center = 2*int(Cam_Centre[1])
        else:
            x_center = int(Cam_Centre[0]) 
            y_center = int(Cam_Centre[1])
        r = int((x.shape[1]) / 8)  # rate = 16/8 = 2
        # Fill 0 around x
        x_fill = torch.zeros([x.shape[0],x.shape[1]+ 6*r, x.shape[2]+6*r, x.shape[3]]).cuda()
        x_fill[:,3*r:(3+8)*r, 3*r:(3+8)*r, :] = x
        xc = int(x_center+3*r)
        yc = int(y_center+3*r)
        xh2 = x_fill[:,xc-r: xc+r, yc-r: yc+r, :].contiguous()
        xh4 = x_fill[:, xc - 2 * r:xc + 2 * r, yc - 2 * r:yc + 2 * r, :].contiguous()
        xh6 = x_fill[:, xc - 3 * r:xc + 3 * r, yc - 3 * r:yc + 3 * r, :].contiguous()
        # xh8 is different,it is constant, but cut xh6 ,and add 0
        x_fill[:, xc - 3 * r:xc + 3 * r, yc - 3 * r:yc + 3 * r, :] = 0
        xh8 =  x_fill[:, 3 * r:3 * r + 8 * r, 3 * r:3 * r + 8 * r, :].contiguous()

        # cut xh2 from xh4, cut xh4 from xh6 ,and add 0
        xh4[:, int(xh4.shape[1] / 2) - r:int(xh4.shape[1] / 2) + r, int(xh4.shape[1] / 2) - r:int(xh4.shape[1] / 2) + r,:] = 0  # add0 size = [2, 32, 32, 96]
        xh6[:, int(xh6.shape[1] / 2) - 2 * r:int(xh6.shape[1] / 2) + 2 * r, int(xh6.shape[1] / 2) - 2 * r:int(xh6.shape[1] / 2) + 2 * r, :] = 0  # add0 size = [2, 64, 64, 96]

        xh2 = xh2.view(xh2.shape[0], xh2.shape[1] * xh2.shape[2], xh2.shape[3])  # [2, 16, 768]
        xh4 = xh4.view(xh4.shape[0], xh4.shape[1] * xh4.shape[2], xh4.shape[3])  # [2, 64, 768]
        xh6 = xh6.view(xh6.shape[0], xh6.shape[1] * xh6.shape[2], xh6.shape[3])  # [2, 144, 768]
        xh8 = xh8.view(xh8.shape[0], xh8.shape[1] * xh8.shape[2], xh8.shape[3])  # [2, 256, 768]
        return xh2, xh4, xh6, xh8, x_0 

class Eye_Attention_Overflow(nn.Module):  
    def __init__(self, C, config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Eye_Attention_Overflow, self).__init__()
        self.num_heads = config.transformer.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = config.eyedim // config.transformer.num_heads
        self.bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.atten = Atten(C)
        self.rate = config.eye_moverate
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.get_attn_weight = Get_Weight(C) 
        Kernel=1
        if Kernel==1:
            Pad = 0
        else:
            Pad = 1
        
        self.conv96 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=Kernel, stride=1, padding=Pad, bias=False)
        self.conv192 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=Kernel, stride=1, padding=Pad, bias=False)
        self.conv384 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=Kernel, stride=1, padding=Pad, bias=False)
        self.conv768 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=Kernel, stride=1, padding=Pad, bias=False)
        
    def norm_maxmin(self,x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return x
    def norm_Zscore(self,x):
        x = x.cpu().detach().numpy() # (2, 8, 8, 768)
        stand = StandardScaler()
        if x.shape[0]  == 1:
            x_stand = x.reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand = stand.fit_transform(x_stand)
            out = x_stand.reshape(x.shape[0], x.shape[1], x.shape[2],  x.shape[3])
            out = torch.from_numpy(out)
        else:
            x_stand1 = x[0,:].reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand2 = x[1,:].reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand1 = stand.fit_transform(x_stand1)
            x_stand2 = stand.fit_transform(x_stand2)
            out1 = x_stand1.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out2 = x_stand2.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out1 = torch.from_numpy(out1)
            out2 = torch.from_numpy(out2)
            out = torch.cat([out1, out2], dim=0)
        return out.cuda()    
    

    def forward(self, x0, xh2, xh4, xh6, xh8, Cam_Centre):
        '''
        print("xh2 = ",xh2.shape)
        print("xh4 = ",xh4.shape)
        print("xh6 = ",xh6.shape)
        print("xh8 = ",xh8.shape)
        '''
        attn_weight = self.get_attn_weight(x0, self.num_heads, self.bias, self.scale)
        xh2 = self.atten(xh2, self.num_heads, self.bias, self.scale)  # [2, 256, 96]   16
        xh4 = self.atten(xh4, self.num_heads, self.bias, self.scale)  # [2, 1024, 96]  32
        xh6 = xh6.reshape(xh6.shape[0], int(math.sqrt(xh6.shape[1])), int(math.sqrt(xh6.shape[1])), xh6.shape[2])
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh6 = avgpool(xh6)
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        xh6 = xh6.reshape(xh6.shape[0], int((xh6.shape[1]) * (xh6.shape[2])), xh6.shape[3])
        xh6 = self.atten(xh6, self.num_heads, self.bias, self.scale)  # [2, 4096, 96]  64   [2, 32, 32, 96]
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        xh6 = self.up(xh6)  # [2, 96, 64, 64]
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        xh8 = xh8.reshape(xh8.shape[0], int(math.sqrt(xh8.shape[1])), int(math.sqrt(xh8.shape[1])), xh8.shape[2])
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh8 = avgpool(xh8)
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()
        xh8 = xh8.reshape(xh8.shape[0], int((xh8.shape[1]) * (xh8.shape[2])), xh8.shape[3])
        xh8 = self.atten(xh8, self.num_heads, self.bias, self.scale)  # [2, 4096, 96]  64   [2, 32, 32, 96]
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        xh8 = self.up(xh8)  # [2, 96, 64, 64]
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()
        
        # norm
        xh2 = self.norm_Zscore(xh2)
        xh4 = self.norm_Zscore(xh4)
        xh6 = self.norm_Zscore(xh6)
        xh8 = self.norm_Zscore(xh8)
        
        # Give the value of xh2 to xh4
        r = int((xh8.shape[1]) / 8)
        x_fill = torch.zeros([xh8.shape[0],xh8.shape[1]+ 6*r, xh8.shape[2]+6*r, xh8.shape[3]])
        x_center = int(Cam_Centre[0])
        y_center = int(Cam_Centre[1])
        xc = int(x_center+3*r)
        yc = int(y_center+3*r)
        # x_fill makes it easier to assign the value of xh2 to xh4, xh4 to xh6, and xh6 to xh8
        x_fill[:, 3 * r:3 * r + 8 * r, 3 * r:3 * r + 8 * r, :] =xh8
        x_fill[:, xc - 3 * r:xc + 3 * r, yc - 3 * r:yc + 3 * r, :] = xh6
        x_fill[:, xc - 2 * r:xc + 2 * r, yc - 2 * r:yc + 2 * r, :] = xh4
        x_fill[:,xc-r: xc+r, yc-r: yc+r, :] = xh2
        xh8 =  x_fill[:, 3 * r:3 * r + 8 * r, 3 * r:3 * r + 8 * r, :].contiguous()
        xh8 = xh8.cuda()

        # Filling edge
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        if xh8.shape[1] == 96:
            xh8 = self.conv96(xh8)
        elif xh8.shape[1] == 192:
            cam_class = self.conv192(xh8)
        elif xh8.shape[1] == 384:
            xh8 = self.conv384(xh8)
        elif xh8.shape[1] == 768:
            xh8 = self.conv768(xh8)
        if x0.shape[2] == 4:
            avgpool = nn.AvgPool2d(2, stride=2)
            xh8 = avgpool(xh8)        
            
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()
        
        return xh8, attn_weight










class Constant(nn.Module):  # input [2, 128,128, 96] --size[32,64,96,128]
    def __init__(self, C, size=[2, 4, 6, 8]):
        super(Constant, self).__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, Cam_Centre):
        x_0 = x
        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 128,128, 96] --> [2, 96, 128, 128]
        x = self.conv(x)
        if x.shape[2] == 4:
            x = self.up(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        xc = int((x.shape[1]) / 2)
        r = int((x.shape[1]) / 8)  # rate = 128/8 = 16
        #print("r = ",r)
        # h = 2
        xh2 = x[:, xc - r:xc + r, xc - r:xc + r, :].contiguous()  # [2, 32, 32, 96]
        xh4 = x[:, xc - 2 * r:xc + 2 * r, xc - 2 * r:xc + 2 * r, :].contiguous()  # [2, 64, 64, 96]
        xh6 = x[:, xc - 3 * r:xc + 3 * r, xc - 3 * r:xc + 3 * r, :].contiguous()  # [2, 96, 96, 96]
        xh8 = x
        # add 0
        xh4[:, int(xh4.shape[1] / 2) - r:int(xh4.shape[1] / 2) + r, int(xh4.shape[1] / 2) - r:int(xh4.shape[1] / 2) + r,
        :] = 0  # add0 size = [2, 32, 32, 96]
        xh6[:, int(xh6.shape[1] / 2) - 2 * r:int(xh6.shape[1] / 2) + 2 * r,
        int(xh6.shape[1] / 2) - 2 * r:int(xh6.shape[1] / 2) + 2 * r, :] = 0  # add0 size = [2, 64, 64, 96]
        xh8[:, int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r,
        int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r, :] = 0  # add0 size = [2, 96, 96, 96]
        xh2 = xh2.view(xh2.shape[0], xh2.shape[1] * xh2.shape[2], xh2.shape[3])  # [2, 64, 768]
        xh4 = xh4.view(xh4.shape[0], xh4.shape[1] * xh4.shape[2], xh4.shape[3])  # [2, 256, 768]
        xh6 = xh6.view(xh6.shape[0], xh6.shape[1] * xh6.shape[2], xh6.shape[3])  # [2, 576, 768]
        xh8 = xh8.view(xh8.shape[0], xh8.shape[1] * xh8.shape[2], xh8.shape[3])  # [2, 1024, 768]

        return xh2, xh4, xh6, xh8, x_0


class Eye_Attention_Constant(nn.Module):  # input [2, 1024, 96], [2, 1024, 96], [2, 9216, 96], [2, 16384, 96] ,out xh8 [2, 64, 64, 96]
    def __init__(self, C, config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Eye_Attention_Constant, self).__init__()
        self.num_heads = config.transformer.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        head_dim = config.eyedim // config.transformer.num_heads
        self.bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.atten = Atten(C)
        self.get_attn_weight = Get_Weight(C) 
        self.rate = config.eye_moverate
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv96 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv192 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv384 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv768 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1, bias=False)
        
    def norm_maxmin(self,x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return x
    def norm_Zscore(self,x):
        x = x.cpu().detach().numpy() # (2, 8, 8, 768)
        stand = StandardScaler()
        if x.shape[0]  == 1:
            x_stand = x.reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand = stand.fit_transform(x_stand)
            out = x_stand.reshape(x.shape[0], x.shape[1], x.shape[2],  x.shape[3])
            out = torch.from_numpy(out)
        else:
            x_stand1 = x[0,:].reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand2 = x[1,:].reshape(x.shape[1]* x.shape[2],  x.shape[3])
            x_stand1 = stand.fit_transform(x_stand1)
            x_stand2 = stand.fit_transform(x_stand2)
            out1 = x_stand1.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out2 = x_stand2.reshape(1, x.shape[1], x.shape[2], x.shape[3])
            out1 = torch.from_numpy(out1)
            out2 = torch.from_numpy(out2)
            out = torch.cat([out1, out2], dim=0)
        return out.cuda()    
    

    def forward(self, x0, xh2, xh4, xh6, xh8, Cam_Centre):
        '''
        print("xh2 = ",xh2.shape)
        print("xh4 = ",xh4.shape)
        print("xh6 = ",xh6.shape)
        print("xh8 = ",xh8.shape)
        '''
        attn_weight = self.get_attn_weight(x0, self.num_heads, self.bias, self.scale)
        xh2 = self.atten(xh2, self.num_heads, self.bias, self.scale)  # xh2 [2, 8, 8, 768]  attn2 [2, 12, 64, 64]
        xh4 = self.atten(xh4, self.num_heads, self.bias, self.scale)  # attn4 [2, 12, 256, 256]        
        xh6 = xh6.reshape(xh6.shape[0], int(math.sqrt(xh6.shape[1])), int(math.sqrt(xh6.shape[1])), xh6.shape[2])
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh6 = avgpool(xh6)
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        xh6 = xh6.reshape(xh6.shape[0], int((xh6.shape[1]) * (xh6.shape[2])), xh6.shape[3])
        xh6 = self.atten(xh6, self.num_heads, self.bias, self.scale)  # attn6 [2, 12, 144, 144]
        xh6 = xh6.permute(0, 3, 1, 2).contiguous()
        xh6 = self.up(xh6)  # [2, 96, 64, 64]
        xh6 = xh6.permute(0, 2, 3, 1).contiguous()
        
        xh8 = xh8.reshape(xh8.shape[0], int(math.sqrt(xh8.shape[1])), int(math.sqrt(xh8.shape[1])), xh8.shape[2])
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        avgpool = nn.AvgPool2d(2, stride=2)
        xh8 = avgpool(xh8)
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()
        xh8 = xh8.reshape(xh8.shape[0], int((xh8.shape[1]) * (xh8.shape[2])), xh8.shape[3])
        xh8 = self.atten(xh8, self.num_heads, self.bias, self.scale)  # attn8 [2, 12, 256, 256]
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        xh8 = self.up(xh8)  # [2, 96, 64, 64]
        xh8 = xh8.permute(0, 2, 3, 1).contiguous()



        # norm
        xh2 = self.norm_Zscore(xh2)      # [2, 8, 8, 768]
        xh4 = self.norm_Zscore(xh4)      # [2, 16, 16, 768]
        xh6 = self.norm_Zscore(xh6)      # [2, 24, 24, 768]
        xh8 = self.norm_Zscore(xh8)      # [2, 32, 32, 768]

        r = int((xh8.shape[1]) / 8)
        xh8[:, int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r, int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r, :] = (1-self.rate)*xh6 + self.rate * xh8[:, int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r, int(xh8.shape[1] / 2) - 3 * r:int(xh8.shape[1] / 2) + 3 * r, :]
        xh8[:, int(xh8.shape[1] / 2) - 2 * r:int(xh8.shape[1] / 2) + 2 * r, int(xh8.shape[1] / 2) - 2 * r:int(xh8.shape[1] / 2) + 2 * r, :] = (1-self.rate)*xh4 + self.rate * xh8[:, int(xh8.shape[1] / 2) - 2 * r:int(xh8.shape[1] / 2) + 2 * r, int(xh8.shape[1] / 2) - 2 * r:int(xh8.shape[1] / 2) + 2 * r, :]
        xh8[:, int(xh8.shape[1] / 2) - r:int(xh8.shape[1] / 2) + r, int(xh8.shape[1] / 2) - r:int(xh8.shape[1] / 2) + r,:] = (1-self.rate)*xh2 + self.rate * xh8[:, int(xh8.shape[1] / 2) - r:int(xh8.shape[1] / 2) + r,int(xh8.shape[1] / 2) - r:int(xh8.shape[1] / 2) + r, :]
        
        xh8 = xh8.permute(0, 3, 1, 2).contiguous()
        # Filling edge 
        if xh8.shape[1] == 96:
            xh8 = self.conv96(xh8)
        elif xh8.shape[1] == 192:
            cam_class = self.conv192(xh8)
        elif xh8.shape[1] == 384:
            xh8 = self.conv384(xh8)
        elif xh8.shape[1] == 768:
            xh8 = self.conv768(xh8)
        if x0.shape[2] == 4:
            avgpool = nn.AvgPool2d(2, stride=2)
            xh8 = avgpool(xh8)
        xh8 = xh8.permute(0, 2, 3, 1).contiguous() # [2, 32, 32, 768]
        

        return xh8, attn_weight  # attn_weight[2, 12, 1024, 1024]
        
        
class Get_Weight(nn.Module):
    def __init__(self, C, attn_drop=0., proj_drop=0.):
        super(Get_Weight, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)

    def forward(self, x, num_heads, bias, scale):  
        # input [2, 32, 32, 768]--> [2, 1024, 768]
        
        x = x.permute(0, 3, 1, 2).contiguous()  
        avgpool = nn.AvgPool2d(2, stride=2)
        x = avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()  
        
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        qkv = self.Qkv(x).reshape(x.shape[0], x.shape[1], 3, num_heads, x.shape[2] // num_heads).permute(2, 0, 3, 1, 4)  # [3, 2, 6, 1024, 16]
        q, k, v = qkv[0], qkv[1], qkv[2]  # q[2, 6, 1024, 16], k [2, 6, 1024, 16] , v [2, 6, 1024, 16]
        q = q * scale
        if q.shape[0] == 1:
            attn = (q @ k.transpose(-2, -1))  # matrix multiplication  attn[2, 6, 1024, 1024]
            attn = self.softmax(attn)
        else:
            attn = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[2])).cuda()
            attn0 = (q[0, ...] @ k[0, ...].transpose(-2, -1))
            attn1 = (q[1, ...] @ k[1, ...].transpose(-2, -1))
            attn[0, ...] = self.softmax(attn0)
            attn[1, ...] = self.softmax(attn1)
        
        return attn

class Atten(nn.Module):
    def __init__(self, C, attn_drop=0., proj_drop=0.):
        super(Atten, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)

    def forward(self, x, num_heads, bias, scale):  

        qkv = self.Qkv(x).reshape(x.shape[0], x.shape[1], 3, num_heads, x.shape[2] // num_heads).permute(2, 0, 3, 1, 4)  # [3, 2, 6, 1024, 16]
        q, k, v = qkv[0], qkv[1], qkv[2]  # q[2, 6, 1024, 16], k [2, 6, 1024, 16] , v [2, 6, 1024, 16]
        q = q * scale
        if q.shape[0] == 1:
            attn = (q @ k.transpose(-2, -1))  # matrix multiplication  attn[2, 6, 1024, 1024]
            attn = self.softmax(attn)
            x = self.attn_drop(x)
            x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2]).contiguous()  # [2, 1024, 96]
        else:
            attn = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[2])).cuda()
            attn0 = (q[0, ...] @ k[0, ...].transpose(-2, -1))
            attn1 = (q[1, ...] @ k[1, ...].transpose(-2, -1))
            attn[0, ...] = self.softmax(attn0)
            attn[1, ...] = self.softmax(attn1)
            x = self.attn_drop(x)
            X = v
            X[0, ...] = (attn[0, ...] @ v[0, ...])
            X[1, ...] = (attn[1, ...] @ v[1, ...])
            x = X.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2]).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2])
        
        return x 