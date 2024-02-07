
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

save_picture = 0 # 1 --> save; 0 --> not save

cam_rate = 0.5
filename_Pseudo_Label = './show_img/Pseudo_Label/'
filename_Pseudo_Cam = './show_img/Pseudo_Cam/'
filename_Avg_CAM = './show_img/Avg_CAM/'
filename_Layer_Cam = './show_img/Layer_Cam/'

#cam_center = "same_calss_max_num"
cam_center = "complexity"

class Show_Pseudo_Label(nn.Module):
    def __init__(self):
        super(Show_Pseudo_Label, self).__init__()
        
    def forward(self, cam):
        if save_picture == 1:
            b, n, h, w = cam.shape
            size_upsample = (512, 512)
      
            for i in range(n):
                output_cam = []
                cam_layer = cam[:, i, :, :].reshape(b, h, w)
                cam_layer = cam_layer - torch.min(cam_layer)
                cam_img = cam_layer / torch.max(cam_layer)  # [2, 32, 32]
                cam_img = torch.as_tensor(255 * cam_img, dtype=torch.uint8)  # [2, 32, 32]
                cam_img = cam_img.cpu().detach().numpy()
      
      
                # size_upsample
                if cam_img.shape[0] != 1:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))
                    output_cam.append(cv2.resize(cam_img[1], (512, 512)))
                    heatmap0 = cv2.applyColorMap(output_cam[0], cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap1 = cv2.applyColorMap(output_cam[1], cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap = [heatmap0, heatmap1]
                
                    cv2.imwrite(filename_Pseudo_Label+'Pseudo_Label1'+'the'+str(i)+'.jpg', heatmap[0])
                    cv2.imwrite(filename_Pseudo_Label+'Pseudo_Label2'+'the'+str(i)+'.jpg', heatmap[1])
                    
                else:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))   
                    heatmap = cv2.applyColorMap(output_cam[0], cv2.COLORMAP_JET)  # (512, 512, 3)
                    cv2.imwrite(filename_Pseudo_Label+'Pseudo_Label'+'the'+str(i)+'.jpg', heatmap) 

        return cam



class Show_Pseudo_Cam(nn.Module):
    def __init__(self):
        super(Show_Pseudo_Cam, self).__init__()
        
    def forward(self, cam):
        if save_picture == 1:
            b, n, h, w = cam.shape
            size_upsample = (512, 512)
            for i in range(n):
                output_cam = []
                cam_layer = cam[:, i, :, :].reshape(b, h, w)
                # print(torch.min(cam_layer))
                cam_layer = cam_layer - torch.min(cam_layer)
                cam_img = cam_layer / torch.max(cam_layer)  # [2, 32, 32]
                cam_img = torch.as_tensor(255 * cam_img, dtype=torch.uint8)  # [2, 32, 32]
                cam_img = cam_img.cpu().detach().numpy()
    
                # size_upsample
                if cam_img.shape[0] != 1:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))
                    output_cam.append(cv2.resize(cam_img[1], (512, 512)))
                else:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))    
                if cam_img.shape[0] != 1:
                    heatmap0 = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap1 = cv2.applyColorMap(cv2.resize(output_cam[1], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap = [heatmap0, heatmap1]
                
                    cv2.imwrite(filename_Pseudo_Cam+'Pseudo_Cam1'+'the'+str(i)+'.jpg', heatmap[0])
                    cv2.imwrite(filename_Pseudo_Cam+'Pseudo_Cam2'+'the'+str(i)+'.jpg', heatmap[1])
    
                else:
                    heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    cv2.imwrite(filename_Pseudo_Cam+'Pseudo_Cam'+'the'+str(i)+'.jpg', heatmap)
        return cam

class Show_Avg_CAM_Mapping(nn.Module):
    def __init__(self):
        super(Show_Avg_CAM_Mapping, self).__init__()
        
    def forward(self, cam, layer):
        if save_picture == 1:
            b, h, w = cam.shape
            size_upsample = (512, 512)
    
            output_cam = []
            cam_layer = cam[:, :, :].reshape(b, h, w)
             # print(torch.min(cam_layer))
            cam_layer = cam_layer - torch.min(cam_layer)
            cam_img = cam_layer / torch.max(cam_layer)  # [2, 32, 32]
            cam_img = torch.as_tensor(255 * cam_img, dtype=torch.uint8)  # [2, 32, 32]
            cam_img = cam_img.cpu().detach().numpy()
            i = 1
            # size_upsample
            if cam_img.shape[0] != 1:
                output_cam.append(cv2.resize(cam_img[0], (512, 512)))
                output_cam.append(cv2.resize(cam_img[1], (512, 512)))
            else:
                output_cam.append(cv2.resize(cam_img[0], (512, 512)))    
            if cam_img.shape[0] != 1:
                heatmap0 = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                heatmap1 = cv2.applyColorMap(cv2.resize(output_cam[1], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                heatmap = [heatmap0, heatmap1]
                if layer == "vb":
                    cv2.imwrite(filename_Avg_CAM+'avgCAMvb1'+'the'+str(i)+'.jpg', heatmap[0])
                    cv2.imwrite(filename_Avg_CAM+'avgCAMvb2'+'the'+str(i)+'.jpg', heatmap[1])
                elif layer == "multi":
                    cv2.imwrite(filename_Avg_CAM+'avgCAMmulti1'+'the'+str(i)+'.jpg', heatmap[0])
                    cv2.imwrite(filename_Avg_CAM+'avgCAMmulti2'+'the'+str(i)+'.jpg', heatmap[1])
            else:
                heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                if layer == "vb":
                    cv2.imwrite(filename_Avg_CAM+'avgCAMvb'+'the'+str(i)+'.jpg', heatmap)
                elif layer == "multi":
                    cv2.imwrite(filename_Avg_CAM+'avgCAMmulti'+'the'+str(i)+'.jpg', heatmap)
        return cam
    

class Class_Activation_Mapping(nn.Module):
    def __init__(self):
        super(Class_Activation_Mapping, self).__init__()
        self.num_classes = 6
        self.attn_proj = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, bias=True)
        self.classifier768 = nn.Conv2d(in_channels=768, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier96 = nn.Conv2d(in_channels=96, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier192 = nn.Conv2d(in_channels=192, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier384 = nn.Conv2d(in_channels=384, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier64 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier12 = nn.Conv2d(in_channels=12, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier4 = nn.Conv2d(in_channels=4, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv2d(in_channels=3, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.pooling = F.adaptive_avg_pool2d
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.classifiervb = nn.Conv2d(in_channels=768, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.complex_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.complex_avgpool = torch.nn.AvgPool2d(kernel_size=2)
        # show cam img

        # cam cam_input_layers
        self.down_cam = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, stride=4, padding=1)
        self.up_cam = nn.UpsamplingBilinear2d(scale_factor=4)

    def show_cam(self, cam, layer):
        # generate the class activation maps upsample to 512x512
        if save_picture == 1:
            b, n, h, w = cam.shape
            size_upsample = (512, 512)
    
    
            for i in range(n):
                output_cam = []
                cam_layer = cam[:, i, :, :].reshape(b, h, w)
                # print(torch.min(cam_layer))
                cam_layer = cam_layer - torch.min(cam_layer)
                cam_img = cam_layer / torch.max(cam_layer)  # [2, 32, 32]
                cam_img = torch.as_tensor(255 * cam_img, dtype=torch.uint8)  # [2, 32, 32]
                cam_img = cam_img.cpu().detach().numpy()
    
                # size_upsample
                if cam_img.shape[0] != 1:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))
                    output_cam.append(cv2.resize(cam_img[1], (512, 512)))
                else:
                    output_cam.append(cv2.resize(cam_img[0], (512, 512)))
    
                if cam_img.shape[0] != 1:
                    heatmap0 = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap1 = cv2.applyColorMap(cv2.resize(output_cam[1], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    heatmap = [heatmap0, heatmap1]
                    
                    cv2.imwrite(filename_Layer_Cam+'CAM1multi'+str(layer)+'the'+str(i)+'.jpg', heatmap[0])
                    cv2.imwrite(filename_Layer_Cam+'CAM2multi'+str(layer)+'the'+str(i)+'.jpg', heatmap[1])
                    
    
                else:
                    heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (512, 512)), cv2.COLORMAP_JET)  # (512, 512, 3)
                    cv2.imwrite(filename_Layer_Cam+'CAM'+str(layer)+'the'+str(i)+'.jpg', heatmap)

        return cam

    def centre(self, cam):  # [2, 5, 32, 32]
        cam_relu = torch.relu(cam)
        cam_maxpool = self.maxpool(cam_relu)
        cam_max = 0
        cam_max1 = 0
        cam_max2 = 0
        layer_max = 0
        layer_max1 = 0
        layer_max2 = 0
        for i in range(cam_maxpool.shape[1]):  # cam_maxpool.shape[1] = 6
            if cam_maxpool.shape[0] == 2:
                cam_layer_sum1 = torch.sum(cam_maxpool[0, i, :, :])
                cam_layer_sum2 = torch.sum(cam_maxpool[1, i, :, :])
                if cam_layer_sum1 > cam_max1:
                    cam_max1 = cam_layer_sum1
                    layer_max1 = i
                if cam_layer_sum2 > cam_max2:
                    cam_max2 = cam_layer_sum2
                    layer_max2 = i

            else:
                cam_layer_sum = torch.sum(cam_maxpool[:, i, :, :])
                if cam_layer_sum > cam_max:
                    cam_max = cam_layer_sum
                    layer_max = i

        # Computational center of gravity
        if cam_maxpool.shape[0] == 2:
            if layer_max1 > layer_max2:
                layer_max = layer_max1
                cam_max_layer = cam_maxpool[0, layer_max, :, :]
            else:
                layer_max = layer_max2
                cam_max_layer = cam_maxpool[1, layer_max, :, :]
        else:
            cam_max_layer = cam_maxpool[0, layer_max, :, :]

        x_sum = torch.sum(cam_max_layer, dim=1)
        y_sum = torch.sum(cam_max_layer, dim=0) # Calculate the sum of each row to get a vertical list
        max_x_sum = torch.argmax(x_sum)
        max_y_sum = torch.argmax(y_sum)  #The number of rows with the maximum sum
        center = [2 * max_y_sum, 2 * max_x_sum] # first is the row index, second is column

        return center

    def complex_centre(self, cam):
        cam_relu = torch.relu(cam)
        cam_maxpool = self.complex_maxpool(cam_relu)
        complex_avgpool = self.complex_avgpool(cam_relu)
        cam_diff = cam_maxpool - complex_avgpool
        x_sum = torch.sum(cam_diff, dim=1)
        y_sum = torch.sum(cam_diff, dim=0) # Calculate the sum of each row to get a vertical list
        max_x_sum = torch.argmax(x_sum)
        max_y_sum = torch.argmax(y_sum)
        center = [2 * max_y_sum, 2 * max_x_sum]
        return center

    def Aff(self,cam, CAM_Last):
        
        if cam.shape[2] < CAM_Last.shape[2]:
            CAM_Last = self.down_cam(CAM_Last)
        elif cam.shape[2] > CAM_Last.shape[2]:
            CAM_Last = self.up_cam(CAM_Last)
        CAM_Last = CAM_Last.mul(cam)
        if torch.sum(CAM_Last) == 0:
            cam = cam 
        else:
            cam = CAM_Last + cam * cam_rate
        #print (cam)
        aff = cam
        return aff
        

    def forward(self, x, layer, CAM_Last):  # x[2, 1024, 768]

        '''
        if layer != "vb":
            x = x.permute(0, 3, 1, 2).contiguous()  # [1, 8, 8, 96] --> [1, 96, 8, 8]
        '''
        # classifier --> 5 class
        if x.shape[1] == 96:
            cam_class = F.conv2d(x, self.classifier96.weight).detach()
        elif x.shape[1] == 192:
            cam_class = F.conv2d(x, self.classifier192.weight).detach()
        elif x.shape[1] == 384:
            cam_class = F.conv2d(x, self.classifier384.weight).detach()
        elif x.shape[1] == 768:
            cam_class = F.conv2d(x, self.classifier768.weight).detach()
        elif x.shape[1] == 64:
            cam_class = F.conv2d(x, self.classifier64.weight).detach()
        elif x.shape[1] == 12:
            cam_class = F.conv2d(x, self.classifier12.weight).detach()
        elif x.shape[1] == 4:
            cam_class = F.conv2d(x, self.classifier4.weight).detach()
        elif x.shape[1] == 3:
            cam_class = F.conv2d(x, self.classifier3.weight).detach()
            
        # pooling
        cam_gap = self.pooling(cam_class, (1, 1))  # [2, 5, 1, 1]
        # cam_gap = cam_gap.view(-1, self.num_classes-1) # [2, 5]
        # cam_gap * cam_class
        cam = cam_gap * cam_class  # [2, 5, 32, 32]
        # print (cam)
        cam = self.Aff(cam , CAM_Last)

        cam_img = self.show_cam(cam, layer)
        if cam_center == "same_calss_max_num":
            cam_centre = self.centre(cam)
        else:
            cam_centre = self.complex_centre(cam)


        
        return cam, cam_centre

