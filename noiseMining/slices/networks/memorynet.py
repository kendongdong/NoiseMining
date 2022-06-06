from sqlite3 import connect
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
 
 


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()





####################################  RGB Network  #####################################
class ConvLSTM(nn.Module):
    def __init__(self,n_class=2):
        super(ConvLSTM, self).__init__()
        # original image's size = 256*256*3
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers
        # conv5 add
        self.conv5_c = nn.Conv2d(512, 64, 3, padding=1)
        # conv3 add
        self.conv4_c = nn.Conv2d(512, 64, 3, padding=1)
        # conv3 add
        self.conv3_c = nn.Conv2d(256, 64, 3, padding=1)
        # conv2_add
        self.conv2_c = nn.Conv2d(128, 64, 3, padding=1)
        # down sample
        self.down = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ---------------------------- ConvLSTM1 ------------------------------ #

        # ------------------ ConvLSTM cell parameter ---------------------- #
        # self.conv_cell1 =  nn.Conv2d(64 + 64 , 4 * 64, 5, padding=2)
        self.conv_cell2 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell3 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell4 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell5 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        # attentive convlstm 2
        self.conv_cell = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)

        # 2nd layer light-field features weighted
        self.conv_w2 = nn.Conv2d(64 * 1, 1, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 3rd layer light-field features weighted
        self.conv_w3 = nn.Conv2d(64 * 1, 1, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 4th layer light-field features weighted
        self.conv_w4 = nn.Conv2d(64 * 1, 1, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        # 5th layer light-field features weighted
        self.conv_w5 = nn.Conv2d(64 * 1, 1, 1, padding=0)  # 13 = 1+N (N=12)
        self.pool_avg_w5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        
        # -----------------------------  Multi-scale2  ----------------------------- #
        # part1:
        self.Atrous_c1_2 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_2 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_2 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_2 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_2 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_2 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_2 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_2 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_2 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale3  ----------------------------- #
        # part1:
        self.Atrous_c1_3 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_3 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_3 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_3 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_3 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_3 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_3 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_3 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_3 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale4  ----------------------------- #
        # part1:
        self.Atrous_c1_4 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_4 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_4 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_4 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_4 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_4 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_4 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_4 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_4 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale5  ----------------------------- #
        # part1:
        self.Atrous_c1_5 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_5 = nn.ReLU(inplace=True)
        # part2:
        self.Atrous_c3_5 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_5 = nn.ReLU(inplace=True)
        # part3:
        self.Atrous_c5_5 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_5 = nn.ReLU(inplace=True)
        # part4:
        self.Atrous_c7_5 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_5 = nn.ReLU(inplace=True)
        # conv
        self.Aconv_5 = nn.Conv2d(64 * 5, 64, 1, padding=0)
 
        # ----------------------------- Attentive ConvLSTM 2 -------------------------- #
        # ConvLSTM-2
        # 2-1
        self.conv_fcn2_1 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_1 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_1 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-2
        self.conv_fcn2_2 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_2 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_2 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-3
        self.conv_fcn2_3 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_3 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_3 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-4
        self.conv_fcn2_4 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_4 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_4 = nn.Conv2d(64, 64, 1, padding=0)
        # 2-5
        self.conv_fcn2_5 = nn.Conv2d(64, 64, 1, padding=0)  # 13 = 1+N (N=12)
        self.conv_h_5 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_5 = nn.Conv2d(64, 64, 1, padding=0)

        self.prediction = nn.Conv2d(64, 2, 1, padding=0)    

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
   
    def convlstm_cell(self,A, new_c):
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ag)
        g = torch.tanh(ag)
        new_c = f * new_c + i * g
        new_h = o * torch.tanh(new_c)
        return new_c , new_h


    def forward(self, x):
       
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        pool1 = h
        
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        pool2 = h
        
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        pool3 = h
       
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        h = self.pool4(h)
        pool4 = h
        
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        pool5 = h
        

        # side out
        pool1 = self.down(pool1)

        pool2 = self.conv2_c(pool2)
        pool2 = pool2 + pool1

        pool3 = self.conv3_c(pool3)
        pool3 = F.upsample(pool3, scale_factor=2, mode='bilinear')

        pool4 = self.conv4_c(pool4)
        pool4 = F.upsample(pool4, scale_factor=2, mode='bilinear')
        pool4 = F.upsample(pool4, scale_factor=2, mode='bilinear')

        pool5 = self.conv5_c(pool5)
        pool5 = F.upsample(pool5, scale_factor=2, mode='bilinear')
        pool5 = F.upsample(pool5, scale_factor=2, mode='bilinear')
        pool5 = pool5 + pool4 +pool3

        # ------------------- the 2nd ConvLSTM layer ------------------ #
        b1 = pool2 # level2 split
        c1 = pool3	# level3 split
        d1 = pool4	# level4 split
        e1 = pool5	# level5 split

        # -----------------------------  level 2 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = b1
        h_state0 = b1
        combined = torch.cat((b1, h_state0), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        out2 = new_h

        # -----------------------------  level 3 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = c1
        h_state0 = c1
        combined = torch.cat((c1, h_state0), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        out3 = new_h

        # -----------------------------  level 4 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = d1
        h_state0 = d1
        combined = torch.cat((d1, h_state0), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        out4 = new_h

        # -----------------------------  level 5 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = e1
        h_state0 = e1
        combined = torch.cat((e1, h_state0), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        out5 = new_h
       
        # --------------------------- Multi-scale ---------------------------- #

        # out2
        A1 = self.Atrous_r1_2(self.Atrous_b1_2(self.Atrous_c1_2(out2)))
        A3 = self.Atrous_r3_2(self.Atrous_b3_2(self.Atrous_c3_2(out2)))
        A5 = self.Atrous_r5_2(self.Atrous_b5_2(self.Atrous_c5_2(out2)))
        A7 = self.Atrous_r7_2(self.Atrous_b7_2(self.Atrous_c7_2(out2)))
        out2 = torch.cat([out2, A1, A3, A5, A7], dim=1)
        out2 = self.Aconv_2(out2)
        # out3
        A1 = self.Atrous_r1_3(self.Atrous_b1_3(self.Atrous_c1_3(out3)))
        A3 = self.Atrous_r3_3(self.Atrous_b3_3(self.Atrous_c3_3(out3)))
        A5 = self.Atrous_r5_3(self.Atrous_b5_3(self.Atrous_c5_3(out3)))
        A7 = self.Atrous_r7_3(self.Atrous_b7_3(self.Atrous_c7_3(out3)))
        out3 = torch.cat([out3, A1, A3, A5, A7], dim=1)
        out3 = self.Aconv_3(out3)
        # out4
        A1 = self.Atrous_r1_4(self.Atrous_b1_4(self.Atrous_c1_4(out4)))
        A3 = self.Atrous_r3_4(self.Atrous_b3_4(self.Atrous_c3_4(out4)))
        A5 = self.Atrous_r5_4(self.Atrous_b5_4(self.Atrous_c5_4(out4)))
        A7 = self.Atrous_r7_4(self.Atrous_b7_4(self.Atrous_c7_4(out4)))
        out4 = torch.cat([out4, A1, A3, A5, A7], dim=1)
        out4 = self.Aconv_4(out4)
        # out5
        A1 = self.Atrous_r1_5(self.Atrous_b1_5(self.Atrous_c1_5(out5)))
        A3 = self.Atrous_r3_5(self.Atrous_b3_5(self.Atrous_c3_5(out5)))
        A5 = self.Atrous_r5_5(self.Atrous_b5_5(self.Atrous_c5_5(out5)))
        A7 = self.Atrous_r7_5(self.Atrous_b7_5(self.Atrous_c7_5(out5)))
        out5 = torch.cat([out5, A1, A3, A5, A7], dim=1)
        out5 = self.Aconv_5(out5)

        # ------------------------ Attentive ConvLSTM 2 ------------------------- #
        new_h = out5
        # cell 1
        out5_ori = out5
        f5 = self.conv_fcn2_5(out5)
        h_c = self.conv_h_5(new_h)

        fh5 = f5 + h_c
        fh5 = self.pool_avg_5(fh5)
        fh5 = self.conv_c_5(fh5)
        # Scene Context weighted Module
        w5 = torch.mul(F.softmax(fh5, dim=1), 64)
        fw5 = torch.mul(w5, out5_ori)

        combined = torch.cat((fw5, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        

        # cell 2
        out4_ori = out4 +out5
        f4 = self.conv_fcn2_4(out4_ori)
        h_c = self.conv_h_4(new_h)

        fh4 = f4 + h_c
        fh4 = self.pool_avg_4(fh4)
        fh4 = self.conv_c_4(fh4)
        # Scene Context weighted Module
        w4 = torch.mul(F.softmax(fh4, dim=1), 64)
        fw4 = torch.mul(w4, out4_ori)

        combined = torch.cat((fw4, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
       

        # cell 3
        out3_ori = out4 + out5 + out3
        f3 = self.conv_fcn2_3(out3_ori)
        h_c = self.conv_h_3(new_h)

        fh3 = f3 + h_c
        fh3 = self.pool_avg_3(fh3)
        fh3 = self.conv_c_3(fh3)
        # Scene Context weighted Module
        w3 = torch.mul(F.softmax(fh3, dim=1), 64)
        fw3 = torch.mul(w3, out3_ori)

        combined = torch.cat((fw3, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
       
        # cell 2
        out2_ori = out2 + out4 + out5 + out3
        f2 = self.conv_fcn2_2(out2_ori)
        h_c = self.conv_h_2(new_h)

        fh2 = f2 + h_c
        fh2 = self.pool_avg_2(fh2)
        fh2 = self.conv_c_2(fh2)
        # Scene Context weighted Module
        w2 = torch.mul(F.softmax(fh2, dim=1), 64)
        fw2 = torch.mul(w2, out2_ori)

        combined = torch.cat((fw2, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        output = new_h

        # -------------------------- prediction ----------------------------#
        # final
        output = self.prediction(output)
        outputs_o = F.upsample(output, scale_factor=4, mode='bilinear')
        #结果需要sigmoid

        return outputs_o

        #return pool1,pool2,pool3,pool4,pool5



    def copy_params_from_vgg19_bn(self, vgg19_bn):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.conv3_4, self.bn3_4, self.relu3_4,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.conv4_4, self.bn4_4, self.relu4_4,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            self.conv5_4, self.bn5_4, self.relu5_4,
            # self.conv6,
            # self.conv7,
        ]
        for l1, l2 in zip(vgg19_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


def memorynet(params : dict ={}):
    memorynet = ConvLSTM()
    if params['train']:
        vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
        memorynet.copy_params_from_vgg19_bn(vgg19_bn)
    return memorynet





if __name__ == '__main__':

    params = {'train':False}
    memorynet = memorynet(params).cuda()
   
    target = torch.randn(1,1,256,256).cuda()
    rgb = torch.randn(1,3,256,256).cuda()
  
    
    
    out = memorynet(rgb)
    print(out.shape)
    
    
    summary(memorynet,input_size=(3,256,256))
    print(memorynet)







