# References:
# DBDA: https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/global_module/network.py


import torch
from torch import nn
import math


class PAM_Module(nn.Module):
    """Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        x = x.squeeze(-1)
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma * out + x).unsqueeze(-1)
        return out


class CAM_Module(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = self.gamma * out + x 
        return out


class DbdaWithCrossAttention(nn.Module):
    def __init__(self, band, classes, flatten_out = False):
        super().__init__()

        self.flatten_out = flatten_out

        self.params = {
            "band": band,
            "classes": classes
        }

        # spectral branch

        self.conv11 = nn.Conv3d(
            in_channels=1, out_channels=24, kernel_size=(1, 1, 7), stride=(1, 1, 2)
        )
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  
            nn.Mish(inplace=True),
        )
        self.conv12 = nn.Conv3d(
            in_channels=24,
            out_channels=12,
            padding=(0, 0, 3),
            kernel_size=(1, 1, 7),
            stride=(1, 1, 1),
        )
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        self.conv13 = nn.Conv3d(
            in_channels=36,
            out_channels=12,
            padding=(0, 0, 3),
            kernel_size=(1, 1, 7),
            stride=(1, 1, 1),
        )
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        self.conv14 = nn.Conv3d(
            in_channels=48,
            out_channels=12,
            padding=(0, 0, 3),
            kernel_size=(1, 1, 7),
            stride=(1, 1, 1),
        )
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(
            in_channels=60,
            out_channels=60,
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1),
        )  

        # Spatial Branch
        self.conv21 = nn.Conv3d(
            in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1)
        )
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        self.conv22 = nn.Conv3d(
            in_channels=24,
            out_channels=12,
            padding=(1, 1, 0),
            kernel_size=(3, 3, 1),
            stride=(1, 1, 1),
        )
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        self.conv23 = nn.Conv3d(
            in_channels=36,
            out_channels=12,
            padding=(1, 1, 0),
            kernel_size=(3, 3, 1),
            stride=(1, 1, 1),
        )
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True),
        )
        self.conv24 = nn.Conv3d(
            in_channels=48,
            out_channels=12,
            padding=(1, 1, 0),
            kernel_size=(3, 3, 1),
            stride=(1, 1, 1),
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.cross_attention_1to2 = nn.MultiheadAttention(embed_dim=60, num_heads=4, batch_first=True)
        self.cross_attention_2to1 = nn.MultiheadAttention(embed_dim=60, num_heads=4, batch_first=True)

        # Classifier on concatenated attended vectors (60 + 60 = 120)
        self.classifier = nn.Sequential(
            nn.LayerNorm(120),
            nn.Dropout(0.5),
            nn.Linear(120, classes)
        )

    def forward(self, X):
        X = X.permute(0, 2, 3, 1).unsqueeze(1)
        # spectral
        x11 = self.conv11(X)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)

        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)

        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)

        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # Global pooling → (B, 60)
        x1 = self.global_pooling(x1).squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x2).squeeze(-1).squeeze(-1).squeeze(-1)

        # Add sequence dimension: (B, 1, 60)
        x1_seq = x1.unsqueeze(1)
        x2_seq = x2.unsqueeze(1)

        # Bidirectional attention
        attn_1, _ = self.cross_attention_1to2(query=x1_seq, key=x2_seq, value=x2_seq)  # x1 attends to x2
        attn_2, _ = self.cross_attention_2to1(query=x2_seq, key=x1_seq, value=x1_seq)  # x2 attends to x1

        # Concatenate both attended outputs
        attn_1 = attn_1.squeeze(1)  # (B, 60)
        attn_2 = attn_2.squeeze(1)  # (B, 60)
        attn_concat = torch.cat([attn_1, attn_2], dim=1)  # (B, 120)

        # Classify
        output = self.classifier(attn_concat)

        if self.flatten_out:
            return output.reshape(-1)
        else:
            return output

    def get_params(self):
        return self.params
