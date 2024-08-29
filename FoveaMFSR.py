import torch
import torch.nn as nn
import torch.nn.functional as F

from TransNext import transnext_tiny, transnext_small, transnext_base
from module import *

    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class FeatureFusion(nn.Module):
    def __init__(self, inchannel):
        super(FeatureFusion, self).__init__()
        self.fusion_convs = nn.Conv2d(inchannel, inchannel//2, kernel_size=1, padding=0)
        self.csam = CSAM(inchannel//2)

    def forward(self, tar, ref):

        x = torch.cat((tar, ref), dim=1)
        x = self.fusion_convs(x)
        x = self.csam(x)
        
        return x

class CSAM(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

class SegFormerEncoder(nn.Module):
    def __init__(self, img_size):
        super(SegFormerEncoder, self).__init__()
        self.img_size = img_size
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.backbone = transnext_tiny(self.img_size)

    def forward(self, inputs):
        fea_map0 = self.conv2d(inputs)
        inputs = fea_map0
        x = self.backbone.forward(inputs)
        fea_map1, fea_map2, fea_map3, fea_map4 = x
        return fea_map0, fea_map1, fea_map2, fea_map3, fea_map4

class RefSegFormerEncoder(nn.Module):
    def __init__(self, img_size):
        super(RefSegFormerEncoder, self).__init__()
        self.img_size = img_size
        self.backbone = transnext_tiny(self.img_size)

    def forward(self, inputs):
        fea_map0 = inputs
        x = self.backbone.forward(inputs)
        fea_map1, fea_map2, fea_map3, fea_map4 = x
        return fea_map0, fea_map1, fea_map2, fea_map3, fea_map4

class MultiModalSegFormer(nn.Module):
    def __init__(self, scale_factor, img_size):
        super(MultiModalSegFormer, self).__init__()
        self.scale_factor = scale_factor
        self.img_size = img_size
        self.in_channels = {
            'tiny': [72, 144, 288, 576], 'small': [72, 144, 288, 576], 'base': [96, 192, 384, 768]
        }['tiny']
        self.embedding_dim = {
            'tiny': 256, 'small': 256, 'base': 768
        }['tiny']
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_first = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_second = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv_third = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.encoder_tar = SegFormerEncoder(self.img_size)
        self.encoder_ref = RefSegFormerEncoder(self.img_size)
        self.FeatureFusion = nn.ModuleList([FeatureFusion(in_channels * 2) for in_channels in [32, 32, 32, 72, 144, 288, 576]])

    def process_ref(self, ref):
        ref_0 = self.conv_first(ref)
        ref_1 = self.conv_second(ref_0)
        ref_2 = self.conv_third(ref_1)
        if self.scale_factor == 2:
            return [ref_0, ref_1] + list(self.encoder_ref(ref_1))
        elif self.scale_factor == 4:
            return [ref_0, ref_1] + list(self.encoder_ref(ref_2))

    def fusion_and_upsampling(self, tar_mid, tar_features, ref_features):
        fused_features = []
        out = []

        if self.scale_factor == 2:
            tar_mid1 = tar_mid # [1, 32, 112, 112]
            tar_mid0 = self.up(tar_mid1) # [1, 32, 224, 224]
            fused_ref0 = self.FeatureFusion[0](tar_mid0, ref_features[0])  # fuse tar_mid0 with ref_0 [1, 32, 224, 224]
            fused_ref1 = self.FeatureFusion[1](tar_mid1, ref_features[1])  # fuse tar_mid with ref_1 [1, 32, 112, 112]
            fused_fea0 = self.FeatureFusion[2](tar_features[0], ref_features[2])  # fuse tar_fea0 with ref_fea0 [1, 32, 112, 112]
            fused_fea1 = self.FeatureFusion[3](tar_features[1], ref_features[3])  # fuse tar_fea1 with ref_fea1 [1, 72, 56, 56]
            fused_fea2 = self.FeatureFusion[4](tar_features[2], ref_features[4])  # fuse tar_fea2 with ref_fea2 [1, 144, 28, 28]
            fused_fea3 = self.FeatureFusion[5](tar_features[3], ref_features[5])  # fuse tar_fea3 with ref_fea3 [1, 288, 14, 14]
            fused_fea4 = self.FeatureFusion[6](tar_features[4], ref_features[6])  # fuse tar_fea4 with ref_fea4 [1, 576, 7, 7]

            out = [tar_mid, fused_ref0, fused_ref1, fused_fea0, fused_fea1, fused_fea2, fused_fea3, fused_fea4]
        
        elif self.scale_factor == 4:
            tar_mid1 = self.up(tar_mid) # [1, 32, 112, 112]
            tar_mid0 = self.up(tar_mid1) # [1, 32, 224, 224]
            fused_ref0 = self.FeatureFusion[0](tar_mid0, ref_features[0])  # fuse tar_mid0 with ref_0 [1, 32, 224, 224]
            fused_ref1 = self.FeatureFusion[1](tar_mid1, ref_features[1])  # fuse tar_mid1 with ref_1 [1, 32, 112, 112]
            fused_fea0 = self.FeatureFusion[2](tar_features[0], ref_features[2])  # fuse tar_fea0 with ref_2 [1, 32, 56, 56]
            fused_fea1 = self.FeatureFusion[3](tar_features[1], ref_features[3])  # fuse tar_fea1 with ref_3 [1, 72, 28, 28]
            fused_fea2 = self.FeatureFusion[4](tar_features[2], ref_features[4])  # fuse tar_fea2 with ref_4 [1, 144, 14, 14]
            fused_fea3 = self.FeatureFusion[5](tar_features[3], ref_features[5])  # fuse tar_fea3 with ref_5 [1, 288, 7, 7]
            fused_fea4 = self.FeatureFusion[6](tar_features[4], ref_features[6])  # fuse tar_fea4 with ref_6 [1, 576, 4, 4]

            out = [tar_mid, fused_ref0, fused_ref1, fused_fea0, fused_fea1, fused_fea2, fused_fea3, fused_fea4]

        return out

    def forward(self, tar, ref):
        H, W = tar.size(2), tar.size(3)
        tar_mid = self.conv_first(tar)
        tar_features = self.encoder_tar(tar)
        ref_features = self.process_ref(ref)

        out = self.fusion_and_upsampling(tar_mid, tar_features, ref_features)

        return out

class FoveaMFSR(nn.Module):
    def __init__(self, scale_factor, img_size):
        super(FoveaMFSR, self).__init__()
        
        self.scale_factor = scale_factor
        self.img_size = img_size 
        self.multisegformer = MultiModalSegFormer(scale_factor=self.scale_factor, img_size=self.img_size)
        # Upsample
        self.upsample = UpsampleModule(in_channels=576, scale=2) # 输入通道数定义为512
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # MFFU
        fea_channel = [32, 32, 32, 72, 144, 288, 576]
        if self.scale_factor == 2:
            self.mffu1 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(14,14))
            self.mffu2 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(28,28))
            self.mffu3 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(56,56))
            self.mffu4 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(112,112))
            self.mffu5 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(224,224))
        elif self.scale_factor == 4:
            self.mffu1 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(7,7))
            self.mffu2 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(14,14))
            self.mffu3 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(28,28))
            self.mffu4 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(56,56))
            self.mffu5 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(112,112))
            self.mffu6 = MFFU(scale=self.scale_factor, cl0=fea_channel[0], cl1=fea_channel[1], cl2=fea_channel[2], cl3=fea_channel[3], cl4=fea_channel[4], cl5=fea_channel[5], cl6=fea_channel[5], out_channels=fea_channel[6], target_size=(224,224))

        # Transformer
        self.transformer = TransformerBlock(dim=576, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

        self.conv_w = nn.Conv2d(576, 32, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, tar, ref):

        out = self.multisegformer(tar, ref)
        
        mid = self.upsample(out[7]) # 4*4

        # MFFU
        mid_fuse = self.mffu1(out, mid) 

        mid = self.transformer(mid_fuse)
        mid = self.transformer(mid)+mid_fuse

        mid = self.upsample(mid) # 7*7
        mid_fuse = self.mffu2(out, mid) 

        mid = self.transformer(mid_fuse)
        mid = self.transformer(mid)+mid_fuse

        mid = self.upsample(mid) # 28*28
        mid_fuse = self.mffu3(out, mid) 
 
        mid = self.transformer(mid_fuse)
        mid = self.transformer(mid)+mid_fuse


        mid = self.upsample(mid) # 112*112
        mid_fuse = self.mffu4(out, mid)
        mid = self.transformer(mid_fuse)
        mid = self.transformer(mid)+mid_fuse

        mid = self.upsample(mid) # 224*224   
        mid_fuse = self.mffu5(out, mid)

        if self.scale_factor == 2:
            mid = self.conv_w(mid_fuse)+self.up(out[0])

        if self.scale_factor == 4:
            mid = self.transformer(mid_fuse)
            mid = self.transformer(mid)+mid_fuse
            mid = self.upsample(mid) # 448*448
            mid = self.mffu6(out, mid)
            mid = self.conv_w(mid)+self.up(self.up(out[0]))
        
        sup_output = self.conv_out(mid)

        return sup_output
    


# 创建输入特征图
# tar = torch.randn(1, 3, 56, 56).cuda()
# ref = torch.randn(1, 3, 224, 224).cuda()
# sup_branch = MultiFusionSup(scale_factor=4, img_size=56).cuda()
# output = sup_branch(tar, ref)
# print("the shape of mid:", output.shape)