import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers


class MFFU(nn.Module):
    def __init__(self, scale, cl0, cl1, cl2, cl3, cl4, cl5, cl6, out_channels, target_size):
        super(MFFU, self).__init__()
        self.target_size = target_size
        self.scale = scale

        # Convolutional layers to adjust the size of c0, c1, c2, c3, c4, c5, and c6
        self.conv_c0 = nn.Conv2d(cl0, out_channels, kernel_size=1)
        self.conv_c1 = nn.Conv2d(cl1, out_channels, kernel_size=1)
        self.conv_c2 = nn.Conv2d(cl2, out_channels, kernel_size=1)
        self.conv_c3 = nn.Conv2d(cl3, out_channels, kernel_size=1)
        self.conv_c4 = nn.Conv2d(cl4, out_channels, kernel_size=1)
        self.conv_c5 = nn.Conv2d(cl5, out_channels, kernel_size=1)
        self.conv_c6 = nn.Conv2d(cl6, out_channels, kernel_size=1)

        # 1x1 convolution to map the channels of fused features to the same as c4
        self.conv_fused_x2 = nn.Conv2d(out_channels * 6, out_channels, kernel_size=1)
        self.conv_fused_x4 = nn.Conv2d(out_channels * 7, out_channels, kernel_size=1)

    def forward(self, out, mid):
        # Adjust the size of c0, c1, c2, c3, c4, c5, and c6 to match the target size
        if self.scale == 2:
            c0_resized = self._adjust_size(self.conv_c0(out[1]), out[1].size(), self.target_size)
            # c1_resized = self._adjust_size(self.conv_c1(out[2]), out[2].size(), self.target_size)
            c2_resized = self._adjust_size(self.conv_c2(out[3]), out[3].size(), self.target_size)
            c3_resized = self._adjust_size(self.conv_c3(out[4]), out[4].size(), self.target_size)
            c4_resized = self._adjust_size(self.conv_c4(out[5]), out[5].size(), self.target_size)
            c5_resized = self._adjust_size(self.conv_c5(out[6]), out[6].size(), self.target_size)
            c6_resized = self._adjust_size(self.conv_c6(mid), mid.size(), self.target_size)

            # Concatenate the resized feature maps
            fused_features = torch.cat([c0_resized, c2_resized, c3_resized, c4_resized, c5_resized, c6_resized], dim=1)

            # Apply 1x1 convolution to adjust the number of channels
            fused_features = self.conv_fused_x2(fused_features)

        elif self.scale == 4:
            c0_resized = self._adjust_size(self.conv_c0(out[1]), out[1].size(), self.target_size)
            c1_resized = self._adjust_size(self.conv_c1(out[2]), out[2].size(), self.target_size)
            c2_resized = self._adjust_size(self.conv_c2(out[3]), out[3].size(), self.target_size)
            c3_resized = self._adjust_size(self.conv_c3(out[4]), out[4].size(), self.target_size)
            c4_resized = self._adjust_size(self.conv_c4(out[5]), out[5].size(), self.target_size)
            c5_resized = self._adjust_size(self.conv_c5(out[6]), out[6].size(), self.target_size)
            c6_resized = self._adjust_size(self.conv_c6(mid), mid.size(), self.target_size)

            # Concatenate the resized feature maps
            fused_features = torch.cat([c0_resized, c1_resized, c2_resized, c3_resized, c4_resized, c5_resized, c6_resized], dim=1)

            # Apply 1x1 convolution to adjust the number of channels
            fused_features = self.conv_fused_x4(fused_features)

        return fused_features

    def _adjust_size(self, feature_map, original_size, target_size):
        # Adjust size based on target size
        if original_size[2:] < target_size:  # if smaller, upsample
            return F.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=False)
        elif original_size[2:] > target_size:  # if larger, downsample
            return F.adaptive_avg_pool2d(feature_map, target_size)
        else:  # if already the same size
            return feature_map

class invertedBlock(nn.Module):
    def __init__(self, in_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)


    def hifi(self,x):

        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        x3 = x1+x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x)+x
    
class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x

class UpsampleModule(nn.Module):
    def __init__(self, in_channels, scale):
        super(UpsampleModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # mlp_hidden_dim = int(dim * 4)
        # self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=4, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x = x + self.mlp(self.norm2(x))

        return x 

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        
        h, w = x.shape[-2:]
        a = to_4d(self.body(to_3d(x)), h, w)
        return to_4d(self.body(to_3d(x)), h, w)

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv0 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv1 = Conv2dReLU(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.graphfusion = GraphFusionModule(out_channels)

    def forward(self, x, skip=None):
        B, C, H, W = x.shape
        if H == 4:
            x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        else:
            x = self.up(x)
        x = self.conv0(x)
        x, skip = self.graphfusion(x, skip)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AggregatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature)

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        attn_local = (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, C, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1)

        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        # q_norm:[1, 1, 49, 576], attn_local:[1, 1, 49, 9]
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x_local)
        x = self.proj_drop(x).reshape(B, C, H, W)

        return x


class TransNextBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=3, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AggregatedAttention(
        dim,
        window_size=window_size,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        attn_drop=attn_drop,
        proj_drop=drop)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
