# Code for FSDFormer 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """spilt 4D tensor into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """reverse windows back to 4D tensor"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer """
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
from torch.nn import init as init

class frequency_selection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU):
        super(frequency_selection, self).__init__()
        self.act_fft = act_method()

        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        
        self.norm = norm

    def forward(self, x):
        _, hw, _ = x.size()
        hh = int(math.sqrt(hw))
        x1 = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)

        y = torch.fft.rfft2(x1, norm=self.norm)
        dim1 = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)

        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1

        y = torch.cat([y.real, y.imag], dim=dim1)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim1)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2

        y = rearrange(y, 'b h w c -> b c h w')

        y = torch.fft.irfft2(y, s=(hh, hh), norm=self.norm) 
        y = rearrange(y, ' b c h w -> b (h w) c', h = hh, w = hh) # + x
        return y

class AgentAttention(nn.Module):
    """ Agent Attention 模块 """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        
        # Agent attention biases
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 8, 8))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 8, 8))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, mask=None):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Agent tokens
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # Position biases
        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        
        # Agent attention
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # Query attention
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class DFAttention(nn.Module):
    r""" DFAttention for fusion
    Q/V:F1, K:F1+delta, A1:C1, A2:C2
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.win = window_size[0] * window_size[1]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        
        self.Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.K = nn.Linear(dim, dim, bias=qkv_bias)
        self.V = nn.Linear(dim, dim, bias=qkv_bias)
        self.A1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.A2 = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x_F1, x_C2, x_C1, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x_F1.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        q = self.Q(x_F1)
        k = self.K(x_F1 + x_C2 - x_C1)
        v = self.V(x_C2 - x_C1)
        agent_tokens1 = self.pool(self.A1(x_C1).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        agent_tokens2 = self.pool(self.A2(x_C2).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens1 = agent_tokens1.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens2 = agent_tokens2.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        
        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        
        position_bias = position_bias1 + position_bias2
        agent_attn2 = self.softmax((agent_tokens2 * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn2 = self.attn_drop(agent_attn2)
        agent_v = agent_attn2 @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens1.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        q = q.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(q).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SEATBlock(nn.Module):
    """ SEAT Block """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, agent_num=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = AgentAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            agent_num=agent_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.FFT = frequency_selection(dim)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Agent attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)) + self.FFT(self.norm2(x)))
        return x

class SDFBlock(nn.Module):
    """ SDF Block """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, agent_num=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        self.attn = DFAttention(dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    agent_num=agent_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, act_layer=act_layer, drop=drop)
        self.FFT = frequency_selection(dim)

        if self.shift_size > 0:
            
            H, W = self.input_resolution

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_F1, x_C2, x_C1,):
        H, W = self.input_resolution
        B, L, C = x_F1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x_F1 + x_C2 - x_C1

        x_F1 = x_F1.view(B, H, W, C)
        x_C2 = x_C2.view(B, H, W, C)
        x_C1 = x_C1.view(B, H, W, C)

        images = []
        for x in [x_F1, x_C2, x_C1]:
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            images.append(x_windows)

        # W-MSA/SW-MSA
        attn_windows = self.attn(images[0], images[1], images[2], mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = x + shortcut

        x = x + self.mlp(self.norm3(x))  + self.FFT(self.norm3(x))

        return x

class SEATLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, agent_num=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SEATBlock(dim=dim, input_resolution=input_resolution,
                     num_heads=num_heads, window_size=window_size,
                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path,
                     norm_layer=norm_layer, agent_num=agent_num)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SDFLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, agent_num=64):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SDFBlock(dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer, agent_num=agent_num)
            for i in range(depth)])

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

    def forward(self, x_F1, x_C2, x_C1):
        x_lr = self.norm1(x_F1)
        x_hr = self.norm2(x_C2)
        x_c = self.norm3(x_C1)

        for blk in self.blocks:
            x_diff = blk(x_lr, x_hr, x_c)
        return x_diff

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth, agent_num):
        super(EncoderBlock, self).__init__()
        self.layer = SEATLayer(dim=in_channels,
                                input_resolution=(resolution, resolution),
                                depth=cur_depth,
                                num_heads=in_channels // 32,
                                window_size=8,
                                mlp_ratio=4,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=0.,
                                norm_layer=nn.LayerNorm, agent_num=agent_num)

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x_o = self.layer(x)

        if self.downsample is not None:
            x_o = self.downsample(x_o)

        return x_o

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, cur_depth, agent_num):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        self.layer = SDFLayer(dim=out_channels, input_resolution=(resolution, resolution),
                                depth=2, num_heads=in_channels // 32, window_size=8, mlp_ratio=4,
                                qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                norm_layer=nn.LayerNorm, agent_num=agent_num)

        self.layer2 = SEATLayer(dim=out_channels, input_resolution=(resolution, resolution),
                                 depth=cur_depth, num_heads=out_channels // 32, window_size=8, mlp_ratio=4,
                                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                                 norm_layer=nn.LayerNorm, agent_num=agent_num)

        self.proj1 = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x_fine0, x_coarse1, x_coarse0, x_fine1):
        """
        :param x_fine0: feature of F1
        :param x_coarse1: feature of C2
        :param x_coarse0: feature of C1
        :param x_fine1: feature of last stage
        :return:
        """
        B, L, C = x_fine1.shape

        x_f1 = x_fine1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        x_f1 = self.up(x_f1).flatten(2).transpose(1, 2)

        x_f0 = self.layer(x_fine0, x_coarse1, x_coarse0)
        x = torch.cat([x_coarse0, x_f0, x_f1], dim=2)
        x = self.proj1(x)
        x = self.layer2(x)

        return x

class Encoder(nn.Module):
    def __init__(self, down_scale=2, in_dim=64, depths=(2, 2, 6, 2), agent_num=[9, 16, 64, 64], in_channels=4):
        super(Encoder, self).__init__()
        self.inc = PatchEmbed(img_size=256, patch_size=down_scale, in_chans=in_channels, embed_dim=in_dim,
                              norm_layer=nn.LayerNorm)
        self.enc1 = EncoderBlock(in_channels=in_dim, out_channels=in_dim, resolution=256 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[0], agent_num=agent_num[0])
        self.enc2 = EncoderBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=128 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[1], agent_num=agent_num[1])
        self.enc3 = EncoderBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=64 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[2], agent_num=agent_num[2])
        self.enc4 = EncoderBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=32 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[3], agent_num=agent_num[3])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    def __init__(self, in_dim=64, down_scale=2, depths=(2, 2, 6, 2), agent_num=[9, 16, 64, 64], in_channels=4):
        super(Decoder, self).__init__()
        self.down_scale = down_scale
        self.dec1 = DecoderBlock(in_dim * 8, in_dim * 4, 32 // down_scale, depths[3], agent_num=agent_num[3])
        self.dec2 = DecoderBlock(in_dim * 4, in_dim * 2, 64 // down_scale, depths[2], agent_num=agent_num[2])
        self.dec3 = DecoderBlock(in_dim * 2, in_dim, 128 // down_scale, depths[1], agent_num=agent_num[1])
        self.dec4 = DecoderBlock(in_dim, in_dim, 256 // down_scale, depths[0], agent_num=agent_num[0])

        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, in_channels, 3, 1, 1),
            nn.Tanh()
        )

        self.layer = SDFLayer(dim=in_dim * 8, input_resolution=(16 // down_scale, 16 // down_scale),
                              depth=2, num_heads=in_dim * 8 // 32, window_size=8, mlp_ratio=4,
                              qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                              norm_layer=nn.LayerNorm, agent_num=agent_num[3])

    def forward(self, fine_fea, coarse_fea1, coarse_fea0):
        """
        :param fine_fea: feature of F1
        :param coarse_fea1: feature of C2
        :param coarse_fea0: feature of C1
        :return: predict fine feature
        """
        x0 = self.layer(fine_fea[4], coarse_fea1[4], coarse_fea0[4])

        x1 = self.dec1(fine_fea[3], coarse_fea1[3], coarse_fea0[3], x0)
        x2 = self.dec2(fine_fea[2], coarse_fea1[2], coarse_fea0[2], x1)
        x3 = self.dec3(fine_fea[1], coarse_fea1[1], coarse_fea0[1], x2)
        x4 = self.dec4(fine_fea[0], coarse_fea1[0], coarse_fea0[0], x3)
       
        B, L, C = x4.shape

        x4 = x4.transpose(1, 2).view(B, C, 256 // self.down_scale, 256 // self.down_scale)
        output_fine = self.outc(x4)

        return output_fine

class FSDFormer(nn.Module):
    """FSDFormer model"""
    def __init__(self, in_dim=64, down_scale=2, depths=(2, 2, 6, 2), agent_num=[9, 16, 64, 64], in_channels=4):
        super(FSDFormer, self).__init__()
        self.encoder1 = Encoder(down_scale=down_scale, in_dim=in_dim, depths=depths, 
                              agent_num=agent_num, in_channels=in_channels)
        self.encoder2 = Encoder(down_scale=down_scale, in_dim=in_dim, depths=depths, 
                              agent_num=agent_num, in_channels=in_channels)
        self.decoder = Decoder(in_dim=in_dim, down_scale=down_scale, depths=depths,
                              agent_num=agent_num, in_channels=in_channels)

    def forward(self, F1, C2, C1):
        fine_fea = self.encoder1(F1)
        coarse_fea1 = self.encoder2(C2)
        coarse_fea0 = self.encoder2(C1)
        
        output_fine = self.decoder(fine_fea, coarse_fea1, coarse_fea0)

        return output_fine