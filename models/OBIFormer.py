import numbers
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from models.FusedAttention import SKFF, FusedAttentionBlock

List = nn.ModuleList


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


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
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# class FeedForward(nn.Module):
#     def __init__(self, dim, out_dim=128):
#         super().__init__()
#         self.project_in = nn.Conv2d(dim, dim, 1)
#         self.project_out = nn.Sequential(
#             nn.Conv2d(dim, out_dim, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(out_dim, dim, 1)
#         )

#     def forward(self, x):
#         x = self.project_in(x)
#         return self.project_out(x)


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim, LayerNorm_type)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


##########################################################################
class GSNB(nn.Module):
    def __init__(self, dim, out_dim, depth=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim*2, dim, 1)
        if depth == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
        elif depth == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.Conv2d(out_dim, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim*2, 3, padding=1),
                nn.BatchNorm2d(dim*2),
                nn.ReLU(),
                nn.Conv2d(dim*2, dim*4, 3, padding=1),
                nn.BatchNorm2d(dim*4),
                nn.ReLU(),
                nn.Conv2d(dim*4, dim*2, 1),
                nn.BatchNorm2d(dim*2),
                nn.ReLU(),
                nn.Conv2d(dim*2, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            )

    def forward(self, x, skip=None):
        if exists(skip):
            x = torch.cat((x, skip), dim=1)
            x = self.upsample(x)

        return self.conv(x) + x


class RSAB(nn.Module):
    def __init__(self, dim, depth, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RSAB, self).__init__()

        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, LayerNorm_type, Attention(dim, num_heads, bias)),
                PreNorm(dim, LayerNorm_type, FeedForward(
                    dim, ffn_expansion_factor, bias))
            ]))

    def forward(self, x, skip):
        if skip is not None:
            x += skip
        for atten, ff in self.layers:
            x = x + atten(x)
            x = x + ff(x)

        return x


class CFB(nn.Module):
    def __init__(self, dim=64, reduction=16, depth_RSAB=2, depth_GSNB=2, heads=8):

        super().__init__()

        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'

        self.fusedAtt = SKFF(dim, height=2)
        # self.fusedAtt = FusedAttentionBlock(dim, reduction)
        self.RSAB = RSAB(dim, depth=depth_RSAB, num_heads=heads,
                         ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.GSNB = GSNB(dim, dim*4, depth=depth_GSNB)

        # self.conv = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1)

    def forward(self, input_rsab, input_gsnb, skip_rsab=None, skip_gsnb=None):
        RSAB_r = self.RSAB(input_rsab, skip_rsab)
        GSNB_r = self.GSNB(input_gsnb, skip_gsnb)

        # Two options to obtain the combined feature, we process simple add function (or can be concat as commented)
        # fused_att = torch.cat((RSAB_r, GSNB_r),dim=1)
        # fused_att = self.conv(fused_att)
        # fused_att = self.fuseAtt(RSAB_r + GSNB_r)
        # fused_att = RSAB_r + GSNB_r
        fused_att = self.fusedAtt([RSAB_r, GSNB_r])

        return RSAB_r + fused_att, GSNB_r + fused_att

# classes


class CharFormer(nn.Module):
    def __init__(
            self,
            dim=64,
            channels=3,
            stages=4,
            depth_RSAB=2,
            depth_GSNB=2,
            dim_head=64,
            window_size=16,
            heads=8,
            input_channels=None,
            output_channels=None
    ):
        super().__init__()
        input_channels = default(input_channels, channels)
        output_channels = default(output_channels, channels)

        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 3, padding=1),
            nn.GELU()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

        self.feature_corrector = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

        self.downs = List([])
        self.ups = List([])

        heads, window_size, dim_head, depth_RSAB, depth_GSNB = map(partial(cast_tuple, depth=stages),
                                                                   (heads, window_size, dim_head, depth_RSAB, depth_GSNB))

        for ind, heads, window_size, dim_head, depth_RSAB, depth_GSNB in zip(range(stages), heads, window_size, dim_head,
                                                                             depth_RSAB, depth_GSNB):
            is_last = ind == (stages - 1)
            self.downs.append(List([
                CFB(dim, depth_RSAB=depth_RSAB,
                    depth_GSNB=depth_GSNB, heads=heads),
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)
            ]))

            self.ups.append(List([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride=2),
                CFB(dim, depth_RSAB=depth_RSAB,
                    depth_GSNB=depth_GSNB, heads=heads)
            ]))

            dim *= 2

            if is_last:
                self.mid = CFB(dim, depth_RSAB=depth_RSAB,
                               depth_GSNB=depth_GSNB, heads=heads)

    def forward(self, x):

        rsab = self.project_in(x)
        gsnb = torch.clone(rsab)

        skip_rsab = []
        skip_gsnb = []

        for block, downsample in self.downs:
            rsab, gsnb = block(rsab, gsnb)

            skip_rsab.append(rsab)
            skip_gsnb.append(gsnb)

            rsab = downsample(rsab)
            gsnb = downsample(gsnb)

        rsab, gsnb = self.mid(rsab, gsnb)

        for (upsample, block), skip1, skip2 in zip(reversed(self.ups), reversed(skip_rsab), reversed(skip_gsnb)):
            rsab = upsample(rsab)
            gsnb = upsample(gsnb)

            rsab, gsnb = block(rsab, gsnb, skip_rsab=skip1, skip_gsnb=skip2)

        rsab = self.project_out(rsab)
        gsnb = self.feature_corrector(gsnb)

        return rsab, gsnb


if __name__ == "__main__":
    model = CharFormer(
        dim=64,  # initial dimensions after input projection, which increases by 2x each stage
        stages=3,  # number of stages
        depth_RSAB=2,  # number of transformer blocks per RSAB
        depth_GSNB=1,  # number of Conv2d blocks per GSNB
        # set window size (along one side) for which to do the attention within
        window_size=16,
        dim_head=32,
        heads=2,
    ).cuda()

    x = torch.randn(1, 3, 256, 256).cuda()
    output, add_feature = model(x)  # (1, 3, 256, 256)

    print(output.shape)
    print(add_feature.shape)
