"""Flatten Transformer / Swin v1 version

Code taken from https://github.com/LeapLabTHU/FLatten-Transformer
"""

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import re
from copy import deepcopy
from typing import Callable, Iterable, List, Optional, Tuple, Union
from collections.abc import Sequence


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import (
    resample_patch_embed,
)
import einops
from src.models.components.utils.utils import (
    get_first_layer,
    infer_output,
    set_first_layer,
)
from src.utils import RankedLogger


from src.models.components.multiscaleDoubleNet import extract_and_upsample_center
from src.models.components.multiscaleViT import (
    RunWithFlattenTemporal,
    TemporalAttention,
)
from src.models.components.utils.seg_blocks import SimpleSegmentationHead

log = RankedLogger(__name__, rank_zero_only=True)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_register_tokens: int = 0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.num_register_tokens = num_register_tokens

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.num_register_tokens > 0:
            relative_position_bias = nn.functional.pad(
                relative_position_bias,
                (0, self.num_register_tokens, 0, self.num_register_tokens),
                "constant",
                0,
            )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class FocusedLinearAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        focusing_factor=3,
        kernel_size=5,
        num_register_tokens: int = 0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            groups=head_dim,
            padding=kernel_size // 2,
        )
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(
            torch.zeros(size=(1, window_size[0] * window_size[1], dim))
        )

        self.num_register_tokens = num_register_tokens
        if self.num_register_tokens > 0:
            self.positional_encoding_ = nn.functional.pad(
                self.positional_encoding,
                (0, 0, 0, self.num_register_tokens),
                "constant",
                0,
            )
        else:
            self.positional_encoding_ = self.positional_encoding
        print(
            "Linear Attention window{} f{} kernel{}".format(
                window_size, focusing_factor, kernel_size
            )
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q**focusing_factor
        k = k**focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N**-0.5)) @ (v * (N**-0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)

        # Exclude register_tokens from the depthwise convolution
        N_non_register = N - self.num_register_tokens
        H = W = int((N_non_register) ** 0.5)
        if self.num_register_tokens > 0:
            non_register_x, register_x = torch.tensor_split(x, (N_non_register,), dim=1)
            non_register_v = (
                v[:, :, :N_non_register]
                .reshape(B * self.num_heads, H, W, -1)
                .permute(0, 3, 1, 2)
            )
            non_register_x = non_register_x + self.dwc(non_register_v).reshape(
                B, C, N_non_register
            ).permute(0, 2, 1)
            x = torch.concat((non_register_x, register_x), axis=1)
        else:
            v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
            x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def eval(self):
        super().eval()
        print("eval")

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        focusing_factor=3,
        kernel_size=5,
        attn_type="L",
        num_register_tokens=0,
    ):
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
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        assert attn_type in ["L", "S"]
        if attn_type == "L":
            self.attn = FocusedLinearAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                focusing_factor=focusing_factor,
                kernel_size=kernel_size,
                num_register_tokens=num_register_tokens,
            )
        else:
            self.attn = WindowAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                num_register_tokens=num_register_tokens,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
            if num_register_tokens > 0:
                # register can attend to all windows
                attn_mask = nn.functional.pad(
                    attn_mask,
                    (0, num_register_tokens, 0, num_register_tokens),
                    "constant",
                    0,
                )
        else:
            attn_mask = None

        self.num_register_tokens = num_register_tokens
        self.register_buffer("attn_mask", attn_mask)

    def _attn(self, x: torch.Tensor, register_token: torch.Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # fmt: off

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll( x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if self.num_register_tokens > 0:
            nW = x_windows.shape[0] // B
            register_token = einops.repeat(register_token, "B R C -> (B nW) R C", nW=nW)
            x_windows = torch.concat((x_windows, register_token), dim=1)
            # B*nW, window_size*window_size+R, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        if self.num_register_tokens > 0:
            register_token = attn_windows[:, -self.num_register_tokens :, : ]  # B*nW, R, C
            attn_windows = attn_windows[:, : -self.num_register_tokens, : ]  # B*nW, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll( shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        if self.num_register_tokens > 0:
            register_token = einops.reduce(
                register_token, "(B nW) R C -> B R C", reduction="mean", B=B
            )
        # fmt: on
        return x, register_token

    def forward(self, x: torch.Tensor, register_token: torch.Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Hacky fuse/unfuse layernorm
        if register_token is not None:
            assert register_token.shape[1] == self.num_register_tokens
            fused_nrm = self.norm1(torch.concat((x, register_token), dim=1))
            x_nrm, register_token_nrm = torch.tensor_split(fused_nrm, (L,), dim=1)
        else:
            x_nrm, register_token_nrm = self.norm1(x), None

        x_att, register_token_att = self._attn(x_nrm, register_token_nrm)

        # Fuse again
        if register_token is not None:
            fused = torch.concat((x, register_token), dim=1)
            fused_att = torch.concat((x_att, register_token_att), dim=1)
        else:
            fused = x
            fused_att = x_att

        # FFN
        fused = fused + self.drop_path(fused_att)
        fused = fused + self.drop_path(self.mlp(self.norm2(fused)))

        if register_token is not None:
            x, register_token = torch.tensor_split(fused, (L,), dim=1)
        else:
            x = fused

        return x, register_token

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        focusing_factor=3,
        kernel_size=5,
        attn_type="L",
        swindow_size=None,  # If None -> default 7/12 behaviour
        num_register_tokens=0,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.Xregister_type = None

        # build blocks
        attn_types = [
            (
                attn_type
                if attn_type[0] != "M"
                else ("L" if i < int(attn_type[1:]) else "S")
            )
            for i in range(depth)
        ]
        if swindow_size is None:  # default hardcoded behaviour
            swindow_size = 7 if window_size <= 56 else 12
        window_sizes = [
            (window_size if attn_types[i] == "L" else swindow_size)
            for i in range(depth)
        ]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

        # register token
        self.use_register_token = num_register_tokens > 0
        self.out_dim = dim  # Hack
        if self.use_register_token and (downsample is not None):
            self.out_dim *= 2
            self.downsample_token = nn.Linear(dim, self.out_dim)
        # self.Xregister_type = "merge"

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    shift_size=0 if (i % 2 == 0) else window_sizes[i] // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    focusing_factor=focusing_factor,
                    kernel_size=kernel_size,
                    attn_type=attn_types[i],
                    num_register_tokens=num_register_tokens,
                )
                for i in range(depth)
            ]
        )

    def add_context_scale(
        self,
        window_size=None,
        Xregister_type="zigzag",
    ):
        self.reg_token_fusion = lambda x: einops.reduce(
            x, "B R (n C) -> B R C", "mean", C=self.out_dim
        )

        # Set the register propagation type
        assert Xregister_type in ["merge", "zigzag"]
        self.Xregister_type = Xregister_type

    def forward(self, x: List[torch.Tensor], register_token: Optional[torch.Tensor]):
        assert (self.Xregister_type == "zigzag") or (self.Xregister_type is None)
        for blk in self.blocks:
            if self.use_checkpoint:
                for i in range(len(x) - 1, -1, -1):
                    x[i], register_token = checkpoint.checkpoint(
                        blk, x[i], register_token, use_reentrant=False
                    )
            else:
                for i in range(len(x) - 1, -1, -1):
                    x[i], register_token = blk(x[i], register_token)

        if self.downsample is not None:
            for i in range(len(x)):
                x[i] = self.downsample(x[i])
            if self.use_register_token:
                register_token = self.downsample_token(register_token)
        return x, register_token

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class FlattenSwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        focusing_factor=3,
        kernel_size=5,
        attn_type="LLLL",
        swindow_sizes=None,
        num_register_tokens=0,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Propagate to all layers if not indexable
        if not isinstance(swindow_sizes, Sequence):
            swindow_sizes = [swindow_sizes] * self.num_layers

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # register token
        self.use_register_token = num_register_tokens > 0
        if self.use_register_token:
            self.register_token = nn.Parameter(
                torch.randn(1, num_register_tokens, self.embed_dim), requires_grad=True
            )
        else:
            self.register_token = None

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                focusing_factor=focusing_factor,
                kernel_size=kernel_size,
                attn_type=attn_type[i_layer]
                + (attn_type[self.num_layers :] if attn_type[i_layer] == "M" else ""),
                swindow_size=swindow_sizes[i_layer],
                num_register_tokens=num_register_tokens,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        # No need for this, as we use segmentation head
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = (
        #     nn.Linear(self.num_features, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_context_scale(
        self,
        img_size,
        num_channels,
        patch_size=None,
        window_size=None,
        temporal_dim=1,
        pre_mlp=False,
        Xregister_type="merge",
        share_patch_embed: Optional[bool] = False,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int): New image size.
            num_channels (int): Number of channels.
            patch_size (int): Patch size.
            window_size (int): Window size.
            temporal_dim (int): Number of temporal dimensions.
            pre_mlp (bool): Whether to add an MLP before the temporal attention layer.
            Xregister_type (str): Type of register token propagation. Default is "merge".
        """
        if patch_size is None:
            patch_size = self.patch_embed.patch_size

        if share_patch_embed:
            patch_embed = self.patch_embed
            self.patch_embed = nn.ModuleList([patch_embed, patch_embed])
        else:
            main_patch_embed = deepcopy(self.patch_embed)
            context_patch_embed = self.patch_embed
            assert temporal_dim <= 1
            self.patch_embed = nn.ModuleList([main_patch_embed, context_patch_embed])

        for index, stage in enumerate(self.layers):
            stage.add_context_scale(
                window_size=window_size,
                Xregister_type=Xregister_type,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        if isinstance(x, Iterable) and not isinstance(x, torch.Tensor):
            x = x  # [high_res, low_res]
        else:
            x = [x]  # [high_res]

        B, H, W, C = x[0].shape
        if self.use_register_token:
            register_token = self.register_token.repeat(B, 1, 1)
        else:
            register_token = None

        for i in range(len(x)):
            if isinstance(self.patch_embed, nn.ModuleList):
                x[i] = self.patch_embed[i](x[i])
            else:
                x[i] = self.patch_embed(x[i])
            if self.ape:
                x[i] = x[i] + self.absolute_pos_embed
            x[i] = self.pos_drop(x[i])

        for layer in self.layers:
            x, register_token = layer(x, register_token)

        for i in range(len(x)):
            x[i] = self.norm(x[i])  # B L C
            # == Instead of doing pooling like they do in flatten
            # x = self.avgpool(x.transpose(1, 2))  # B C 1
            # x = torch.flatten(x, 1)
            # == Unpack for segmentation
            H, W = self.layers[-1].input_resolution
            x[i] = einops.rearrange(x[i], "B (H W) C -> B C H W", H=H, W=W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


def checkpoint_filter_fn(checkpoint, model):
    state_dict = checkpoint["model"].copy()

    # delete these since we always re-init them
    bad_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    bad_keys.extend([k for k in state_dict.keys() if "relative_coords_table" in k])
    bad_keys.extend([k for k in state_dict.keys() if "attn_mask" in k])
    for k in bad_keys:
        del state_dict[k]

    # bicubic interpolate positional_encoding if not match
    positional_encoding_keys = [
        k for k in state_dict.keys() if "positional_encoding" in k
    ]
    for k in positional_encoding_keys:
        positional_encoding_pretrained = state_dict[k]
        positional_encoding_current = model.state_dict()[k]
        _, L1, nH1 = positional_encoding_pretrained.size()
        _, L2, nH2 = positional_encoding_current.size()
        if nH1 != nH2:
            log.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                positional_encoding_pretrained_resized = (
                    torch.nn.functional.interpolate(
                        positional_encoding_pretrained.permute(0, 2, 1).view(
                            1, nH1, S1, S1
                        ),
                        size=(S2, S2),
                        mode="bicubic",
                    )
                )
                state_dict[k] = positional_encoding_pretrained_resized.view(
                    1, nH2, L2
                ).permute(0, 2, 1)

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            log.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                relative_position_bias_table_pretrained_resized = (
                    torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(
                            1, nH1, S1, S1
                        ),
                        size=(S2, S2),
                        mode="bicubic",
                    )
                )
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2
                ).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k
    ]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            log.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1
                )
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2
                )
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic"
                )
                absolute_pos_embed_pretrained_resized = (
                    absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                )
                absolute_pos_embed_pretrained_resized = (
                    absolute_pos_embed_pretrained_resized.flatten(1, 2)
                )
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # If head exists (instead of seg_head) -> drop it
    bad_keys = [k for k in state_dict.keys() if k.startswith("head")]
    for k in bad_keys:
        del state_dict[k]

    return state_dict


class Flatten_swin_module(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_channels=1,
        segmentation_head=SimpleSegmentationHead,
        pretrained=False,
        pretrained_path=None,
        img_size=512,
        patch_size=4,
        window_size=96,
        drop_path_rate=0.1,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_register_tokens=0,
        multi_scale=False,
        temporal_dim=0,
        pre_temporal_mlp=False,
        cross_scale_token_fusion="mean",
        decision_fusion=False,
        Xregister_type="merge",
        share_patch_embed=False,
        la=None,
        swindow_sizes=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.multi_scale = multi_scale
        self.Xregister_type = Xregister_type
        if isinstance(img_size, Iterable) and not self.multi_scale:
            self.img_size = img_size[0]
        else:
            self.img_size = img_size

        if self.multi_scale:
            assert len(img_size) == 2, "Multi-scale requires two image sizes."
            assert (
                len(num_channels) == 2
            ), "Multi-scale requires two channel number (on per scale)"

        assert decision_fusion is False or isinstance(
            decision_fusion, int
        ), "decision_fusion must be False or an integer"
        self.decision_fusion = decision_fusion

        self.model = FlattenSwinTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            # // Flatten defaults
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            focusing_factor=la.focusing_factor,
            kernel_size=la.kernel_size,
            attn_type=la.attn_type,
            swindow_sizes=swindow_sizes,
            num_register_tokens=num_register_tokens,
        )

        if pretrained:
            assert pretrained_path is not None, "Pretrained weights path is required."
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(
                checkpoint_filter_fn(checkpoint, self.model), strict=False
            )

        if multi_scale:
            assert cross_scale_token_fusion == "mean"
            self.model.add_context_scale(
                self.img_size[1],
                num_channels[1],
                temporal_dim=temporal_dim,
                pre_mlp=pre_temporal_mlp,
                Xregister_type=self.Xregister_type,
                share_patch_embed=share_patch_embed,
            )

            set_first_layer(
                self.model.patch_embed[0], num_channels[0]
            )  # convert to correct number of features

            if not share_patch_embed:
                set_first_layer(self.model.patch_embed[1], num_channels[1])

        else:
            set_first_layer(self.model.patch_embed, num_channels)

        # Measure downsample factor
        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(
            self.model,
            self.num_channels,
            self.img_size,
            temporal_dim=[0, 0] if multi_scale else 0,
            indices=[0],
        )
        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,  # use only one scale at a time scale
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
            layer="first",
        )

    def forward(self, x, metas=None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the output tensor.
        """
        if (
            not self.multi_scale
            and isinstance(x, Iterable)
            and not isinstance(x, torch.Tensor)
        ):
            x = x[0]

        x = self.model.forward_features(x)

        if self.multi_scale:
            x_main, x_cont = x[0], x[1]
            x_main_pred = self.seg_head(x_main)
            x_cont_pred = self.seg_head(x_cont)

            if self.decision_fusion:
                x_cont_center = extract_and_upsample_center(
                    x_cont_pred, x_main_pred.size()[2:], self.decision_fusion
                )
                x_main_pred = x_main_pred + x_cont_center

            return {
                "out": x_main_pred,
                "out_context": x_cont_pred,
                "embed": x_main,
                "embed_context": x_cont,
            }
        else:
            out = self.seg_head(x[0])
            return {"out": out, "embed": x[0]}
