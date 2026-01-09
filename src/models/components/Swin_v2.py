"""Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""

import math

# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from copy import deepcopy
from typing import Callable, Iterable, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    ClassifierHead,
    DropPath,
    Mlp,
    PatchEmbed,
    get_act_layer,
    resample_patch_embed,
    to_2tuple,
    trunc_normal_,
)

from src.models.components.utils.seg_blocks import SimpleSegmentationHead
from src.models.components.utils.utils import (
    get_first_layer,
    infer_output,
    set_first_layer,
)

_int_or_tuple_2_t = Union[int, Tuple[int, int]]
_input_with_context = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    windows = einops.rearrange(
        x,
        "B (H wH) (W wW) C -> (B H W) wH wW C",
        wH=window_size[0],
        wW=window_size[1],
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    img_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    x = einops.rearrange(
        windows,
        "(B H W) wH wW C -> B (H wH) (W wW) C",
        H=H // window_size[0],
        W=W // window_size[1],
        wH=window_size[0],
        wW=window_size[1],
    )
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qkv_bias_separate: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: Tuple[int, int] = (0, 0),
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.num_heads = num_heads
        self.qkv_bias_separate = qkv_bias_separate

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer("k_bias", torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self._make_pair_wise_relative_positions()
        self.num_register_tokens = num_register_tokens

    def _make_pair_wise_relative_positions(self, scale: float = 1.0):
        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0]
        ).to(torch.float32)
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1]
        ).to(torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        relative_coords_table = (
            relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        # fmt: off
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1) * scale
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1) * scale
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1) * scale
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1) * scale
        # fmt: on
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / math.log2(8)
        )
        self.register_buffer(
            "relative_coords_table", relative_coords_table, persistent=False
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
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
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        window_size = to_2tuple(window_size)
        if window_size != self.window_size:
            self.window_size = window_size
            self._make_pair_wise_relative_positions()

    def add_context_scale(
        self, context_window_size: Tuple[int, int]
    ):
        self.register_buffer(
            "relative_coords_table_main",
            self.relative_coords_table,
            persistent=False,
        )
        self.register_buffer(
            "relative_position_index_main",
            self.relative_position_index,
            persistent=False,
        )
        self.window_size_list = [self.window_size, context_window_size]
        self.window_size = self.window_size_list[1]
        self._make_pair_wise_relative_positions(
            scale=1.0
            * self.window_size_list[1][0]
            / self.window_size_list[0][0]
        )
        self.register_buffer(
            "relative_coords_table_context",
            self.relative_coords_table,
            persistent=False,
        )
        self.register_buffer(
            "relative_position_index_context",
            self.relative_position_index,
            persistent=False,
        )
        self.set_scale(0)

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        self.window_size = self.window_size_list[scale]
        self.relative_coords_table = (
            self.relative_coords_table_main
            if scale == 0
            else self.relative_coords_table_context
        )
        self.relative_position_index = (
            self.relative_position_index_main
            if scale == 0
            else self.relative_position_index_context
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        if self.num_register_tokens > 0:
            relative_position_bias = nn.functional.pad(
                relative_position_bias,
                (0, self.num_register_tokens, 0, self.num_register_tokens),
                "constant",
                0,
            )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
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


class SwinTransformerV2Block(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: _int_or_tuple_2_t,
        num_heads: int,
        window_size: _int_or_tuple_2_t = 7,
        shift_size: _int_or_tuple_2_t = 0,
        always_partition: bool = False,
        dynamic_mask: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer="gelu",
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: _int_or_tuple_2_t = 0,
        num_register_tokens: int = 0,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            always_partition: Always partition into full windows and shift
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
            num_register_tokens: How many register token are use
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.target_shift_size = to_2tuple(shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(
            window_size, shift_size
        )
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        act_layer = get_act_layer(act_layer)

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            num_register_tokens=num_register_tokens,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.num_register_tokens = num_register_tokens
        self.register_buffer(
            "attn_mask",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

        self.current_scale = 0

    def get_attn_mask(
        self, x: Optional[torch.Tensor] = None, num_register_tokens: int = 0
    ) -> Optional[torch.Tensor]:
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            # (needed because we use ciclic shift and we need to avoid oposite side of the image to attend to one another)
            if x is None:
                img_mask = torch.zeros((1, *self.input_resolution, 1))  # 1 H W 1
            else:
                img_mask = torch.zeros(
                    (1, x.shape[1], x.shape[2], 1),
                    dtype=x.dtype,
                    device=x.device,
                )  # 1 H W 1

            cnt = 0
            for h in (
                (0, -self.window_size[0]),
                (-self.window_size[0], -self.shift_size[0]),
                (-self.shift_size[0], None),
            ):
                for w in (
                    (0, -self.window_size[1]),
                    (-self.window_size[1], -self.shift_size[1]),
                    (-self.shift_size[1], None),
                ):
                    img_mask[:, h[0] : h[1], w[0] : w[1], :] = cnt
                    cnt += 1
            # add padding
            pad_h = (
                self.window_size[0] - img_mask.shape[1] % self.window_size[0]
            ) % self.window_size[0]
            pad_w = (
                self.window_size[1] - img_mask.shape[2] % self.window_size[1]
            ) % self.window_size[1]
            img_mask = torch.nn.functional.pad(img_mask, (0, 0, 0, pad_w, 0, pad_h))
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
            # Attention mask is nW, window_area, window_area
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
        return attn_mask

    def _calc_window_shift(
        self,
        target_window_size: _int_or_tuple_2_t,
        target_shift_size: Optional[_int_or_tuple_2_t] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was active
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                # if there was previously a non-zero shift, recalculate based on current window_size
                target_shift_size = (
                    target_window_size[0] // 2,
                    target_window_size[1] // 2,
                )
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [
            r if r <= w else w
            for r, w in zip(self.input_resolution, target_window_size)
        ]
        shift_size = [
            0 if r <= w else s
            for r, w, s in zip(self.input_resolution, window_size, target_shift_size)
        ]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(
        self,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        always_partition: Optional[bool] = None,
    ):
        """Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        # Update input resolution
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def add_context_scale(
        self,
        feat_size,
        window_size=None,
        reg_token_fusion=None,
    ):
        """Add a context scale to the stage.

        Args:
            feat_size (Tuple[int, int]): Feature size for the context scale.
            window_size (Tuple[int, int], optional): Window size. Defaults to None.
            reg_token_fusion (Callable, optional): Function to fuse register tokens from multiple scales. Defaults to None (mean fusion).
        """
        self.input_resolution = [self.input_resolution, feat_size]
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if window_size is not None:
            self.window_size = to_2tuple(window_size)
            self._make_pair_wise_relative_positions(
                scale=1.0
                * self.window_size[0]
                / self.pretrained_window_size[0]
            )
        if reg_token_fusion is not None:
            self.register_token_fusion = reg_token_fusion
        else:
            self.register_token_fusion = lambda x: x.mean(dim=1)

        self.register_buffer(
            "attn_mask",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        self.window_size = self.window_size_list[scale]
        self.relative_coords_table = (
            self.relative_coords_table_main
            if scale == 0
            else self.relative_coords_table_context
        )
        self.relative_position_index = (
            self.relative_position_index_main
            if scale == 0
            else self.relative_position_index_context
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        if self.num_register_tokens > 0:
            relative_position_bias = nn.functional.pad(
                relative_position_bias,
                (0, self.num_register_tokens, 0, self.num_register_tokens),
                "constant",
                0,
            )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
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


class SwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: _int_or_tuple_2_t,
        depth: int,
        num_heads: int,
        window_size: _int_or_tuple_2_t,
        always_partition: bool = False,
        dynamic_mask: bool = False,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: _int_or_tuple_2_t = 0,
        output_nchw: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
            use_register_token: Whether to use register token
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # register token
        self.use_register_token = num_register_tokens > 0
        if self.use_register_token and downsample:
            self.downsample_token = nn.Linear(dim, out_dim)

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerV2Block(
                    dim=out_dim,
                    input_resolution=self.output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    always_partition=always_partition,
                    dynamic_mask=dynamic_mask,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    num_register_tokens=num_register_tokens,
                )
                for i in range(depth)
            ]
        )

    def set_input_size(
        self,
        feat_size: Tuple[int, int],
        window_size: int,
        always_partition: Optional[bool] = None,
    ):
        """Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def add_context_scale(
        self,
        feat_size,
        window_size=None,
        reg_token_fusion=None,
        share_patch_embed: Optional[bool] = False,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int or Tuple[int, int]): New image size for the context scale.
            patch_size (int, optional): Patch size. Defaults to None (uses existing patch size).
            window_size (int, optional): Window size. Defaults to None.
            reg_token_fusion (Callable, optional): Function to fuse register tokens from multiple scales. Defaults to None (mean fusion).
            share_patch_embed (bool, optional): Whether to share patch embedding between scales. Defaults to False.
        """
        if window_size is not None and isinstance(window_size, int):
            window_size = (window_size, window_size)
        context_window_size, context_shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.input_resolutions = [self.input_resolution, feat_size]
        self.window_size_list = [self.window_size, context_window_size]
        self.shift_size_list = [self.shift_size, context_shift_size]
        self.attn.add_context_scale(context_window_size)

        self.set_scale(1)  # switch to context scale
        self.register_buffer(
            "attn_mask_context",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )
        self.set_scale(0)  # switch back to main scale
        self.register_buffer(
            "attn_mask_main",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        if self.current_scale == scale:
            return
        self.current_scale = scale
        self.input_resolution = self.input_resolutions[scale]
        self.window_size = self.window_size_list[scale]
        self.window_area = self.window_size[0] * self.window_size[1]
        self.shift_size = self.shift_size_list[scale]
        self.attn.set_scale(scale)
        if hasattr(self, "attn_mask_main"):  # if context scale has been fully added
            self.attn_mask = (
                self.attn_mask_main if scale == 0 else self.attn_mask_context
            )

    def _attn(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # B*nW, window_size, window_size, C
        x_windows = einops.rearrange(
            x_windows, "BnW nH nW C -> BnW (nH nW) C"
        )  # B*nW, window_size*window_size, C
        if self.num_register_tokens > 0:
            nW = x_windows.shape[0] // B
            register_token = einops.repeat(register_token, "B R C -> (B nW) R C", nW=nW)
            x_windows = torch.concat((x_windows, register_token), dim=1)
            # B*nW, window_size*window_size+R, C

        # W-MSA/SW-MSA
        if getattr(self, "dynamic_mask", False):
            attn_mask = self.get_attn_mask(
                shifted_x, num_register_tokens=self.num_register_tokens
            )
        else:
            if hasattr(self, "scale"):
                attn_mask = self.attn_mask[self.scale]
            else:
                attn_mask = self.attn_mask
        # Attention step
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # B*nW, window_size*window_size+R, C

        if self.num_register_tokens > 0:
            register_token = attn_windows[
                :, -self.num_register_tokens :, :
            ]  # B*nW, R, C
            attn_windows = attn_windows[
                :, : -self.num_register_tokens, :
            ]  # B*nW, window_size*window_size, C
        # merge windows
        attn_windows = einops.rearrange(
            attn_windows,
            "BnW (wH wW) C -> BnW wH wW C",
            wH=self.window_size[0],
            wW=self.window_size[1],
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, (Hp, Wp)
        )  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x

        if self.num_register_tokens > 0:
            register_token = einops.reduce(
                register_token, "(B nW) R C -> B R C", reduction="mean", B=B
            )
        return x, register_token

    def forward(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if register_token is not None:
            R = register_token.shape[1]
        else:
            R = 0
        assert R == self.num_register_tokens

        x_att, register_token_att = self._attn(x, register_token)
        # B H W C -> B (HW+R) C
        if self.num_register_tokens > 0:
            fused = torch.concat((x.reshape(B, -1, C), register_token), dim=1)
            fused_att = torch.concat(
                (x_att.reshape(B, -1, C), register_token_att), dim=1
            )
        else:
            fused = x.reshape(B, -1, C)
            fused_att = x_att.reshape(B, -1, C)

        fused = fused + self.drop_path1(self.norm1(fused_att))
        fused = fused + self.drop_path2(self.norm2(self.mlp(fused)))
        x = fused[:, : H * W, :].reshape(B, H, W, C)
        if register_token is not None:
            register_token = fused[:, H * W :, :].reshape(B, R, C)

        return x, register_token


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: _int_or_tuple_2_t,
        depth: int,
        num_heads: int,
        window_size: _int_or_tuple_2_t,
        always_partition: bool = False,
        dynamic_mask: bool = False,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: _int_or_tuple_2_t = 0,
        output_nchw: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
            use_register_token: Whether to use register token
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # register token
        self.use_register_token = num_register_tokens > 0
        if self.use_register_token and downsample:
            self.downsample_token = nn.Linear(dim, out_dim)

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerV2Block(
                    dim=out_dim,
                    input_resolution=self.output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    always_partition=always_partition,
                    dynamic_mask=dynamic_mask,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    num_register_tokens=num_register_tokens,
                )
                for i in range(depth)
            ]
        )

    def set_input_size(
        self,
        feat_size: Tuple[int, int],
        window_size: int,
        always_partition: Optional[bool] = None,
    ):
        """Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def add_context_scale(
        self,
        feat_size,
        window_size=None,
        reg_token_fusion=None,
        share_patch_embed: Optional[bool] = False,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int or Tuple[int, int]): New image size for the context scale.
            patch_size (int, optional): Patch size. Defaults to None (uses existing patch size).
            window_size (int, optional): Window size. Defaults to None.
            reg_token_fusion (Callable, optional): Function to fuse register tokens from multiple scales. Defaults to None (mean fusion).
            share_patch_embed (bool, optional): Whether to share patch embedding between scales. Defaults to False.
        """
        if window_size is not None and isinstance(window_size, int):
            window_size = (window_size, window_size)
        context_window_size, context_shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.input_resolutions = [self.input_resolution, feat_size]
        self.window_size_list = [self.window_size, context_window_size]
        self.shift_size_list = [self.shift_size, context_shift_size]
        self.attn.add_context_scale(context_window_size)

        self.set_scale(1)  # switch to context scale
        self.register_buffer(
            "attn_mask_context",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )
        self.set_scale(0)  # switch back to main scale
        self.register_buffer(
            "attn_mask_main",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        if self.current_scale == scale:
            return
        self.current_scale = scale
        self.input_resolution = self.input_resolutions[scale]
        self.window_size = self.window_size_list[scale]
        self.window_area = self.window_size[0] * self.window_size[1]
        self.shift_size = self.shift_size_list[scale]
        self.attn.set_scale(scale)
        if hasattr(self, "attn_mask_main"):  # if context scale has been fully added
            self.attn_mask = (
                self.attn_mask_main if scale == 0 else self.attn_mask_context
            )

    def _attn(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # B*nW, window_size, window_size, C
        x_windows = einops.rearrange(
            x_windows, "BnW nH nW C -> BnW (nH nW) C"
        )  # B*nW, window_size*window_size, C
        if self.num_register_tokens > 0:
            nW = x_windows.shape[0] // B
            register_token = einops.repeat(register_token, "B R C -> (B nW) R C", nW=nW)
            x_windows = torch.concat((x_windows, register_token), dim=1)
            # B*nW, window_size*window_size+R, C

        # W-MSA/SW-MSA
        if getattr(self, "dynamic_mask", False):
            attn_mask = self.get_attn_mask(
                shifted_x, num_register_tokens=self.num_register_tokens
            )
        else:
            if hasattr(self, "scale"):
                attn_mask = self.attn_mask[self.scale]
            else:
                attn_mask = self.attn_mask
        # Attention step
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # B*nW, window_size*window_size+R, C

        if self.num_register_tokens > 0:
            register_token = attn_windows[
                :, -self.num_register_tokens :, :
            ]  # B*nW, R, C
            attn_windows = attn_windows[
                :, : -self.num_register_tokens, :
            ]  # B*nW, window_size*window_size, C
        # merge windows
        attn_windows = einops.rearrange(
            attn_windows,
            "BnW (wH wW) C -> BnW wH wW C",
            wH=self.window_size[0],
            wW=self.window_size[1],
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, (Hp, Wp)
        )  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x

        if self.num_register_tokens > 0:
            register_token = einops.reduce(
                register_token, "(B nW) R C -> B R C", reduction="mean", B=B
            )
        return x, register_token

    def forward(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if register_token is not None:
            R = register_token.shape[1]
        else:
            R = 0
        assert R == self.num_register_tokens

        x_att, register_token_att = self._attn(x, register_token)
        # B H W C -> B (HW+R) C
        if self.num_register_tokens > 0:
            fused = torch.concat((x.reshape(B, -1, C), register_token), dim=1)
            fused_att = torch.concat(
                (x_att.reshape(B, -1, C), register_token_att), dim=1
            )
        else:
            fused = x.reshape(B, -1, C)
            fused_att = x_att.reshape(B, -1, C)

        fused = fused + self.drop_path1(self.norm1(fused_att))
        fused = fused + self.drop_path2(self.norm2(self.mlp(fused)))
        x = fused[:, : H * W, :].reshape(B, H, W, C)
        if register_token is not None:
            register_token = fused[:, H * W :, :].reshape(B, R, C)

        return x, register_token


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: _int_or_tuple_2_t,
        depth: int,
        num_heads: int,
        window_size: _int_or_tuple_2_t,
        always_partition: bool = False,
        dynamic_mask: bool = False,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: _int_or_tuple_2_t = 0,
        output_nchw: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
            use_register_token: Whether to use register token
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # register token
        self.use_register_token = num_register_tokens > 0
        if self.use_register_token and downsample:
            self.downsample_token = nn.Linear(dim, out_dim)

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerV2Block(
                    dim=out_dim,
                    input_resolution=self.output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    always_partition=always_partition,
                    dynamic_mask=dynamic_mask,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    num_register_tokens=num_register_tokens,
                )
                for i in range(depth)
            ]
        )

    def set_input_size(
        self,
        feat_size: Tuple[int, int],
        window_size: int,
        always_partition: Optional[bool] = None,
    ):
        """Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def add_context_scale(
        self,
        feat_size,
        window_size=None,
        reg_token_fusion=None,
        share_patch_embed: Optional[bool] = False,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int or Tuple[int, int]): New image size for the context scale.
            patch_size (int, optional): Patch size. Defaults to None (uses existing patch size).
            window_size (int, optional): Window size. Defaults to None.
            reg_token_fusion (Callable, optional): Function to fuse register tokens from multiple scales. Defaults to None (mean fusion).
            share_patch_embed (bool, optional): Whether to share patch embedding between scales. Defaults to False.
        """
        if window_size is not None and isinstance(window_size, int):
            window_size = (window_size, window_size)
        context_window_size, context_shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.input_resolutions = [self.input_resolution, feat_size]
        self.window_size_list = [self.window_size, context_window_size]
        self.shift_size_list = [self.shift_size, context_shift_size]
        self.attn.add_context_scale(context_window_size)

        self.set_scale(1)  # switch to context scale
        self.register_buffer(
            "attn_mask_context",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )
        self.set_scale(0)  # switch back to main scale
        self.register_buffer(
            "attn_mask_main",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        if self.current_scale == scale:
            return
        self.current_scale = scale
        self.input_resolution = self.input_resolutions[scale]
        self.window_size = self.window_size_list[scale]
        self.window_area = self.window_size[0] * self.window_size[1]
        self.shift_size = self.shift_size_list[scale]
        self.attn.set_scale(scale)
        if hasattr(self, "attn_mask_main"):  # if context scale has been fully added
            self.attn_mask = (
                self.attn_mask_main if scale == 0 else self.attn_mask_context
            )

    def _attn(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # B*nW, window_size, window_size, C
        x_windows = einops.rearrange(
            x_windows, "BnW nH nW C -> BnW (nH nW) C"
        )  # B*nW, window_size*window_size, C
        if self.num_register_tokens > 0:
            nW = x_windows.shape[0] // B
            register_token = einops.repeat(register_token, "B R C -> (B nW) R C", nW=nW)
            x_windows = torch.concat((x_windows, register_token), dim=1)
            # B*nW, window_size*window_size+R, C

        # W-MSA/SW-MSA
        if getattr(self, "dynamic_mask", False):
            attn_mask = self.get_attn_mask(
                shifted_x, num_register_tokens=self.num_register_tokens
            )
        else:
            if hasattr(self, "scale"):
                attn_mask = self.attn_mask[self.scale]
            else:
                attn_mask = self.attn_mask
        # Attention step
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # B*nW, window_size*window_size+R, C

        if self.num_register_tokens > 0:
            register_token = attn_windows[
                :, -self.num_register_tokens :, :
            ]  # B*nW, R, C
            attn_windows = attn_windows[
                :, : -self.num_register_tokens, :
            ]  # B*nW, window_size*window_size, C
        # merge windows
        attn_windows = einops.rearrange(
            attn_windows,
            "BnW (wH wW) C -> BnW wH wW C",
            wH=self.window_size[0],
            wW=self.window_size[1],
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, (Hp, Wp)
        )  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x

        if self.num_register_tokens > 0:
            register_token = einops.reduce(
                register_token, "(B nW) R C -> B R C", reduction="mean", B=B
            )
        return x, register_token

    def forward(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if register_token is not None:
            R = register_token.shape[1]
        else:
            R = 0
        assert R == self.num_register_tokens

        x_att, register_token_att = self._attn(x, register_token)
        # B H W C -> B (HW+R) C
        if self.num_register_tokens > 0:
            fused = torch.concat((x.reshape(B, -1, C), register_token), dim=1)
            fused_att = torch.concat(
                (x_att.reshape(B, -1, C), register_token_att), dim=1
            )
        else:
            fused = x.reshape(B, -1, C)
            fused_att = x_att.reshape(B, -1, C)

        fused = fused + self.drop_path1(self.norm1(fused_att))
        fused = fused + self.drop_path2(self.norm2(self.mlp(fused)))
        x = fused[:, : H * W, :].reshape(B, H, W, C)
        if register_token is not None:
            register_token = fused[:, H * W :, :].reshape(B, R, C)

        return x, register_token


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: _int_or_tuple_2_t,
        depth: int,
        num_heads: int,
        window_size: _int_or_tuple_2_t,
        always_partition: bool = False,
        dynamic_mask: bool = False,
        downsample: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: _int_or_tuple_2_t = 0,
        output_nchw: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
            use_register_token: Whether to use register token
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # register token
        self.use_register_token = num_register_tokens > 0
        if self.use_register_token and downsample:
            self.downsample_token = nn.Linear(dim, out_dim)

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerV2Block(
                    dim=out_dim,
                    input_resolution=self.output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    always_partition=always_partition,
                    dynamic_mask=dynamic_mask,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    num_register_tokens=num_register_tokens,
                )
                for i in range(depth)
            ]
        )

    def set_input_size(
        self,
        feat_size: Tuple[int, int],
        window_size: int,
        always_partition: Optional[bool] = None,
    ):
        """Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def add_context_scale(
        self,
        feat_size,
        window_size=None,
        reg_token_fusion=None,
        share_patch_embed: Optional[bool] = False,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int or Tuple[int, int]): New image size for the context scale.
            patch_size (int, optional): Patch size. Defaults to None (uses existing patch size).
            window_size (int, optional): Window size. Defaults to None.
            reg_token_fusion (Callable, optional): Function to fuse register tokens from multiple scales. Defaults to None (mean fusion).
            share_patch_embed (bool, optional): Whether to share patch embedding between scales. Defaults to False.
        """
        if window_size is not None and isinstance(window_size, int):
            window_size = (window_size, window_size)
        context_window_size, context_shift_size = self._calc_window_shift(
            to_2tuple(window_size)
        )
        self.input_resolutions = [self.input_resolution, feat_size]
        self.window_size_list = [self.window_size, context_window_size]
        self.shift_size_list = [self.shift_size, context_shift_size]
        self.attn.add_context_scale(context_window_size)

        self.set_scale(1)  # switch to context scale
        self.register_buffer(
            "attn_mask_context",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )
        self.set_scale(0)  # switch back to main scale
        self.register_buffer(
            "attn_mask_main",
            (
                None
                if self.dynamic_mask
                else self.get_attn_mask(num_register_tokens=self.num_register_tokens)
            ),
            persistent=False,
        )

    def set_scale(self, scale: int):
        assert scale in [0, 1], "Scale must be 0 or 1"
        if self.current_scale == scale:
            return
        self.current_scale = scale
        self.input_resolution = self.input_resolutions[scale]
        self.window_size = self.window_size_list[scale]
        self.window_area = self.window_size[0] * self.window_size[1]
        self.shift_size = self.shift_size_list[scale]
        self.attn.set_scale(scale)
        if hasattr(self, "attn_mask_main"):  # if context scale has been fully added
            self.attn_mask = (
                self.attn_mask_main if scale == 0 else self.attn_mask_context
            )

    def _attn(self, x: torch.Tensor, register_token: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # B*nW, window_size, window_size, C
        x_windows = einops.rearrange(
            x_windows, "BnW nH nW C -> BnW (nH nW) C"
        )  # B*nW, window_size*window_size, C
        if self.num_register_tokens > 0:
            nW = x_windows.shape[0] // B
            register_token = einops.repeat(register_token, "B R C -> (B nW) R C", nW=nW)
            x_windows = torch.concat((x_windows, register_token), dim=1)
            # B*nW, window_size*window_size+R, C

        # W-MSA/SW-MSA
        if getattr(self, "dynamic_mask", False):
            attn_mask = self.get_attn_mask(
                shifted_x, num_register_tokens=self.num_register_tokens
            )
        else