"""Vision Transformer (ViT) in PyTorch


A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import copy
import logging
import math
from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from src.models.components.multiscaleDoubleNet import extract_and_upsample_center
from src.models.components.utils.seg_blocks import SimpleSegmentationHead
from src.models.components.utils.utils import (
    get_first_layer,
    infer_output,
    set_first_layer,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    AttentionPoolLatent,
    DropPath,
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    use_fused_attn,
)
from torch.jit import Final

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = nn.ModuleList([norm_layer(dim)])
        self.attn = nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )
            ]
        )
        self.ls1 = nn.ModuleList(
            [
                (
                    LayerScale(dim, init_values=init_values)
                    if init_values
                    else nn.Identity()
                )
            ]
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = nn.ModuleList([norm_layer(dim)])
        self.mlp = nn.ModuleList(
            [
                mlp_layer(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    bias=proj_bias,
                    drop=proj_drop,
                )
            ]
        )
        self.ls2 = nn.ModuleList(
            [
                (
                    LayerScale(dim, init_values=init_values)
                    if init_values
                    else nn.Identity()
                )
            ]
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def add_scale_specific_params(self):
        """Add parameters for additional scale if they don't exist already"""
        if len(self.norm1) < 2:
            dim = self.norm1[0].normalized_shape[0]
            # Clone all parameters for the second scale
            self.norm1.append(copy.deepcopy(self.norm1[0]))
            self.attn.append(copy.deepcopy(self.attn[0]))
            self.ls1.append(copy.deepcopy(self.ls1[0]))
            self.norm2.append(copy.deepcopy(self.norm2[0]))
            self.mlp.append(copy.deepcopy(self.mlp[0]))
            self.ls2.append(copy.deepcopy(self.ls2[0]))

    def forward(
        self,
        x: List[torch.Tensor],
        reg_token: torch.Tensor,
        num_scales: int = 1,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        R = reg_token.shape[1] if reg_token is not None else 0
        # If we have multiple scales but not enough parameters, add them

        # Default merge pattern - process all scales and merge register tokens
        reg_tokens_list = []
        x_out = []  # Create new list to avoid in-place updates

        for i in range(num_scales):
            idx = min(
                i, len(self.norm1) - 1
            )  # Use appropriate parameter set
            val = x[i]
            val = torch.concat([reg_token, val], dim=1)
            val = val + self.drop_path1(
                self.ls1[idx](self.attn[idx](self.norm1[idx](val)))
            )
            val = val + self.drop_path2(
                self.ls2[idx](self.mlp[idx](self.norm2[idx](val)))
            )
            x_out.append(val[:, R:])
            reg_tokens_list.append(val[:, :R])

        if num_scales > 1:
            reg_token = torch.stack(reg_tokens_list, dim=0).mean(dim=0)
        else:
            reg_token = reg_tokens_list[0]

        return x_out, reg_token


def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal[
            "", "avg", "avgmax", "max", "token", "map"
        ] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max")
            if fc_norm is None
            else fc_norm
        )
        norm_layer = get_norm_layer(norm_layer) or partial(
            nn.LayerNorm, eps=1e-6
        )
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim))
            if reg_tokens
            else None
        )
        embed_len = (
            num_patches
            if no_embed_class
            else num_patches + self.num_prefix_tokens
        )
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nn.ParameterList(
                [nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)]
            )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(depth)
        ]
        self.norm = (
            norm_layer(embed_dim)
            if final_norm and not use_fc_norm
            else nn.Identity()
        )

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = (
            norm_layer(embed_dim)
            if final_norm and use_fc_norm
            else nn.Identity()
        )
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.is_multiscale = False
        self.num_scales = 1  # Initialize with 1 scale

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, "set_grad_checkpointing"):
            self.patch_embed.set_grad_checkpointing(enable)

    def set_input_size(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[Tuple[int, int]] = None,
    ):
        """Method updates the input image resolution, patch size

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(
            img_size=img_size, patch_size=patch_size
        )
        if self.pos_embed[0] is not None:
            num_prefix_tokens = (
                0 if self.no_embed_class else self.num_prefix_tokens
            )
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed[0].shape[1]:
                self.pos_embed[0] = nn.Parameter(
                    resample_abs_pos_embed(
                        self.pos_embed[0],
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    )
                )

    def add_context_scale(
        self,
        img_size,
        patch_size=None,
        share_patch_embed=False,
        share_scale_parameters=True,
    ):
        """Add a new scale to the model.

        Args:
            img_size (int or tuple): New image size for context scale.
            patch_size (int or tuple, optional): Patch size. Defaults to None (uses existing).
            share_patch_embed (bool, optional): Whether to share patch embedding between scales. Defaults to False.
            share_scale_parameters (bool, optional): Whether to share transformer parameters between scales. Defaults to True.
        """
        self.single_patch_embed = share_patch_embed
        if not share_patch_embed:
            if patch_size is None:
                patch_size = self.patch_embed.patch_size
            main_patch_embeded = copy.deepcopy(self.patch_embed)
            context_patch_embeded = self.patch_embed
            context_patch_embeded.set_input_size(
                img_size=img_size, patch_size=patch_size
            )
            new_grid_size = context_patch_embeded.grid_size
            self.patch_embed = nn.ModuleList(
                [main_patch_embeded, context_patch_embeded]
            )

            old_pos_embed = self.pos_embed[0]
            new_pos_embed = nn.Parameter(
                resample_abs_pos_embed(
                    self.pos_embed[0],
                    new_size=new_grid_size,
                    old_size=main_patch_embeded.grid_size,
                    num_prefix_tokens=self.num_prefix_tokens,
                    verbose=True,
                )
            )

            self.pos_embed = nn.ParameterList([old_pos_embed, new_pos_embed])

        self.is_multiscale = True
        self.num_scales = 2  # Update to 2 scales after adding context scale

        # Add parameters for the new scale
        if not share_scale_parameters:
            print("not sharing parameters between scales")
            for block in self.blocks:
                block.add_scale_specific_params()

    def _pos_embed(self, x: torch.Tensor, scale: int = 0) -> torch.Tensor:
        multiscale = self.is_multiscale
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = (
                self.patch_embed.grid_size
                if not multiscale
                else (self.patch_embed[scale].grid_size)
            )
            pos_embed = resample_abs_pos_embed(
                self.pos_embed[scale],
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=(
                    0 if self.no_embed_class else self.num_prefix_tokens
                ),
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed[scale]

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        R = self.reg_token.shape[1] if self.reg_token is not None else 0

        if isinstance(x, Iterable) and not isinstance(x, torch.Tensor):
            x = x  # [high_res, low_res]
        else:
            x = [x]  # [high_res]
            self.num_scales = 1

        x_patch = []
        for i in range(
            self.num_scales
        ):  # Use self.num_scales instead of len(x)
            x_tmp = self.patch_drop(x[i])
            if self.is_multiscale and not self.single_patch_embed:
                x_tmp = self.patch_embed[i](x_tmp)
                x_tmp = self._pos_embed(x_tmp, scale=i)
            else:
                x_tmp = self.patch_embed(x_tmp)
                x_tmp = self._pos_embed(x_tmp)
            x_tmp = self.norm_pre(x_tmp)
            reg_token = x_tmp[:, :R, :]
            x_patch.append(x_tmp[:, R:, :])
        x = x_patch

        # Process through transformer blocks, passing num_scales
        for block in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, reg_token = torch.utils.checkpoint.checkpoint(
                    block, x, reg_token, self.num_scales, use_reentrant=False
                )
            else:
                x, reg_token = block(x, reg_token, self.num_scales)

        x = self.norm(x)
        return x

    def pool(
        self, x: torch.Tensor, pool_type: Optional[str] = None
    ) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens
        )
        return x

    def forward_head(
        self, x: torch.Tensor, pre_logits: bool = False
    ) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    out_dict = {}

    for k, v in state_dict.items():
        if any([n in k for n in ("attn_mask",)]):
            continue  # skip buffers that should not be persistent

        if k.startswith("pos_embed"):
            HW = v.shape[1]
            H = int(math.sqrt(HW))
            if H * H != HW:
                # class token
                if H * H == HW - 1:
                    cls_token = True
                else:
                    raise ValueError(f"pos_embed shape {v.shape} not square")
            else:
                cls_token = False
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                old_size=(H, H),
                num_prefix_tokens=(1 if cls_token else 0),
            )
            # add register tokens
            cls_token = 1 if cls_token else 0
            tokens_to_add = model.num_prefix_tokens - cls_token
            if tokens_to_add > 0:
                # get from model tokens
                tokens = model.pos_embed[0][
                    :, cls_token : cls_token + tokens_to_add, :
                ]
                if cls_token:
                    v = torch.cat([v[:, :1], tokens, v[:, 1:]], dim=1)
                else:
                    v = torch.cat([tokens, v], dim=1)
            k = "pos_embed.0"
        out_dict[k] = v

    return out_dict


class RT_ViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_channels=1,
        segmentation_head=SimpleSegmentationHead,
        pretrained=False,
        pretrained_path=None,
        img_size=512,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=0,
        multi_scale=False,
        share_patch_embed=False,
        class_token=False,
        share_scale_parameters=True,
    ):
        """Initialize the RT_ViT model.

        Args:
            num_classes (int): Number of output classes. Default is 4.
            num_channels (int or tuple): Number of input channels. Default is 1. If multi_scale is True, must be a tuple of 2 integers.
            segmentation_head (nn.Module): Segmentation head class. Default is SimpleSegmentationHead.
            pretrained (bool): Whether to use pretrained weights. Default is False.
            pretrained_path (str, optional): Path to the pretrained weights file. Default is None.
            img_size (int or tuple): Size of the input image(s). Default is 512. If multi_scale is True, must be a tuple of 2 integers.
            patch_size (int): Patch size. Default is 8.
            embed_dim (int): Embedding dimension. Default is 768.
            depth (int): Number of transformer blocks. Default is 12.
            num_heads (int): Number of attention heads. Default is 12.
            mlp_ratio (float): MLP expansion ratio. Default is 4.0.
            num_register_tokens (int): Number of register tokens. Default is 0.
            multi_scale (bool): Whether to use multi-scale processing. Default is False.
            share_patch_embed (bool): Whether to share patch embedding between scales. Default is False.
            class_token (bool): Whether to use a class token. Default is False.
            share_scale_parameters (bool): Whether to share transformer block parameters between scales. Default is True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.patchsize = patch_size
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.multi_scale = multi_scale

        if isinstance(img_size, Iterable) and not self.multi_scale:
            self.img_size = img_size[0]
        else:
            self.img_size = img_size

        if self.multi_scale:
            assert len(img_size) == 2, "Multi-scale requires two image sizes."
            assert (
                len(num_channels) == 2
            ), "Multi-scale requires two channel number (on per scale)"

        self.model = VisionTransformer(
            img_size=self.img_size[0] if self.multi_scale else self.img_size,
            patch_size=self.patchsize,
            in_chans=3,
            num_classes=1000,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_token=class_token,
            global_pool="avg",
            reg_tokens=num_register_tokens,
        )

        if pretrained:
            assert (
                pretrained_path is not None
            ), "Pretrained weights path is required."
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(
                checkpoint_filter_fn(checkpoint, self.model), strict=False
            )

        if num_register_tokens > 0:
            assert self.model.reg_token.shape[1] == num_register_tokens
        else:
            assert self.model.reg_token is None

        if multi_scale:
            self.model.add_context_scale(
                self.img_size[1],
                share_patch_embed=share_patch_embed,
                share_scale_parameters=share_scale_parameters,
            )

            if not share_patch_embed:
                set_first_layer(
                    self.model.patch_embed[0], num_channels[0]
                )  # convert to correct number of features
                set_first_layer(self.model.patch_embed[1], num_channels[1])
            else:
                set_first_layer(self.model.patch_embed, num_channels[0])
        else:
            set_first_layer(self.model.patch_embed, num_channels)

        # if use_FPN:
        #     # Add FPN
        #     self.model = FPN(self.model, self.num_channels, self.img_size)

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
            temporal_dim=[0,0],
            indices=[0],
        )
        print(self.embed_dim, self.downsample_factor, self.feature_size)
        # Add segmentation head
        self.seg_head = segmentation_head(
            self.embed_dim,
            self.downsample_factor,
            self.remove_cls_token,
            self.features_format,
            self.feature_size,
            self.num_classes,
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

            return {
                "out": x_main_pred,
                "out_context": x_cont_pred,
                "embed": x_main,
                "embed_context": x_cont,
            }
        else:
            x = self.seg_head(x[0])
            return {"out": x, "embed": x}
