from dataclasses import dataclass, field

import torch.nn as nn

from src.models.components.utils.seg_blocks import SimpleSegmentationHead
from src.models.components.utils.utils import infer_output, set_first_layer

from . import backbones
from .context_encoders import ContextEncoderConfig
from .decoders.decoder import xT


@dataclass
class BackboneConfig:
    """Configuration for feature extracting backbone."""

    in_chans: int = 3
    """Number of channels in input data."""
    input_dim: int = 2
    """Input dimension."""
    drop_path_rate: float = 0.0
    """Drop path rate for stochastic depth."""
    pretrained: str = ""
    """Path to pretrained weights, empty for none."""
    channel_last: bool = True
    """If channels are last in data format."""
    input_size: int = 256
    """Expected input size of data."""


@dataclass
class ModelConfig:
    name: str = "xT"
    """Name of overarching model architecture."""
    resume: str = ""
    """Path to checkpoint to resume training from. Empty for none."""
    tiling: str = "naive"
    """Transformer-XL tiling strategy"""
    backbone_class: str = "swinv2_tiny_window16_256_timm"
    """Class name for backbone."""
    patch_size: int = 16
    """Patch size used for transformer XL."""  # TODO: properly derive this
    num_classes: int = 9999
    cls_head: str = "naive"
    """Number of classes for head on dataset."""
    mlp_ratio: int = 4
    """MLP ratio for Enc/Dec."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    context: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)


def build_model(config: ModelConfig, dataset: str = "inaturalist"):
    backbone_class = config.backbone_class
    backbone = backbones.__dict__[backbone_class](**vars(config.backbone))

    if config.name == "xT":
        model = xT(
            backbone=backbone,
            xl_config=config.context,
            channels_last=config.backbone.channel_last,
            crop_size=config.backbone.input_size,
            skip_decoder=False,
            backbone_name=config.backbone_class,
            dataset=dataset,
            num_classes=config.num_classes,
            mlp_ratio=config.mlp_ratio,
            cls_head=config.cls_head,
        )
    return model


class SegmentationXT(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        img_size,
        encoder_img_size,
        patch_size,
        mlp_ratio,
        backbone_name,
        decoder_name,
        pretrained,
        pretrained_path,
        segmentation_head=SimpleSegmentationHead,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.encoder_img_size = encoder_img_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.backbone_name = backbone_name
        self.decoder_name = decoder_name
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path

        backbone_config = BackboneConfig(
            in_chans=num_channels,
            input_size=encoder_img_size,
        )

        if not pretrained:
            pretrained_path = ""
        model_config = ModelConfig(
            name=decoder_name,
            resume=pretrained_path,
            backbone_class=backbone_name,
            backbone=backbone_config,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            cls_head="xl",
        )
        self.model = build_model(model_config)

        # if features_only is True, map the output to forward_features
        if not hasattr(self.model, "forward_features"):
            self.model.forward_features = self.model.forward

        set_first_layer(self.model, num_channels)

        (
            self.embed_dim,
            self.downsample_factor,
            self.feature_size,
            self.features_format,
            self.remove_cls_token,
        ) = infer_output(self.model, self.num_channels, self.img_size)

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
        x = self.model.forward(x)

        x = self.seg_head(x)
        return {"out": x}
