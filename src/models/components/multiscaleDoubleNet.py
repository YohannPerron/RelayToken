import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.utils.lora import apply_lora
from src.models.components.utils.utils import load_state_dict, set_first_layer
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class DoubleNet(nn.Module):
    def __init__(
        self,
        net_high_res: nn.Module,
        net_low_res: nn.Module,
        net_low_res_scale: int,
        chkpt_net_high_res: str = None,
        chkpt_net_low_res: str = None,
        img_size: int = 224,
        num_channels: int = 4,
        num_classes: int = 1,
    ):
        super().__init__()
        self.net_high_res = net_high_res
        self.net_low_res = net_low_res
        if net_low_res is None:
            # We use the same network for high and low resolution
            self.net_low_res = net_high_res

        self.net_low_res_scale = net_low_res_scale
        if chkpt_net_high_res is not None:
            self.net_high_res.load_state_dict(
                load_state_dict(chkpt_net_high_res, "net")
            )
        if chkpt_net_low_res is not None:
            assert (
                net_low_res is not None
            ), "net_low_res must be provided to load a checkpoint"
            self.net_low_res.load_state_dict(
                load_state_dict(chkpt_net_low_res, "net")
            )

    def forward(self, x, metas=None):
        x_high_res = self.net_high_res(x[0])["out"]
        x_low_res = self.net_low_res(x[1])["out"]

        # Use the helper function to extract and upsample the center patch
        x_low_res = extract_and_upsample_center(
            x_low_res, x_high_res.size()[2:], self.net_low_res_scale
        )

        return {"out": x_high_res + x_low_res}


def extract_and_upsample_center(x_low_res, target_size, scale):
    """
    Extract the center patch from low resolution feature map and upsample it.

    Args:
        x_low_res: Low resolution feature map
        target_size: Target size to upsample to (high resolution size)
        scale: Scale factor for center extraction

    Returns:
        Upsampled center patch
    """
    # Extract center patch from x_low_res (BCHW)
    extracted_size = (
        x_low_res.shape[2] // (2 * scale),
        x_low_res.shape[3] // (2 * scale),
    )
    center = (
        x_low_res.shape[2] // 2,
        x_low_res.shape[3] // 2,
    )
    x_low_res = x_low_res[
        :,
        :,
        center[0] - extracted_size[0] : center[0] + extracted_size[0],
        center[1] - extracted_size[1] : center[1] + extracted_size[1],
    ]

    # Upsample center to match target size
    x_low_res = F.interpolate(
        x_low_res,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return x_low_res
