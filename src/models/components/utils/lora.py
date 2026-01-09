import timm
import torch
import torch.nn as nn


class _LORA_linear(nn.Module):
    def __init__(
        self, old_linear, r, multiplicity=1, freeze_not_lora=True
    ) -> None:
        super().__init__()
        self.r = r
        self.factor = 16 / r
        self.multiplicity = multiplicity
        self.old_linear = old_linear

        self.in_layer = nn.ModuleList(
            [
                nn.Linear(self.old_linear.in_features, r, bias=False)
                for i in range(multiplicity)
            ]
        )
        self.out_layer = nn.ModuleList(
            [
                nn.Linear(
                    r, self.old_linear.out_features // multiplicity, bias=False
                )
                for i in range(multiplicity)
            ]
        )
        # init weights
        for l in self.out_layer:
            nn.init.zeros_(l.weight)
        if freeze_not_lora:
            self.freeze_not_lora()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_org = self.old_linear(x)
        x_lora = torch.cat(
            [
                self.out_layer[i](self.in_layer[i](x))
                for i in range(self.multiplicity)
            ],
            dim=-1,
        )
        return x_org + self.factor * x_lora

    def freeze_not_lora(self):
        for param in self.old_linear.parameters():
            param.requires_grad = False


def apply_lora_linear(attention, r, freeze_not_lora=True):
    applied = 0
    if hasattr(attention, "qkv"):
        attention.qkv = _LORA_linear(
            attention.qkv, r, 3, freeze_not_lora=freeze_not_lora
        )
        applied += 3
    if hasattr(attention, "kv"):
        attention.kv = _LORA_linear(
            attention.kv, r, 2, freeze_not_lora=freeze_not_lora
        )
        applied += 2
    if hasattr(attention, "q"):
        attention.q = _LORA_linear(
            attention.q, r, 1, freeze_not_lora=freeze_not_lora
        )
        applied += 1
    if hasattr(attention, "k"):
        attention.k = _LORA_linear(
            attention.k, r, 1, freeze_not_lora=freeze_not_lora
        )
        applied += 1
    if hasattr(attention, "v"):
        attention.v = _LORA_linear(
            attention.v, r, 1, freeze_not_lora=freeze_not_lora
        )
        applied += 1

    assert (
        applied == 3
    ), f"lora not applied to enough linear layer: {applied}/3 "


def apply_lora(model, r, assert_num_att=True):
    num_att = 0

    if isinstance(model, nn.ModuleDict):
        list = model.values()
    elif hasattr(model, "blocks"):
        list = model.blocks
    elif hasattr(model, "stages"):
        list = model.stages
    elif hasattr(model, "layers"):
        list = model.layers
    else:
        list = []
    for i, block in enumerate(list):
        if hasattr(block, "attn"):
            if isinstance(block.attn, nn.ModuleList):
                for attn_module in block.attn:
                    apply_lora_linear(attn_module, r)
                    num_att += 1
            else:
                apply_lora_linear(model.blocks[i].attn, r)
                num_att += 1
        elif isinstance(block, nn.Module):
            num_att += apply_lora(block, r, assert_num_att=False)

    if assert_num_att:
        assert num_att > 0, "Model must have attention blocks to apply lora"
        print(f"Number of attention blocks: {num_att}")
    return num_att
