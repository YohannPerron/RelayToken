import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from src.models.components.utils.utils import set_first_layer


# Self made
class ConvModule(nn.Module):
    """A conv module with conv, norm and activation layers.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int | tuple[int]): Kernel size of conv layer.
        stride (int | tuple[int]): Stride of conv layer.
        padding (int | tuple[int]): Padding of conv layer.
        dilation (int | tuple[int]): Dilation of conv layer.
        groups (int): Groups of conv layer.
        bias (bool): Whether to add a bias term to the conv layer.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict | None): Config of activation layers.
    """
    # Implementation details omitted for brevity
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(ConvModule, self).__init__()
        assert conv_cfg is None, 'conv_cfg is not supported in ConvModule'
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=bias)
        if 'type' in norm_cfg:
            assert norm_cfg['type'] in ['BN', 'SyncBN']
            if norm_cfg['type'] == 'SyncBN':
                self.norm = nn.SyncBatchNorm(out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels, **norm_cfg.get('args', {}))
        else:
            self.norm = nn.Identity()
        if act_cfg is None:
            self.act = nn.ReLU(inplace=True)
        elif 'type' in act_cfg:
            act_layer = act_cfg.get('type', 'ReLU')
            if act_layer == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            else:
                raise NotImplementedError(f'Activation layer {act_layer} is not implemented.')

    def forward(self, x):
        """Forward function."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

# https://github.com/cedricgsh/ISDNet/blob/main/mmseg/models/decode_heads/aspp_head.py
class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs

# https://github.com/cedricgsh/ISDNet/blob/main/mmseg/models/decode_heads/aspp_refine_head.py
class RefineASPPHead(nn.Module):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), channels=256, in_channels=2048,num_classes=19, dropout=0.1,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, align_corners=False):
        super(RefineASPPHead, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        # x = inputs[3]
        x = inputs
        # 用来存储decoder的中间特征图
        fm_middle = []
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        # fm_middle: 初期验证，采用最后cls_seg之前最近的一个特征图
        fm_middle.append(output)
        output = self.dropout(output)
        output = self.cls_seg(output)
        fm_middle.append(output)
        return output, fm_middle


# https://github.com/cedricgsh/ISDNet/blob/main/mmseg/models/decode_heads/stdc_head.py
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, sync=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        if sync:
            self.bn = nn.SyncBatchNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


# From https://github.com/MichaelFan01/STDC-Seg
class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


# From https://github.com/MichaelFan01/STDC-Seg
class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

class ShallowNet(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, type="cat", dropout=0.20, pretrain_model=''):
        super(ShallowNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.in_channels = in_channels
        self.features = self._make_layers(base, layers, block_num, block)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'features.0.conv.weight' and self.in_channels != 3:
                assert self.in_channels % 2 == 0, "Input channels must be even"
                half_channels = self.in_channels // 2
                # extent v with random values to match half_channels
                rand = torch.rand(v.shape[0], half_channels - v.shape[1], v.shape[2], v.shape[3], device=v.device)
                v = torch.cat([v, rand], dim=1)
                v = torch.cat([v, v], dim=1)
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(self.in_channels, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8, feat16

# https://github.com/cedricgsh/ISDNet/blob/main/mmseg/models/decode_heads/isdhead.py
class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.register_buffer('kernel', self.gauss_kernel())

    def gauss_kernel(self, device=torch.device('cpu')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, n_classes=3, up_lists=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_lists[0])
        self.conv2 = ConvModule(channels // 2, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_lists[1])
        self.conv3 = ConvModule(channels // 2, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_lists[2])
        self.conv_sr = SegmentationHead(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, n_classes, kernel_size=1)

    def forward(self, x, fa=False):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)
        if fa:
            return feats, outs
        else:
            return outs

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten

class RelationAwareFusion(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
        super(RelationAwareFusion, self).__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_mlp = nn.Sequential(*[nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)
        c_att_split = c_att.view(b, self.r, c // self.r)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        out = self.smooth(s_feat + c_feat)
        return s_feat, c_feat, out

class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):

        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x

class ISDHead(nn.Module):
    def __init__(self, in_channels, down_ratio, prev_channels, reduce=False, channels=128,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, num_classes=19, dropout_ratio=0.1, index_to_ignore=255):
        super(ISDHead, self).__init__()
        self.down_ratio = down_ratio
        self.prev_channels = prev_channels
        self.channels = channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio

        self.fuse8 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
        self.fuse16 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=4)
        self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg,
                                    channels=self.channels, n_classes=in_channels, up_lists=[4, 2, 2])
        # shallow branch
        self.stdc_net = ShallowNet(in_channels=in_channels*2, pretrain_model="datasets/Models/STDCNet813M_73.91.tar")
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2, gauss_chl=in_channels)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg_aux_8 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                               self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                         self.channels // 2, self.num_classes, kernel_size=1)

        self.reduce = Reducer() if reduce else None

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(self.channels, self.num_classes, kernel_size=1, bias=True)
        )

        self.losses = nn.CrossEntropyLoss(ignore_index=index_to_ignore, reduction='mean')

    def forward(self, inputs, prev_output, train_flag=True):
        """Forward function."""
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        shallow_feat8, shallow_feat16 = self.stdc_net(high_residual_input)
        deep_feat = prev_output[0]
        if self.reduce is not None:
            deep_feat = self.reduce(deep_feat)
        # stage 1
        _, aux_feat16, fused_feat_16 = self.fuse16(shallow_feat16, deep_feat)
        # stage 2
        _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)
        output = self.cls_seg(fused_feat_8)
        output = resize(
            input=output,
            size=prymaid_results[0].shape[2:],
            mode='bilinear',
            align_corners=False)
        if train_flag:
            output_aux16 = self.conv_seg_aux_16(aux_feat8)
            output_aux8 = self.conv_seg_aux_8(aux_feat16)
            feats, output_sr = self.sr_decoder(deep_feat, True)
            losses_re = self.image_recon_loss(high_residual_1 + high_residual_2, output_sr, re_weight=0.1)
            losses_fa = self.feature_affinity_loss(deep_feat, feats)
            output_aux16 = resize(
                input=output_aux16,
                size=prymaid_results[0].shape[2:],
                mode='bilinear',
                align_corners=False)
            output_aux8 = resize(
                input=output_aux8,
                size=prymaid_results[0].shape[2:],
                mode='bilinear',
                align_corners=False)
            return output, output_aux16, output_aux8, losses_re, losses_fa
        else:
            return output

    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight
        loss['recon_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        loss = dict()
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, seg_logits_aux16, seg_logits_aux8, losses_recon, losses_fa = self.forward(inputs, prev_output)
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
        losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
        return seg_logits, losses, losses_aux16, losses_aux8, losses_recon, losses_fa

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output, False)

class ISDNet_model(nn.Module):
    def __init__(
            self,
            img_size=512,
            num_classes=4,
            num_channels=1,
            deep_backbone='resnet18',
            deep_pretrained_path=None,
            ASPP_head:RefineASPPHead=None,
            ISDHead:ISDHead=None,
            l_deep_segmentation:float=1.0,
            l_reconstruction:float=0.1,
            l_distillation:float=1.0,
    ):
        super(ISDNet_model, self).__init__()
        self.need_targets = True
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.deep_backbone = deep_backbone
        self.deep_pretrained_path = deep_pretrained_path
        self.ASPP_head = ASPP_head
        self.ISDHead = ISDHead

        self.l_deep_segmentation = l_deep_segmentation
        self.l_reconstruction = l_reconstruction
        self.l_distillation = l_distillation

        if self.deep_pretrained_path is not None:
            pretrained_cfg_overlay = dict(file=self.deep_pretrained_path)
        else:
            pretrained_cfg_overlay = None

        self.deep_backbone = timm.create_model(
                self.deep_backbone,
                pretrained=self.deep_pretrained_path is not None,
                in_chans=3,  # load as RGB
                pretrained_cfg_overlay=pretrained_cfg_overlay,
            )

        set_first_layer(self.deep_backbone, num_channels)

    def forward(self, x, metas=None, targets=None):
        train = self.training
        if train:
            assert targets is not None, "ISDNet_model requires targets for training."

        assert isinstance(x, torch.Tensor), "ISDNet_model doesn't support multiscale inputs."

        deep_features = self.deep_backbone.forward_features(x)

        deep_pred, fm_middle = self.ASPP_head(deep_features)

        if train:
            seg_logits, loss, loss_aux16, loss_aux8, loss_recon, loss_fa = self.ISDHead.forward_train(x, fm_middle, None, targets, None)
            deep_pred = resize(
                input=deep_pred,
                size=targets.size()[1:],
                mode='bilinear',
                align_corners=False)
            loss_deep = self.ISDHead.losses(deep_pred, targets)

            total_loss = loss + loss_recon['recon_losses'] * self.l_reconstruction + \
                loss_fa['fa_loss'] * self.l_distillation + loss_deep * self.l_deep_segmentation
            return {'loss': total_loss, 'out': seg_logits}

        else:
            seg_logits = self.ISDHead.forward_test(x, fm_middle, None, None)
            return {'loss': 0.0, 'out': seg_logits}
