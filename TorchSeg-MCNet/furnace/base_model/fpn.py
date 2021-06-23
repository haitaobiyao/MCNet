import torch.nn as nn
import torch.nn.functional as F
from seg_opr.seg_oprs import ConvBnRelu
import torch

class FPN(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__()
        self.start_level = 0
        self.backbone_end_level = 4
        in_channels = [256, 512, 1024, 2048]
        out_channel = 256
        self.dila_FCN = True
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvBnRelu(in_channels[i], out_channel, 1, 1, 0, 1,
                                has_bn=True, inplace=False,
                                has_relu=True, has_bias=False, norm_layer=norm_layer)
            fpn_conv = ConvBnRelu(out_channel, out_channel, 3, 1, 1, 1,
                                  has_bn=True, inplace=False,
                                  has_relu=True, has_bias=False, norm_layer=norm_layer)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        laterals = [
            laterals_conv(inputs[i+self.start_level])
            for i, laterals_conv in enumerate(self.lateral_convs)
        ]
        for i in range(3, 0, -1):
            if self.dila_FCN:
                if i == 3:
                    laterals[i - 1] += laterals[i]
                else:
                    laterals[i - 1] += F.interpolate(
                        laterals[i], scale_factor=2, mode='nearest')

        outs = [self.fpn_convs[i](laterals[i]) for i in range(4)]
        outs_temp = torch.cat((F.max_pool2d(outs[0], 1, stride=2), outs[1]), 1)
        out = torch.cat((F.max_pool2d(outs_temp, 1, stride=2), outs[2], outs[3]), 1)
        return out


