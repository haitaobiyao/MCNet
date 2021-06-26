from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50, resnet101
from seg_opr.seg_oprs import ConvBnRelu, ShareSeqConv
import numpy as np

class MCNet(nn.Module):
    def __init__(self, out_planes, criterion, edge_criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(MCNet, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=False, stem_width=64)
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))  # output_stride=16
        self.business_layer = []
        self.decoder_level1 = Decoder_level1(out_planes, norm_layer=norm_layer)
        self.meem = MEEM(norm_layer=norm_layer)
        self.cam_1 = CAM1(norm_layer=norm_layer)
        self.cmcm = CMCM()
        self.decoder_layer_level2 = Decoder_level2(out_planes, norm_layer=norm_layer)
        self.business_layer.append(self.decoder_level1)
        self.business_layer.append(self.meem)
        self.business_layer.append(self.cam_1)
        self.business_layer.append(self.decoder_layer_level2)
        self.criterion = criterion
        self.edge_criterion = edge_criterion

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)
        low_level_feat = blocks[-4]
        x = blocks[-1]
        road_attention, energy, proj_query, proj_key, proj_value, road_reduce = self.cam_1(x)
        edge_feature, edge_score = self.meem(low_level_feat, x, road_attention)
        x = self.decoder_level1(low_level_feat, edge_feature, road_attention)

        level1_seg_result = F.interpolate(x, size=data.size()[2:], mode='bilinear', align_corners=True)
        '''
        Level 2 model
        '''
        refine_energy, refine_attention_map = self.cmcm(level1_seg_result, energy)
        x_second = self.decoder_layer_level2(low_level_feat, edge_feature, blocks[-1], proj_query, proj_key, proj_value, road_reduce, refine_attention_map)
        level2_seg_result = F.interpolate(x_second, size=data.size()[2:], mode='bilinear', align_corners=True)
        edge_score = F.interpolate(edge_score, size=data.size()[2:], mode='bilinear', align_corners=True)
        if label is not None and aux_label is not None:
            level1_seg_loss = self.criterion(level1_seg_result, label)
            level2__seg_loss = self.criterion(level2_seg_result, label)
            edge_loss = self.edge_criterion(edge_score, aux_label)
            return level1_seg_loss + edge_loss*5 + level2__seg_loss, edge_loss*5, level1_seg_loss, level2__seg_loss
        return F.log_softmax(level2_seg_result, dim=1)

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class Decoder_level1(nn.Module):
    def __init__(self, outplanes, norm_layer=nn.BatchNorm2d):
        super(Decoder_level1, self).__init__()
        low_level_inplanes = 256
        self.conv1 = ConvBnRelu(low_level_inplanes, 48, 1, 1, 0,
                                has_bn=True,
                                has_relu=True, has_bias=False, norm_layer=norm_layer)

        self.last_conv = nn.Sequential(ConvBnRelu(560, 256, 3, 1, 1,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer),
                                       ConvBnRelu(256, 256, 3, 1, 1,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer),
                                       nn.Dropout(0.1, inplace=False),
                                       nn.Conv2d(256, outplanes, kernel_size=1, stride=1)
                                       )

    def forward(self, low_level_feat, edge_feature, road_attention):
        low_level_feat = self.conv1(low_level_feat)
        road_attention = F.interpolate(road_attention, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((low_level_feat, road_attention), dim=1)
        x = torch.cat((x, edge_feature), dim=1)
        x = self.last_conv(x)

        return x

class Decoder_level2(nn.Module):
    def __init__(self, outplanes, norm_layer=nn.BatchNorm2d):
        super(Decoder_level2, self).__init__()
        low_level_inplanes = 256
        self.conv1 = ConvBnRelu(low_level_inplanes, 48, 1, 1, 0,
                                has_bn=True,
                                has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.last_conv = nn.Sequential(ConvBnRelu(560, 256, 3, 1, 1,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer),
                                       ConvBnRelu(256, 256, 3, 1, 1,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer),
                                       nn.Dropout(0.1, inplace=False),
                                       nn.Conv2d(256, outplanes, kernel_size=1, stride=1)
                                       )

    def forward(self, low_level_feat, edge_feature, high_level_feat, proj_query, proj_key, proj_value, road_reduce, refine_attention_map):
        low_level_feat = self.conv1(low_level_feat)
        refine_road_feat = CAM2(proj_query, proj_key, proj_value, road_reduce, refine_attention_map)
        road_attention = F.interpolate(refine_road_feat, size=low_level_feat.size()[2:], mode='bilinear',
                                       align_corners=True)
        x = torch.cat((low_level_feat, road_attention), dim=1)
        x = torch.cat((x, edge_feature), dim=1)
        x = self.last_conv(x)

        return x

class MEEM(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(MEEM, self).__init__()
        self.toptodown = nn.Sequential(ConvBnRelu(2048, 512, 1, 1, 0,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer),
                                       ConvBnRelu(512, 256, 1, 1, 0,
                                                  has_bn=True,
                                                  has_relu=True, has_bias=False, norm_layer=norm_layer)
                                       )
        self.roadfeat_down = ConvBnRelu(256, 256, 3, 1, 1,
                                        has_bn=True,
                                        has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.reduce_concat = ConvBnRelu(512, 256, 3, 1, 1,
                                        has_bn=True,
                                        has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.fusion_layer = ConvBnRelu(256, 256, 3, 1, 1,
                                       has_bn=True,
                                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.score_layer = nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, low_feat, high_feat, road_feat):
        tran_feat = self.toptodown(high_feat)
        roadarea_feat = self.roadfeat_down(road_feat)
        cat_feat = self.reduce_concat(torch.cat((tran_feat, roadarea_feat), dim=1))
        edge_feature = low_feat + F.interpolate(cat_feat, low_feat.size()[2:], mode='bilinear', align_corners=True)
        edge_feature = self.fusion_layer(edge_feature)
        edge_score = self.score_layer(edge_feature)
        return edge_feature, edge_score


class CMCM(nn.Module):
    def __init__(self):
        super(CMCM, self).__init__()

    def forward(self, label, energy):
        label = F.log_softmax(label, dim=1)
        b, c, h, w = label.size()
        attention_b, attention_pixel, attention_map = energy.size()
        downsample = nn.AdaptiveAvgPool2d((int(h/16), int(w/16)))
        label = downsample(label)
        label = label.argmax(1)
        label = label[:, 10:20, :]
        label = label.view(b, -1)
        label = label.detach().cpu().numpy()
        energy = energy.detach().cpu().numpy()
        for i in range(0, b):
            single_label = label[i, :]
            for j in range(0, attention_pixel):
                index = np.where(single_label != label[i, j])
                pos_index = np.where(single_label == label[i, j])
                for k in range(0, index[0].shape[0]):
                    if energy[i, j, index[0][k]] > 0:
                        energy[i, j, index[0][k]] = -0.5
                for t in range(0, pos_index[0].shape[0]):
                    if energy[i, j, pos_index[0][t]] < 0:
                        energy[i, j, pos_index[0][t]] = 0.5

        energy = torch.from_numpy(energy).cuda()
        attention_map = torch.softmax(energy, dim=-1)
        return energy, attention_map


class CAM1(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CAM1, self).__init__()
        self.collect_reduction = ConvBnRelu(2048, 256, 3, 1, 1,
                                            has_bn=True, has_relu=True,
                                            has_bias=False,
                                            norm_layer=norm_layer)
        self.query = ConvBnRelu(256, 64, 1, 1, 0,
                                has_bn=False, has_relu=False,
                                has_bias=False,
                                norm_layer=norm_layer)
        self.key = ConvBnRelu(256, 64, 1, 1, 0,
                              has_bn=False, has_relu=False,
                              has_bias=False,
                              norm_layer=norm_layer)
        self.value = ConvBnRelu(256, 256, 1, 1, 0,
                                has_bn=False, has_relu=False,
                                has_bias=False,
                                norm_layer=norm_layer)

    def forward(self, x):
        collect_reduce_x = self.collect_reduction(x)
        road_refine_feature = self.roadAttention(collect_reduce_x)
        return road_refine_feature

    def roadAttention(self, road_reduce):
        b, c, h, w = road_reduce.size()
        finnaly_tensor = torch.FloatTensor(np.zeros((b, c, h, w))).cuda()
        proj_query = self.query(road_reduce)
        proj_key = self.key(road_reduce)
        road_proj_query = proj_query[:, :, 10:20, :].view(b, -1, w*10).permute(0, 2, 1)
        road_proj_key = proj_key[:, :, 10:20, :].view(b, -1, w*10)
        energy = torch.bmm(road_proj_query, road_proj_key)
        attention_map = torch.softmax(energy, dim=-1)
        proj_value = self.value(road_reduce)[:, :, 10:20, :].view(b, -1, w*10)
        out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        finnaly_tensor[:, :, 10:20, :] = out.view(b, c, 10, w)
        finnaly_tensor = finnaly_tensor + road_reduce
        return finnaly_tensor, energy, proj_query, proj_key, proj_value, road_reduce

def CAM2(proj_query, proj_key, proj_value, road_reduce, refine_attention_map):
    b, c, h, w = road_reduce.size()
    finnaly_tensor = torch.FloatTensor(np.zeros((b, c, h, w))).cuda()
    road_query = proj_query[:, :, 10:20, :].view(b, -1, w * 10).permute(0, 2, 1)
    road_key = proj_key[:, :, 10:20, :].view(b, -1, w * 10)
    out = torch.bmm(proj_value, refine_attention_map.permute(0, 2, 1))
    finnaly_tensor[:, :, 10:20, :] = out.view(b, c, 10, w)
    finnaly_tensor = finnaly_tensor + road_reduce
    return finnaly_tensor














