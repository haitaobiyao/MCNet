#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn



def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr, smooth_dilation=False):
    group_decay = []
    group_no_decay = []
    zero_gradient = []  # when the code use position encoding
    if isinstance(module, nn.Parameter):  # gamma参数的学习过程
        group_no_decay.append(module)
        weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
        return weight_group
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        # position encoding
        elif isinstance(m, nn.Embedding):
            group_decay.append(m.weight)
            # zero_gradient.append(m.weight)

    # my add module for smooth dilation conv
    if smooth_dilation and hasattr(module, 'smooth2'):
        group_decay.append(module.smooth2.weight)
        group_decay.append(module.smooth3.weight)
        group_decay.append(module.smooth4.weight)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay) + len(zero_gradient)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
